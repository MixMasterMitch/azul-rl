"""Mixed-table games and fixed-weight, seed-blocked tournament summaries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import permutations
import math
import time
from typing import Callable

import numpy as np
import torch

from .arena import paired_interval
from .builtin_opponents import BUILTIN_BOTS
from .multiplayer import outcome_record
from ..env.engine import GameEngine
from ..env.outcomes import final_values_binary
from ..net.encoder import encode_state
from ..search.config import SearchConfig
from ..search.gumbel_mcts import gumbel_root_act
from ..train.reanalysis import SnapshotRecorder, finished_sample_mask
from ..train.replay_buffer import ReplayBuffer
from ..train.selfplay import _rotate_for_cp, _default_temperature_schedule


@dataclass(frozen=True)
class Table:
    name: str
    opponents: tuple[str, ...]
    weight: float

    def __post_init__(self) -> None:
        if not self.name or len(self.opponents) not in (1, 2, 3):
            raise ValueError("A table needs a name and one to three opponents")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Table weights must be finite and positive")
        if "candidate" in self.opponents:
            raise ValueError("candidate is reserved for the replaced seat")


def default_tables(players: int) -> list[Table]:
    """Frozen panel weights, independent of results and allocated game counts."""
    if players == 2:
        return [
            Table("specialist", ("specialist",), 0.5),
            Table("astra", ("astra",), 0.35),
            Table("opus", ("opus",), 0.15),
        ]
    if players == 3:
        return [
            Table("astra_specialist", ("astra", "specialist"), 0.4),
            Table("astra_opus", ("astra", "opus"), 0.3),
            Table("specialist_opus", ("specialist", "opus"), 0.2),
            Table("double_astra", ("astra", "astra"), 0.1),
        ]
    if players == 4:
        return [
            Table("astra_opus_specialist", ("astra", "opus", "specialist"), 0.4),
            Table("double_astra_specialist", ("astra", "astra", "specialist"), 0.3),
            Table("astra_double_opus", ("astra", "opus", "opus"), 0.2),
            Table(
                "double_specialist_astra", ("specialist", "specialist", "astra"), 0.1
            ),
        ]
    raise ValueError("Unsupported player count")


def seat_assignments(table: Table) -> list[tuple[str, ...]]:
    """All distinct seat permutations, including every opponent ordering."""
    return sorted(set(permutations(("candidate", *table.opponents))))


def summarize_table(records: list[dict], table: Table) -> dict:
    expected = set(seat_assignments(table))
    grouped: dict[int, list[dict]] = {}
    for row in records:
        grouped.setdefault(row["group_seed"], []).append(row)
    if not records or any(
        len(rows) != len(expected)
        or {tuple(r["seats"]) for r in rows} != expected
        or any(r["outcome"] == "unfinished" for r in rows)
        for rows in grouped.values()
    ):
        raise ValueError(
            "Table summary requires complete, finished seat-permutation blocks"
        )
    scores = {
        str(seed): float(np.mean([r["win_share"] for r in rows]))
        for seed, rows in sorted(grouped.items())
    }
    return {
        "games": len(records),
        "unfinished": 0,
        "win_share": float(np.mean(list(scores.values()))),
        "win_share_ci95": paired_interval(list(scores.values())),
        "sole_winrate": sum(r["outcome"] == "win" for r in records) / len(records),
        "shared_wins": sum(r["outcome"] == "shared" for r in records),
        "group_scores": scores,
        "table": asdict(table),
    }


def weighted_summary(
    reports: dict[str, dict],
    tables: list[Table],
    reference: dict[str, dict] | None = None,
) -> dict:
    """Bootstrap shared seed blocks jointly across tables; optionally pair models.

    No renormalization after missing/unfinished games and no adaptive weights.
    The returned difference interval is descriptive, not a noninferiority verdict.
    """
    if (
        not tables
        or len({t.name for t in tables}) != len(tables)
        or set(reports) != {t.name for t in tables}
    ):
        raise ValueError("Missing or duplicate tables")
    if reference is not None and set(reference) != set(reports):
        raise ValueError("Reference panel differs")
    keys = set(reports[tables[0].name]["summary"]["group_scores"])
    if len(keys) < 2:
        raise ValueError("At least two independent seed blocks required")
    total = sum(t.weight for t in tables)
    combined = {k: 0.0 for k in keys}
    for table in tables:
        report = reports[table.name]
        if report["table"] != asdict(table):
            # JSON round trips turn tuples into lists.
            if (
                report["table"]["name"],
                tuple(report["table"]["opponents"]),
                report["table"]["weight"],
            ) != (table.name, table.opponents, table.weight):
                raise ValueError("Table identity/weight changed")
        values = report["summary"]["group_scores"]
        if set(values) != keys or report["summary"]["unfinished"]:
            raise ValueError("Seed blocks differ or games are unfinished")
        if reference is not None:
            other = reference[table.name]
            if (
                other["num_players"],
                other["search"],
                other["identities"],
                other.get("opponent_search"),
                tuple(other["table"]["opponents"]),
            ) != (
                report["num_players"],
                report["search"],
                report["identities"],
                report.get("opponent_search"),
                tuple(table.opponents),
            ):
                raise ValueError(
                    "Paired evaluation settings or opponent identities differ"
                )
            other_values = other["summary"]["group_scores"]
            if set(other_values) != keys or other["summary"]["unfinished"]:
                raise ValueError("Reference seed blocks differ or games are unfinished")
        for key in keys:
            value = (
                values[key] - other_values[key]
                if reference is not None
                else values[key]
            )
            combined[key] += table.weight / total * value
    values = [combined[k] for k in sorted(keys, key=int)]
    return {
        "metric": "weighted_win_share_difference"
        if reference is not None
        else "weighted_win_share",
        "score": float(np.mean(values)),
        "ci95": paired_interval(values),
        "seed_blocks": len(values),
        "group_scores": combined,
        "weights": {t.name: t.weight / total for t in tables},
    }


def validate_table_prefix(
    records: list[dict],
    table: Table,
    *,
    seed: int,
    groups: int,
    batch_size: int,
    max_turns: int,
) -> None:
    """A resume must be a finished prefix at the original batch boundaries."""
    assignments = seat_assignments(table)
    total = groups * len(assignments)
    if (
        groups < 1
        or batch_size < 1
        or max_turns < 1
        or batch_size % len(assignments)
        or len(records) > total
        or (len(records) != total and len(records) % batch_size)
    ):
        raise ValueError("Resume requires complete original batches and seat blocks")
    n = len(table.opponents) + 1
    for index, row in enumerate(records):
        seats = assignments[index % len(assignments)]
        if (row["group_seed"], tuple(row["seats"]), row["candidate_seat"]) != (
            seed + index // len(assignments),
            seats,
            seats.index("candidate"),
        ):
            raise ValueError("Resume records differ from the frozen seat/seed schedule")
        winners = row["winning_seats"]
        if (
            not winners
            or len(set(winners)) != len(winners)
            or any(type(seat) is not int or not 0 <= seat < n for seat in winners)
        ):
            raise ValueError("Invalid winning seats in resume")
        won = row["candidate_seat"] in winners
        outcome = ("win" if len(winners) == 1 else "shared") if won else "loss"
        share = 1 / len(winners) if won else 0.0
        if (row["outcome"], row["win_share"]) != (outcome, share):
            raise ValueError("Unfinished or inconsistent outcome in resume")
        if (
            len(row["scores"]) != n
            or type(row["turns"]) is not int
            or not 1 <= row["turns"] <= max_turns
        ):
            raise ValueError("Invalid completed game in resume")


@torch.no_grad()
def play_table(
    candidate: Callable,
    opponents: dict[str, Callable],
    table: Table,
    search: SearchConfig,
    *,
    seed: int,
    groups: int,
    batch_size: int = 96,
    buffer: ReplayBuffer | None = None,
    training: bool = False,
    max_turns: int | None = None,
    bot_workers: int = 8,
    opponent_search: SearchConfig | None = None,
    progress: Callable[[dict], None] | None = None,
    resume_records: list[dict] | None = None,
    on_batch: Callable[[dict], None] | None = None,
) -> dict:
    """Replace one seat only; fixed other players, draws, seat order and RNG streams.

    Replay records candidate positions only and keeps terminal binary utilities.
    All production engine/search work uses Rust. Network wrappers may infer on CUDA.
    Evaluation resumes preserve original batch offsets and therefore search seeds.
    on_batch runs before progress, so a stop in progress cannot lose a saved batch.
    """
    if groups < 1 or batch_size < 1 or not 1 <= bot_workers <= 8:
        raise ValueError("Invalid game limits")
    if search.move_deadline_s is not None:
        raise ValueError("Batched games require fixed simulation budgets")
    n = len(table.opponents) + 1
    max_turns = max_turns or 100 * n
    assignments = seat_assignments(table)
    schedule = [
        (seed + group, seats) for group in range(groups) for seats in assignments
    ]
    if resume_records is not None or on_batch is not None:
        if training or buffer is not None:
            raise ValueError("Batch resumption is only supported for evaluation")
        validate_table_prefix(
            resume_records or [],
            table,
            seed=seed,
            groups=groups,
            batch_size=batch_size,
            max_turns=max_turns,
        )
    records = deepcopy(resume_records) if resume_records is not None else []
    samples_added = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=bot_workers) as pool:
        for start in range(len(records), len(schedule), batch_size):
            batch_started = time.monotonic()
            batch = schedule[start : start + batch_size]
            seeds, seats = zip(*batch)
            engine = GameEngine(len(batch), n, "cpu", game_seeds=list(seeds))
            candidate_seats = torch.tensor([s.index("candidate") for s in seats])
            bots = {
                (i, seat): BUILTIN_BOTS[name](
                    seed=seeds[i] + 17 + 1009 * ((seat - int(candidate_seats[i])) % n)
                )
                for i, row in enumerate(seats)
                for seat, name in enumerate(row)
                if name in BUILTIN_BOTS
            }
            snapshots = SnapshotRecorder(
                buffer.snapshot_capacity if buffer else 0, seed + start
            )
            trajectory = []
            turns = torch.zeros(len(batch), dtype=torch.long)
            for turn in range(max_turns):
                alive = ~engine.ended
                if not alive.any():
                    break
                current = engine.current_player.tolist()
                roles = [seats[i][cp] for i, cp in enumerate(current)]
                actions = torch.zeros(len(batch), dtype=torch.long)
                # Stable role order and seeds independent of policy checkpoint identity.
                for role_index, role in enumerate(sorted(set(roles))):
                    idx = torch.tensor(
                        [i for i, r in enumerate(roles) if r == role and alive[i]],
                        dtype=torch.long,
                    )
                    if not len(idx):
                        continue
                    sub = engine.index_select(idx)
                    if role in BUILTIN_BOTS:
                        selected = torch.tensor(
                            list(
                                pool.map(
                                    lambda i: bots[i, current[i]].select_action(
                                        engine, i
                                    ),
                                    idx.tolist(),
                                )
                            )
                        )
                    else:
                        model = candidate if role == "candidate" else opponents[role]
                        cfg = (
                            search
                            if role == "candidate"
                            else (opponent_search or search)
                        )
                        cfg = replace(
                            cfg,
                            seed=seed * 100003 + start * 1009 + turn * 31 + role_index,
                            temperature=_default_temperature_schedule(turn)
                            if training
                            else 0.25,
                        )
                        selected, policy = gumbel_root_act(
                            sub, model, search_config=cfg
                        )
                        if role == "candidate" and buffer is not None:
                            g, s = encode_state(sub)
                            snapshots.capture(engine, idx)
                            trajectory.append(
                                (
                                    g,
                                    s,
                                    sub.legal_action_mask(),
                                    policy,
                                    idx,
                                    sub.current_player.long(),
                                )
                            )
                    if not sub.legal_action_mask().gather(1, selected[:, None]).all():
                        raise RuntimeError("Mixed table produced an illegal action")
                    actions[idx] = selected
                engine.step(actions)
                turns[alive] += 1
            utilities = final_values_binary(engine, n)
            for i, (game_seed, row) in enumerate(batch):
                records.append(
                    {
                        "group_seed": game_seed,
                        "seats": list(row),
                        "candidate_seat": int(candidate_seats[i]),
                        **outcome_record(
                            utilities[i, :n].tolist(),
                            int(candidate_seats[i]),
                            ended=bool(engine.ended[i]),
                        ),
                        "scores": engine.scores[i, :n].tolist(),
                        "turns": int(turns[i]),
                    }
                )
            if trajectory:
                g, s, legal, policy, game_idx, cp = (
                    torch.cat([t[j] for t in trajectory]) for j in range(6)
                )
                values = _rotate_for_cp(utilities[game_idx], cp, n)
                keep = finished_sample_mask(
                    legal, policy, values, game_idx, engine.ended
                )
                if keep.any():
                    buffer.add(
                        g[keep],
                        s[keep],
                        legal[keep],
                        policy[keep],
                        values[keep],
                        policy_sims=search.num_simulations,
                        snapshots=snapshots.selected(keep),
                    )
                    samples_added += int(keep.sum())
            if on_batch:
                # Fail rather than persist stalled games or an incomplete seed block.
                validate_table_prefix(
                    records,
                    table,
                    seed=seed,
                    groups=groups,
                    batch_size=batch_size,
                    max_turns=max_turns,
                )
                on_batch(
                    {
                        "start": start,
                        "records": deepcopy(records[start:]),
                        "wall_s": time.monotonic() - batch_started,
                    }
                )
            if progress:
                progress(
                    {
                        "completed": len(records),
                        "total": len(schedule),
                        "table": table.name,
                    }
                )
    report = {
        "table": asdict(table),
        "num_players": n,
        "search": asdict(search),
        "opponent_search": asdict(opponent_search or search),
        "records": records,
        "samples_added": samples_added,
        "wall_s": time.monotonic() - started,
        "finished": sum(r["outcome"] != "unfinished" for r in records),
        "games_total": len(records),
    }
    if not training:
        report["summary"] = summarize_table(records, table)
    return report
