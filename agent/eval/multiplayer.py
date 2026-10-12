"""Seat-rotated multiplayer evaluation with correctly attributed shared wins."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import time
from typing import Callable

import torch

from ..env.engine import GameEngine
from ..env.outcomes import final_values_binary, REWARD_SEMANTICS_VERSION
from ..net.encoder import encode_state
from ..search.config import SearchConfig
from ..search.gumbel_mcts import gumbel_root_act
from ..train.checkpointing import load_net_from_checkpoint
from ..train.reproducibility import provenance
from .arena import checkpoint_hash, paired_interval
from .builtin_opponents import BUILTIN_BOTS, builtin_identity
from .inference import InferenceModel


@dataclass(frozen=True)
class MultiplayerArenaConfig:
    num_players: int = 3
    num_games: int = 192
    seed: int = 20261008
    max_turns: int = 360
    game_batch_size: int = 96
    inference_device: str = "cpu"
    bot_workers: int = 8
    search: SearchConfig = field(default_factory=SearchConfig)
    greedy: bool = False
    split: str = "development"

    def __post_init__(self) -> None:
        if self.num_players not in (3, 4):
            raise ValueError("Multiplayer arena requires three or four players")
        if any(
            n < self.num_players or n % self.num_players
            for n in (self.num_games, self.game_batch_size)
        ):
            raise ValueError("Game counts must contain complete seat rotations")
        if self.max_turns < 1 or not 1 <= self.bot_workers <= 8:
            raise ValueError("Invalid arena limits")
        if self.search.move_deadline_s is not None:
            raise ValueError("Use single-game latency qualification for move budgets")
        if self.split not in {"development", "confirmation", "replication"}:
            raise ValueError("Invalid evaluation split")


def summarize_multiplayer(records: list[dict], num_players: int) -> dict:
    """Share one win among actual tied winners; bootstrap whole seed rotations."""
    finished = [r for r in records if r["outcome"] != "unfinished"]
    groups: dict[int, list[dict]] = {}
    for row in finished:
        groups.setdefault(row["group_seed"], []).append(row)
    group_scores = [
        sum(r["win_share"] for r in group) / num_players
        for group in groups.values()
        if len(group) == num_players
        and {r["candidate_seat"] for r in group} == set(range(num_players))
    ]
    return {
        "games": len(records),
        "unfinished": len(records) - len(finished),
        "sole_wins": sum(r["outcome"] == "win" for r in finished),
        "shared_wins": sum(r["outcome"] == "shared" for r in finished),
        "sole_winrate": sum(r["outcome"] == "win" for r in finished)
        / max(1, len(finished)),
        "win_share": sum(r["win_share"] for r in finished) / max(1, len(finished)),
        "win_share_ci95": paired_interval(group_scores),
        "group_scores": group_scores,
        "complete_rotations": len(group_scores),
        "equal_strength_reference": 1 / num_players,
    }


def outcome_record(utilities: list[float], candidate_seat: int, *, ended: bool) -> dict:
    if not ended:
        return {"outcome": "unfinished", "win_share": None, "winning_seats": []}
    winners = [seat for seat, value in enumerate(utilities) if value >= 0]
    if not winners:
        raise ValueError("Finished game has no winning seat")
    won = candidate_seat in winners
    return {
        "outcome": ("win" if len(winners) == 1 else "shared") if won else "loss",
        "win_share": 1 / len(winners) if won else 0.0,
        "winning_seats": winners,
    }


@torch.inference_mode()
def evaluate_multiplayer(
    candidate: str,
    opponent: str,
    config: MultiplayerArenaConfig,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """One candidate versus N-1 copies of an opponent, rotated through all seats."""
    started = time.monotonic()
    n_players = config.num_players
    net, payload = load_net_from_checkpoint(candidate)
    model = InferenceModel(net, config.inference_device)
    other = None
    if opponent not in BUILTIN_BOTS:
        other_net, _ = load_net_from_checkpoint(opponent)
        other = InferenceModel(other_net, config.inference_device)
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=config.bot_workers) as workers:
        for start in range(0, config.num_games, config.game_batch_size):
            count = min(config.game_batch_size, config.num_games - start)
            seeds = [config.seed + (start + i) // n_players for i in range(count)]
            seats = torch.tensor([(start + i) % n_players for i in range(count)])
            engine = GameEngine(count, n_players, "cpu", game_seeds=seeds)
            bots = (
                [
                    [
                        BUILTIN_BOTS[opponent](
                            seed=seed + 17 + 1009 * ((seat - int(seats[i])) % n_players)
                        )
                        for seat in range(n_players)
                    ]
                    for i, seed in enumerate(seeds)
                ]
                if other is None
                else []
            )
            turns = torch.zeros(count, dtype=torch.long)
            for turn in range(config.max_turns):
                alive = ~engine.ended
                if not alive.any():
                    break
                candidate_turn = engine.current_player.long() == seats
                actions = torch.zeros(count, dtype=torch.long)
                for is_candidate in (True, False):
                    idx = (alive & (candidate_turn == is_candidate)).nonzero().flatten()
                    if not len(idx):
                        continue
                    sub = engine.index_select(idx)
                    if is_candidate or other is not None:
                        active_model = model if is_candidate else other
                        if config.greedy:
                            g, s = encode_state(sub)
                            selected = active_model(
                                g, s, sub.legal_action_mask(), n_players
                            )[0].argmax(1)
                        else:
                            search = replace(
                                config.search,
                                seed=config.seed * 100003
                                + start * 1009
                                + turn * 2
                                + int(is_candidate),
                            )
                            selected = gumbel_root_act(
                                sub, active_model, search_config=search
                            )[0]
                    else:
                        player_seats = sub.current_player.tolist()

                        def bot_action(item: tuple[int, int]) -> int:
                            local, global_index = item
                            return bots[global_index][
                                player_seats[local]
                            ].select_action(sub, local)

                        selected = torch.tensor(
                            list(workers.map(bot_action, enumerate(idx.tolist())))
                        )
                    if not sub.legal_action_mask().gather(1, selected[:, None]).all():
                        raise RuntimeError(
                            "Multiplayer evaluation produced an illegal action"
                        )
                    actions[idx] = selected
                engine.step(actions)
                turns[alive] += 1
            utilities = final_values_binary(engine, n_players)[:, :n_players].tolist()
            scores = engine.scores[:, :n_players].tolist()
            for i in range(count):
                records.append(
                    {
                        "group_seed": seeds[i],
                        "candidate_seat": int(seats[i]),
                        **outcome_record(
                            utilities[i], int(seats[i]), ended=bool(engine.ended[i])
                        ),
                        "scores": scores[i],
                        "turns": int(turns[i]),
                    }
                )
            if progress:
                progress({"completed": len(records), "total": config.num_games})
    return {
        "schema_version": 1,
        "candidate": str(Path(candidate).resolve()),
        "candidate_sha256": checkpoint_hash(candidate),
        "opponent": opponent,
        "opponent_sha256": checkpoint_hash(opponent) if other is not None else None,
        "opponent_identity": builtin_identity(opponent, n_players),
        "reward_semantics_version": REWARD_SEMANTICS_VERSION,
        "candidate_trained_player_counts": payload.get("trained_player_counts", []),
        "config": asdict(config),
        "summary": summarize_multiplayer(records, n_players),
        "records": records,
        "wall_s": time.monotonic() - started,
        "provenance": provenance(),
    }
