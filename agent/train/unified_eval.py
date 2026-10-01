"""Unified evaluation: mixed opponents, async CPU subprocess."""

from __future__ import annotations

import dataclasses
import logging
import math
import multiprocessing
import queue
import random as stdlib_random
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..env import engine as BE
from ..eval import bots as B
from ..eval.heuristic_opus import HeuristicOpusBot
from ..eval.builtin_opponents import BUILTIN_BOTS, builtin_identity
from ..net import model as M
from ..search import gumbel_mcts as G
from ..search.config import SearchConfig
from ..eval.inference import InferenceModel
from ..env.outcomes import _shared_victory_mask
from . import checkpointing as CK
from .instrumentation import PerfCounters, maybe_time, resource_delta, resource_snapshot

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class UnifiedEvalConfig:
    total_games: int = 512
    num_sims: int = 32
    max_turns: int = 200
    temperature: float = 0.25
    turns_per_player: int = 60
    weight_2p: float = 1.0
    weight_3p: float = 0.0
    weight_4p: float = 0.0
    league_opponents: int = 4
    timeout_s: float = 900.0
    q_scale: float = 10.0
    profile: bool = False
    num_workers: int = 1
    inference_device: str = "cpu"
    search_backend: str = "one_ply"
    search_tree_core: str = "python"
    root_noise_scale: float = 1.0
    max_root_candidates: int = 16
    chance_samples: int = 4
    inference_cache_size: int = 0
    astra_opponent_fraction: float = 0.125

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.astra_opponent_fraction)
            or not 0 <= self.astra_opponent_fraction <= 1
        ):
            raise ValueError("astra_opponent_fraction must be between zero and one")


@dataclasses.dataclass
class PairwiseResult:
    winner: str
    loser: str
    weight: float
    num_players: int = 2
    is_tie: bool = False


def _distribute_games(
    total: int, w2: float, w3: float, w4: float
) -> Tuple[int, int, int]:
    total_weight = w2 + w3 + w4
    if total_weight <= 0:
        return total, 0, 0
    n2 = int(round(total * w2 / total_weight))
    n3 = int(round(total * w3 / total_weight))
    n4 = total - n2 - n3
    return n2, n3, n4


def extract_pairwise_results(
    seat_names: List[List[str]],
    winner_seats: torch.Tensor,
    finished: torch.Tensor,
    num_players: int,
    winning_mask: torch.Tensor | None = None,
) -> List[PairwiseResult]:
    results: List[PairwiseResult] = []
    for b in range(winner_seats.shape[0]):
        if not finished[b]:
            continue
        w = int(winner_seats[b].item())
        if w == BE.SHARED_VICTORY:
            if winning_mask is None and num_players != 2:
                continue
            winners = (
                winning_mask[b].nonzero().flatten().tolist()
                if winning_mask is not None
                else [0, 1]
            )
            for p in winners:
                for q in range(num_players):
                    left, right = seat_names[b][p], seat_names[b][q]
                    if p == q or left == right:
                        continue
                    tied = q in winners
                    results.append(
                        PairwiseResult(
                            left, right, 0.5 if tied else 1.0, num_players, tied
                        )
                    )
            continue
        if w < 0:
            continue
        winner_name = seat_names[b][w]
        for p in range(num_players):
            if p == w:
                continue
            loser_name = seat_names[b][p]
            if loser_name == winner_name:
                continue
            results.append(
                PairwiseResult(
                    winner=winner_name,
                    loser=loser_name,
                    weight=1.0,
                    num_players=num_players,
                )
            )
    return results


def _run_multiplayer_games(
    num_players: int,
    num_games: int,
    policies: Dict[str, Callable[[BE.BatchedEngine], torch.Tensor]],
    eval_agent_name: str,
    seed: int,
    max_turns: int,
    perf: PerfCounters | None = None,
    astra_opponent_fraction: float = 0.0,
) -> Tuple[List[PairwiseResult], Dict[str, float]]:
    if num_games == 0:
        return [], {}

    device = "cpu"
    rng = stdlib_random.Random(seed)
    other_names = [n for n in policies if n != eval_agent_name]
    non_astra_names = [name for name in other_names if name != "astra"]
    if astra_opponent_fraction > 0 and "astra" not in other_names:
        raise ValueError(
            "Astra evaluation was requested but no Astra policy was supplied"
        )
    if astra_opponent_fraction < 1 and not non_astra_names:
        raise ValueError(
            "Evaluation requires a non-Astra opponent when Astra fraction is below one"
        )

    seat_names: List[List[str]] = []
    for _ in range(num_games):
        seats = [""] * num_players
        eval_seat = rng.randint(0, num_players - 1)
        seats[eval_seat] = eval_agent_name
        for p in range(num_players):
            if p != eval_seat:
                if "astra" in other_names and astra_opponent_fraction > 0:
                    if rng.random() < astra_opponent_fraction:
                        seats[p] = "astra"
                    else:
                        seats[p] = rng.choice(non_astra_names)
                else:
                    seats[p] = rng.choice(non_astra_names)
        seat_names.append(seats)

    engine = BE.BatchedEngine(num_games, num_players, device=device, seed=seed)

    turn = 0
    while turn < max_turns and (~engine.ended).any():
        alive = ~engine.ended
        cp = engine.current_player.long()
        actions = torch.zeros((num_games,), dtype=torch.long, device=device)

        with maybe_time(perf, f"eval_{num_players}p_group_policies"):
            policy_for_game: Dict[str, List[int]] = {}
            for b in range(num_games):
                if not alive[b]:
                    continue
                name = seat_names[b][int(cp[b].item())]
                policy_for_game.setdefault(name, []).append(b)

        for name, game_indices in policy_for_game.items():
            stage_name = (
                name
                if name
                in {"eval_agent", "random", "heuristic", "heuristic_opus", "astra"}
                else "league"
            )
            with maybe_time(perf, f"eval_{num_players}p_policy_{stage_name}"):
                idx = torch.tensor(game_indices, dtype=torch.long, device=device)
                sub = engine.index_select(idx)
                sub_actions = policies[name](sub)
                if not sub.legal_action_mask().gather(1, sub_actions[:, None]).all():
                    raise RuntimeError(
                        f"Evaluation policy {name} returned an illegal action"
                    )
                actions.index_copy_(0, idx, sub_actions)
            if perf is not None:
                perf.add_count(
                    f"eval_{num_players}p_policy_{stage_name}_positions",
                    len(game_indices),
                )

        with maybe_time(perf, f"eval_{num_players}p_env_step"):
            engine.step(actions)
        turn += 1

    winners = engine.get_winners()
    finished = engine.ended
    shared_mask = _shared_victory_mask(engine, torch.arange(num_games), num_players)
    pairwise = extract_pairwise_results(
        seat_names, winners, finished, num_players, shared_mask
    )

    wins = losses = shared = 0
    shares = [0.0] * num_games
    for b in range(num_games):
        if not finished[b]:
            continue
        w = int(winners[b])
        eval_seat = seat_names[b].index(eval_agent_name)
        if w == BE.SHARED_VICTORY and shared_mask[b, eval_seat]:
            shared += 1
            shares[b] = 1.0 / int(shared_mask[b].sum())
        elif w >= 0 and seat_names[b][w] == eval_agent_name:
            wins += 1
            shares[b] = 1.0
        else:
            losses += 1
    count = wins + losses + shared
    metrics = {
        f"games_{num_players}p": float(num_games),
        f"finished_{num_players}p": float(count),
        f"wins_{num_players}p": float(wins),
        f"losses_{num_players}p": float(losses),
        f"shared_{num_players}p": float(shared),
        f"unfinished_{num_players}p": float(num_games - count),
        f"win_share_sum_{num_players}p": sum(shares),
        f"eval_winrate_{num_players}p": wins / max(count, 1),
        f"eval_match_score_{num_players}p": sum(shares) / max(count, 1),
        f"eval_win_share_{num_players}p": sum(shares) / max(count, 1),
    }
    if "astra" in policies:
        suffix = f"{num_players}p_{'vs' if num_players == 2 else 'with'}_astra"
        selected = [b for b, names in enumerate(seat_names) if "astra" in names]
        completed = [b for b in selected if finished[b]]
        credit = sum(shares[b] for b in completed)
        metrics.update(
            {
                f"games_{suffix}": float(len(selected)),
                f"finished_{suffix}": float(len(completed)),
                f"unfinished_{suffix}": float(len(selected) - len(completed)),
                f"win_share_sum_{suffix}": credit,
                f"eval_win_share_{suffix}": credit / max(len(completed), 1),
            }
        )
    return pairwise, metrics


def run_unified_eval(
    state_dict: Dict[str, torch.Tensor],
    league_checkpoint_paths: List[str],
    config: UnifiedEvalConfig,
    seed: int,
    hidden: int = 256,
    arch: str = "attn",
) -> Dict:
    torch.set_num_threads(1)
    t_start = time.monotonic()
    perf = PerfCounters(enabled=config.profile, device="cpu")
    resource_start = resource_snapshot() if config.profile else None
    eval_net = M.AzulNet(hidden=hidden, arch=arch)
    eval_net.load_state_dict(state_dict)
    eval_net.eval()
    torch.manual_seed(seed)

    eval_agent_name = "eval_agent"

    def _make_net_policy(net: M.AzulNet, num_sims: int, num_players: int):
        inference = InferenceModel(net, config.inference_device)
        calls = 0

        def choose(engine: BE.BatchedEngine) -> torch.Tensor:
            nonlocal calls
            calls += 1
            with torch.no_grad():
                act, _ = G.gumbel_root_act(
                    engine,
                    inference,
                    num_sims=num_sims,
                    temperature=config.temperature,
                    q_scale=config.q_scale,
                    search_config=SearchConfig(
                        backend=config.search_backend,
                        tree_core=config.search_tree_core,
                        root_noise_scale=config.root_noise_scale,
                        max_root_candidates=config.max_root_candidates,
                        chance_samples=config.chance_samples,
                        inference_cache_size=config.inference_cache_size,
                        num_simulations=num_sims,
                        temperature=config.temperature,
                        q_scale=config.q_scale,
                        seed=seed * 100003 + calls,
                    ),
                )
            return act

        return choose

    class _BotPolicy:
        def __init__(self, bot: B.Bot, num_players: int) -> None:
            self.bot = bot
            self.num_players = num_players

        def __call__(self, engine: BE.BatchedEngine) -> torch.Tensor:
            actions = torch.zeros(
                (engine.batch_size,), dtype=torch.long, device=engine.device
            )
            for b in range(engine.batch_size):
                if engine.ended[b]:
                    continue
                actions[b] = self.bot.select_action(engine, b)
            return actions

    policies: Dict[str, Callable[[BE.BatchedEngine], torch.Tensor]] = {
        eval_agent_name: _make_net_policy(eval_net, config.num_sims, 2),
        "random": _BotPolicy(B.RandomBot(seed=seed), 2),
        "heuristic": _BotPolicy(B.HeuristicBot(seed=seed + 1), 2),
        "heuristic_opus": _BotPolicy(HeuristicOpusBot(seed=seed + 2), 2),
    }
    astra_identities = {}
    if config.astra_opponent_fraction > 0:
        astra_identities = {str(n): builtin_identity("astra", n) for n in (2, 3, 4)}
        policies["astra"] = _BotPolicy(BUILTIN_BOTS["astra"](seed=seed + 3), 2)

    league_names: List[str] = []
    for i, path in enumerate(league_checkpoint_paths):
        try:
            net, payload = CK.load_net_from_checkpoint(path, map_location="cpu")
            net.eval()
            name = f"league_{i}"
            policies[name] = _make_net_policy(net, config.num_sims, 2)
            league_names.append(name)
        except Exception as exc:
            logger.warning("Failed to load league checkpoint %s: %s", path, exc)

    n2, n3, n4 = _distribute_games(
        config.total_games, config.weight_2p, config.weight_3p, config.weight_4p
    )

    all_pairwise: List[PairwiseResult] = []
    all_metrics: Dict[str, float] = {}

    for num_players, num_games in [(2, n2), (3, n3), (4, n4)]:
        if num_games <= 0:
            continue
        effective_max_turns = (
            config.turns_per_player * num_players
            if config.turns_per_player > 0
            else config.max_turns
        )
        pairwise, metrics = _run_multiplayer_games(
            num_players=num_players,
            num_games=num_games,
            policies=policies,
            eval_agent_name=eval_agent_name,
            seed=seed + num_players * 1000,
            max_turns=effective_max_turns,
            perf=perf,
            astra_opponent_fraction=config.astra_opponent_fraction,
        )
        all_pairwise.extend(pairwise)
        all_metrics.update(metrics)

    wall_s = time.monotonic() - t_start
    all_metrics["eval_wall_s"] = round(wall_s, 3)
    if config.profile and resource_start is not None:
        all_metrics.update(
            resource_delta(resource_start, resource_snapshot(), prefix="eval_profile")
        )
        all_metrics.update(perf.snapshot(prefix="eval_profile"))

    pairwise_list = [
        {
            "winner": pr.winner,
            "loser": pr.loser,
            "weight": pr.weight,
            "num_players": pr.num_players,
            "is_tie": pr.is_tie,
        }
        for pr in all_pairwise
    ]

    return {
        "pairwise": pairwise_list,
        "metrics": all_metrics,
        "eval_wall_s": wall_s,
        "league_opponents": league_names,
        "astra_identity": astra_identities,
    }


def _unified_eval_worker(
    queue: multiprocessing.Queue,
    state_dict: Dict[str, torch.Tensor],
    league_checkpoint_paths: List[str],
    config: UnifiedEvalConfig,
    iteration: int,
    seed: int,
    hidden: int,
    arch: str,
    worker_id: int = 0,
) -> None:
    try:
        results = run_unified_eval(
            state_dict,
            league_checkpoint_paths,
            config,
            seed + worker_id * 100003,
            hidden=hidden,
            arch=arch,
        )
        queue.put((iteration, worker_id, results))
    except Exception as exc:
        import traceback

        queue.put(
            (
                iteration,
                worker_id,
                {"error": str(exc), "traceback": traceback.format_exc()},
            )
        )


def _merge_eval_results(parts: List[Dict]) -> Dict:
    if not parts:
        return {
            "pairwise": [],
            "metrics": {},
            "eval_wall_s": 0.0,
            "league_opponents": [],
        }
    merged_pairwise: List[dict] = []
    merged_metrics: Dict[str, float] = {}
    league_opponents: List[str] = []
    wall_s = 0.0
    for part in parts:
        if "error" in part:
            return part
        merged_pairwise.extend(part.get("pairwise", []))
        for k, v in part.get("metrics", {}).items():
            if isinstance(v, (int, float)) and not k.startswith(
                ("eval_winrate_", "eval_match_score_", "eval_win_share_")
            ):
                merged_metrics[k] = merged_metrics.get(k, 0.0) + float(v)
        wall_s = max(wall_s, float(part.get("eval_wall_s", 0.0)))
        if not league_opponents:
            league_opponents = part.get("league_opponents", [])
    for players in (2, 3, 4):
        n = merged_metrics.get(f"finished_{players}p", 0)
        if f"games_{players}p" in merged_metrics:
            wins = merged_metrics.get(f"wins_{players}p", 0)
            shared = merged_metrics.get(f"shared_{players}p", 0)
            merged_metrics[f"eval_winrate_{players}p"] = wins / max(n, 1)
            credit = merged_metrics.get(
                f"win_share_sum_{players}p", wins + 0.5 * shared
            )
            merged_metrics[f"eval_match_score_{players}p"] = credit / max(n, 1)
            merged_metrics[f"eval_win_share_{players}p"] = credit / max(n, 1)
        suffix = f"{players}p_{'vs' if players == 2 else 'with'}_astra"
        if f"games_{suffix}" in merged_metrics:
            merged_metrics[f"eval_win_share_{suffix}"] = merged_metrics[
                f"win_share_sum_{suffix}"
            ] / max(merged_metrics[f"finished_{suffix}"], 1)
    identities = [part.get("astra_identity", {}) for part in parts]
    if any(identity != identities[0] for identity in identities):
        return {
            "error": "Astra configuration/native identity differed between evaluation workers"
        }
    merged_metrics["eval_wall_s"] = wall_s
    return {
        "pairwise": merged_pairwise,
        "metrics": merged_metrics,
        "eval_wall_s": wall_s,
        "league_opponents": league_opponents,
        "astra_identity": identities[0],
    }


class UnifiedEvalHandle:
    def __init__(
        self, config: UnifiedEvalConfig, hidden: int = 256, arch: str = "attn"
    ) -> None:
        self._config, self._hidden, self._arch = config, hidden, arch
        self._mp_ctx = multiprocessing.get_context("spawn")
        self._processes: list[multiprocessing.Process] = []
        self._queue = None
        self._pending: dict[int, dict] = {}
        self._iteration_tag: int | None = None
        self._expected_workers = 0
        self._context: dict = {}
        self._started = 0.0
        self._immediate = None

    def is_active(self) -> bool:
        # Completed subprocesses may still have undelivered results.
        return self._iteration_tag is not None

    def launch(
        self,
        net_state_dict: Dict[str, torch.Tensor],
        league_checkpoint_paths: List[str],
        iteration: int,
        seed: int,
        context: dict | None = None,
    ) -> bool:
        if self.is_active():
            return False
        if self._config.total_games <= 0:
            return False
        if self._config.inference_device.startswith("cuda"):
            from .reproducibility import capture_rng_state, restore_rng_state

            rng = capture_rng_state()
            self._iteration_tag = iteration
            self._context = dict(context or {})
            try:
                self._immediate = run_unified_eval(
                    net_state_dict,
                    league_checkpoint_paths,
                    self._config,
                    seed,
                    self._hidden,
                    self._arch,
                )
            except Exception as exc:
                self._immediate = {"error": str(exc)}
            finally:
                restore_rng_state(rng)
            return True
        self._queue = self._mp_ctx.Queue()
        self._context = dict(context or {})
        self._iteration_tag = iteration
        self._started = time.monotonic()
        n = min(max(1, self._config.num_workers), self._config.total_games)
        self._expected_workers = n
        self._pending = {}
        for worker in range(n):
            shard = self._config.total_games // n + int(
                worker < self._config.total_games % n
            )
            config = dataclasses.replace(self._config, total_games=shard)
            proc = self._mp_ctx.Process(
                target=_unified_eval_worker,
                args=(
                    self._queue,
                    net_state_dict,
                    league_checkpoint_paths,
                    config,
                    iteration,
                    seed,
                    self._hidden,
                    self._arch,
                    worker,
                ),
                daemon=True,
            )
            proc.start()
            self._processes.append(proc)
        return True

    def _drain(self) -> None:
        while self._queue is not None:
            try:
                iteration, worker, result = self._queue.get_nowait()
            except queue.Empty:
                break
            if iteration == self._iteration_tag:
                self._pending[worker] = result

    def try_collect(self) -> Optional[Tuple[int, Dict]]:
        if not self.is_active():
            return None
        if self._immediate is not None:
            result, iteration = self._immediate, self._iteration_tag
            result["job_context"] = self._context
            self._immediate = None
            self.cleanup()
            return iteration, result
        self._drain()
        if len(self._pending) == self._expected_workers:
            return self._finish()
        if time.monotonic() - self._started > self._config.timeout_s:
            return self._finish(
                error="evaluation timed out before all workers returned"
            )
        if all(not p.is_alive() for p in self._processes):
            # join flushes worker queue feeder threads before the final drain.
            for p in self._processes:
                p.join(timeout=0.1)
            self._drain()
            if len(self._pending) == self._expected_workers:
                return self._finish()
            return self._finish(error="evaluation worker exited without a result")
        return None

    def wait_and_collect(
        self, timeout: float | None = None
    ) -> Optional[Tuple[int, Dict]]:
        deadline = time.monotonic() + (
            timeout if timeout is not None else self._config.timeout_s
        )
        while self.is_active():
            result = self.try_collect()
            if result is not None:
                return result
            if time.monotonic() >= deadline:
                return self._finish(error="evaluation timed out")
            time.sleep(0.05)
        return None

    def _finish(self, error: str | None = None) -> Tuple[int, Dict]:
        iteration = self._iteration_tag
        result = (
            {"error": error}
            if error
            else _merge_eval_results([self._pending[k] for k in sorted(self._pending)])
        )
        result["job_context"] = self._context
        self.cleanup()
        return iteration, result

    def cleanup(self) -> None:
        for proc in self._processes:
            proc.join(timeout=0.2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        if self._queue is not None:
            self._queue.close()
            self._queue.join_thread()
        self._queue = None
        self._processes = []
        self._pending = {}
        self._iteration_tag = None
