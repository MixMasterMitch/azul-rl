"""Unified evaluation: mixed opponents, async CPU subprocess."""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import random as stdlib_random
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..env import batched_engine as BE
from ..eval import bots as B
from ..eval.heuristic_opus import HeuristicOpusBot
from ..net import model as M
from ..search import gumbel_mcts as G
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


@dataclasses.dataclass
class PairwiseResult:
    winner: str
    loser: str
    weight: float
    num_players: int = 2


def _distribute_games(total: int, w2: float, w3: float, w4: float) -> Tuple[int, int, int]:
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
) -> List[PairwiseResult]:
    results: List[PairwiseResult] = []
    for b in range(winner_seats.shape[0]):
        if not finished[b]:
            continue
        w = int(winner_seats[b].item())
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
                PairwiseResult(winner=winner_name, loser=loser_name, weight=1.0, num_players=num_players)
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
) -> Tuple[List[PairwiseResult], Dict[str, float]]:
    if num_games == 0:
        return [], {}

    device = "cpu"
    rng = stdlib_random.Random(seed)
    other_names = [n for n in policies if n != eval_agent_name]

    seat_names: List[List[str]] = []
    for _ in range(num_games):
        seats = [""] * num_players
        eval_seat = rng.randint(0, num_players - 1)
        seats[eval_seat] = eval_agent_name
        for p in range(num_players):
            if p != eval_seat:
                seats[p] = rng.choice(other_names)
        seat_names.append(seats)

    engine = BE.BatchedEngine(num_games, num_players, device=device, seed=seed)
    prev_ended = torch.zeros(num_games, dtype=torch.bool, device=device)

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
            stage_name = name if name in {"eval_agent", "random", "heuristic", "heuristic_opus"} else "league"
            with maybe_time(perf, f"eval_{num_players}p_policy_{stage_name}"):
                idx = torch.tensor(game_indices, dtype=torch.long, device=device)
                sub = engine.index_select(idx)
                sub_actions = policies[name](sub)
                actions.index_copy_(0, idx, sub_actions)
            if perf is not None:
                perf.add_count(f"eval_{num_players}p_policy_{stage_name}_positions", len(game_indices))

        with maybe_time(perf, f"eval_{num_players}p_env_step"):
            engine.step(actions)
        prev_ended = engine.ended.clone()
        turn += 1

    winners = engine.get_winners()
    finished = engine.ended
    pairwise = extract_pairwise_results(seat_names, winners, finished, num_players)

    eval_wins = 0
    eval_finished = 0
    for b in range(num_games):
        if not finished[b]:
            continue
        w = int(winners[b].item())
        if w < 0:
            continue
        eval_finished += 1
        if seat_names[b][w] == eval_agent_name:
            eval_wins += 1

    metrics = {
        f"games_{num_players}p": float(num_games),
        f"finished_{num_players}p": float(int(finished.sum().item())),
        f"eval_winrate_{num_players}p": eval_wins / max(eval_finished, 1),
    }
    return pairwise, metrics


def run_unified_eval(
    state_dict: Dict[str, torch.Tensor],
    league_checkpoint_paths: List[str],
    config: UnifiedEvalConfig,
    seed: int,
    hidden: int = 256,
    arch: str = "attn",
) -> Dict:
    from ..net import encoder as ENC

    t_start = time.monotonic()
    perf = PerfCounters(enabled=config.profile, device="cpu")
    resource_start = resource_snapshot() if config.profile else None
    eval_net = M.AzulNet(hidden=hidden, arch=arch)
    eval_net.load_state_dict(state_dict)
    eval_net.eval()

    eval_agent_name = "eval_agent"

    def _make_net_policy(net: M.AzulNet, num_sims: int, num_players: int):
        def choose(engine: BE.BatchedEngine) -> torch.Tensor:
            with torch.no_grad():
                act, _ = G.gumbel_root_act(
                    engine,
                    net,
                    num_sims=num_sims,
                    temperature=config.temperature,
                    q_scale=config.q_scale,
                )
            return act

        return choose

    class _BotPolicy:
        def __init__(self, bot: B.RandomBot | B.HeuristicBot, num_players: int):
            self.bot = bot
            self.num_players = num_players

        def __call__(self, engine: BE.BatchedEngine) -> torch.Tensor:
            actions = torch.zeros((engine.batch_size,), dtype=torch.long, device=engine.device)
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

    league_names: List[str] = []
    for i, path in enumerate(league_checkpoint_paths):
        try:
            net, payload = CK.load_net_from_checkpoint(path, map_location="cpu")
            spec = CK.checkpoint_net_spec(payload)
            net = M.AzulNet(hidden=spec.hidden, arch=spec.arch)
            net.load_state_dict(CK.checkpoint_net_state_dict(payload))
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
        )
        all_pairwise.extend(pairwise)
        all_metrics.update(metrics)

    wall_s = time.monotonic() - t_start
    all_metrics["eval_wall_s"] = round(wall_s, 3)
    if config.profile and resource_start is not None:
        all_metrics.update(resource_delta(resource_start, resource_snapshot(), prefix="eval_profile"))
        all_metrics.update(perf.snapshot(prefix="eval_profile"))

    pairwise_list = [
        {"winner": pr.winner, "loser": pr.loser, "weight": pr.weight, "num_players": pr.num_players}
        for pr in all_pairwise
    ]

    return {
        "pairwise": pairwise_list,
        "metrics": all_metrics,
        "eval_wall_s": wall_s,
        "league_opponents": league_names,
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

        queue.put((iteration, worker_id, {"error": str(exc), "traceback": traceback.format_exc()}))


def _merge_eval_results(parts: List[Dict]) -> Dict:
    if not parts:
        return {"pairwise": [], "metrics": {}, "eval_wall_s": 0.0, "league_opponents": []}
    merged_pairwise: List[dict] = []
    merged_metrics: Dict[str, float] = {}
    league_opponents: List[str] = []
    wall_s = 0.0
    for part in parts:
        if "error" in part:
            return part
        merged_pairwise.extend(part.get("pairwise", []))
        for k, v in part.get("metrics", {}).items():
            if isinstance(v, (int, float)):
                merged_metrics[k] = merged_metrics.get(k, 0.0) + float(v)
        wall_s = max(wall_s, float(part.get("eval_wall_s", 0.0)))
        if not league_opponents:
            league_opponents = part.get("league_opponents", [])
    return {
        "pairwise": merged_pairwise,
        "metrics": merged_metrics,
        "eval_wall_s": wall_s,
        "league_opponents": league_opponents,
    }


class UnifiedEvalHandle:
    def __init__(self, config: UnifiedEvalConfig, hidden: int = 256, arch: str = "attn") -> None:
        self._config = config
        self._hidden = hidden
        self._arch = arch
        self._processes: List[multiprocessing.Process] = []
        ctx = multiprocessing.get_context("spawn")
        self._mp_ctx = ctx
        self._queue: multiprocessing.Queue = ctx.Queue()
        self._iteration_tag: Optional[int] = None
        self._expected_workers: int = 0

    def is_active(self) -> bool:
        return any(p.is_alive() for p in self._processes)

    def launch(
        self,
        net_state_dict: Dict[str, torch.Tensor],
        league_checkpoint_paths: List[str],
        iteration: int,
        seed: int,
    ) -> bool:
        if self.is_active():
            return False
        self._reap()
        n_workers = max(1, self._config.num_workers)
        total = self._config.total_games
        base = total // n_workers
        extra = total % n_workers
        self._expected_workers = n_workers
        self._iteration_tag = iteration

        for worker_id in range(n_workers):
            shard_games = base + (1 if worker_id < extra else 0)
            if shard_games <= 0:
                continue
            shard_config = dataclasses.replace(self._config, total_games=shard_games)
            proc = self._mp_ctx.Process(
                target=_unified_eval_worker,
                args=(
                    self._queue,
                    net_state_dict,
                    league_checkpoint_paths,
                    shard_config,
                    iteration,
                    seed,
                    self._hidden,
                    self._arch,
                    worker_id,
                ),
                daemon=True,
            )
            proc.start()
            self._processes.append(proc)
        return True

    def try_collect(self) -> Optional[Tuple[int, Dict]]:
        if not self._processes:
            return None
        collected = self._drain_all_pending()
        if len(collected) >= self._expected_workers:
            self._reap()
            return self._merge_collected(collected)
        if not self.is_active():
            self._reap()
            if collected:
                return self._merge_collected(collected)
        return None

    def wait_and_collect(self, timeout: float | None = None) -> Optional[Tuple[int, Dict]]:
        if not self._processes:
            return None
        effective_timeout = timeout if timeout is not None else self._config.timeout_s
        deadline = time.monotonic() + effective_timeout
        collected: List[Tuple[int, int, Dict]] = []
        while time.monotonic() < deadline:
            collected.extend(self._drain_all_pending())
            if len(collected) >= self._expected_workers:
                break
            if not self.is_active() and self._queue.empty():
                break
            time.sleep(0.05)

        for proc in self._processes:
            if proc.is_alive():
                proc.join(timeout=1)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        collected.extend(self._drain_all_pending())
        self._reap()
        if not collected:
            return None
        return self._merge_collected(collected)

    def cleanup(self) -> None:
        for proc in self._processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        self._reap()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def _drain_all_pending(self) -> List[Tuple[int, int, Dict]]:
        out: List[Tuple[int, int, Dict]] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except Exception:
                break
        return out

    def _merge_collected(self, collected: List[Tuple[int, int, Dict]]) -> Tuple[int, Dict]:
        iteration = collected[0][0]
        parts = [item[2] for item in collected]
        return iteration, _merge_eval_results(parts)

    def _reap(self) -> None:
        for proc in self._processes:
            if not proc.is_alive():
                proc.join(timeout=1)
        self._processes = []
        self._iteration_tag = None
        self._expected_workers = 0
