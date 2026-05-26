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
from ..net import model as M
from ..search import gumbel_mcts as G
from . import checkpointing as CK

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class UnifiedEvalConfig:
    total_games: int = 512
    num_sims: int = 64
    max_turns: int = 200
    turns_per_player: int = 60
    weight_2p: float = 1.0
    weight_3p: float = 0.0
    weight_4p: float = 0.0
    league_opponents: int = 4
    timeout_s: float = 900.0


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

        policy_for_game: Dict[str, List[int]] = {}
        for b in range(num_games):
            if not alive[b]:
                continue
            name = seat_names[b][int(cp[b].item())]
            policy_for_game.setdefault(name, []).append(b)

        for name, game_indices in policy_for_game.items():
            idx = torch.tensor(game_indices, dtype=torch.long, device=device)
            sub = engine.index_select(idx)
            sub_actions = policies[name](sub)
            actions.index_copy_(0, idx, sub_actions)

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
        eval_finished += 1
        if seat_names[b][int(winners[b].item())] == eval_agent_name:
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
    eval_net = M.AzulNet(hidden=hidden, arch=arch)
    eval_net.load_state_dict(state_dict)
    eval_net.eval()

    eval_agent_name = "eval_agent"

    def _make_net_policy(net: M.AzulNet, num_sims: int, num_players: int):
        def choose(engine: BE.BatchedEngine) -> torch.Tensor:
            with torch.no_grad():
                act, _ = G.gumbel_root_act(engine, net, num_sims=num_sims)
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
        )
        all_pairwise.extend(pairwise)
        all_metrics.update(metrics)

    wall_s = time.monotonic() - t_start
    all_metrics["eval_wall_s"] = round(wall_s, 3)

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
) -> None:
    try:
        results = run_unified_eval(
            state_dict,
            league_checkpoint_paths,
            config,
            seed,
            hidden=hidden,
            arch=arch,
        )
        queue.put((iteration, results))
    except Exception as exc:
        import traceback

        queue.put((iteration, {"error": str(exc), "traceback": traceback.format_exc()}))


class UnifiedEvalHandle:
    def __init__(self, config: UnifiedEvalConfig, hidden: int = 256, arch: str = "attn") -> None:
        self._config = config
        self._hidden = hidden
        self._arch = arch
        self._process: Optional[multiprocessing.Process] = None
        ctx = multiprocessing.get_context("spawn")
        self._mp_ctx = ctx
        self._queue: multiprocessing.Queue = ctx.Queue()
        self._iteration_tag: Optional[int] = None

    def is_active(self) -> bool:
        return self._process is not None and self._process.is_alive()

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
        self._process = self._mp_ctx.Process(
            target=_unified_eval_worker,
            args=(
                self._queue,
                net_state_dict,
                league_checkpoint_paths,
                self._config,
                iteration,
                seed,
                self._hidden,
                self._arch,
            ),
            daemon=True,
        )
        self._iteration_tag = iteration
        self._process.start()
        return True

    def try_collect(self) -> Optional[Tuple[int, Dict]]:
        if self._process is None:
            return None
        result = self._drain_one()
        if result is not None:
            self._reap()
            return result
        if not self._process.is_alive():
            self._reap()
        return None

    def wait_and_collect(self, timeout: float | None = None) -> Optional[Tuple[int, Dict]]:
        if self._process is None:
            return None
        effective_timeout = timeout if timeout is not None else self._config.timeout_s
        self._process.join(timeout=effective_timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        result = self._drain_one()
        self._reap()
        return result

    def cleanup(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._reap()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def _drain_one(self) -> Optional[Tuple[int, Dict]]:
        try:
            return self._queue.get_nowait()
        except Exception:
            return None

    def _reap(self) -> None:
        if self._process is not None:
            if not self._process.is_alive():
                self._process.join(timeout=1)
            self._process = None
            self._iteration_tag = None
