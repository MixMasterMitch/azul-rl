"""Single-game search qualification on opening, middle, and late positions."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import time

import numpy as np
import torch

from ..env.engine import BatchedEngine
from ..net.encoder import encode_state
from ..search.config import SearchConfig
from ..search.gumbel_mcts import gumbel_root_act
from ..train.checkpointing import load_net_from_checkpoint
from ..train.instrumentation import PerfCounters
from .arena import checkpoint_hash
from .bots import HeuristicBot
from .inference import InferenceModel


def benchmark_latency(
    checkpoint: str,
    search: SearchConfig,
    *,
    device: str = "cpu",
    seed: int = 204913,
    games: int = 3,
    deadline_s: float = 5.0,
    wall_budget_s: float | None = None,
) -> dict:
    """Qualify full budgets against a configurable per-move serving deadline.

    Fixed-budget arena evaluations remain batched. A profile must also finish its
    requested budget on every latency position before it can be selected.
    """
    wall_budget_s = deadline_s if wall_budget_s is None else wall_budget_s
    if not (
        math.isfinite(deadline_s)
        and math.isfinite(wall_budget_s)
        and 0 < deadline_s <= wall_budget_s
        and games >= 1
    ):
        raise ValueError("Invalid internal deadline or outer move budget")
    model, _ = load_net_from_checkpoint(checkpoint, "cpu")
    wrapped = InferenceModel(model, device)
    positions = []
    for game in range(games):
        engine = BatchedEngine(1, 2, "cpu", seed=seed + game)
        bot = HeuristicBot(seed=seed + game)
        trajectory = []
        for _ in range(300):
            if engine.ended[0]:
                break
            trajectory.append(engine.clone())
            engine.step(torch.tensor([bot.select_action(engine, 0)]))
        if not engine.ended[0]:
            raise RuntimeError("Latency reference game did not finish")
        for phase, turn in [
            ("opening", 0),
            ("middle", len(trajectory) // 2),
            ("endgame", len(trajectory) - 2),
        ]:
            positions.append((game, phase, max(turn, 0), trajectory[max(turn, 0)]))
    # Warm the inference path so model construction is excluded from move time.
    g, s = encode_state(positions[0][3])
    wrapped(g, s, positions[0][3].legal_action_mask(), 2)
    records = []
    for game, phase, turn, engine in positions:
        cfg = replace(
            search, seed=seed + game * 1000 + turn, move_deadline_s=deadline_s
        )
        perf = PerfCounters(True)
        started = time.monotonic()
        actions, _ = gumbel_root_act(engine, wrapped, search_config=cfg, perf=perf)
        elapsed = time.monotonic() - started
        if not engine.legal_action_mask()[0, actions[0]]:
            raise RuntimeError("Latency search produced an illegal move")
        stats = perf.snapshot()
        completed = stats.get(
            "profile_native_tree_simulations_per_root",
            stats.get(
                "profile_tree_simulations_per_root",
                stats.get("profile_one_ply_candidates", 0),
            ),
        )
        records.append(
            {
                "game_seed": seed + game,
                "phase": phase,
                "turn": turn,
                "wall_s": elapsed,
                "completed_simulations": completed,
                **stats,
            }
        )
    return {
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "search": asdict(search),
        "inference_device": device,
        "torch_threads": torch.get_num_threads(),
        "records": records,
        "deadline_s": deadline_s,
        "wall_budget_s": wall_budget_s,
        "p95_s": float(np.quantile([r["wall_s"] for r in records], 0.95)),
        "qualified": all(
            r["wall_s"] <= wall_budget_s
            and r["completed_simulations"] >= search.num_simulations
            for r in records
        ),
    }
