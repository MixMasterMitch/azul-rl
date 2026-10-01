"""Bounded state capture and policy-only reanalysis of live replay entries."""

from __future__ import annotations

import random
import time
from dataclasses import replace

import torch

from ..env.engine import GameEngine
from ..net.encoder import encode_state
from ..net.model import AzulNet
from ..search.config import SearchConfig
from ..search.gumbel_mcts import gumbel_root_act
from .replay_buffer import ReplayBuffer
from .policy_surprise import policy_surprise


def finished_sample_mask(
    legal: torch.Tensor,
    policy: torch.Tensor,
    value: torch.Tensor,
    game_idx: torch.Tensor,
    ended: torch.Tensor,
) -> torch.Tensor:
    return (
        ended[game_idx]
        & legal.any(-1)
        & torch.isfinite(policy).all(-1)
        & torch.isfinite(value).all(-1)
    )


class SnapshotRecorder:
    """Capture a random 1/16 of recorded positions, with a strict memory bound.

    Keys refer to the unfiltered trajectory. Filtering uses the very same mask
    as replay insertion, so aborted games can never enter the reanalysis pool.
    A private RNG avoids changing self-play's action or environment streams.
    """

    def __init__(self, capacity: int, seed: int | None) -> None:
        self.capacity = capacity
        self.rng = random.Random((seed or 0) ^ 0x5245414E)
        self.count = 0
        self.snapshots: dict[int, list[int]] = {}

    def capture(self, engine: GameEngine, indices: torch.Tensor) -> None:
        if not self.capacity:
            return
        rows = indices.tolist()
        chosen = [i for i in range(len(rows)) if self.rng.randrange(16) == 0]
        if chosen:
            sub = engine.index_select(
                torch.tensor([rows[i] for i in chosen], device=engine.device)
            )
            for i, snapshot in zip(chosen, sub.state_dict()["snapshots"]):
                self.snapshots[self.count + i] = snapshot
            while len(self.snapshots) > self.capacity:
                del self.snapshots[next(iter(self.snapshots))]
        self.count += len(rows)

    def selected(self, keep: torch.Tensor) -> dict[int, list[int]]:
        if not self.capacity:
            return {}
        if len(keep) != self.count:
            raise ValueError("Snapshot/trajectory length mismatch")
        keep = keep.cpu()
        destination = keep.long().cumsum(0) - 1
        return {
            int(destination[i]): s for i, s in self.snapshots.items() if bool(keep[i])
        }


@torch.no_grad()
def reanalyse_policy_targets(
    net: AzulNet,
    buffer: ReplayBuffer,
    num_players: int,
    search: SearchConfig,
    positions: int,
    batch_size: int = 64,
    seed: int = 0,
) -> dict:
    """Refresh policies in place, retaining observed game outcomes and replay age.

    Search resamples future chance events; stored private draw RNG is never a
    teacher hint. The caller must not mutate the replay/model concurrently.
    """
    if positions < 0 or batch_size < 1:
        raise ValueError("Invalid reanalysis batch size or position count")
    if search.backend != "gumbel_tree":
        raise ValueError("Reanalysis requires tree search")
    started = time.monotonic()
    rng = random.Random(seed)
    indices = rng.sample(list(buffer.snapshots), min(positions, len(buffer.snapshots)))
    was_training = net.training
    net.eval()
    divergence = 0.0
    try:
        for start in range(0, len(indices), batch_size):
            rows = indices[start : start + batch_size]
            idx = torch.tensor(rows, dtype=torch.long, device=buffer.device)
            engine = GameEngine.from_state_dict(
                {
                    "backend": "rust",
                    "version": 1,
                    "num_players": num_players,
                    "snapshots": [buffer.snapshots[i] for i in rows],
                },
                device=str(buffer.device),
            )
            g, s = encode_state(engine)
            legal = engine.legal_action_mask()
            if (
                engine.ended.any()
                or not torch.equal(legal, buffer.legal_mask[idx])
                or not torch.allclose(g, buffer.global_feat[idx], atol=1e-6)
                or not torch.allclose(s, buffer.source_feat[idx], atol=1e-6)
            ):
                raise ValueError(
                    "Reanalysis snapshot does not match its replay position"
                )
            _, policy = gumbel_root_act(
                engine,
                net,
                search_config=replace(
                    search, seed=seed + start, dirichlet_mix=0.0, root_noise_scale=0.0
                ),
            )
            if (
                not torch.isfinite(policy).all()
                or (policy < 0).any()
                or (policy.masked_select(~legal) != 0).any()
                or not torch.allclose(
                    policy.sum(-1),
                    torch.ones(len(rows), device=policy.device),
                    atol=1e-5,
                )
            ):
                raise ValueError("Invalid reanalysis policy")
            divergence += float((buffer.policy_target[idx] - policy).abs().sum())
            buffer.policy_target[idx] = policy
            buffer.policy_sims[idx] = search.num_simulations
            surprise = policy_surprise(
                net, buffer, g, s, legal, policy, search.num_simulations, num_players
            )
            buffer.policy_surprise[idx] = float("nan") if surprise is None else surprise
            buffer._surprise_mean_cache = None
    finally:
        net.train(was_training)
    return {
        "positions": len(indices),
        "eligible": len(buffer.snapshots),
        "mean_policy_l1_change": divergence / max(1, len(indices)),
        "simulations": search.num_simulations,
        "wall_s": round(time.monotonic() - started, 3),
    }
