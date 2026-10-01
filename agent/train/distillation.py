"""Frozen search teacher with a bounded, explicitly sampled policy replay bank."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..env import actions as A
from ..net import encoder as ENC
from ..search.config import SearchConfig
from .checkpointing import load_net_from_checkpoint
from .reanalysis import reanalyse_policy_targets
from .replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from .loop import LoopConfig


def file_hash(path: str) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def make_bank(capacity: int, device: str, *, snapshots: int = 0) -> ReplayBuffer:
    return ReplayBuffer(
        capacity,
        ENC.D_GLOBAL,
        ENC.NUM_SOURCES,
        ENC.D_SOURCE,
        A.NUM_ACTIONS,
        ENC.MAX_PLAYERS,
        device,
        snapshot_capacity=snapshots,
    )


class PolicyTeacher:
    def __init__(self, config: LoopConfig, device: str) -> None:
        self.digest = file_hash(config.distillation_teacher)
        if self.digest != config.distillation_teacher_sha256:
            raise ValueError("Distillation teacher hash mismatch")
        self.net, payload = load_net_from_checkpoint(
            config.distillation_teacher, device
        )
        if payload.get("trained_player_counts") != [config.num_players]:
            raise ValueError("Teacher player-count heads do not match training")
        self.net.eval().requires_grad_(False)
        self.bank = make_bank(config.distillation_capacity, device)

    def state_dict(self) -> dict:
        return {"teacher_sha256": self.digest, "bank": self.bank.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if state.get("teacher_sha256") != self.digest:
            raise ValueError("Saved distillation teacher differs")
        self.bank.load_state_dict(state["bank"])

    @torch.no_grad()
    def refresh(
        self,
        replay: ReplayBuffer,
        num_players: int,
        search: SearchConfig,
        positions: int,
        batch_size: int,
        seed: int,
    ) -> dict:
        # Copy live states before relabelling: original replay policies/outcomes
        # are untouched, and the bank preserves observed outcomes, not values
        # predicted by the teacher. Private RNG keeps self-play RNG unchanged.
        rows = random.Random(seed).sample(
            list(replay.snapshots), min(positions, len(replay.snapshots))
        )
        if not rows:
            raise ValueError(
                "Distillation requires finished replay positions with snapshots"
            )
        idx = torch.tensor(rows, device=replay.device)
        temporary = make_bank(len(rows), str(replay.device), snapshots=len(rows))
        temporary.iteration = replay.iteration
        temporary.add(
            replay.global_feat[idx],
            replay.source_feat[idx],
            replay.legal_mask[idx],
            replay.policy_target[idx],
            replay.value_target[idx],
            snapshots={i: replay.snapshots[row] for i, row in enumerate(rows)},
            policy_sims=replay.policy_sims[idx],
            score_margin=replay.score_margin[idx].masked_fill(
                ~replay.score_margin_valid[idx], float("nan")
            ),
        )
        result = reanalyse_policy_targets(
            self.net,
            temporary,
            num_players,
            replace(search, move_deadline_s=None),
            len(rows),
            batch_size,
            seed,
        )
        self.bank.iteration = replay.iteration
        self.bank.add(
            temporary.global_feat,
            temporary.source_feat,
            temporary.legal_mask,
            temporary.policy_target,
            temporary.value_target,
            policy_sims=search.num_simulations,
            score_margin=temporary.score_margin.masked_fill(
                ~temporary.score_margin_valid, float("nan")
            ),
        )
        return {**result, "bank_size": self.bank.size, "teacher_sha256": self.digest}
