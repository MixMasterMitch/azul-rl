"""Balanced multi-teacher consolidation and joint 2p/3p/4p learning."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from ..net.model import AzulNet
from ..net import encoder as ENC
from .checkpointing import (
    load_net_from_checkpoint,
    load_checkpoint_payload,
    save_checkpoint_payload,
)
from .replay_buffer import ReplayBuffer
from .reproducibility import capture_rng_state, restore_rng_state
from ..env.outcomes import REWARD_SEMANTICS_VERSION

COUNTS = (2, 3, 4)


def new_buffer(capacity: int = 200_000, snapshots: int = 4096) -> ReplayBuffer:
    return ReplayBuffer(
        capacity,
        ENC.D_GLOBAL,
        ENC.NUM_SOURCES,
        ENC.D_SOURCE,
        300,
        4,
        snapshot_capacity=snapshots,
    )


def initialize_shared(
    paths: dict[int, str], device: str = "cuda"
) -> tuple[AzulNet, dict[int, AzulNet]]:
    if set(paths) != set(COUNTS):
        raise ValueError("Exactly three specialists are required")
    teachers = {}
    for n, path in paths.items():
        teacher, payload = load_net_from_checkpoint(path)
        if n not in payload.get("trained_player_counts", []):
            raise ValueError(f"Teacher has not been trained for {n} players")
        if teacher.arch != "source_attn" or teacher.aux_score:
            raise ValueError(
                "Shared consolidation requires source-attention teachers without auxiliary heads"
            )
        teachers[n] = teacher.to(device).eval().requires_grad_(False)
    student = deepcopy(teachers[2]).requires_grad_(True)
    with torch.no_grad():
        for n, teacher in teachers.items():
            if teacher.hidden != student.hidden:
                raise ValueError("Teacher widths must match")
            key = str(n - 2)
            student.policy_heads[key].load_state_dict(
                teacher.policy_heads[key].state_dict()
            )
            student.value_heads[key].load_state_dict(
                teacher.value_heads[key].state_dict()
            )
            student.pc_embed.weight[n - 2].copy_(teacher.pc_embed.weight[n - 2])
    student.trained_player_counts = []
    return student, teachers


@torch.no_grad()
def label_corpus(
    teacher: AzulNet, batch: tuple[torch.Tensor, ...], n: int, chunk: int = 1024
) -> tuple[torch.Tensor, ...]:
    """Policy AND value imitation of the selected teacher on donated observations."""
    device = next(teacher.parameters()).device
    g, s, legal = batch[:3]
    policies, values = [], []
    teacher.eval()
    for i in range(0, len(g), chunk):
        p, v = teacher(*(x[i : i + chunk].to(device) for x in (g, s, legal)), n)
        policies.append(p.softmax(-1).cpu())
        values.append(v.cpu())
    return g.cpu(), s.cpu(), legal.cpu(), torch.cat(policies), torch.cat(values)


def donor_corpus(
    path: str,
    teacher: AzulNet,
    n: int,
    *,
    seed: int,
    train_size: int = 100_000,
    validation_size: int = 4096,
) -> tuple[tuple, tuple]:
    payload = load_checkpoint_payload(path)
    if payload.get("trained_player_counts") != [n]:
        raise ValueError("Donor player count differs")
    state = payload["buffer"]
    if state.get("reward_semantics_version") != REWARD_SEMANTICS_VERSION:
        raise ValueError("Donor reward semantics differ")
    size = int(state["size"])
    if size < train_size + validation_size:
        raise ValueError("Donor does not contain enough distinct positions")
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed))[
        : train_size + validation_size
    ]
    batch = tuple(
        state[key][indices].clone()
        for key in ("global_feat", "source_feat", "legal_mask")
    )
    del payload, state
    labelled = label_corpus(teacher, batch, n)
    return tuple(x[:train_size].clone() for x in labelled), tuple(
        x[train_size:].clone() for x in labelled
    )


def sample_corpus(
    corpus: tuple[torch.Tensor, ...], count: int, device: torch.device | str
) -> tuple[torch.Tensor, ...]:
    idx = torch.randint(len(corpus[0]), (count,))
    return tuple(x[idx].to(device) for x in corpus)


def imitation_loss(
    net: AzulNet, batch: tuple[torch.Tensor, ...], n: int
) -> tuple[torch.Tensor, dict]:
    g, s, legal, target_p, target_v = batch[:5]
    logits, values = net(g, s, legal, n)
    kl = F.kl_div(logits.log_softmax(-1), target_p, reduction="batchmean")
    mse = F.mse_loss(values[:, :n], target_v[:, :n])
    return kl + mse, {
        "policy_kl": float(kl.detach()),
        "value_mse": float(mse.detach()),
        "top1_agreement": float(
            (logits.argmax(-1) == target_p.argmax(-1)).float().mean()
        ),
    }


def joint_step(
    net: AzulNet,
    optimizer: torch.optim.Optimizer,
    corpora: dict[int, tuple],
    *,
    batch_size: int = 256,
    buffers: dict[int, ReplayBuffer] | None = None,
    anchor_weight: float = 0.15,
) -> dict:
    """One optimizer step after three equally weighted player-count gradients.

    Each count gets its own homogeneous minibatch; values average over active
    seats. Anchor batches are independent frozen teacher policy/value targets.
    """
    if set(corpora) != set(COUNTS) or (
        buffers is not None and set(buffers) != set(COUNTS)
    ):
        raise ValueError("Joint learning needs all three player counts")
    if not 0 <= anchor_weight <= 1:
        raise ValueError("Invalid anchor weight")
    device = next(net.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    # Deterministic teachers can be consolidated without dropout noise.
    net.train(buffers is not None)
    metrics = {}
    for n in COUNTS:
        anchor = sample_corpus(corpora[n], batch_size, device)
        anchor_loss, stats = imitation_loss(net, anchor, n)
        loss = anchor_loss
        if buffers is not None:
            batch = tuple(
                x.to(device)
                for x in buffers[n].sample(batch_size, include_policy_sims=True)
            )
            g, s, legal, policy, value, sims = batch
            logits, prediction = net(g, s, legal, n)
            per_position = F.kl_div(
                logits.log_softmax(-1), policy, reduction="none"
            ).sum(-1)
            weight = torch.where((sims > 0) & (sims < 256), 0.25, 1.0)
            policy_loss = (per_position * weight).sum() / weight.sum()
            value_loss = F.mse_loss(prediction[:, :n], value[:, :n])
            entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1).mean()
            rl_loss = policy_loss + value_loss - 0.014274964936918592 * entropy
            loss = (1 - anchor_weight) * rl_loss + anchor_weight * anchor_loss
            stats.update(
                rl_policy_loss=float(policy_loss.detach()),
                rl_value_loss=float(value_loss.detach()),
            )
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite joint loss")
        (loss / len(COUNTS)).backward()
        stats["loss"] = float(loss.detach())
        metrics[str(n)] = stats
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite joint gradient")
    optimizer.step()
    net.trained_player_counts = list(COUNTS)
    return {"counts": metrics, "grad_norm": float(norm)}


@torch.no_grad()
def validate_imitation(
    net: AzulNet, corpora: dict[int, tuple], chunk: int = 512
) -> dict:
    net.eval()
    device = next(net.parameters()).device
    result = {}
    for n, corpus in corpora.items():
        totals = {}
        for i in range(0, len(corpus[0]), chunk):
            batch = tuple(x[i : i + chunk].to(device) for x in corpus)
            _, stats = imitation_loss(net, batch, n)
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + len(batch[0]) * v
        result[str(n)] = {k: v / len(corpus[0]) for k, v in totals.items()}
    return result


def save_shared(
    path: Path,
    net: AzulNet,
    optimizer: torch.optim.Optimizer,
    buffers: dict[int, ReplayBuffer],
    progress: dict,
    config: dict,
) -> None:
    payload = {
        "model_state_dict": net.state_dict(),
        "hidden": net.hidden,
        "arch": net.arch,
        "model_version": net.model_version,
        "encoder_version": 3,
        "aux_score": False,
        "trained_player_counts": list(net.trained_player_counts),
        "reward_semantics_version": REWARD_SEMANTICS_VERSION,
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        # The buffer key selects bounded compressed atomic serialization.
        "buffer": {str(n): b.state_dict() for n, b in buffers.items()},
        "progress": progress,
        "config": config,
        "shared_core_schema": 1,
    }
    save_checkpoint_payload(path, payload)


def restore_shared(
    path: Path,
    net: AzulNet,
    optimizer: torch.optim.Optimizer,
    buffers: dict[int, ReplayBuffer],
    config: dict,
) -> dict:
    payload = load_checkpoint_payload(path)
    if payload.get("shared_core_schema") != 1 or payload["config"] != config:
        raise ValueError("Shared resume configuration differs")
    if payload["trained_player_counts"] != list(COUNTS) and not (
        payload["trained_player_counts"] == []
        and payload["progress"]["phase"] == "consolidation"
    ):
        raise ValueError("Shared resume is missing player counts")
    net.load_state_dict(payload["model_state_dict"])
    net.trained_player_counts = payload["trained_player_counts"]
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for n, buffer in buffers.items():
        buffer.load_state_dict(payload["buffer"][str(n)])
    restore_rng_state(payload["rng_state"])
    return payload["progress"]
