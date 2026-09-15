"""Learner: loss computation and optimizer step."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..net.model import AzulNet
from .instrumentation import PerfCounters, maybe_time, tensor_nbytes
from .replay_buffer import ReplayBuffer


def make_optimizer(
    net: AzulNet,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)


def _sanitize_policy_target(
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    pt = policy_target.clamp(min=0).float()
    pt = pt * legal_mask.to(pt.dtype)
    total = pt.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return pt / total


def step(
    net: AzulNet,
    optimizer: torch.optim.Optimizer,
    global_feat: torch.Tensor,
    source_feat: torch.Tensor,
    legal_mask: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    num_players: int = 2,
    entropy_bonus: float = 0.015,
    max_grad_norm: float = 1.0,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
    perf: PerfCounters | None = None,
) -> dict[str, float]:
    """One learner step: compute loss and update weights (fp32 for stability)."""
    valid = legal_mask.any(dim=-1)
    if not valid.any():
        return {
            "loss": float("nan"),
            "policy_loss": float("nan"),
            "value_loss": float("nan"),
            "entropy": float("nan"),
            "skipped": 1.0,
        }

    if not valid.all():
        idx = valid.nonzero(as_tuple=True)[0]
        global_feat = global_feat[idx]
        source_feat = source_feat[idx]
        legal_mask = legal_mask[idx]
        policy_target = policy_target[idx]
        value_target = value_target[idx]

    net.train()
    # Full fp32 learner step — AMP here was associated with NaN/Inf gradients.
    with maybe_time(perf, "learner_forward"):
        policy_logits, value_pred = net(global_feat, source_feat, legal_mask, num_players)

    with maybe_time(perf, "learner_loss"):
        target_p = _sanitize_policy_target(policy_target, legal_mask)
        target_v = value_target.float().clamp(-1.0, 1.0)

        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = F.kl_div(log_probs, target_p, reduction="batchmean")
        value_loss = F.mse_loss(value_pred[:, :num_players], target_v[:, :num_players])

        probs = F.softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = policy_loss + value_loss - entropy_bonus * entropy

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss": float("nan"),
            "policy_loss": float("nan"),
            "value_loss": float("nan"),
            "entropy": float("nan"),
            "skipped": 1.0,
            "nonfinite_loss": 1.0,
        }

    with maybe_time(perf, "learner_zero_grad"):
        optimizer.zero_grad()
    with maybe_time(perf, "learner_backward"):
        loss.backward()
    with maybe_time(perf, "learner_clip_grad"):
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "skipped": 1.0,
            "nonfinite_gradient": 1.0,
            "grad_norm": grad_norm.item(),
        }
    with maybe_time(perf, "learner_optimizer_step"):
        optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "skipped": 0.0,
        "value_bias": (value_pred[:, :num_players] - target_v[:, :num_players]).mean().item(),
        "value_sign_accuracy": ((value_pred[:, :num_players] >= 0) == (target_v[:, :num_players] >= 0)).float().mean().item(),
        "grad_norm": grad_norm.item(),
    }


def step_from_buffer(
    net: AzulNet,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_players: int = 2,
    entropy_bonus: float = 0.015,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
    perf: PerfCounters | None = None,
) -> dict[str, float]:
    del grad_scaler
    with maybe_time(perf, "learner_sample_batch"):
        batch = buffer.sample(batch_size)
    if perf is not None:
        perf.add_count("learner_samples", batch_size)
        perf.add_count("learner_batch_mb", sum(tensor_nbytes(t) for t in batch) / (1024**2))
    return step(
        net,
        optimizer,
        *batch,
        num_players=num_players,
        entropy_bonus=entropy_bonus,
        perf=perf,
    )
