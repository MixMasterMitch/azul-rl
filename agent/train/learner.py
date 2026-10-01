"""Learner: loss computation and optimizer step."""

from __future__ import annotations

import math
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
    policy_weights: torch.Tensor | None = None,
    score_margin: torch.Tensor | None = None,
    aux_score_weight: float = 0.0,
    aux_score_scale: float = 50.0,
) -> dict[str, float]:
    """One learner step: compute loss and update weights (fp32 for stability)."""
    if (
        not math.isfinite(aux_score_weight)
        or aux_score_weight < 0
        or not math.isfinite(aux_score_scale)
        or aux_score_scale <= 0
    ):
        raise ValueError("Invalid auxiliary score loss settings")
    if aux_score_weight and (
        score_margin is None or not net.aux_score or num_players != 2
    ):
        raise ValueError(
            "Auxiliary score loss needs labels and a two-player score head"
        )
    if score_margin is not None:
        if (
            score_margin.shape != (global_feat.shape[0],)
            or torch.isinf(score_margin).any()
        ):
            raise ValueError("Invalid score margins")
        score_margin = score_margin.to(global_feat.device)
    if policy_weights is not None:
        if (
            policy_weights.shape != (global_feat.shape[0],)
            or not torch.isfinite(policy_weights).all()
            or (policy_weights <= 0).any()
        ):
            raise ValueError(
                "policy_weights must be finite and positive, one per sample"
            )
        policy_weights = policy_weights.to(global_feat.device)
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
        if score_margin is not None:
            score_margin = score_margin[idx]
        if policy_weights is not None:
            policy_weights = policy_weights[idx]

    net.train()
    # Full fp32 learner step — AMP here was associated with NaN/Inf gradients.
    with maybe_time(perf, "learner_forward"):
        if aux_score_weight:
            policy_logits, value_pred, score_pred = net.forward_with_score(
                global_feat, source_feat, legal_mask, num_players
            )
        else:
            policy_logits, value_pred = net(
                global_feat, source_feat, legal_mask, num_players
            )

    with maybe_time(perf, "learner_loss"):
        target_p = _sanitize_policy_target(policy_target, legal_mask)
        target_v = value_target.float().clamp(-1.0, 1.0)

        log_probs = F.log_softmax(policy_logits, dim=-1)
        if policy_weights is None:
            policy_loss = F.kl_div(log_probs, target_p, reduction="batchmean")
        else:
            per_position = F.kl_div(log_probs, target_p, reduction="none").sum(-1)
            weights = policy_weights.to(
                device=per_position.device, dtype=per_position.dtype
            )
            policy_loss = (per_position * weights).sum() / weights.sum()
        value_loss = F.mse_loss(value_pred[:, :num_players], target_v[:, :num_players])

        probs = F.softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = policy_loss + value_loss - entropy_bonus * entropy
        score_loss = loss.new_zeros(())
        labelled = torch.isfinite(score_margin) if score_margin is not None else None
        score_fraction = float(labelled.float().mean()) if labelled is not None else 0.0
        if aux_score_weight and labelled.any():
            # Divide by the whole batch: the extra loss ramps naturally as fresh,
            # labelled games replace historical replay. Never train on missing labels.
            score_loss = (
                F.smooth_l1_loss(
                    score_pred[labelled],
                    score_margin[labelled] / aux_score_scale,
                    reduction="sum",
                )
                / global_feat.shape[0]
            )
            loss = loss + aux_score_weight * score_loss

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
        "value_bias": (value_pred[:, :num_players] - target_v[:, :num_players])
        .mean()
        .item(),
        "value_sign_accuracy": (
            (value_pred[:, :num_players] >= 0) == (target_v[:, :num_players] >= 0)
        )
        .float()
        .mean()
        .item(),
        "grad_norm": grad_norm.item(),
        "aux_score_loss": score_loss.item(),
        "aux_score_label_fraction": score_fraction,
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
    policy_fast_weight: float = 1.0,
    policy_full_sims: int = 256,
    policy_surprise_fraction: float = 0.0,
    policy_surprise_min_sims: int = 256,
    policy_surprise_max_weight: float = 4.0,
    teacher_buffer: ReplayBuffer | None = None,
    teacher_fraction: float = 0.0,
    aux_score_weight: float = 0.0,
    aux_score_scale: float = 50.0,
) -> dict[str, float]:
    del grad_scaler
    if (
        not math.isfinite(policy_fast_weight)
        or not 0 < policy_fast_weight <= 1
        or type(policy_full_sims) is not int
        or policy_full_sims < 1
    ):
        raise ValueError("Invalid policy search-budget weighting configuration")
    if not math.isfinite(teacher_fraction) or not 0 <= teacher_fraction < 1:
        raise ValueError("teacher_fraction must be finite and in [0, 1)")
    teacher_count = round(batch_size * teacher_fraction)
    if teacher_count and (
        teacher_buffer is None or not teacher_buffer.size or teacher_count >= batch_size
    ):
        raise ValueError(
            "Teacher mixture requires a populated bank and ordinary samples"
        )
    with maybe_time(perf, "learner_sample_batch"):
        sampling = dict(
            surprise_fraction=policy_surprise_fraction,
            surprise_min_sims=policy_surprise_min_sims,
            surprise_max_weight=policy_surprise_max_weight,
        )
        if teacher_count and policy_surprise_fraction:
            raise ValueError(
                "Policy surprise and distillation must be tested separately"
            )
        margins = None
        if getattr(net, "aux_score", False):
            if teacher_count:
                raise ValueError(
                    "Auxiliary score and distillation must be tested separately"
                )
            *batch, budgets, margins = buffer.sample(
                batch_size,
                include_policy_sims=True,
                include_score_margin=True,
                **sampling,
            )
        else:
            *batch, budgets = buffer.sample(
                batch_size - teacher_count, include_policy_sims=True, **sampling
            )
        fast = (budgets > 0) & (budgets < policy_full_sims)
        weights = (
            torch.where(fast, policy_fast_weight, 1.0)
            if policy_fast_weight != 1.0
            else None
        )
        if teacher_count:
            *teacher_batch, teacher_budgets = teacher_buffer.sample(
                teacher_count, include_policy_sims=True
            )
            # Preserve fast/full weighting within ordinary replay, while keeping
            # the teacher's total policy contribution exactly the declared mix.
            ordinary_weights = (
                weights
                if weights is not None
                else torch.ones_like(budgets, dtype=torch.float32)
            )
            ordinary_weights = ordinary_weights / ordinary_weights.mean()
            weights = torch.cat(
                (ordinary_weights, torch.ones(teacher_count, device=budgets.device))
            )
            batch = [torch.cat((a, b)) for a, b in zip(batch, teacher_batch)]
            budgets = torch.cat((budgets, teacher_budgets))
    if perf is not None:
        perf.add_count("learner_samples", batch_size)
        perf.add_count(
            "learner_batch_mb", sum(tensor_nbytes(t) for t in batch) / (1024**2)
        )
    metrics = step(
        net,
        optimizer,
        *batch,
        num_players=num_players,
        entropy_bonus=entropy_bonus,
        perf=perf,
        policy_weights=weights,
        score_margin=margins,
        aux_score_weight=aux_score_weight,
        aux_score_scale=aux_score_scale,
    )
    metrics.update(
        distillation_fraction=teacher_count / batch_size,
        policy_surprise_sampled_fraction=buffer.last_sample_surprise_fraction,
        policy_surprise_sampled_mean=buffer.last_sample_surprise_mean,
        policy_budget_known_fraction=float((budgets > 0).float().mean()),
        policy_full_fraction=float((budgets >= policy_full_sims).float().mean()),
        policy_weight_mean=float(weights.mean()) if weights is not None else 1.0,
    )
    return metrics
