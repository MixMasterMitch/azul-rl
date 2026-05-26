"""Learner: loss computation and optimizer step."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..net.model import AzulNet
from .replay_buffer import ReplayBuffer


def make_optimizer(
    net: AzulNet,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)


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
) -> dict[str, float]:
    """One learner step: compute loss and update weights."""
    net.train()
    device = global_feat.device
    use_amp = grad_scaler is not None and device.type == "cuda"

    with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
        policy_logits, value_pred = net(global_feat, source_feat, legal_mask, num_players)

        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = F.kl_div(log_probs, policy_target, reduction="batchmean")
        value_loss = F.mse_loss(value_pred[:, :num_players], value_target[:, :num_players])

        probs = F.softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = policy_loss + value_loss - entropy_bonus * entropy

    optimizer.zero_grad()
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }


def step_from_buffer(
    net: AzulNet,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_players: int = 2,
    entropy_bonus: float = 0.015,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
) -> dict[str, float]:
    batch = buffer.sample(batch_size)
    return step(
        net,
        optimizer,
        *batch,
        num_players=num_players,
        entropy_bonus=entropy_bonus,
        grad_scaler=grad_scaler,
    )
