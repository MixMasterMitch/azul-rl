"""Training stability helpers (finite weights, sanitized replay targets)."""

from __future__ import annotations

import pathlib
from typing import Optional

import torch

from ..net.model import AzulNet
from .checkpointing import (
    checkpoint_net_state_dict,
    load_checkpoint_payload,
    load_model_state_dict_compatible,
)
from .replay_buffer import ReplayBuffer


def net_parameters_finite(net: AzulNet) -> bool:
    for param in net.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def sanitize_value_targets(value_target: torch.Tensor) -> torch.Tensor:
    return value_target.clamp(-1.0, 1.0)


def sanitize_policy_targets(
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    pt = policy_target.clamp(min=0).float()
    pt = pt * legal_mask.to(pt.dtype)
    total = pt.sum(dim=-1, keepdim=True)
    valid = (total.squeeze(-1) > 1e-8) & legal_mask.any(dim=-1)
    safe = total.clamp_min(1e-8)
    pt = pt / safe
    if (~valid).any():
        bad = (~valid).nonzero(as_tuple=True)[0]
        pt[bad] = 0.0
    return pt


def restore_net_from_checkpoint(
    net: AzulNet,
    path: str | pathlib.Path,
    device: str,
) -> None:
    payload = load_checkpoint_payload(path, map_location=device)
    load_model_state_dict_compatible(net, checkpoint_net_state_dict(payload))


def reset_buffer(buffer: ReplayBuffer) -> None:
    buffer.size = 0
    buffer.pos = 0
