"""Checkpoint save/load utilities."""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Optional

import torch

from ..net import encoder as ENC
from ..net.model import AzulNet
from .replay_buffer import ReplayBuffer


_V2_D_GLOBAL = 171
_V2_COMMON_GLOBAL = 19
_V2_D_SEAT_FEAT = 38
_V2_PATTERN_STRIDE = 2
_V2_PATTERN_OFFSET = 0
_V2_WALL_OFFSET = 10
_V2_FLOOR_COUNT_OFFSET = 35
_V2_SCORE_OFFSET = 36
_V2_IS_CURRENT_OFFSET = 37

_V3_COMMON_GLOBAL = 19
_V3_PATTERN_STRIDE = ENC.D_PATTERN_LINE
_V3_PATTERN_OFFSET = 0
_V3_WALL_OFFSET = ENC.D_PATTERN
_V3_FLOOR_COUNT_OFFSET = ENC.D_PATTERN + ENC.D_WALL_FLAT
_V3_SCORE_OFFSET = _V3_FLOOR_COUNT_OFFSET + ENC.D_FLOOR
_V3_IS_CURRENT_OFFSET = _V3_SCORE_OFFSET + ENC.D_SCORE


@dataclasses.dataclass(frozen=True)
class NetSpec:
    hidden: int
    arch: str


def checkpoint_net_spec(payload: dict) -> NetSpec:
    return NetSpec(
        hidden=int(payload.get("hidden", 192)),
        arch=str(payload.get("arch", "attn")),
    )


def checkpoint_net_state_dict(payload: dict) -> dict[str, torch.Tensor]:
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    if "net" in payload:
        return payload["net"]
    raise KeyError("checkpoint payload missing model weights")


def _v2_to_current_global_column_map() -> list[tuple[int, int]]:
    """Map old 171-d global encoder columns into the current richer layout."""
    if ENC.D_GLOBAL == _V2_D_GLOBAL:
        return [(i, i) for i in range(ENC.D_GLOBAL)]

    mapping: list[tuple[int, int]] = [(i, i) for i in range(_V2_COMMON_GLOBAL)]
    for seat in range(ENC.MAX_PLAYERS):
        old_base = _V2_COMMON_GLOBAL + seat * _V2_D_SEAT_FEAT
        new_base = _V3_COMMON_GLOBAL + seat * ENC.D_SEAT_FEAT

        for row in range(5):
            old_fill = old_base + _V2_PATTERN_OFFSET + row * _V2_PATTERN_STRIDE
            old_has_color = old_fill + 1
            new_fill = new_base + _V3_PATTERN_OFFSET + row * _V3_PATTERN_STRIDE
            mapping.append((old_fill, new_fill))
            for color in range(ENC.NUM_COLORS):
                mapping.append((old_has_color, new_fill + 1 + color))

        for idx in range(ENC.D_WALL_FLAT):
            mapping.append((old_base + _V2_WALL_OFFSET + idx, new_base + _V3_WALL_OFFSET + idx))

        mapping.append(
            (old_base + _V2_FLOOR_COUNT_OFFSET, new_base + _V3_FLOOR_COUNT_OFFSET)
        )
        mapping.append((old_base + _V2_SCORE_OFFSET, new_base + _V3_SCORE_OFFSET))
        mapping.append((old_base + _V2_IS_CURRENT_OFFSET, new_base + _V3_IS_CURRENT_OFFSET))

    return mapping


def _adapt_linear_input_weight(
    old_weight: torch.Tensor,
    new_weight: torch.Tensor,
    *,
    old_global_dim: int,
    new_global_dim: int,
) -> torch.Tensor | None:
    """Copy compatible old input columns into a new first-layer weight matrix."""
    if old_weight.shape[0] != new_weight.shape[0]:
        return None
    if old_weight.shape[1] == new_weight.shape[1]:
        return old_weight
    if old_global_dim != _V2_D_GLOBAL or new_global_dim != ENC.D_GLOBAL:
        return None

    old_source_dim = old_weight.shape[1] - old_global_dim
    new_source_dim = new_weight.shape[1] - new_global_dim
    if old_source_dim != new_source_dim:
        return None

    adapted = torch.zeros_like(new_weight)
    for old_col, new_col in _v2_to_current_global_column_map():
        adapted[:, new_col] = old_weight[:, old_col]

    if old_source_dim > 0:
        adapted[:, new_global_dim:] = old_weight[:, old_global_dim:]
    return adapted


def load_model_state_dict_compatible(net: AzulNet, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Load a checkpoint, migrating known old encoder input layouts when needed.

    The v3 encoder adds pattern-line color and floor-detail features. Old v2
    checkpoints can still initialize the model by copying equivalent columns
    and zero-initializing the new columns so the starting policy/value behavior
    is as close as possible to the original checkpoint.
    """
    current = net.state_dict()
    migrated: list[str] = []
    adapted_state = dict(state_dict)

    for key, old_global_dim, new_global_dim in (
        ("g_in.0.weight", _V2_D_GLOBAL, ENC.D_GLOBAL),
        ("flat_trunk.0.weight", _V2_D_GLOBAL, ENC.D_GLOBAL),
    ):
        if key not in adapted_state or key not in current:
            continue
        old_weight = adapted_state[key]
        new_weight = current[key]
        if old_weight.shape == new_weight.shape:
            continue
        adapted = _adapt_linear_input_weight(
            old_weight,
            new_weight,
            old_global_dim=old_global_dim,
            new_global_dim=new_global_dim,
        )
        if adapted is not None:
            adapted_state[key] = adapted
            migrated.append(key)

    net.load_state_dict(adapted_state)
    return migrated


def save_checkpoint(
    path: pathlib.Path | str,
    net: AzulNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
    iteration: int = 0,
    config: Optional[dict] = None,
    buffer: Optional[ReplayBuffer] = None,
) -> None:
    """Save a training checkpoint atomically."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "model_state_dict": net.state_dict(),
        "hidden": net.hidden,
        "arch": net.arch,
        "iteration": iteration,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = config
    if buffer is not None:
        payload["buffer"] = buffer.state_dict()

    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint_payload(
    path: pathlib.Path | str,
    map_location: str | torch.device = "cpu",
) -> dict:
    path = pathlib.Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict checkpoint, got {type(payload)}")
    return payload


def load_net_from_checkpoint(
    path: pathlib.Path | str,
    map_location: str | torch.device = "cpu",
) -> tuple[AzulNet, dict]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    spec = checkpoint_net_spec(payload)
    net = AzulNet(hidden=spec.hidden, arch=spec.arch)
    load_model_state_dict_compatible(net, checkpoint_net_state_dict(payload))
    net = net.to(map_location)
    return net, payload


def load_checkpoint(
    path: pathlib.Path | str,
    net: AzulNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
    buffer: Optional[ReplayBuffer] = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load weights (and optionally optimizer/buffer) into existing objects."""
    payload = load_checkpoint_payload(path, map_location=map_location)
    spec = checkpoint_net_spec(payload)
    if spec.arch != net.arch or spec.hidden != net.hidden:
        raise ValueError(
            f"checkpoint arch/hidden mismatch: ckpt={spec}, net={net.arch}/{net.hidden}"
        )
    migrated = load_model_state_dict_compatible(net, checkpoint_net_state_dict(payload))
    if optimizer is not None and "optimizer_state_dict" in payload and not migrated:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if buffer is not None and "buffer" in payload and not migrated:
        buffer.load_state_dict(payload["buffer"])
    return payload
