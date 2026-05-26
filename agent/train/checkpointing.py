"""Checkpoint save/load utilities."""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Optional

import torch

from ..net.model import AzulNet
from .replay_buffer import ReplayBuffer


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
    net.load_state_dict(checkpoint_net_state_dict(payload))
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
    net.load_state_dict(checkpoint_net_state_dict(payload))
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if buffer is not None and "buffer" in payload:
        buffer.load_state_dict(payload["buffer"])
    return payload
