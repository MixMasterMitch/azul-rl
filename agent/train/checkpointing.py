"""Checkpoint save/load utilities."""

from __future__ import annotations

import dataclasses
import io
import os
import pathlib
import zipfile
from typing import Optional

import torch

from ..net import encoder as ENC
from ..net.model import AzulNet
from ..env.outcomes import REWARD_SEMANTICS_VERSION
from .replay_buffer import ReplayBuffer
from .reproducibility import (
    capture_rng_state,
    restore_rng_state,
    require_disk_space,
    tensor_bytes,
)


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
    model_version: int = 1
    encoder_version: int = 3
    aux_score: bool = False


def checkpoint_net_spec(payload: dict) -> NetSpec:
    return NetSpec(
        hidden=int(payload.get("hidden", 192)),
        arch=str(payload.get("arch", "attn")),
        model_version=int(payload.get("model_version", 1)),
        encoder_version=int(payload.get("encoder_version", 3)),
        aux_score=bool(payload.get("aux_score", False)),
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
            mapping.append(
                (old_base + _V2_WALL_OFFSET + idx, new_base + _V3_WALL_OFFSET + idx)
            )

        mapping.append(
            (old_base + _V2_FLOOR_COUNT_OFFSET, new_base + _V3_FLOOR_COUNT_OFFSET)
        )
        mapping.append((old_base + _V2_SCORE_OFFSET, new_base + _V3_SCORE_OFFSET))
        mapping.append(
            (old_base + _V2_IS_CURRENT_OFFSET, new_base + _V3_IS_CURRENT_OFFSET)
        )

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


def load_model_state_dict_compatible(
    net: AzulNet, state_dict: dict[str, torch.Tensor]
) -> list[str]:
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


def warm_start_net(net: AzulNet, payload: dict) -> list[str]:
    """Explicit weights-only initialization; never restore optimizer or replay."""
    spec = checkpoint_net_spec(payload)
    state = checkpoint_net_state_dict(payload)
    if spec.hidden != net.hidden:
        raise ValueError("Warm-start width must match")
    if spec.arch == net.arch:
        if net.aux_score and not spec.aux_score:
            state = {
                **state,
                **{
                    k: v
                    for k, v in net.state_dict().items()
                    if k.startswith("score_head.")
                },
            }
        return load_model_state_dict_compatible(net, state)
    if spec.arch != "attn" or net.arch != "source_attn":
        raise ValueError(f"Unsupported warm start: {spec.arch} -> {net.arch}")
    legacy = AzulNet(hidden=spec.hidden, arch="attn")
    load_model_state_dict_compatible(legacy, state)
    weights = {
        k: v
        for k, v in legacy.state_dict().items()
        if not k.startswith("policy_heads.")
    }
    missing, unexpected = net.load_state_dict(weights, strict=False)
    if unexpected or any(
        not k.startswith(("policy_heads.", "source_type.", "source_policy_norm."))
        for k in missing
    ):
        raise ValueError(f"Invalid warm-start mapping: {missing}, {unexpected}")
    return list(missing)


def save_checkpoint(
    path: pathlib.Path | str,
    net: AzulNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
    iteration: int = 0,
    config: Optional[dict] = None,
    buffer: Optional[ReplayBuffer] = None,
    progress: dict | None = None,
    distillation_state: dict | None = None,
) -> None:
    """Save a training checkpoint atomically."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "reward_semantics_version": REWARD_SEMANTICS_VERSION,
        "model_state_dict": net.state_dict(),
        "hidden": net.hidden,
        "arch": net.arch,
        "aux_score": net.aux_score,
        "iteration": iteration,
        "model_version": net.model_version,
        "encoder_version": 3,
        "trained_player_counts": [int(config["num_players"])]
        if config and "num_players" in config
        else getattr(net, "trained_player_counts", []),
        "rng_state": capture_rng_state(),
        "progress": progress or {},
        "checkpoint_compression": "deflate" if buffer is not None else "stored",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if config is not None:
        payload["config"] = config
    if buffer is not None:
        payload["buffer"] = buffer.state_dict()
    if distillation_state is not None:
        payload["distillation"] = distillation_state

    save_checkpoint_payload(path, payload)


def save_checkpoint_payload(path: pathlib.Path | str, payload: dict) -> None:
    """Atomically save complete state without regenerating RNG or optimizer data.

    Used when forking an explicitly declared experiment from a durable resume.
    Ordinary resumes still enforce immutable training settings in the loop.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = "buffer" in payload
    require_disk_space(path, 0 if compressed else tensor_bytes(payload))
    tmp = path.with_suffix(".tmp")
    try:
        if compressed:
            _write_compressed_checkpoint(payload, tmp)
        else:
            torch.save(payload, tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_compressed_checkpoint(payload: dict, path: pathlib.Path) -> None:
    """Losslessly compress a standard PyTorch ZIP archive without a raw disk copy.

    Sparse Azul observations and policy targets compress substantially. PyTorch's
    regular loader reads DEFLATE records directly; mmap must not be used on them.
    Serialization uses host RAM temporarily, retaining the old durable checkpoint
    until the compressed replacement has been flushed and atomically installed.
    """
    with io.BytesIO() as raw:
        torch.save(payload, raw)
        raw.seek(0)
        level = (
            6
            if payload.get("config", {}).get("bounded_checkpoint_storage", False)
            else 1
        )
        with (
            zipfile.ZipFile(raw) as source,
            zipfile.ZipFile(
                path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=level,
            ) as target,
        ):
            for member in source.infolist():
                with (
                    source.open(member) as incoming,
                    target.open(member.filename, "w", force_zip64=True) as outgoing,
                ):
                    while chunk := incoming.read(8 * 1024**2):
                        require_disk_space(path, len(chunk))
                        outgoing.write(chunk)


def load_checkpoint_payload(
    path: pathlib.Path | str,
    map_location: str | torch.device = "cpu",
) -> dict:
    path = pathlib.Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict checkpoint, got {type(payload)}")
    return payload


def checkpoint_launch_headroom(
    path: pathlib.Path | None, replay_bytes: int, *, bounded_storage: bool = False
) -> int:
    """Use observed compressed size for a resume; new runs reserve raw replay size.

    Default to 1 GiB or twice the existing archive for growth. A campaign with
    an explicitly bounded retention plan can reserve 125% of its observed
    archive instead; atomic writes still enforce the separate 256 MiB reserve.
    """
    if path is not None and path.exists() and replay_bytes:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if members and all(
                member.compress_type == zipfile.ZIP_DEFLATED for member in members
            ):
                if bounded_storage:
                    return min(
                        replay_bytes,
                        max(256 * 1024**2, int(1.25 * path.stat().st_size)),
                    )
                return min(replay_bytes, max(1024**3, 2 * path.stat().st_size))
    return replay_bytes


def load_net_from_checkpoint(
    path: pathlib.Path | str,
    map_location: str | torch.device = "cpu",
) -> tuple[AzulNet, dict]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    spec = checkpoint_net_spec(payload)
    with torch.random.fork_rng(devices=[]):
        net = AzulNet(hidden=spec.hidden, arch=spec.arch, aux_score=spec.aux_score)
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
    if (
        spec.arch != net.arch
        or spec.hidden != net.hidden
        or spec.aux_score != net.aux_score
    ):
        raise ValueError(
            f"checkpoint arch/hidden mismatch: ckpt={spec}, net={net.arch}/{net.hidden}"
        )
    migrated = load_model_state_dict_compatible(net, checkpoint_net_state_dict(payload))
    if optimizer is not None and "optimizer_state_dict" in payload and not migrated:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if buffer is not None and "buffer" in payload and not migrated:
        buffer.load_state_dict(payload["buffer"])
    if "rng_state" in payload and not migrated:
        restore_rng_state(payload["rng_state"])
    return payload
