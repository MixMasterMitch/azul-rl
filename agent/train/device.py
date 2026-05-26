"""Device resolution utilities (CPU and NVIDIA CUDA only)."""

from __future__ import annotations

import os

import torch

SUPPORTED_DEVICES = frozenset({"cpu", "cuda", "auto"})


def resolve_device(requested: str = "cpu") -> str:
    """Resolve the training device: ``cuda`` if available, else ``cpu``.

    ``auto`` prefers CUDA when present. MPS (Apple GPU) is not supported.
    """
    req = requested.lower().strip()
    if req == "mps":
        raise ValueError(
            "MPS is not supported. Use --device cpu or --device cuda (or --device auto)."
        )
    if req not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device {requested!r}. Choose from: {', '.join(sorted(SUPPORTED_DEVICES))}."
        )
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return req


def configure_device(device: str) -> dict:
    """Configure device-specific optimizations."""
    info: dict = {"device": device}
    if device.startswith("cuda"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        info["cudnn_benchmark"] = True
        info["matmul_precision"] = "high"
        info["tf32"] = True
        info["cuda_alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    return info
