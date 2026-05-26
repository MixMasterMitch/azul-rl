"""Device resolution utilities."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "cpu") -> str:
    """Resolve the best available device name."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def configure_device(device: str) -> dict:
    """Configure device-specific optimizations."""
    info: dict = {"device": device}
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        info["cudnn_benchmark"] = True
        info["matmul_precision"] = "high"
    return info
