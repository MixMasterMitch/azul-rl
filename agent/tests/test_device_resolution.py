"""Device resolution supports CPU and CUDA only."""

from __future__ import annotations

import pytest

from agent.train.device import resolve_device


def test_auto_never_returns_mps() -> None:
    device = resolve_device("auto")
    assert device in ("cpu", "cuda")
    assert device != "mps"


def test_mps_rejected() -> None:
    with pytest.raises(ValueError, match="MPS is not supported"):
        resolve_device("mps")
