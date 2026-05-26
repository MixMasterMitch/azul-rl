"""Self-play value targets must be rotated to acting player seat 0."""

from __future__ import annotations

import torch

from agent.train.selfplay import _rotate_for_cp


def test_rotate_for_cp_2p() -> None:
    values = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
    cp = torch.tensor([0, 1], dtype=torch.long)
    rotated = _rotate_for_cp(values, cp)
    assert rotated[0, 0].item() == 1.0
    assert rotated[0, 1].item() == -1.0
    assert rotated[1, 0].item() == -0.5
    assert rotated[1, 1].item() == 0.5
