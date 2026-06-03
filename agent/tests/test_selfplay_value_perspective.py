"""Self-play value targets must be rotated to acting player seat 0."""

from __future__ import annotations

import torch

from agent.train.selfplay import _rotate_for_cp


def test_rotate_for_cp_2p() -> None:
    values = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
    cp = torch.tensor([0, 1], dtype=torch.long)
    rotated = _rotate_for_cp(values, cp, num_players=2)
    assert rotated[0, 0].item() == 1.0
    assert rotated[0, 1].item() == -1.0
    assert rotated[1, 0].item() == -0.5
    assert rotated[1, 1].item() == 0.5


def test_rotate_for_cp_2p_ignores_padded_seats() -> None:
    values = torch.tensor([[1.0, -1.0, -9.0, -9.0], [0.5, -0.5, -9.0, -9.0]])
    cp = torch.tensor([0, 1], dtype=torch.long)
    rotated = _rotate_for_cp(values, cp, num_players=2)
    assert rotated[0, :2].tolist() == [1.0, -1.0]
    assert rotated[1, :2].tolist() == [-0.5, 0.5]


def test_rotate_for_cp_3p_ignores_padded_seats() -> None:
    values = torch.tensor([[10.0, 20.0, 30.0, -9.0]])
    cp = torch.tensor([2], dtype=torch.long)
    rotated = _rotate_for_cp(values, cp, num_players=3)
    assert rotated[0, :3].tolist() == [30.0, 10.0, 20.0]
