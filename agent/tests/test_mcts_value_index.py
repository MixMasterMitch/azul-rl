"""MCTS root value index must match encoder perspective rotation."""

from __future__ import annotations

import torch

from agent.env import batched_engine as BE
from agent.search import gumbel_mcts as G


def test_root_value_index_same_player() -> None:
    parent = torch.tensor([1, 2], dtype=torch.long)
    child = parent.clone()
    idx = G._root_value_index(parent, child)
    assert idx.tolist() == [0, 0]


def test_root_value_index_advances_2p() -> None:
    parent = torch.tensor([0], dtype=torch.long)
    child = torch.tensor([1], dtype=torch.long)
    idx = G._root_value_index(parent, child)
    assert idx.item() == BE.MAX_PLAYERS - 1


def test_root_value_index_advances_3p() -> None:
    parent = torch.tensor([0], dtype=torch.long)
    child = torch.tensor([1], dtype=torch.long)
    idx = G._root_value_index(parent, child)
    assert idx.item() == BE.MAX_PLAYERS - 1
