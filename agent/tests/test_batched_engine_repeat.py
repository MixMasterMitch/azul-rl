"""Tests for batched engine batching helpers."""

from __future__ import annotations

import torch

from agent.env import batched_engine as BE


def test_repeat_interleave_expands_batch() -> None:
    engine = BE.BatchedEngine(4, num_players=2, device="cpu", seed=0)
    expanded = engine.repeat_interleave(3)
    assert expanded.batch_size == 12
    assert torch.equal(expanded.scores[0], expanded.scores[4])
    assert torch.equal(expanded.scores[0], expanded.scores[8])


def test_index_select_subset() -> None:
    engine = BE.BatchedEngine(6, num_players=2, device="cpu", seed=1)
    sub = engine.index_select(torch.tensor([1, 3, 5]))
    assert sub.batch_size == 3
    assert torch.equal(sub.scores[0], engine.scores[1])
