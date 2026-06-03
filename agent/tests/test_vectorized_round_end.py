"""Round finalization invariants for the batched engine."""

from __future__ import annotations

import torch

from agent.env.batched_engine import BatchedEngine

def test_vectorized_next_round_refill_conserves_tiles() -> None:
    eng = BatchedEngine(batch_size=4, num_players=2, device="cpu", seed=11)
    rows = torch.tensor([0, 2, 3], dtype=torch.long)

    eng.factory_tiles.zero_()
    eng.center_tiles.zero_()
    eng.bag[rows] = torch.tensor(
        [
            [20, 0, 0, 0, 0],
            [1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.int8,
    )
    eng.box_lid[rows] = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [5, 4, 3, 2, 1],
            [4, 4, 4, 4, 4],
        ],
        dtype=torch.int8,
    )
    before = eng.total_tile_count().clone()

    eng._prepare_next_round_batch(rows)

    assert torch.equal(before, eng.total_tile_count())
    assert torch.equal(
        eng.factory_tiles[rows, : eng.num_factories].sum(dim=(1, 2)),
        torch.tensor([20, 20, 20], dtype=torch.int64),
    )
    assert eng.center_first[rows].all()
