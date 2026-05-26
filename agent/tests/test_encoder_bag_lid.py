"""Encoder exposes bag and box-lid (discard) tile counts."""

from __future__ import annotations

import torch

from agent.env import batched_engine as BE
from agent.env import tiles as T
from agent.net import encoder as ENC


def test_encode_state_includes_bag_and_lid() -> None:
    engine = BE.BatchedEngine(2, num_players=2, device="cpu", seed=0)
    global_feat, source_feat = ENC.encode_state(engine)

    assert global_feat.shape == (2, ENC.D_GLOBAL)
    assert source_feat.shape == (2, ENC.NUM_SOURCES, ENC.D_SOURCE)
    assert ENC.D_GLOBAL == 171

    # Bag/lid slice sits after center (5), center_first (1), pc_oh (3).
    bag_slice = slice(9, 9 + T.NUM_COLORS)
    lid_slice = slice(14, 14 + T.NUM_COLORS)
    assert torch.allclose(global_feat[:, bag_slice], engine.bag.float() / T.TILES_PER_COLOR)
    assert torch.allclose(global_feat[:, lid_slice], engine.box_lid.float() / T.TILES_PER_COLOR)


def test_tile_conservation_per_color() -> None:
    """Each color has exactly TILES_PER_COLOR tiles across all zones."""
    engine = BE.BatchedEngine(1, num_players=2, device="cpu", seed=1)
    g, s = ENC.encode_state(engine)

    center = g[0, : T.NUM_COLORS] * T.TILES_PER_COLOR
    bag = g[0, 9:14] * T.TILES_PER_COLOR
    lid = g[0, 14:19] * T.TILES_PER_COLOR
    factories = s[0, : engine.num_factories].sum(dim=0)

    on_board = (
        engine.pattern_count[0].sum(dim=0).float()
        + engine.wall[0].sum(dim=(0, 1)).float()
        + engine.floor_tiles[0].sum(dim=0).float()
    )

    total = center + bag + lid + factories + on_board
    assert torch.allclose(total, torch.full((T.NUM_COLORS,), float(T.TILES_PER_COLOR)))
