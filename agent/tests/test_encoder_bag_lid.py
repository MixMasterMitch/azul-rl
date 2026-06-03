"""Encoder exposes bag and box-lid (discard) tile counts."""

from __future__ import annotations

import torch

from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.env import tiles as T
from agent.net import encoder as ENC


def test_encode_state_includes_bag_and_lid() -> None:
    engine = BE.BatchedEngine(2, num_players=2, device="cpu", seed=0)
    global_feat, source_feat = ENC.encode_state(engine)

    assert global_feat.shape == (2, ENC.D_GLOBAL)
    assert source_feat.shape == (2, ENC.NUM_SOURCES, ENC.D_SOURCE)
    assert ENC.D_GLOBAL == 275

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


def test_encode_state_includes_pattern_color_and_floor_details() -> None:
    engine = BE.BatchedEngine(1, num_players=2, device="cpu", seed=2)
    engine.current_player[0] = 1
    engine.pattern_count[0, 1, 2] = 2
    engine.pattern_color[0, 1, 2] = 3
    engine.floor_count[0, 1] = 4
    engine.floor_tiles[0, 1] = torch.tensor([0, 1, 0, 2, 1], dtype=torch.int8)
    engine.floor_slots[0, 1] = torch.tensor(
        [1, 3, A.FLOOR_MARKER, 3, 4, -1, -1],
        dtype=torch.int8,
    )
    engine.floor_first[0, 1] = True

    global_feat, _ = ENC.encode_state(engine)
    seat0 = 19

    row2 = seat0 + 2 * ENC.D_PATTERN_LINE
    assert torch.isclose(global_feat[0, row2], torch.tensor(2 / 3))
    assert global_feat[0, row2 + 1 : row2 + 1 + ENC.NUM_COLORS].tolist() == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]

    floor = seat0 + ENC.D_PATTERN + ENC.D_WALL_FLAT
    assert torch.isclose(global_feat[0, floor], torch.tensor(4 / A.FLOOR_SIZE))
    assert torch.allclose(
        global_feat[0, floor + 1 : floor + 1 + ENC.NUM_COLORS],
        engine.floor_tiles[0, 1].float() / A.FLOOR_SIZE,
    )
    assert global_feat[0, floor + 1 + ENC.NUM_COLORS].item() == 1.0
