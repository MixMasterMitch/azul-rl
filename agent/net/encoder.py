"""Encodes batched engine state into feature tensors for the network.

Produces two tensors per batch:
- `global_feat` (B, D_GLOBAL): fixed-size dense vector with factory state,
  center state, player-count one-hot, current player info, etc.
- `source_feat` (B, NUM_SOURCES, D_SOURCE): per-source feature rows for
  each factory display and the center.

Perspective: encoding is from the current player's point of view. Other
players' state is attached as additional per-seat features ordered cyclically.
"""

from __future__ import annotations

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..env import tiles as T

MAX_PLAYERS = BE.MAX_PLAYERS
NUM_COLORS = A.NUM_COLORS
NUM_SOURCES = A.NUM_SOURCES
MAX_FACTORIES = A.MAX_FACTORIES
FLOOR_SIZE = A.FLOOR_SIZE

# Per-seat features
D_PATTERN_LINE = 1 + NUM_COLORS  # count_normalized + committed color one-hot
D_PATTERN = 5 * D_PATTERN_LINE
D_WALL_FLAT = 25  # 5x5 wall grid flattened
D_FLOOR_COUNT = 1  # normalized floor count
D_FLOOR_TILES = NUM_COLORS  # floor tile color counts
D_FLOOR_FIRST = 1  # whether this floor has the first-player marker
D_FLOOR = D_FLOOR_COUNT + D_FLOOR_TILES + D_FLOOR_FIRST
D_SCORE = 1
D_IS_CURRENT = 1
D_SEAT_FEAT = D_PATTERN + D_WALL_FLAT + D_FLOOR + D_SCORE + D_IS_CURRENT  # 64

# Global features
D_PC_OH = 3  # one-hot for 2p/3p/4p
D_CENTER_FIRST = 1  # whether first-player marker is still in center
D_BAG = NUM_COLORS  # draw bag counts per color (normalized)
D_BOX_LID = NUM_COLORS  # discard / box-lid counts per color (normalized)
D_GLOBAL = (
    NUM_COLORS  # center tile counts (5)
    + D_CENTER_FIRST  # 1
    + D_PC_OH  # 3
    + D_BAG  # 5
    + D_BOX_LID  # 5
    + MAX_PLAYERS * D_SEAT_FEAT  # 4 * 64 = 256
)  # total: 275

# Per-source features (factory displays + center)
D_SOURCE = NUM_COLORS  # tile counts at each source (5)


def encode_state(
    engine: BE.BatchedEngine,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (global_feat, source_feat).

    global_feat: (B, D_GLOBAL) float32
    source_feat: (B, NUM_SOURCES, D_SOURCE) float32
    """
    native_encode = getattr(engine, "encode_state", None)
    if native_encode is not None:
        return native_encode()
    device = engine.device
    B = engine.batch_size
    nP = engine.num_players
    cp = engine.current_player.long()  # (B,)
    batch_idx = torch.arange(B, device=device)

    source_feat = torch.zeros((B, NUM_SOURCES, D_SOURCE), dtype=torch.float32, device=device)
    source_feat[:, :MAX_FACTORIES, :] = engine.factory_tiles.float()
    source_feat[:, engine.num_factories, :] = engine.center_tiles.float()

    center = engine.center_tiles.float()
    center_first = engine.center_first.float().unsqueeze(-1)
    pc_idx = torch.full((B,), nP - 2, dtype=torch.long, device=device)
    pc_oh = torch.nn.functional.one_hot(pc_idx, num_classes=3).float()

    seat_blocks: list[torch.Tensor] = []
    capacities = torch.arange(1, 6, device=device, dtype=torch.float32)

    for seat_offset in range(MAX_PLAYERS):
        active = seat_offset < nP
        player_idx = (cp + seat_offset) % nP

        pattern_count = engine.pattern_count[batch_idx, player_idx]  # (B, 5)
        pattern_color = engine.pattern_color[batch_idx, player_idx]  # (B, 5)

        fill_ratio = pattern_count.float() / capacities.unsqueeze(0)
        has_color = pattern_color >= 0
        color_oh = torch.nn.functional.one_hot(
            pattern_color.clamp_min(0).long(),
            num_classes=NUM_COLORS,
        ).float()
        color_oh = color_oh * has_color.unsqueeze(-1).float()
        pattern_feat = torch.cat([fill_ratio.unsqueeze(-1), color_oh], dim=-1).reshape(
            B, D_PATTERN
        )
        if not active:
            pattern_feat = torch.zeros_like(pattern_feat)

        wall_feat = engine.wall[batch_idx, player_idx].float().reshape(B, D_WALL_FLAT)
        if not active:
            wall_feat = torch.zeros_like(wall_feat)

        floor_count_feat = (
            engine.floor_count[batch_idx, player_idx].float() / FLOOR_SIZE
        ).unsqueeze(-1)
        floor_tiles_feat = engine.floor_tiles[batch_idx, player_idx].float() / FLOOR_SIZE
        floor_first_feat = engine.floor_first[batch_idx, player_idx].float().unsqueeze(-1)
        floor_feat = torch.cat(
            [floor_count_feat, floor_tiles_feat, floor_first_feat],
            dim=-1,
        )
        score_feat = (engine.scores[batch_idx, player_idx].float() / 100.0).unsqueeze(-1)
        if not active:
            floor_feat = torch.zeros_like(floor_feat)
            score_feat = torch.zeros_like(score_feat)

        is_current = torch.zeros((B, 1), dtype=torch.float32, device=device)
        if seat_offset == 0:
            is_current[:] = 1.0

        seat_blocks.append(
            torch.cat([pattern_feat, wall_feat, floor_feat, score_feat, is_current], dim=-1)
        )

    bag_feat = engine.bag.float() / float(T.TILES_PER_COLOR)
    lid_feat = engine.box_lid.float() / float(T.TILES_PER_COLOR)

    seats_flat = torch.cat(seat_blocks, dim=-1)
    global_feat = torch.cat(
        [center, center_first, pc_oh, bag_feat, lid_feat, seats_flat], dim=-1
    )
    return global_feat, source_feat


def encode_state_with_legal(
    engine: BE.BatchedEngine,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (global_feat, source_feat, legal_mask)."""
    g, s = encode_state(engine)
    legal = engine.legal_action_mask()
    return g, s, legal
