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

MAX_PLAYERS = BE.MAX_PLAYERS
NUM_COLORS = A.NUM_COLORS
NUM_SOURCES = A.NUM_SOURCES
MAX_FACTORIES = A.MAX_FACTORIES
FLOOR_SIZE = A.FLOOR_SIZE

# Per-seat features
D_PATTERN = 5 * 2  # for each pattern line: count_normalized, has_color
D_WALL_FLAT = 25  # 5x5 wall grid flattened
D_FLOOR = 1  # normalized floor count
D_SCORE = 1
D_IS_CURRENT = 1
D_SEAT_FEAT = D_PATTERN + D_WALL_FLAT + D_FLOOR + D_SCORE + D_IS_CURRENT  # 38

# Global features
D_PC_OH = 3  # one-hot for 2p/3p/4p
D_CENTER_FIRST = 1  # whether first-player marker is still in center
D_GLOBAL = (
    NUM_COLORS  # center tile counts (5)
    + D_CENTER_FIRST  # 1
    + D_PC_OH  # 3
    + MAX_PLAYERS * D_SEAT_FEAT  # 4 * 38 = 152
)  # total: 161

# Per-source features (factory displays + center)
D_SOURCE = NUM_COLORS  # tile counts at each source (5)


def encode_state(
    engine: BE.BatchedEngine,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (global_feat, source_feat).

    global_feat: (B, D_GLOBAL) float32
    source_feat: (B, NUM_SOURCES, D_SOURCE) float32
    """
    device = engine.device
    B = engine.batch_size
    nP = engine.num_players
    cp = engine.current_player.long()  # (B,)
    batch_idx = torch.arange(B, device=device)

    source_feat = torch.zeros((B, NUM_SOURCES, D_SOURCE), dtype=torch.float32, device=device)
    source_feat[:, :MAX_FACTORIES, :] = engine.factory_tiles.float()
    source_feat[:, MAX_FACTORIES, :] = engine.center_tiles.float()

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
        has_color = (pattern_color >= 0).float()
        pattern_feat = torch.stack([fill_ratio, has_color], dim=-1).reshape(B, D_PATTERN)
        if not active:
            pattern_feat = torch.zeros_like(pattern_feat)

        wall_feat = engine.wall[batch_idx, player_idx].float().reshape(B, D_WALL_FLAT)
        if not active:
            wall_feat = torch.zeros_like(wall_feat)

        floor_feat = (engine.floor_count[batch_idx, player_idx].float() / FLOOR_SIZE).unsqueeze(-1)
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

    seats_flat = torch.cat(seat_blocks, dim=-1)
    global_feat = torch.cat([center, center_first, pc_oh, seats_flat], dim=-1)
    return global_feat, source_feat


def encode_state_with_legal(
    engine: BE.BatchedEngine,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (global_feat, source_feat, legal_mask)."""
    g, s = encode_state(engine)
    legal = engine.legal_action_mask()
    return g, s, legal
