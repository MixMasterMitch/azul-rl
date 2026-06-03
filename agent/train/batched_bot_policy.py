"""Vectorized heuristic bot policy for training bot self-play."""

from __future__ import annotations

import torch

from ..env import actions as A
from ..env import batched_engine as BE


def _action_layout(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sources: list[int] = []
    colors: list[int] = []
    targets: list[int] = []
    for action in range(A.NUM_ACTIONS):
        s, c, t = A.decode_action(action)
        sources.append(s)
        colors.append(c)
        targets.append(t)
    src = torch.tensor(sources, device=device, dtype=torch.long)
    col = torch.tensor(colors, device=device, dtype=torch.long)
    tgt = torch.tensor(targets, device=device, dtype=torch.long)
    row = torch.clamp(tgt, max=A.NUM_PATTERN_LINES - 1)
    cap = (row + 1).to(torch.float32)
    return src, col, tgt, row, cap


_CACHE: dict[str, tuple[torch.Tensor, ...]] = {}


def batched_heuristic_actions(engine: BE.BatchedEngine) -> torch.Tensor:
    """Pick greedy heuristic actions for every row in the batch (legal actions only)."""
    dev = engine.device
    legal = engine.legal_action_mask()
    b_idx = torch.arange(engine.batch_size, device=dev)
    cp = engine.current_player.to(torch.long)

    cache_key = str(dev)
    if cache_key not in _CACHE:
        _CACHE[cache_key] = _action_layout(dev)
    src, col, tgt, row, cap = _CACHE[cache_key]
    nonfloor = tgt < A.NUM_PATTERN_LINES

    counts = torch.zeros((engine.batch_size, A.NUM_SOURCES, A.NUM_COLORS), dtype=torch.float32, device=dev)
    if engine.num_factories > 0:
        counts[:, : engine.num_factories] = engine.factory_tiles[:, : engine.num_factories].float()
    counts[:, engine.num_factories] = engine.center_tiles.float()

    n = counts[:, src, col]
    pc = engine.pattern_count[b_idx, cp].float()
    cur = pc[:, row]
    space = (cap.unsqueeze(0) - cur).clamp_min(0)
    placed = torch.minimum(n, space)
    excess = (n - placed).clamp_min(0)
    completes = (cur + placed) >= cap.unsqueeze(0)

    pattern_score = (placed / cap.unsqueeze(0)) * 10.0 - excess * 3.0
    pattern_score = pattern_score + torch.where(
        completes & nonfloor.unsqueeze(0),
        20.0 + cap.unsqueeze(0),
        torch.zeros_like(pattern_score),
    )
    floor_score = torch.full((engine.batch_size, A.NUM_ACTIONS), -10.0, device=dev)
    score = torch.where(nonfloor.unsqueeze(0), pattern_score, floor_score)
    score = score.masked_fill(~legal, -1e9)
    return score.argmax(dim=-1)
