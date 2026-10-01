"""Observed final score differences; independent of binary win/tie targets."""

from __future__ import annotations

import torch


def final_score_margins(
    scores: torch.Tensor,
    game_indices: torch.Tensor,
    current_players: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """Two-player final score margin from the player acting at each recorded state.

    Callers apply the same finished/valid-position mask as ordinary replay.
    Scores include endgame bonuses. Margins are undiscounted and stored in points.
    """
    if num_players != 2:
        return torch.full(game_indices.shape, float("nan"), device=game_indices.device)
    scores = scores.to(device=game_indices.device, dtype=torch.float32)
    cp = current_players.long()
    return scores[game_indices, cp] - scores[game_indices, 1 - cp]
