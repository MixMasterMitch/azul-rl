"""Official game outcomes shared by search and training (absolute seat order)."""
from __future__ import annotations

import torch
from . import batched_engine as BE

REWARD_MODES = ("binary", "score_scaled")

def final_values_binary(
    engine: BE.BatchedEngine, num_players: int
) -> torch.Tensor:
    """Winner +1, losers -1 per absolute seat."""
    native_values = getattr(engine, "final_values", None)
    if native_values is not None:
        return native_values("binary")
    B = engine.batch_size
    dev = engine.device
    values = torch.full((B, BE.MAX_PLAYERS), -1.0, dtype=torch.float32, device=dev)
    winners = engine.get_winners().to(torch.long)
    ended = engine.ended
    if ended.any():
        rows = ended.nonzero(as_tuple=True)[0]
        w = winners[rows]
        single = w >= 0
        if single.any():
            r = rows[single]
            values[r, w[single].long()] = 1.0
        shared = w == BE.SHARED_VICTORY
        if shared.any():
            r = rows[shared]
            tied = _shared_victory_mask(engine, r, num_players)
            values[r, :num_players] = torch.where(
                tied,
                torch.ones_like(tied, dtype=torch.float32),
                -torch.ones_like(tied, dtype=torch.float32),
            )
    return values


def _shared_victory_mask(
    engine: BE.BatchedEngine,
    rows: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """Return active seats sharing the official win on ended tied rows."""
    scores = engine.scores[rows, :num_players].to(torch.int32)
    rows_complete = engine.wall[rows, :num_players].all(dim=-1).sum(dim=-1).to(torch.int32)
    max_score = scores.max(dim=-1, keepdim=True).values
    score_best = scores == max_score
    rows_for_best = rows_complete.masked_fill(~score_best, -1)
    max_rows = rows_for_best.max(dim=-1, keepdim=True).values
    return score_best & (rows_complete == max_rows)


def final_values_score_scaled(
    engine: BE.BatchedEngine, num_players: int
) -> torch.Tensor:
    """Winner(s) +1; losers -1/(n-1) + (score/winner_score)^2."""
    native_values = getattr(engine, "final_values", None)
    if native_values is not None:
        return native_values("score_scaled")
    B = engine.batch_size
    dev = engine.device
    values = torch.full((B, BE.MAX_PLAYERS), -1.0, dtype=torch.float32, device=dev)
    winners = engine.get_winners().to(torch.long)
    scores = engine.scores[:, :num_players].float()
    ended = engine.ended
    if not ended.any():
        return values

    rows = ended.nonzero(as_tuple=True)[0]
    w = winners[rows]
    single = w >= 0
    if single.any():
        r = rows[single]
        ws = w[single].long()
        winner_score = scores[r].gather(1, ws.unsqueeze(1)).squeeze(1).clamp_min(1.0)
        loss_base = -1.0 / max(num_players - 1, 1)
        ratio = (scores[r] / winner_score.unsqueeze(1)).pow(2)
        row_values = (loss_base + ratio).clamp(-1.0, 1.0)
        row_values.scatter_(1, ws.unsqueeze(1), 1.0)
        values[r, :num_players] = row_values
    shared = w == BE.SHARED_VICTORY
    if shared.any():
        r = rows[shared]
        tied = _shared_victory_mask(engine, r, num_players)
        winner_score = scores[r].masked_fill(~tied, -1).max(dim=-1).values.clamp_min(1.0)
        loss_base = -1.0 / max(num_players - 1, 1)
        ratio = (scores[r] / winner_score.unsqueeze(1)).pow(2)
        row_values = (loss_base + ratio).clamp(-1.0, 1.0)
        values[r, :num_players] = torch.where(tied, torch.ones_like(row_values), row_values)
    return values


def final_values(
    engine: BE.BatchedEngine, num_players: int, reward_mode: str
) -> torch.Tensor:
    if reward_mode == "binary":
        return final_values_binary(engine, num_players)
    if reward_mode == "score_scaled":
        return final_values_score_scaled(engine, num_players)
    raise ValueError(f"Unknown reward_mode={reward_mode!r}. Choose from {REWARD_MODES}.")


