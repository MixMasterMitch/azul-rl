"""Gumbel-root MCTS for Azul.

Uses a "Gumbel root + 1-ply learned value" scheme:
1. Compute prior logits from the policy head
2. Add Gumbel noise to select top-K candidate actions at the root
3. Expand all B×K children in one batched engine step
4. Score child states with a single value-network pass
5. Combine Q-estimates with priors to select final action
6. Produce an improved policy as the training target
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import encoder as ENC
from ..net import model as M
from ..train.instrumentation import PerfCounters, maybe_time

NUM_ACTIONS = A.NUM_ACTIONS


def _sample_gumbel(shape, device):
    u = torch.rand(shape, device=device).clamp_min(1e-9)
    return -torch.log(-torch.log(u))


def _score_completed_placements(
    wall: torch.Tensor,
    row: int,
    col: torch.Tensor,
) -> torch.Tensor:
    """Vectorized Azul placement score for one row with variable columns."""
    n = wall.shape[0]
    batch_idx = torch.arange(n, device=wall.device)

    h_count = torch.ones((n,), dtype=torch.int16, device=wall.device)
    active = torch.ones((n,), dtype=torch.bool, device=wall.device)
    for step in range(1, 5):
        c = col - step
        active = active & (c >= 0)
        hit = active & wall[batch_idx, row, c.clamp(0, 4)]
        h_count += hit.to(torch.int16)
        active = hit

    active = torch.ones((n,), dtype=torch.bool, device=wall.device)
    for step in range(1, 5):
        c = col + step
        active = active & (c < 5)
        hit = active & wall[batch_idx, row, c.clamp(0, 4)]
        h_count += hit.to(torch.int16)
        active = hit

    v_count = torch.ones((n,), dtype=torch.int16, device=wall.device)
    active = torch.ones((n,), dtype=torch.bool, device=wall.device)
    for step in range(1, 5):
        r = row - step
        if r < 0:
            break
        hit = active & wall[batch_idx, r, col]
        v_count += hit.to(torch.int16)
        active = hit

    active = torch.ones((n,), dtype=torch.bool, device=wall.device)
    for step in range(1, 5):
        r = row + step
        if r >= 5:
            break
        hit = active & wall[batch_idx, r, col]
        v_count += hit.to(torch.int16)
        active = hit

    connected = (h_count > 1) | (v_count > 1)
    linked = torch.where(h_count > 1, h_count, torch.zeros_like(h_count)) + torch.where(
        v_count > 1, v_count, torch.zeros_like(v_count)
    )
    return torch.where(connected, linked, torch.ones_like(linked))


def _apply_end_bonuses_for_rows(engine: BE.BatchedEngine, rows: torch.Tensor) -> None:
    """Apply final bonuses for already-ended child rows."""
    if rows.numel() == 0:
        return

    nP = engine.num_players
    wall = engine.wall[rows, :nP]
    bonus = wall.all(dim=-1).sum(dim=-1).to(torch.int16) * 2
    bonus += wall.all(dim=-2).sum(dim=-1).to(torch.int16) * 7

    row_idx = torch.arange(5, device=engine.device)
    for color in range(A.NUM_COLORS):
        col_idx = (row_idx + color) % 5
        bonus += wall[:, :, row_idx, col_idx].all(dim=-1).to(torch.int16) * 10

    engine.scores[rows, :nP] += bonus
    engine.ended[rows] = True


def _finalize_round_for_child_values(engine: BE.BatchedEngine) -> None:
    """Score round-ending child states and advance them to the next playable phase.

    Child values should be encoded from the same phase distribution as real
    self-play states. A round-ending move therefore needs wall-tiling and the
    next factory refill unless it ends the game.
    """
    factories_empty = engine.factory_tiles[:, : engine.num_factories].sum(dim=(1, 2)) == 0
    center_empty = engine.center_tiles.sum(dim=1) == 0
    round_done = factories_empty & center_empty & (~engine.ended)
    rows = round_done.nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return

    floor_penalties = torch.tensor(
        [0, -1, -2, -4, -6, -8, -11, -14],
        dtype=torch.int16,
        device=engine.device,
    )

    for player in range(engine.num_players):
        for row in range(5):
            count = engine.pattern_count[rows, player, row].to(torch.int16)
            complete = count >= row + 1
            if not complete.any():
                continue

            r = rows[complete]
            color = engine.pattern_color[r, player, row].to(torch.long)
            col = (color + row) % 5
            engine.wall[r, player, row, col] = True

            wall = engine.wall[r, player]
            points = _score_completed_placements(wall, row, col)
            engine.scores[r, player] += points
            returned = (count[complete] - 1).to(torch.int8)
            engine._add_to_box_lid(r, color, returned)
            engine.pattern_count.index_put_(
                (r, torch.full_like(r, player), torch.full_like(r, row)),
                torch.zeros(r.shape[0], dtype=torch.int8, device=engine.device),
            )
            engine.pattern_color.index_put_(
                (r, torch.full_like(r, player), torch.full_like(r, row)),
                torch.full((r.shape[0],), -1, dtype=torch.int8, device=engine.device),
            )

        floor_n = engine.floor_count[rows, player].to(torch.long).clamp(0, A.FLOOR_SIZE)
        updated_scores = engine.scores[rows, player] + floor_penalties[floor_n]
        engine.scores[rows, player] = updated_scores.clamp_min(0).to(engine.scores.dtype)
        engine.box_lid[rows] += engine.floor_tiles[rows, player]
        engine.floor_tiles[rows, player].zero_()
        engine.floor_slots[rows, player].fill_(-1)
        engine.floor_count[rows, player] = 0

        has_first = engine.floor_first[rows, player]
        if has_first.any():
            engine.first_player[rows[has_first]] = player
        engine.floor_first[rows, player] = False

    wall_rows_complete = engine.wall[rows, : engine.num_players].all(dim=-1)
    game_over = wall_rows_complete.any(dim=(1, 2))
    _apply_end_bonuses_for_rows(engine, rows[game_over])

    next_round = rows[~game_over]
    if next_round.numel() > 0:
        engine._prepare_next_round_batch(next_round)


def _root_value_index(
    parent_cp: torch.Tensor,
    child_cp: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """Column index for the root player's value in the child's rotated view."""
    diff = parent_cp.to(torch.long) - child_cp.to(torch.long)
    return diff.remainder(num_players)


def _safe_root_actions(topk_idx: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Remap any illegal top-k slot to a guaranteed legal fallback action."""
    any_legal = legal.to(torch.int64).argmax(dim=-1, keepdim=True)
    topk_legal = legal.gather(1, topk_idx)
    return torch.where(topk_legal, topk_idx, any_legal.expand_as(topk_idx))


def _apply_dirichlet_noise(
    prior_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    dirichlet_alpha: float,
    dirichlet_mix: float,
) -> torch.Tensor:
    """Vectorized Dirichlet exploration noise over legal actions."""
    legal_f = legal_mask.to(prior_logits.dtype)
    gamma = torch._standard_gamma(torch.full_like(legal_f, dirichlet_alpha))
    gamma = gamma * legal_f
    gamma = gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    prior_probs = torch.softmax(
        prior_logits.masked_fill(~legal_mask, A.ILLEGAL_LOGIT), dim=-1
    )
    mixed = (1.0 - dirichlet_mix) * prior_probs + dirichlet_mix * gamma
    return torch.log(mixed.clamp_min(1e-9))


def _evaluate_root_children_batched(
    engine: BE.BatchedEngine,
    net: M.AzulNet,
    topk_idx: torch.Tensor,
    legal: torch.Tensor,
    num_players: int,
    perf: PerfCounters | None = None,
) -> torch.Tensor:
    """Expand all root children in one B×K batched engine/value pass."""
    B = engine.batch_size
    K = topk_idx.shape[1]
    q_values = torch.full((B, K), float("-inf"), dtype=torch.float32, device=engine.device)
    if K == 0:
        return q_values

    with maybe_time(perf, "mcts_child_safe_actions"):
        safe_topk = _safe_root_actions(topk_idx, legal)
        parent_cp = engine.current_player.to(torch.long).repeat_interleave(K)
    with maybe_time(perf, "mcts_child_repeat"):
        child_engine = engine.repeat_interleave(K)
    with maybe_time(perf, "mcts_child_step"):
        child_engine.step(safe_topk.reshape(-1), finalize_round=False)
    with maybe_time(perf, "mcts_child_finalize"):
        _finalize_round_for_child_values(child_engine)

    with maybe_time(perf, "mcts_child_encode"):
        g_child, s_child = ENC.encode_state(child_engine)
    with maybe_time(perf, "mcts_child_value_net"):
        with torch.no_grad():
            child_value = net.forward_value(g_child, s_child, num_players)

    with maybe_time(perf, "mcts_child_value_index"):
        child_cp = child_engine.current_player.to(torch.long)
        val_idx = _root_value_index(parent_cp, child_cp, num_players)
        q_values = child_value.gather(1, val_idx.unsqueeze(-1)).squeeze(-1).reshape(B, K)
    return q_values


def gumbel_root_act(
    engine: BE.BatchedEngine,
    net: M.AzulNet,
    num_sims: int = 8,
    temperature: float = 1.0,
    root_noise_scale: float = 1.0,
    dirichlet_alpha: float = 0.0,
    dirichlet_mix: float = 0.0,
    q_scale: float = 10.0,
    precomputed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    perf: PerfCounters | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select actions for each game in the batch.

    Returns:
        actions: (B,) int64 — chosen action per game
        improved_policy: (B, NUM_ACTIONS) float — training target

    When ``precomputed`` is provided as ``(global_feat, source_feat, legal_mask)``,
    skips re-encoding the root state (used by self-play).
    """
    B = engine.batch_size
    device = engine.device
    nP = engine.num_players
    neg_inf = torch.finfo(torch.float32).min

    if precomputed is None:
        with maybe_time(perf, "mcts_root_encode"):
            global_feat, source_feat = ENC.encode_state(engine)
        with maybe_time(perf, "mcts_root_legal_mask"):
            legal_mask = engine.legal_action_mask()
    else:
        global_feat, source_feat, legal_mask = precomputed

    with maybe_time(perf, "mcts_prior_net"):
        with torch.no_grad():
            prior_logits, _root_value = net(global_feat, source_feat, legal_mask, nP)

    prior = prior_logits / max(temperature, 1e-6)

    if dirichlet_alpha > 0 and dirichlet_mix > 0:
        with maybe_time(perf, "mcts_dirichlet"):
            prior = _apply_dirichlet_noise(prior, legal_mask, dirichlet_alpha, dirichlet_mix)

    k = min(num_sims, NUM_ACTIONS)
    with maybe_time(perf, "mcts_gumbel_topk"):
        gumbel = _sample_gumbel((B, NUM_ACTIONS), device) * root_noise_scale
        perturbed = (prior + gumbel).masked_fill(~legal_mask, neg_inf)

        has_legal = legal_mask.any(dim=-1)
        if not has_legal.any():
            actions = torch.zeros((B,), dtype=torch.long, device=device)
            improved = torch.zeros((B, NUM_ACTIONS), dtype=torch.float32, device=device)
            return actions, improved

        top_vals, top_idx = perturbed.topk(k, dim=-1)

    q_values = _evaluate_root_children_batched(engine, net, top_idx, legal_mask, nP, perf=perf)

    with maybe_time(perf, "mcts_select_action"):
        combined = top_vals + q_scale * q_values
        combined = combined.masked_fill(~torch.isfinite(combined), neg_inf)
        best_k = combined.argmax(dim=-1)
        actions = top_idx.gather(1, best_k.unsqueeze(-1)).squeeze(-1)

        fallback = prior.masked_fill(~legal_mask, neg_inf).argmax(dim=-1)
        actions = torch.where(has_legal, actions, fallback)

    with maybe_time(perf, "mcts_improved_policy"):
        improved = torch.full((B, NUM_ACTIONS), neg_inf, dtype=torch.float32, device=device)
        for ki in range(k):
            slot_logits = prior.gather(1, top_idx[:, ki : ki + 1]) + q_scale * q_values[:, ki : ki + 1]
            improved.scatter_(1, top_idx[:, ki : ki + 1], slot_logits)

        improved = improved.masked_fill(~legal_mask, neg_inf)
        improved = torch.softmax(improved, dim=-1)
        improved = torch.where(has_legal.unsqueeze(-1), improved, torch.zeros_like(improved))

    return actions, improved
