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

NUM_ACTIONS = A.NUM_ACTIONS


def _sample_gumbel(shape, device):
    u = torch.rand(shape, device=device).clamp_min(1e-9)
    return -torch.log(-torch.log(u))


def _root_value_index(
    parent_cp: torch.Tensor,
    child_cp: torch.Tensor,
) -> torch.Tensor:
    """Column index for the root player's value in the child's rotated view."""
    diff = parent_cp.to(torch.long) - child_cp.to(torch.long)
    return diff.remainder(BE.MAX_PLAYERS)


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
    prior_probs = torch.softmax(prior_logits.masked_fill(~legal_mask, -1e9), dim=-1)
    mixed = (1.0 - dirichlet_mix) * prior_probs + dirichlet_mix * gamma
    return torch.log(mixed.clamp_min(1e-9))


def _evaluate_root_children_batched(
    engine: BE.BatchedEngine,
    net: M.AzulNet,
    topk_idx: torch.Tensor,
    legal: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """Expand all root children in one B×K batched engine/value pass."""
    B = engine.batch_size
    K = topk_idx.shape[1]
    q_values = torch.full((B, K), float("-inf"), dtype=torch.float32, device=engine.device)
    if K == 0:
        return q_values

    safe_topk = _safe_root_actions(topk_idx, legal)
    parent_cp = engine.current_player.to(torch.long).repeat_interleave(K)
    child_engine = engine.repeat_interleave(K)
    child_engine.step(safe_topk.reshape(-1))

    g_child, s_child = ENC.encode_state(child_engine)
    with torch.no_grad():
        child_value = net.forward_value(g_child, s_child, num_players)

    child_cp = child_engine.current_player.to(torch.long)
    val_idx = _root_value_index(parent_cp, child_cp)
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
        global_feat, source_feat = ENC.encode_state(engine)
        legal_mask = engine.legal_action_mask()
    else:
        global_feat, source_feat, legal_mask = precomputed

    with torch.no_grad():
        prior_logits, _root_value = net(global_feat, source_feat, legal_mask, nP)

    prior = prior_logits / max(temperature, 1e-6)

    if dirichlet_alpha > 0 and dirichlet_mix > 0:
        prior = _apply_dirichlet_noise(prior, legal_mask, dirichlet_alpha, dirichlet_mix)

    k = min(num_sims, NUM_ACTIONS)
    gumbel = _sample_gumbel((B, NUM_ACTIONS), device) * root_noise_scale
    perturbed = (prior + gumbel).masked_fill(~legal_mask, neg_inf)

    has_legal = legal_mask.any(dim=-1)
    if not has_legal.any():
        actions = torch.zeros((B,), dtype=torch.long, device=device)
        improved = torch.zeros((B, NUM_ACTIONS), dtype=torch.float32, device=device)
        return actions, improved

    top_vals, top_idx = perturbed.topk(k, dim=-1)

    q_values = _evaluate_root_children_batched(engine, net, top_idx, legal_mask, nP)

    combined = top_vals + q_scale * q_values
    combined = combined.masked_fill(~torch.isfinite(combined), neg_inf)
    best_k = combined.argmax(dim=-1)
    actions = top_idx.gather(1, best_k.unsqueeze(-1)).squeeze(-1)

    fallback = prior.masked_fill(~legal_mask, neg_inf).argmax(dim=-1)
    actions = torch.where(has_legal, actions, fallback)

    improved = torch.full((B, NUM_ACTIONS), neg_inf, dtype=torch.float32, device=device)
    for ki in range(k):
        slot_logits = prior.gather(1, top_idx[:, ki : ki + 1]) + q_scale * q_values[:, ki : ki + 1]
        improved.scatter_(1, top_idx[:, ki : ki + 1], slot_logits)

    improved = improved.masked_fill(~legal_mask, neg_inf)
    improved = torch.softmax(improved, dim=-1)
    improved = torch.where(has_legal.unsqueeze(-1), improved, torch.zeros_like(improved))

    return actions, improved
