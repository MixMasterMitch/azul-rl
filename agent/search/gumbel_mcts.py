"""Gumbel-root MCTS for Azul.

Uses a "Gumbel root + 1-ply learned value" scheme:
1. Compute prior logits from the policy head
2. Add Gumbel noise to select top-K candidate actions at the root
3. For each candidate, expand one step in the batched engine
4. Score each child state with the value network
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
    """Column index for the root player's value in the child's rotated view.

    The encoder rotates seats so seat 0 = child's current player. The root
    player (parent's current player) sits at index ``(parent_cp - child_cp) % MAX_PLAYERS``.
    """
    diff = parent_cp.to(torch.long) - child_cp.to(torch.long)
    return diff.remainder(BE.MAX_PLAYERS)


def gumbel_root_act(
    engine: BE.BatchedEngine,
    net: M.AzulNet,
    num_sims: int = 8,
    temperature: float = 1.0,
    root_noise_scale: float = 1.0,
    dirichlet_alpha: float = 0.0,
    dirichlet_mix: float = 0.0,
    q_scale: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select actions for each game in the batch.

    Returns:
        actions: (B,) int64 — chosen action per game
        improved_policy: (B, NUM_ACTIONS) float — training target
    """
    B = engine.batch_size
    device = engine.device
    nP = engine.num_players

    global_feat, source_feat = ENC.encode_state(engine)
    legal_mask = engine.legal_action_mask()

    with torch.no_grad():
        prior_logits, _root_value = net(global_feat, source_feat, legal_mask, nP)

    prior = prior_logits / max(temperature, 1e-6)
    neg_inf = torch.finfo(prior.dtype).min

    if dirichlet_alpha > 0 and dirichlet_mix > 0:
        noise = torch.zeros_like(prior)
        for b in range(B):
            legal_indices = legal_mask[b].nonzero(as_tuple=True)[0]
            if len(legal_indices) > 0:
                alpha = torch.full((len(legal_indices),), dirichlet_alpha, device=device)
                dir_noise = torch.distributions.Dirichlet(alpha).sample()
                log_noise = torch.log(dir_noise + 1e-9)
                noise[b, legal_indices] = log_noise * dirichlet_mix
        prior = prior + noise

    # Fixed top-K width per game (do NOT use batch min legal count: ended games
    # have zero legal moves and would force K=0 for the entire batch).
    k = min(num_sims, NUM_ACTIONS)
    gumbel = _sample_gumbel((B, NUM_ACTIONS), device) * root_noise_scale
    perturbed = (prior + gumbel).masked_fill(~legal_mask, neg_inf)

    has_legal = legal_mask.any(dim=-1)
    if not has_legal.any():
        actions = torch.zeros((B,), dtype=torch.long, device=device)
        improved = torch.zeros((B, NUM_ACTIONS), dtype=torch.float32, device=device)
        return actions, improved

    top_vals, top_idx = perturbed.topk(k, dim=-1)  # (B, k)

    parent_cp = engine.current_player.long()
    q_values = torch.full((B, k), float("-inf"), dtype=torch.float32, device=device)

    for ki in range(k):
        child = engine.clone()
        child.step(top_idx[:, ki])

        g_child, s_child = ENC.encode_state(child)
        legal_child = child.legal_action_mask()
        with torch.no_grad():
            _, child_value = net(g_child, s_child, legal_child, nP)

        child_cp = child.current_player.long()
        val_idx = _root_value_index(parent_cp, child_cp)
        q_values[:, ki] = child_value.gather(1, val_idx.unsqueeze(-1)).squeeze(-1)

    combined = top_vals + q_scale * q_values
    combined = combined.masked_fill(~torch.isfinite(combined), neg_inf)
    best_k = combined.argmax(dim=-1)
    actions = top_idx.gather(1, best_k.unsqueeze(-1)).squeeze(-1)

    # Games with no legal moves (ended): fallback action unused by engine.step
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
