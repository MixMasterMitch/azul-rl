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
import time

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..env.engine import GameEngine
from ..env.outcomes import final_values
from .config import SearchConfig
from ..net import encoder as ENC
from ..net import model as M
from ..train.instrumentation import PerfCounters, maybe_time

NUM_ACTIONS = A.NUM_ACTIONS


def _sample_gumbel(shape: tuple[int, ...], device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    u = torch.rand(shape, device=device, generator=generator).clamp_min(1e-9)
    return -torch.log(-torch.log(u))


def _finalize_round_for_child_values(engine: BE.BatchedEngine) -> None:
    """Use exactly the same round transition as a real move."""
    engine.finalize_round()


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
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Vectorized Dirichlet exploration noise over legal actions."""
    legal_f = legal_mask.to(prior_logits.dtype)
    gamma = torch._standard_gamma(torch.full_like(legal_f, dirichlet_alpha), generator=generator)
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
    reward_mode: str = "binary",
    generator: torch.Generator | None = None,
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
    # Search chance samples never inherit the live game's future draws.
    seed = int(torch.randint(2**31, (), device=engine.device, generator=generator).item())
    if isinstance(engine, GameEngine) and perf is None:
        child_engine = engine.expand(safe_topk, seed=seed)
    else:
        with maybe_time(perf, "mcts_child_repeat"):
            child_engine = engine.repeat_interleave(K)
            if isinstance(child_engine, GameEngine):
                child_engine.reseed(seed)
            else:
                child_engine._game_rngs = None
                child_engine._rng.manual_seed(seed)
        with maybe_time(perf, "mcts_child_step"):
            child_engine.step(safe_topk.reshape(-1), finalize_round=False)
        if perf is not None:
            refill = (child_engine.round_done() if isinstance(child_engine, GameEngine) else
                      ((child_engine.factory_tiles.sum((1, 2)) + child_engine.center_tiles.sum(1)) == 0) & ~child_engine.ended)
            perf.add_count("search_round_end_children", int(refill.sum()))
        with maybe_time(perf, "mcts_child_finalize"):
            _finalize_round_for_child_values(child_engine)

    terminal = child_engine.ended
    child_value = torch.zeros((B * K, BE.MAX_PLAYERS), device=engine.device)
    live_idx = (~terminal).nonzero(as_tuple=True)[0]
    if live_idx.numel():
        live = child_engine.index_select(live_idx)
        with maybe_time(perf, "mcts_child_encode"):
            g_child, s_child = ENC.encode_state(live)
        with maybe_time(perf, "mcts_child_value_net"), torch.no_grad():
            child_value[live_idx] = net.forward_value(g_child, s_child, num_players)
    with maybe_time(perf, "mcts_child_value_index"):
        child_cp = child_engine.current_player.to(torch.long)
        val_idx = _root_value_index(parent_cp, child_cp, num_players)
        q_flat = child_value.gather(1, val_idx.unsqueeze(-1)).squeeze(-1)
        if terminal.any():
            exact = final_values(child_engine, num_players, reward_mode)
            q_flat[terminal] = exact.gather(1, parent_cp[:, None]).squeeze(1)[terminal]
        q_values = q_flat.reshape(B, K)
    if perf is not None:
        perf.add_count("search_terminal_children", int(terminal.sum()))
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
    reward_mode: str = "binary",
    search_config: SearchConfig | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select actions for each game in the batch.

    Returns:
        actions: (B,) int64 — chosen action per game
        improved_policy: (B, NUM_ACTIONS) float — training target

    When ``precomputed`` is provided as ``(global_feat, source_feat, legal_mask)``,
    skips re-encoding the root state (used by self-play).
    """
    started = time.monotonic()
    if search_config is not None:
        if search_config.backend == "gumbel_tree":
            from .tree import gumbel_tree_act
            return gumbel_tree_act(engine, net, search_config, perf=perf)
        num_sims = search_config.num_simulations
        temperature = search_config.temperature
        root_noise_scale = search_config.root_noise_scale
        dirichlet_alpha = search_config.dirichlet_alpha
        dirichlet_mix = search_config.dirichlet_mix
        q_scale = search_config.q_scale
        reward_mode = search_config.reward_mode
    if num_sims < 1:
        raise ValueError("num_sims must be positive")
    generator = None
    if search_config is not None and search_config.seed is not None:
        generator = torch.Generator(device=engine.device).manual_seed(search_config.seed)
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
            prior = _apply_dirichlet_noise(prior, legal_mask, dirichlet_alpha, dirichlet_mix, generator)

    k = min(num_sims, NUM_ACTIONS)
    with maybe_time(perf, "mcts_gumbel_topk"):
        gumbel = _sample_gumbel((B, NUM_ACTIONS), device, generator) * root_noise_scale
        perturbed = (prior + gumbel).masked_fill(~legal_mask, neg_inf)

        has_legal = legal_mask.any(dim=-1)
        if not has_legal.any():
            actions = torch.zeros((B,), dtype=torch.long, device=device)
            improved = torch.zeros((B, NUM_ACTIONS), dtype=torch.float32, device=device)
            return actions, improved

        top_vals, top_idx = perturbed.topk(k, dim=-1)

    if search_config is not None and search_config.move_deadline_s is not None:
        deadline = started + search_config.move_deadline_s
        q_values = torch.full_like(top_vals, float('-inf'))
        completed, largest_batch_s = 0, .005
        for start in range(0, k, 64):
            if time.monotonic() + largest_batch_s >= deadline:
                break
            before = time.monotonic()
            stop = min(start + 64, k)
            q_values[:, start:stop] = _evaluate_root_children_batched(engine, net, top_idx[:, start:stop], legal_mask,
                nP, perf=perf, reward_mode=reward_mode, generator=generator)
            completed = stop
            largest_batch_s = max(largest_batch_s, time.monotonic() - before)
        if perf is not None:
            perf.add_count('one_ply_candidates', completed)
        if completed == 0:
            probs = torch.softmax(prior_logits, dim=1).masked_fill(~legal_mask, 0)
            return prior_logits.argmax(1), probs
    else:
        q_values = _evaluate_root_children_batched(engine, net, top_idx, legal_mask, nP, perf=perf, reward_mode=reward_mode, generator=generator)

    with maybe_time(perf, "mcts_select_action"):
        combined = top_vals + q_scale * q_values
        combined = combined.masked_fill(~torch.isfinite(combined), neg_inf)
        best_k = combined.argmax(dim=-1)
        actions = top_idx.gather(1, best_k.unsqueeze(-1)).squeeze(-1)

        fallback = prior.masked_fill(~legal_mask, neg_inf).argmax(dim=-1)
        actions = torch.where(has_legal, actions, fallback)

    with maybe_time(perf, "mcts_improved_policy"):
        improved = torch.full((B, NUM_ACTIONS), neg_inf, dtype=torch.float32, device=device)
        improved.scatter_(1, top_idx, prior.gather(1, top_idx) + q_scale * q_values)

        improved = improved.masked_fill(~legal_mask, neg_inf)
        improved = torch.softmax(improved, dim=-1)
        improved = torch.where(has_legal.unsqueeze(-1), improved, torch.zeros_like(improved))

    if perf is not None:
        perf.add_count("search_positions", int(has_legal.sum()))
        perf.add_count("search_policy_disagreements", int(((actions != prior_logits.argmax(1)) & has_legal).sum()))
    return actions, improved
