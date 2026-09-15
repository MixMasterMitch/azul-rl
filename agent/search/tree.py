"""Gumbel tree search with sampled refill outcomes and absolute-seat backups.

Root sequential halving and interior policy matching implement the algorithms in
Danihelka et al., ICLR 2022: https://openreview.net/forum?id=bERaNdoegnO.
Mixed-value Q completion follows Appendix D. Unlike the one-ply backend, a
simulation revisits and descends through existing decision nodes. Refill edges
average a bounded empirical set of independently sampled chance outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np
import torch

from ..env import actions as A
from ..env.batched_engine import BatchedEngine, _STATE_TENSOR_ATTRS
from ..env.engine import GameEngine
from ..env.outcomes import final_values
from ..net.encoder import encode_state
from ..net.model import AzulNet
from ..train.instrumentation import PerfCounters
from .config import SearchConfig


@dataclass
class Edge:
    visits: int = 0
    total: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    outcomes: list[Node] = field(default_factory=list)
    stochastic: bool = False


@dataclass
class Node:
    state: BatchedEngine
    prior: np.ndarray
    raw_value: np.ndarray
    legal: np.ndarray
    player: int
    terminal: bool = False
    edges: dict[int, Edge] = field(default_factory=dict)


def completed_q(node: Node) -> tuple[np.ndarray, np.ndarray]:
    """Q in this node's acting-player perspective, completed with mixed value."""
    visits = np.zeros(A.NUM_ACTIONS, dtype=np.float64)
    q = np.zeros(A.NUM_ACTIONS, dtype=np.float64)
    for action, edge in node.edges.items():
        visits[action] = edge.visits
        if edge.visits:
            q[action] = edge.total[node.player] / edge.visits
    probs = _softmax(node.prior, node.legal)
    seen = visits > 0
    weighted = float(np.dot(probs[seen], q[seen]) / max(float(probs[seen].sum()), 1e-30))
    mixed = (node.raw_value[node.player] + visits.sum() * weighted) / (1 + visits.sum())
    return np.where(seen, q, mixed), visits


def _softmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    out = np.zeros_like(logits, dtype=np.float64)
    if legal.any():
        scores = np.exp(logits[legal] - logits[legal].max())
        out[legal] = scores / scores.sum()
    return out


def _sigma(node: Node, cfg: SearchConfig) -> tuple[np.ndarray, np.ndarray]:
    q, visits = completed_q(node)
    if node.legal.any():
        lo, hi = q[node.legal].min(), q[node.legal].max()
        q = (q - lo) / max(hi - lo, 1e-8)
    return q * cfg.q_scale * (1 + visits.max() / 50), visits


def considered_visits(candidates: int, budget: int) -> list[int]:
    """Balanced visit levels, with half as many contenders in each next round."""
    if candidates == 1:
        return list(range(budget))
    rounds = math.ceil(math.log2(candidates))
    levels: list[int] = []
    level = 0
    remaining = candidates
    while len(levels) < budget:
        repetitions = max(1, budget // (rounds * remaining))
        for _ in range(repetitions):
            levels.extend([level] * remaining)
            level += 1
        remaining = max(2, remaining // 2)
    return levels[:budget]


def _stack(states: list[BatchedEngine], seed: int) -> BatchedEngine:
    first = states[0]
    if isinstance(first, GameEngine):
        return GameEngine.concatenate(states, seed=seed)
    combined = BatchedEngine.__new__(BatchedEngine)
    combined.batch_size = len(states)
    combined.num_players, combined.num_factories = first.num_players, first.num_factories
    combined.device = torch.device('cpu')
    combined._rng = torch.Generator().manual_seed(seed)
    combined._game_rngs = None
    for key in _STATE_TENSOR_ATTRS:
        setattr(combined, key, torch.cat([getattr(s, key).cpu() for s in states]))
    return combined


def _nodes(states: BatchedEngine, model: Any, cfg: SearchConfig) -> list[Node]:
    live = (~states.ended).nonzero().flatten()
    values = torch.zeros((states.batch_size, 4))
    logits = torch.full((states.batch_size, A.NUM_ACTIONS), A.ILLEGAL_LOGIT)
    legal = states.legal_action_mask()
    if len(live):
        active = states.index_select(live)
        g, s = encode_state(active)
        p, v = model(g, s, legal[live], states.num_players)
        logits[live] = p.cpu()
        # Network outputs relative seats; the tree stores absolute seat vectors.
        cp = active.current_player.long()
        offsets = (torch.arange(states.num_players)[None, :] - cp[:, None]) % states.num_players
        values[live, :states.num_players] = v.cpu()[:, :states.num_players].gather(1, offsets)
    if states.ended.any():
        exact = final_values(states, states.num_players, cfg.reward_mode)
        values[states.ended] = exact[states.ended]
    nodes = []
    for b in range(states.batch_size):
        nodes.append(Node(states.index_select(torch.tensor([b])), logits[b].numpy().astype(float),
                          values[b].numpy().astype(float), legal[b].numpy(),
                          int(states.current_player[b]), bool(states.ended[b])))
    return nodes


@torch.no_grad()
def gumbel_tree_act(engine: BatchedEngine, net: Any, cfg: SearchConfig,
                    perf: PerfCounters | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    started = time.monotonic()
    deadline = started + cfg.move_deadline_s if cfg.move_deadline_s is not None else math.inf
    seed = cfg.seed if cfg.seed is not None else int(torch.randint(2**31, ()).item())
    rng = np.random.default_rng(seed)
    model = net
    if isinstance(net, AzulNet):
        from ..eval.inference import InferenceModel
        model = InferenceModel(net, str(next(net.parameters()).device))
    roots = []
    # Chunk the root inference as well as leaf inference.
    for start in range(0, engine.batch_size, cfg.leaf_batch_size):
        subset = engine.index_select(torch.arange(start, min(start + cfg.leaf_batch_size, engine.batch_size), device=engine.device))
        cpu = _stack([subset.index_select(torch.tensor([b], device=engine.device)) for b in range(subset.batch_size)], seed)
        roots.extend(_nodes(cpu, model, cfg))
    candidates, noises, schedules = [], [], []
    for root in roots:
        root.prior = root.prior / max(cfg.temperature, 1e-6)
        if cfg.dirichlet_mix and root.legal.any():
            prior = _softmax(root.prior, root.legal)
            noise = rng.dirichlet(np.full(int(root.legal.sum()), cfg.dirichlet_alpha))
            prior[root.legal] = (1-cfg.dirichlet_mix)*prior[root.legal] + cfg.dirichlet_mix*noise
            root.prior[root.legal] = np.log(np.maximum(prior[root.legal], 1e-30))
        noise = rng.gumbel(size=A.NUM_ACTIONS) * cfg.root_noise_scale
        k = min(cfg.max_root_candidates, int(root.legal.sum()), cfg.num_simulations)
        chosen = np.argsort(np.where(root.legal, root.prior + noise, -np.inf))[-k:] if k else np.array([], dtype=int)
        candidates.append(chosen)
        noises.append(noise)
        schedules.append(considered_visits(max(k, 1), cfg.num_simulations))
    max_depth = 0
    chance_expansions = terminals = 0
    node_count = len(roots)
    max_batch_s = .005
    for simulation in range(cfg.num_simulations):
        if time.monotonic() + max_batch_s >= deadline:
            break
        for start in range(0, len(roots), cfg.leaf_batch_size):
            if time.monotonic() + max_batch_s >= deadline:
                break
            batch_start = time.monotonic()
            pending: list[tuple[Node, int, Edge, list[Edge]]] = []
            for i in range(start, min(start + cfg.leaf_batch_size, len(roots))):
                root = roots[i]
                if root.terminal or not len(candidates[i]):
                    continue
                sigma, visits = _sigma(root, cfg)
                allowed = candidates[i][visits[candidates[i]] == schedules[i][simulation]]
                if not len(allowed):
                    allowed = candidates[i][visits[candidates[i]] == visits[candidates[i]].min()]
                action = int(allowed[np.argmax((root.prior + noises[i] + sigma)[allowed])])
                node = root
                path: list[Edge] = []
                for depth in range(cfg.max_depth):
                    max_depth = max(max_depth, depth + 1)
                    edge = node.edges.setdefault(action, Edge())
                    path.append(edge)
                    if not edge.outcomes or (edge.stochastic and len(edge.outcomes) < cfg.chance_samples):
                        pending.append((node, action, edge, path))
                        break
                    node = edge.outcomes[int(rng.integers(len(edge.outcomes)))]
                    if node.terminal or depth + 1 == cfg.max_depth:
                        for visited in path:
                            visited.visits += 1
                            visited.total += node.raw_value
                        break
                    sigma, counts = _sigma(node, cfg)
                    improved = _softmax(node.prior + sigma, node.legal)
                    action = int(np.argmax(np.where(node.legal, improved - counts/(1 + counts.sum()), -np.inf)))
            if pending:
                batch = _stack([p[0].state for p in pending], int(rng.integers(2**31)))
                batch.step(torch.tensor([p[1] for p in pending]), finalize_round=False)
                round_end = (batch.round_done() if isinstance(batch, GameEngine) else
                             ((batch.factory_tiles.sum((1, 2)) + batch.center_tiles.sum(1)) == 0) & ~batch.ended)
                batch.finalize_round()
                children = _nodes(batch, model, cfg)
                node_count += len(children)
                chance_expansions += int((round_end & ~batch.ended).sum())
                terminals += int(batch.ended.sum())
                for j, ((_, _, edge, path), child) in enumerate(zip(pending, children)):
                    edge.stochastic = bool(round_end[j]) and not child.terminal
                    edge.outcomes.append(child)
                    for visited in path:
                        visited.visits += 1
                        visited.total += child.raw_value
            max_batch_s = max(max_batch_s, time.monotonic() - batch_start)
    actions = torch.zeros(len(roots), dtype=torch.long)
    policies = torch.zeros((len(roots), A.NUM_ACTIONS))
    for i, root in enumerate(roots):
        if not root.legal.any():
            continue
        sigma, visits = _sigma(root, cfg)
        policies[i] = torch.from_numpy(_softmax(root.prior + sigma, root.legal))
        # Sequential halving recommends among the most-visited survivors.
        survivors = candidates[i][visits[candidates[i]] == visits[candidates[i]].max()]
        actions[i] = int(survivors[np.argmax((root.prior + noises[i] + sigma)[survivors])])
    if perf is not None:
        counts = [sum(e.visits for e in r.edges.values()) for r in roots if not r.terminal]
        perf.add_count('tree_simulations_per_root', min(counts, default=0))
        perf.add_count('tree_simulations_max_per_root', max(counts, default=0))
        perf.add_count('tree_max_depth', max_depth)
        perf.add_count('search_terminal_children', terminals)
        perf.add_count('search_refill_children', chance_expansions)
        perf.add_count('tree_nodes', node_count)
    return actions.to(engine.device), policies.to(engine.device)
