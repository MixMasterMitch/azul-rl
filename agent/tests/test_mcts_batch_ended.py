"""MCTS must keep working for alive games when some batch rows have ended."""

from __future__ import annotations

import pytest
import torch

from agent.env import batched_engine as BE
from agent.net.model import AzulNet
from agent.search import gumbel_mcts as G


def test_mcts_alive_games_not_zero_when_one_ended() -> None:
    engine = BE.BatchedEngine(8, 2, "cpu", 0)
    engine.ended[0] = True
    net = AzulNet(hidden=64, arch="flat")
    net.eval()

    legal = engine.legal_action_mask()
    assert int(legal[0].sum()) == 0
    assert int(legal[1:].sum()) > 0

    actions, _ = G.gumbel_root_act(engine, net, num_sims=4)
    # Alive games should not all be action 0 due to K=0 batch bug
    alive_actions = actions[1:]
    assert not (alive_actions == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_mcts_child_eval_near_round_end_is_fast() -> None:
    """Regression: child eval must not wall-tile thousands of hypothetical states."""
    import time

    from agent.net import encoder as ENC
    from agent.search.gumbel_mcts import _evaluate_root_children_batched

    device = "cuda"
    engine = BE.BatchedEngine(64, 2, device, seed=0)
    net = AzulNet(hidden=128, arch="attn").to(device)
    net.eval()

    for _ in range(10):
        g, s = ENC.encode_state(engine)
        legal = engine.legal_action_mask()
        actions, _ = G.gumbel_root_act(engine, net, num_sims=8, precomputed=(g, s, legal))
        engine.step(actions)

    g, s = ENC.encode_state(engine)
    legal = engine.legal_action_mask()
    with torch.no_grad():
        prior, _ = net(g, s, legal, 2)
    top_idx = (prior + torch.randn_like(prior)).topk(8, dim=-1).indices

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _evaluate_root_children_batched(engine, net, top_idx, legal, 2)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"child eval took {elapsed:.2f}s (expected <1s without wall-tiling)"


def test_selfplay_finishes_most_games_under_cap() -> None:
    from agent.train.replay_buffer import ReplayBuffer
    from agent.train.selfplay import run_selfplay
    from agent.net import encoder as ENC
    from agent.env import actions as A

    net = AzulNet(hidden=64, arch="flat")
    buffer = ReplayBuffer(
        capacity=50_000,
        d_global=ENC.D_GLOBAL,
        n_sources=ENC.NUM_SOURCES,
        d_source=ENC.D_SOURCE,
        num_actions=A.NUM_ACTIONS,
        max_players=BE.MAX_PLAYERS,
        device="cpu",
    )
    metrics = run_selfplay(
        net,
        buffer=buffer,
        num_games=8,
        num_players=2,
        num_sims=4,
        max_turns=500,
        device="cpu",
        seed=123,
    )
    assert metrics["finished"] >= 6
