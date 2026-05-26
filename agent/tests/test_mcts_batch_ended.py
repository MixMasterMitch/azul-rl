"""MCTS must keep working for alive games when some batch rows have ended."""

from __future__ import annotations

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
