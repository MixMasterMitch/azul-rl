"""Improved MCTS policy target invariants."""

from __future__ import annotations

import torch

from agent.env import actions as A
from agent.env.batched_engine import BatchedEngine
from agent.net.model import AzulNet
from agent.search.gumbel_mcts import gumbel_root_act


def test_improved_policy_is_legal_and_normalized() -> None:
    torch.manual_seed(123)
    engine = BatchedEngine(4, 2, "cpu", seed=0)
    net = AzulNet(hidden=64, arch="flat")
    net.eval()

    legal = engine.legal_action_mask()
    actions, improved = gumbel_root_act(engine, net, num_sims=4)

    assert improved.shape == (4, A.NUM_ACTIONS)
    assert legal.gather(1, actions.unsqueeze(1)).all()
    assert torch.all(improved[~legal] == 0)
    assert torch.allclose(improved.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert (improved > 0).sum(dim=-1).max().item() <= 4


def test_improved_policy_zero_for_ended_rows() -> None:
    torch.manual_seed(123)
    engine = BatchedEngine(3, 2, "cpu", seed=0)
    engine.ended[0] = True
    net = AzulNet(hidden=64, arch="flat")
    net.eval()

    legal = engine.legal_action_mask()
    _, improved = gumbel_root_act(engine, net, num_sims=4)

    assert not legal[0].any()
    assert improved[0].sum().item() == 0.0
    assert torch.allclose(improved[1:].sum(dim=-1), torch.ones(2), atol=1e-6)
