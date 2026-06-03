"""Batched training bot policy must always pick legal actions."""

from __future__ import annotations

import torch

from agent.env.batched_engine import BatchedEngine
from agent.train.batched_bot_policy import batched_heuristic_actions


def test_batched_bot_actions_are_legal() -> None:
    engine = BatchedEngine(batch_size=32, num_players=2, device="cpu", seed=99)
    for _ in range(40):
        if engine.ended.all():
            break
        legal = engine.legal_action_mask()
        actions = batched_heuristic_actions(engine)
        alive = ~engine.ended
        assert (legal[alive].gather(1, actions[alive].unsqueeze(-1)).squeeze(-1)).all()
        engine.step(actions)
