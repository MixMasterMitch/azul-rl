"""Encoder source slots must match action-source legality."""

from __future__ import annotations

import pytest
import torch

from agent.env import actions as A
from agent.env.batched_engine import BatchedEngine
from agent.net import encoder as ENC


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_center_source_feature_uses_engine_center_source(num_players: int) -> None:
    engine = BatchedEngine(1, num_players, "cpu", seed=0)
    engine.factory_tiles.zero_()
    engine.center_tiles[0] = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int8)

    _, source_feat = ENC.encode_state(engine)

    assert source_feat[0, engine.num_factories].tolist() == [1, 2, 3, 4, 5]
    if engine.num_factories != A.MAX_FACTORIES:
        assert source_feat[0, A.MAX_FACTORIES].sum().item() == 0.0


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_center_legal_actions_use_same_source_slot_as_encoder(num_players: int) -> None:
    engine = BatchedEngine(1, num_players, "cpu", seed=0)
    engine.factory_tiles.zero_()
    engine.center_tiles.zero_()
    engine.center_tiles[0, 0] = 1

    legal_sources = {
        A.decode_action(int(a))[0]
        for a in engine.legal_action_mask()[0].nonzero(as_tuple=True)[0].tolist()
    }

    assert legal_sources == {engine.num_factories}
