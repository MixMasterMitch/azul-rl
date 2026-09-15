"""MCTS root value index must match encoder perspective rotation."""

from __future__ import annotations

import torch

from agent.env import actions as A
from agent.env import batched_engine as BE
from agent.search import gumbel_mcts as G
from agent.net import encoder as ENC


def test_root_value_index_same_player() -> None:
    parent = torch.tensor([1, 2], dtype=torch.long)
    child = parent.clone()
    idx = G._root_value_index(parent, child, num_players=3)
    assert idx.tolist() == [0, 0]


def test_root_value_index_advances_2p() -> None:
    parent = torch.tensor([0], dtype=torch.long)
    child = torch.tensor([1], dtype=torch.long)
    idx = G._root_value_index(parent, child, num_players=2)
    assert idx.item() == 1


def test_root_value_index_advances_3p() -> None:
    parent = torch.tensor([0], dtype=torch.long)
    child = torch.tensor([1], dtype=torch.long)
    idx = G._root_value_index(parent, child, num_players=3)
    assert idx.item() == 2


class _WallFeatureValueNet:
    def forward_value(
        self,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        num_players: int,
    ) -> torch.Tensor:
        del source_feat, num_players
        wall_start = 19 + ENC.D_PATTERN
        wall_row0_col0 = global_feat[:, wall_start]
        value = wall_row0_col0 * 2.0 - 1.0
        return value.unsqueeze(1).expand(-1, BE.MAX_PLAYERS).clone()


def test_root_children_finalize_round_end_before_value_eval() -> None:
    engine = BE.BatchedEngine(1, 2, "cpu", seed=0)
    engine.factory_tiles.zero_()
    engine.center_tiles.zero_()
    engine.center_tiles[0, 0] = 1
    engine.center_first[0] = False
    engine.bag.zero_()
    engine.box_lid.zero_()
    engine.pattern_count.zero_()
    engine.pattern_color.fill_(-1)
    engine.wall.zero_()
    engine.floor_count.zero_()
    engine.floor_tiles.zero_()
    engine.floor_slots.fill_(-1)
    engine.floor_first.zero_()
    engine.scores.zero_()
    engine.current_player[0] = 0
    engine.first_player[0] = 0

    action = A.encode_action(engine.num_factories, 0, 0)
    legal = engine.legal_action_mask()
    q_values = G._evaluate_root_children_batched(
        engine,
        _WallFeatureValueNet(),
        torch.tensor([[action]], dtype=torch.long),
        legal,
        num_players=2,
    )

    assert q_values.tolist() == [[1.0]]


def test_root_child_round_finalize_matches_full_step_refill() -> None:
    engine = BE.BatchedEngine(1, 2, "cpu", seed=0)
    engine.factory_tiles.zero_()
    engine.center_tiles.zero_()
    engine.center_tiles[0, 0] = 1
    engine.center_first[0] = False
    engine.bag.zero_()
    engine.bag[0, 0] = 20
    engine.box_lid.zero_()
    engine.pattern_count.zero_()
    engine.pattern_color.fill_(-1)
    engine.wall.zero_()
    engine.floor_count.zero_()
    engine.floor_tiles.zero_()
    engine.floor_slots.fill_(-1)
    engine.floor_first.zero_()
    engine.scores.zero_()
    engine.current_player[0] = 0
    engine.first_player[0] = 0

    action = A.encode_action(engine.num_factories, 0, 0)
    full = engine.clone()
    child = engine.clone()

    full.step(torch.tensor([action], dtype=torch.long))
    child.step(torch.tensor([action], dtype=torch.long), finalize_round=False)
    G._finalize_round_for_child_values(child)

    assert torch.equal(child.factory_tiles, full.factory_tiles)
    assert torch.equal(child.center_tiles, full.center_tiles)
    assert torch.equal(child.center_first, full.center_first)
    assert torch.equal(child.current_player, full.current_player)
    assert torch.equal(child.scores, full.scores)
    assert torch.equal(child.wall, full.wall)
