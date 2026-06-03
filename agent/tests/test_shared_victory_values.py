"""Self-play value targets for official shared victories."""

from __future__ import annotations

import pytest

from agent.env.batched_engine import SHARED_VICTORY, BatchedEngine
from agent.train.selfplay import _final_rank_values


def test_binary_shared_victory_marks_tied_winners_positive() -> None:
    engine = BatchedEngine(1, 3, "cpu", seed=0)
    engine.ended[0] = True
    engine.scores[0, :3] = 40
    engine.wall[0, 0, 0, :] = True
    engine.wall[0, 1, 0, :] = True
    engine.wall[0, 2, 0, :] = True

    assert int(engine.get_winners()[0].item()) == SHARED_VICTORY

    values = _final_rank_values(engine, num_players=3, reward_mode="binary")

    assert values[0, :3].tolist() == [1.0, 1.0, 1.0]


def test_binary_shared_victory_only_marks_tied_leaders_positive() -> None:
    engine = BatchedEngine(1, 3, "cpu", seed=0)
    engine.ended[0] = True
    engine.scores[0, :3] = 40
    engine.wall[0, 0, 0, :] = True
    engine.wall[0, 1, 0, :] = True
    engine.wall[0, 2, 0, 0] = True

    assert int(engine.get_winners()[0].item()) == SHARED_VICTORY

    values = _final_rank_values(engine, num_players=3, reward_mode="binary")

    assert values[0, :3].tolist() == [1.0, 1.0, -1.0]


def test_score_scaled_shared_victory_marks_tied_winners_positive() -> None:
    engine = BatchedEngine(1, 3, "cpu", seed=0)
    engine.ended[0] = True
    engine.scores[0, :3] = 40
    engine.wall[0, 0, 0, :] = True
    engine.wall[0, 1, 0, :] = True
    engine.wall[0, 2, 0, 0] = True

    values = _final_rank_values(engine, num_players=3, reward_mode="score_scaled")

    assert values[0, 0].item() == 1.0
    assert values[0, 1].item() == 1.0
    assert values[0, 2].item() == pytest.approx(0.5)
