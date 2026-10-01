"""Self-play value targets for official shared victories."""

from __future__ import annotations

import pytest
import torch

from agent.env.batched_engine import SHARED_VICTORY, BatchedEngine
from agent.train.selfplay import _final_rank_values
from agent.env.engine import GameEngine


def test_binary_shared_victory_marks_tied_winners_zero() -> None:
    engine = BatchedEngine(1, 3, "cpu", seed=0)
    engine.ended[0] = True
    engine.scores[0, :3] = 40
    engine.wall[0, 0, 0, :] = True
    engine.wall[0, 1, 0, :] = True
    engine.wall[0, 2, 0, :] = True

    assert int(engine.get_winners()[0].item()) == SHARED_VICTORY

    values = _final_rank_values(engine, num_players=3, reward_mode="binary")

    assert values[0, :3].tolist() == [0.0, 0.0, 0.0]


def test_binary_shared_victory_only_marks_tied_leaders_zero() -> None:
    engine = BatchedEngine(1, 3, "cpu", seed=0)
    engine.ended[0] = True
    engine.scores[0, :3] = 40
    engine.wall[0, 0, 0, :] = True
    engine.wall[0, 1, 0, :] = True
    engine.wall[0, 2, 0, 0] = True

    assert int(engine.get_winners()[0].item()) == SHARED_VICTORY

    values = _final_rank_values(engine, num_players=3, reward_mode="binary")

    assert values[0, :3].tolist() == [0.0, 0.0, -1.0]


def test_score_scaled_shared_victory_marks_tied_winners_zero() -> None:
    engine = BatchedEngine(1, 3, "cpu", seed=0)
    engine.ended[0] = True
    engine.scores[0, :3] = 40
    engine.wall[0, 0, 0, :] = True
    engine.wall[0, 1, 0, :] = True
    engine.wall[0, 2, 0, 0] = True

    values = _final_rank_values(engine, num_players=3, reward_mode="score_scaled")

    assert values[0, 0].item() == 0.0
    assert values[0, 1].item() == 0.0
    assert values[0, 2].item() == pytest.approx(0.5)


@pytest.mark.parametrize("players", [2, 3, 4])
@pytest.mark.parametrize("mode", ["binary", "score_scaled"])
@pytest.mark.parametrize("leaders", [1, 2])
def test_native_and_reference_official_tiebreak_rewards(
    players: int, mode: str, leaders: int
) -> None:
    reference = BatchedEngine(1, players, "cpu", seed=7)
    reference.bag += reference.factory_tiles.sum(1).to(reference.bag.dtype)
    reference.factory_tiles.zero_()
    reference.ended[0] = True
    reference.scores[0, :players] = 40
    for seat in range(leaders):
        reference.wall[0, seat, 0, :] = True
        reference.bag[0] -= 1
    native = GameEngine.from_batched(reference, seed=7)
    expected = _final_rank_values(reference, players, mode)
    actual = _final_rank_values(native, players, mode)
    torch.testing.assert_close(actual, expected)
    assert actual[0, :leaders].tolist() == ([1.0] if leaders == 1 else [0.0, 0.0])
    if mode == "binary":
        assert actual[0, leaders:players].tolist() == [-1.0] * (players - leaders)
