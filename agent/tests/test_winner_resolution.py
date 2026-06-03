"""Tests for official Azul winner / tiebreaker resolution."""

from __future__ import annotations

import torch

from agent.env.batched_engine import NO_WINNER, SHARED_VICTORY, BatchedEngine


def test_get_winners_not_ended() -> None:
    engine = BatchedEngine(2, 2, "cpu", seed=1)
    winners = engine.get_winners()
    assert winners.tolist() == [NO_WINNER, NO_WINNER]


def test_get_winners_single_leader() -> None:
    engine = BatchedEngine(1, 2, "cpu", seed=1)
    engine.ended[0] = True
    engine.scores[0, 0] = 50
    engine.scores[0, 1] = 30
    engine.wall[0, 0, 0, 0] = True
    engine.wall[0, 1, 0, 0] = True
    assert int(engine.get_winners()[0].item()) == 0


def test_get_winners_row_tiebreak() -> None:
    engine = BatchedEngine(1, 2, "cpu", seed=1)
    engine.ended[0] = True
    engine.scores[0, 0] = 40
    engine.scores[0, 1] = 40
    engine.wall[0, 0, 0, :] = True
    engine.wall[0, 1, 0, :] = True
    engine.wall[0, 1, 1, :] = True
    assert int(engine.get_winners()[0].item()) == 1


def test_get_winners_shared_victory() -> None:
    engine = BatchedEngine(1, 2, "cpu", seed=1)
    engine.ended[0] = True
    engine.scores[0, 0] = 40
    engine.scores[0, 1] = 40
    engine.wall[0, 0, 0, :] = True
    engine.wall[0, 1, 0, :] = True
    assert int(engine.get_winners()[0].item()) == SHARED_VICTORY


def test_single_engine_shared_victory_matches_batched() -> None:
    from agent.env.single_engine import SingleEngine

    single = SingleEngine(num_players=2, seed=1)
    single.ended = True
    single.players[0].score = 40
    single.players[1].score = 40
    for r in range(5):
        single.players[0].wall[r] = [True] * 5
        single.players[1].wall[r] = [True] * 5

    batched = BatchedEngine(1, 2, "cpu", seed=1)
    batched.ended[0] = True
    batched.scores[0, 0] = 40
    batched.scores[0, 1] = 40
    batched.wall[0, 0, :, :] = True
    batched.wall[0, 1, :, :] = True

    assert single.get_winner() == SHARED_VICTORY
    assert int(batched.get_winners()[0].item()) == SHARED_VICTORY
