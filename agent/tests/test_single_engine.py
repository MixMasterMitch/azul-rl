"""Tests for the single-game reference engine."""

import pytest

from agent.env import actions as A
from agent.env.batched_engine import SHARED_VICTORY
from agent.env.single_engine import SingleEngine


def test_initial_state():
    """Verify initial game state is set up correctly."""
    engine = SingleEngine(num_players=2, seed=42)
    assert engine.num_factories == 5
    assert engine.num_players == 2
    assert not engine.ended

    # Each factory should have exactly 4 tiles
    for f in range(5):
        assert sum(engine.factory_tiles[f]) == 4

    # Bag should have 100 - 20 = 80 tiles (5 factories * 4 tiles = 20)
    assert sum(engine.bag) == 80


def test_legal_actions_nonempty():
    """At game start, there should be legal actions available."""
    engine = SingleEngine(num_players=2, seed=42)
    actions = engine.legal_actions()
    assert len(actions) > 0


def test_play_random_game():
    """A random game should terminate within a reasonable number of turns."""
    import random
    rng = random.Random(123)
    engine = SingleEngine(num_players=2, seed=42)

    turns = 0
    while not engine.ended and turns < 500:
        actions = engine.legal_actions()
        assert len(actions) > 0, f"No legal actions at turn {turns}"
        action = rng.choice(actions)
        engine.step(action)
        turns += 1

    assert engine.ended, f"Game did not end within 500 turns"
    winner = engine.get_winner()
    assert winner == SHARED_VICTORY or 0 <= winner < 2


def test_scores_nonnegative():
    """Scores should never go below 0."""
    import random
    rng = random.Random(456)
    engine = SingleEngine(num_players=2, seed=99)

    while not engine.ended:
        actions = engine.legal_actions()
        if not actions:
            break
        engine.step(rng.choice(actions))

    for player in engine.players:
        assert player.score >= 0


def test_wall_tiling_scores():
    """Placing a tile on the wall should score at least 1 point."""
    engine = SingleEngine(num_players=2, seed=42)

    # Play until wall-tiling happens (complete a pattern line)
    import random
    rng = random.Random(789)
    initial_score = engine.players[0].score

    for _ in range(200):
        if engine.ended:
            break
        actions = engine.legal_actions()
        if not actions:
            break
        engine.step(rng.choice(actions))

    # At least one player should have scored
    total_score = sum(p.score for p in engine.players)
    assert total_score > 0


def test_three_player_game():
    """3-player games should work correctly."""
    import random
    rng = random.Random(111)
    engine = SingleEngine(num_players=3, seed=55)

    assert engine.num_factories == 7

    turns = 0
    while not engine.ended and turns < 500:
        actions = engine.legal_actions()
        if not actions:
            break
        engine.step(rng.choice(actions))
        turns += 1

    assert engine.ended


def test_four_player_game():
    """4-player games should work correctly."""
    import random
    rng = random.Random(222)
    engine = SingleEngine(num_players=4, seed=77)

    assert engine.num_factories == 9

    turns = 0
    while not engine.ended and turns < 600:
        actions = engine.legal_actions()
        if not actions:
            break
        engine.step(rng.choice(actions))
        turns += 1

    assert engine.ended
