"""Tests for batched engine parity with single engine."""

import random

import torch
import pytest

from agent.env import actions as A
from agent.env.single_engine import SingleEngine
from agent.env.batched_engine import BatchedEngine


def _copy_single_to_batched(single: SingleEngine, batched: BatchedEngine) -> None:
    """Copy single engine state into batched engine slot 0 for parity testing."""
    b = 0
    for f in range(single.num_factories):
        for c in range(A.NUM_COLORS):
            batched.factory_tiles[b, f, c] = single.factory_tiles[f][c]
    for c in range(A.NUM_COLORS):
        batched.center_tiles[b, c] = single.center_tiles[c]
    batched.center_first[b] = single.center_has_first

    for p in range(single.num_players):
        player = single.players[p]
        for row in range(5):
            batched.pattern_count[b, p, row] = player.pattern_count[row]
            batched.pattern_color[b, p, row] = player.pattern_color[row]
            for col in range(5):
                batched.wall[b, p, row, col] = player.wall[row][col]
        batched.floor_count[b, p] = player.floor_count
        for c in range(A.NUM_COLORS):
            batched.floor_tiles[b, p, c] = player.floor_tiles[c]
        batched.floor_first[b, p] = player.floor_has_first
        batched.scores[b, p] = player.score

    for c in range(A.NUM_COLORS):
        batched.bag[b, c] = single.bag[c]
        batched.box_lid[b, c] = single.box_lid[c]
    batched.current_player[b] = single.current_player
    batched.first_player[b] = single.first_player
    batched.ended[b] = single.ended


def test_legal_actions_match():
    """Batched and single engine should agree on legal actions for identical states."""
    single = SingleEngine(num_players=2, seed=42)
    batched = BatchedEngine(batch_size=1, num_players=2, seed=0)
    _copy_single_to_batched(single, batched)

    single_legal = set(single.legal_actions())
    batched_mask = batched.legal_action_mask()[0]
    batched_legal = set(batched_mask.nonzero(as_tuple=True)[0].tolist())

    assert single_legal == batched_legal, (
        f"Mismatch: single has {single_legal - batched_legal} extra, "
        f"batched has {batched_legal - single_legal} extra"
    )


def test_legal_actions_match_mid_game():
    """Legal actions should match after several moves."""
    rng = random.Random(99)
    single = SingleEngine(num_players=2, seed=42)

    # Play a few moves in single engine
    for _ in range(10):
        actions = single.legal_actions()
        if not actions or single.ended:
            break
        single.step(rng.choice(actions))

    if single.ended:
        return  # game ended too fast, skip

    # Copy state to batched engine and compare
    batched = BatchedEngine(batch_size=1, num_players=2, seed=0)
    _copy_single_to_batched(single, batched)

    single_legal = set(single.legal_actions())
    batched_mask = batched.legal_action_mask()[0]
    batched_legal = set(batched_mask.nonzero(as_tuple=True)[0].tolist())

    assert single_legal == batched_legal


def test_play_parallel_games():
    """Multiple parallel games should all terminate."""
    B = 8
    engine = BatchedEngine(batch_size=B, num_players=2, seed=42)

    rng = random.Random(100)
    for turn in range(400):
        if engine.ended.all():
            break
        mask = engine.legal_action_mask()
        actions = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            if engine.ended[b]:
                continue
            legal = mask[b].nonzero(as_tuple=True)[0].tolist()
            if legal:
                actions[b] = rng.choice(legal)
        engine.step(actions)

    assert engine.ended.all(), f"Not all games ended: {engine.ended}"


def test_scores_reasonable():
    """Scores should be reasonable after a full game."""
    engine = BatchedEngine(batch_size=4, num_players=2, seed=123)
    rng = random.Random(456)

    for _ in range(400):
        if engine.ended.all():
            break
        mask = engine.legal_action_mask()
        actions = torch.zeros(4, dtype=torch.long)
        for b in range(4):
            if engine.ended[b]:
                continue
            legal = mask[b].nonzero(as_tuple=True)[0].tolist()
            if legal:
                actions[b] = rng.choice(legal)
        engine.step(actions)

    for b in range(4):
        if engine.ended[b]:
            for p in range(2):
                score = engine.scores[b, p].item()
                assert 0 <= score <= 200, f"Unreasonable score {score} in game {b} player {p}"
