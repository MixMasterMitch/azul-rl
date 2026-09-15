"""Step-by-step parity between single and batched engines."""

from __future__ import annotations

import random

import pytest
import torch

from agent.env import actions as A
from agent.env import tiles as T
from agent.env.batched_engine import BatchedEngine
from agent.env.single_engine import SingleEngine
from agent.tests.test_batched_parity import _copy_single_to_batched


def _copy_batched_to_single(batched: BatchedEngine, single: SingleEngine, b: int = 0) -> None:
    for f in range(single.num_factories):
        for c in range(A.NUM_COLORS):
            single.factory_tiles[f][c] = int(batched.factory_tiles[b, f, c].item())
    for c in range(A.NUM_COLORS):
        single.center_tiles[c] = int(batched.center_tiles[b, c].item())
    single.center_has_first = bool(batched.center_first[b].item())

    for p in range(single.num_players):
        player = single.players[p]
        for row in range(5):
            player.pattern_count[row] = int(batched.pattern_count[b, p, row].item())
            player.pattern_color[row] = int(batched.pattern_color[b, p, row].item())
            for col in range(5):
                player.wall[row][col] = bool(batched.wall[b, p, row, col].item())
        player.floor_count = int(batched.floor_count[b, p].item())
        for c in range(A.NUM_COLORS):
            player.floor_tiles[c] = int(batched.floor_tiles[b, p, c].item())
        player.floor_has_first = bool(batched.floor_first[b, p].item())
        player.score = int(batched.scores[b, p].item())

    for c in range(A.NUM_COLORS):
        single.bag[c] = int(batched.bag[b, c].item())
        single.box_lid[c] = int(batched.box_lid[b, c].item())
    single.current_player = int(batched.current_player[b].item())
    single.first_player = int(batched.first_player[b].item())
    single.ended = bool(batched.ended[b].item())


def _play_random_steps(
    *,
    num_players: int,
    num_steps: int,
    single_seed: int,
    batched_seed: int,
    rng_seed: int,
    device: str = "cpu",
) -> None:
    rng = random.Random(rng_seed)
    single = SingleEngine(num_players=num_players, seed=single_seed)
    batched = BatchedEngine(
        batch_size=1,
        num_players=num_players,
        device=device,
        seed=batched_seed,
    )
    _copy_single_to_batched(single, batched)

    for step_i in range(num_steps):
        if single.ended:
            break

        assert int(batched.total_tile_count()[0].item()) == T.TOTAL_TILES, (
            f"tile conservation at step {step_i}"
        )

        single_legal = set(single.legal_actions())
        batched_legal = set(
            batched.legal_action_mask()[0].nonzero(as_tuple=True)[0].tolist()
        )
        assert single_legal == batched_legal, f"legal mismatch at step {step_i}"

        action = rng.choice(list(single_legal))
        single.step(action)
        batched.step(torch.tensor([action], dtype=torch.long, device=device), finalize_round=False)
        round_done = bool((batched.factory_tiles.sum() + batched.center_tiles.sum()) == 0)
        batched.finalize_round()

        expected = BatchedEngine(1, num_players, device=device, seed=0)
        _copy_single_to_batched(single, expected)
        for name in ("center_tiles", "center_first", "pattern_count", "pattern_color",
                     "wall", "floor_count", "floor_tiles", "floor_first", "scores",
                     "current_player", "first_player", "ended"):
            assert torch.equal(getattr(batched, name), getattr(expected, name)), (step_i, name)
        if single.ended:
            assert single.get_winner() == int(batched.get_winners()[0])
        if round_done and not single.ended:
            # Refills can distribute tiles differently, but cannot change the
            # combined per-color draw/discard/factory inventory.
            def inventory(e: BatchedEngine) -> torch.Tensor:
                return e.bag + e.box_lid + e.factory_tiles.sum(1)
            assert torch.equal(inventory(batched), inventory(expected)), (step_i, 'refill inventory')
            # Only stochastic refill results are synchronized AFTER comparisons.
            single.factory_tiles = batched.factory_tiles[0, :single.num_factories].tolist()
            single.bag = batched.bag[0].tolist()
            single.box_lid = batched.box_lid[0].tolist()
        else:
            for name in ('factory_tiles', 'bag', 'box_lid'):
                assert torch.equal(getattr(batched, name), getattr(expected, name)), (step_i, name)


def test_step_parity_random_play() -> None:
    """Batched and single engines stay aligned over many random steps."""
    _play_random_steps(
        num_players=2,
        num_steps=80,
        single_seed=42,
        batched_seed=0,
        rng_seed=7,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("num_players", [2, 3, 4])
@pytest.mark.parametrize("rng_seed", [1, 7, 99])
def test_step_parity_longer_runs(num_players: int, rng_seed: int, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _play_random_steps(
        num_players=num_players,
        num_steps=300,
        single_seed=rng_seed * 11,
        batched_seed=rng_seed,
        rng_seed=rng_seed + 100,
        device=device,
    )
