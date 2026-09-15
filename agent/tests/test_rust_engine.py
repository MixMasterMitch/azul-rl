"""CPU-only full-state, refill and observation parity with both Python engines."""
from __future__ import annotations

import copy
import random
from typing import Iterator

import numpy as np
import pytest
import torch

from agent.env.batched_engine import BatchedEngine, _STATE_TENSOR_ATTRS
from agent.env.rust_engine import RustEngine
from agent.env.outcomes import final_values
from agent.env.single_engine import SingleEngine
from agent.net.encoder import encode_state
from agent.tests.test_batched_parity import _copy_single_to_batched
from agent.tests.test_batched_step_parity import _copy_batched_to_single

native = pytest.importorskip("azul_astra")
if not hasattr(native, "BatchEngine"):
    pytest.skip("Rebuild azul-astra for full simulator tests", allow_module_level=True)


@pytest.fixture(autouse=True)
def cpu_threads() -> Iterator[None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class UniformRanks:
    """Feed the scalar reference the same float32 uniforms as the other engines."""
    def __init__(self, values: np.ndarray) -> None:
        self.values = iter(values)

    def randint(self, low: int, high: int) -> int:
        assert low == 0
        return int(np.floor(next(self.values) * np.float32(high + 1)))


def install_draws(engine: BatchedEngine, draws: np.ndarray) -> None:
    cursors = [0] * engine.batch_size

    def draw(rows: torch.Tensor) -> torch.Tensor:
        values = []
        for row in rows.tolist():
            values.append(float(draws[row, cursors[row]]))
            cursors[row] += 1
        return torch.tensor(values, dtype=torch.float32, device="cpu")

    engine._draw_uniform = draw


def assert_parity(rust: RustEngine, reference: BatchedEngine) -> None:
    actual = rust.to_batched()
    for name in _STATE_TENSOR_ATTRS:
        assert torch.equal(getattr(actual, name), getattr(reference, name)), name
    assert torch.equal(rust.get_winners(), reference.get_winners())
    assert torch.equal(rust.current_player, reference.current_player)
    assert torch.equal(rust.ended, reference.ended)
    assert torch.equal(rust.scores, reference.scores)
    for mode in ("binary", "score_scaled"):
        torch.testing.assert_close(rust.final_values(mode), final_values(reference, reference.num_players, mode),
                                   rtol=1e-7, atol=1e-7)
    assert rust.total_tile_count().tolist() == [100] * reference.batch_size
    g, s, legal = rust.encode_state_with_legal()
    expected_g, expected_s = encode_state(reference)
    torch.testing.assert_close(g, expected_g, rtol=0, atol=0)
    torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
    assert torch.equal(legal, reference.legal_action_mask())
    assert torch.equal(legal, rust.legal_action_mask())


def assert_single_parity(single: SingleEngine, reference: BatchedEngine, scratch: BatchedEngine, row: int) -> None:
    _copy_single_to_batched(single, scratch)
    for name in _STATE_TENSOR_ATTRS:
        if name == "floor_slots":  # The scalar reference does not track floor order.
            continue
        assert torch.equal(getattr(scratch, name)[0], getattr(reference, name)[row]), name
    assert single.get_winner() == int(reference.get_winners()[row])


@pytest.mark.parametrize("players", [2, 3, 4])
@pytest.mark.parametrize("seed", [3, 19, 71])
def test_complete_games_with_identical_draws(players: int, seed: int) -> None:
    """Never resynchronize after initialization: compare every field through all refills."""
    reference = BatchedEngine(4, players, device="cpu", seed=seed)
    rust = RustEngine.from_batched(reference, seed=seed)
    singles = [SingleEngine(players, seed=0) for _ in range(4)]
    for row, single in enumerate(singles):
        _copy_batched_to_single(reference, single, row)
    scratch = BatchedEngine(1, players, device="cpu", seed=0)
    rng = np.random.default_rng(seed)
    moves = random.Random(seed)
    refills = 0
    mixed_terminal = False
    for turn in range(600):
        assert_parity(rust, reference)
        if bool(reference.ended.all()):
            break
        mixed_terminal |= bool(reference.ended.any())
        legal = reference.legal_action_mask()
        actions = []
        for row, single in enumerate(singles):
            if single.ended:
                actions.append(0)
            else:
                available = legal[row].nonzero().flatten().tolist()
                assert available == single.legal_actions()
                actions.append(moves.choice(available))
        draws = rng.random((4, reference.num_factories * 4), dtype=np.float32)
        install_draws(reference, draws)
        for row, single in enumerate(singles):
            single.rng = UniformRanks(draws[row])
            if not single.ended:
                single.step(actions[row])
        # Exercise both the fused and explicitly deferred round paths.
        if turn % 2:
            reference.step(torch.tensor(actions), finalize_round=False)
            rust.step(actions, finalize_round=False)
            assert_parity(rust, reference)
            refills += int(((reference.factory_tiles.sum((1, 2)) + reference.center_tiles.sum(1) == 0)
                            & ~reference.ended).sum())
            reference.finalize_round()
            rust.finalize_round(draw_uniforms=draws.tolist())
        else:
            reference.step(torch.tensor(actions))
            rust.step(actions, draw_uniforms=draws.tolist())
        for row, single in enumerate(singles):
            assert_single_parity(single, reference, scratch, row)
    assert bool(reference.ended.all()), "All seeded games must finish"
    assert refills >= 4
    assert mixed_terminal
    assert_parity(rust, reference)
    before = rust.snapshots()
    rust.step([299] * 4)  # Terminal rows are inert, including their RNG.
    rust.finalize_round()
    assert rust.snapshots() == before


def rebalance(e: BatchedEngine) -> None:
    """Put all tiles not visible on boards/sources/floors into the bag."""
    e.bag.zero_()
    e.box_lid.zero_()
    for color in range(5):
        used = e.factory_tiles[0, :, color].sum() + e.center_tiles[0, color]
        used += e.floor_tiles[0, :, color].sum()
        used += e.pattern_count[0][e.pattern_color[0] == color].sum()
        used += sum(int(e.wall[0, :, row, (row + color) % 5].sum()) for row in range(5))
        assert int(used) <= 20
        e.bag[0, color] = 20 - int(used)


def empty_reference(players: int = 2) -> BatchedEngine:
    e = BatchedEngine(1, players, device="cpu", seed=0)
    e.factory_tiles.zero_()
    e.center_tiles.zero_()
    return e


def test_full_floor_marker_overflow_ordered_scoring_and_recycling() -> None:
    e = empty_reference()
    e.center_tiles[0, 0] = 2
    e.floor_count[0, 0] = 7
    e.floor_slots[0, 0] = 4
    e.floor_tiles[0, 0, 4] = 7
    e.scores[0, 0] = 20
    e.pattern_count[0, 0, :2] = torch.tensor([1, 2])
    e.pattern_color[0, 0, :2] = torch.tensor([0, 4])
    rebalance(e)
    rust = RustEngine.from_batched(e)
    action = e.num_factories * 30 + 5
    rust.step([action], finalize_round=False)
    e.step(torch.tensor([action]), finalize_round=False)
    assert_parity(rust, e)
    assert bool(e.floor_first[0, 0]) and not bool((e.floor_slots == 5).any())
    assert int(e.box_lid[0, 0]) == 2
    draws = np.zeros((1, e.num_factories * 4), dtype=np.float32)
    install_draws(e, draws)
    e.finalize_round()
    rust.finalize_round(draw_uniforms=draws.tolist())
    assert_parity(rust, e)
    assert int(rust.scores[0, 0]) == 9  # 20 + 1 + 2 - 14.
    assert int(rust.current_player[0]) == 0


@pytest.mark.parametrize("players", [2, 3, 4])
def test_bag_recycles_mid_factory_and_empty_supply_is_safe(players: int) -> None:
    e = empty_reference(players)
    e.center_first.zero_()
    rebalance(e)
    e.box_lid.copy_(e.bag)
    e.bag.zero_()
    e.bag[0, 0] = 1
    e.box_lid[0, 0] -= 1
    rust = RustEngine.from_batched(e)
    draws = np.full((1, e.num_factories * 4), np.float32(.99999994))
    install_draws(e, draws)
    e.finalize_round()
    rust.finalize_round(draw_uniforms=draws.tolist())
    assert_parity(rust, e)
    assert int(e.factory_tiles[0, 0, 0]) == 1
    assert int(e.factory_tiles.sum()) == e.num_factories * 4
    assert int(e.box_lid.sum()) == 0

    # Four almost-full walls leave too few tiles to fill nine factories.
    e = empty_reference(4)
    for p in range(4):
        e.wall[0, p] = True
        e.wall[0, p, :, 0] = False  # No complete horizontal row.
    rebalance(e)
    rust = RustEngine.from_batched(e)
    draws = np.zeros((1, 36), dtype=np.float32)
    install_draws(e, draws)
    e.finalize_round()
    rust.finalize_round(draw_uniforms=draws.tolist())
    assert_parity(rust, e)
    assert int(e.factory_tiles.sum()) == 20
    assert int(e.bag.sum() + e.box_lid.sum()) == 0


@pytest.mark.parametrize("shared", [False, True])
def test_terminal_bonuses_tiebreak_and_snapshot_idempotence(shared: bool) -> None:
    e = empty_reference()
    e.center_first.zero_()
    e.wall[0, :2, 0, :4] = True
    e.pattern_count[0, :2, 0] = 1
    e.pattern_color[0, :2, 0] = 4
    e.scores[0, :2] = torch.tensor([30, 30 if shared else 20])
    rebalance(e)
    rust = RustEngine.from_batched(e)
    e.finalize_round()
    rust.finalize_round()
    assert_parity(rust, e)
    assert int(rust.get_winners()[0]) == (-2 if shared else 0)
    state = rust.snapshots()
    restored = RustEngine.from_snapshots(2, state)
    restored.finalize_round()
    assert restored.snapshots() == state


def test_floor_penalties_clamp_before_terminal_bonuses() -> None:
    e = empty_reference()
    e.center_first.zero_()
    e.wall[0, :2, 0, :4] = True
    e.pattern_count[0, :2, 0] = 1
    e.pattern_color[0, :2, 0] = 4
    e.floor_count[0, :2] = 7
    e.floor_slots[0, :2] = 1
    e.floor_tiles[0, :2, 1] = 7
    rebalance(e)
    rust = RustEngine.from_batched(e)
    rust.finalize_round()
    e.finalize_round()
    assert_parity(rust, e)
    assert rust.scores[0, :2].tolist() == [2, 2]
    assert int(rust.get_winners()[0]) == -2


def test_clone_select_repeat_expand_and_checkpoint_rng_independence() -> None:
    root = RustEngine(3, 3, game_seeds=[51, 76, 91])
    restored = RustEngine.from_snapshots(3, root.snapshots())
    assert restored.snapshots() == root.snapshots()
    separate = [root.index_select([i]) for i in range(3)]
    reordered = root.index_select([2, 0, 2])
    assert reordered.snapshots() == [root.snapshots()[i] for i in [2, 0, 2]]
    rng = random.Random(31)
    for _ in range(140):
        masks = root.legal_action_mask()
        actions = [rng.choice(m.nonzero().flatten().tolist()) if m.any() else 0 for m in masks]
        before = root.snapshots()
        repeated = root.repeat_interleave(2)
        seeds = [rng.getrandbits(64) for _ in range(6)]
        repeated.reseed(seeds)
        repeated.step([a for a in actions for _ in range(2)])
        expanded = root.expand([[a, a] for a in actions], game_seeds=seeds)
        assert expanded.snapshots() == repeated.snapshots()
        assert root.snapshots() == before
        root.step(actions)
        restored.step(actions)
        for i, game in enumerate(separate):
            game.step([actions[i]])
        assert root.snapshots() == restored.snapshots() == [s.snapshots()[0] for s in separate]
        if root.ended.all():
            break
    assert root.ended.all()
    assert root.total_tile_count().tolist() == [100, 100, 100]


def test_invalid_inputs_are_atomic_and_observation_buffers_are_owned() -> None:
    rust = RustEngine(2, 2)
    original = rust.snapshots()
    action = int(rust.legal_action_mask()[0].nonzero()[0])
    for actions, draws in [([action], None), ([action, 299], None),
                           ([action, action], [[0.0] * 20]),
                           ([action, action], [[float("nan")] * 20] * 2),
                           ([action, action], [[1.0] * 20] * 2)]:
        with pytest.raises(ValueError):
            rust.step(actions, draw_uniforms=draws)
        assert rust.snapshots() == original
    for mutate in (lambda e: e.index_select([2]), lambda e: e.repeat_interleave(0),
                   lambda e: e.reseed([1]), lambda e: e.expand([[action], [299]])):
        with pytest.raises(ValueError):
            mutate(rust)
        assert rust.snapshots() == original
    public_len = 57 + 14 * 2
    for index, value in [(0, 1), (public_len, 21), (public_len, 0),
                         (public_len + 15, 5), (-1, -1)]:
        bad = copy.deepcopy(original)
        bad[0][index] = value
        with pytest.raises(ValueError):
            RustEngine.from_snapshots(2, bad)
    with pytest.raises(ValueError):
        RustEngine.from_snapshots(3, original)
    g, s, legal = rust.encode_state_with_legal()
    before = (g.clone(), s.clone(), legal.clone())
    rust.step(rust.legal_action_mask().long().argmax(1))
    for actual, expected in zip((g, s, legal), before):
        assert torch.equal(actual, expected)
    g.zero_()
    assert rust.snapshots() != original
    assert rust.encode_state()[0].any()


@pytest.mark.parametrize("players", [2, 3, 4])
def test_empty_batches_and_cpu_network_contract(players: int) -> None:
    empty = RustEngine(0, players)
    g, s, legal = empty.encode_state_with_legal()
    assert (g.shape, s.shape, legal.shape) == ((0, 275), (0, 10, 5), (0, 300))
    empty.step([])
    empty.finalize_round()
    assert empty.clone().snapshots() == empty.index_select([]).snapshots() == []
    assert empty.repeat_interleave(2).snapshots() == empty.expand([]).snapshots() == []
    assert empty.to_batched().batch_size == 0
    from agent.net.model import AzulNet

    rust = RustEngine(3, players)
    e = rust.to_batched()
    net = AzulNet(hidden=32, arch="flat").to("cpu").eval()
    with torch.no_grad():
        g, s, legal = rust.encode_state_with_legal()
        actual = net(g, s, legal, players)
        g_ref, s_ref = encode_state(e)
        expected = net(g_ref, s_ref, e.legal_action_mask(), players)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
