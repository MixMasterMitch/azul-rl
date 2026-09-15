from __future__ import annotations

from dataclasses import replace
import random

import pytest
import torch

from agent.env.batched_engine import BatchedEngine
from agent.env.single_engine import SingleEngine
from agent.eval.heuristic_astra import AstraConfig, AstraWeights, HeuristicAstraBot, snapshot
from agent.eval import astra_reference

native = pytest.importorskip("azul_astra")


def single_from_batched(e: BatchedEngine) -> SingleEngine:
    s = SingleEngine(e.num_players, seed=0)
    s.factory_tiles = e.factory_tiles[0, :e.num_factories].tolist()
    s.center_tiles = e.center_tiles[0].tolist()
    s.center_has_first = bool(e.center_first[0])
    s.current_player = int(e.current_player[0])
    s.first_player = int(e.first_player[0])
    s.ended = bool(e.ended[0])
    s.bag = e.bag[0].tolist()
    s.box_lid = e.box_lid[0].tolist()
    for p, b in enumerate(s.players):
        b.wall = e.wall[0, p].tolist()
        b.score = int(e.scores[0, p])
        b.floor_count = int(e.floor_count[0, p])
        b.floor_has_first = bool(e.floor_first[0, p])
        b.floor_tiles = e.floor_tiles[0, p].tolist()
        b.pattern_count = e.pattern_count[0, p].tolist()
        b.pattern_color = e.pattern_color[0, p].tolist()
    return s


@pytest.mark.parametrize("players", [2, 3, 4])
@pytest.mark.parametrize("seed", [13, 41, 92])
def test_random_transition_and_round_parity(players: int, seed: int) -> None:
    torch.set_num_threads(1)
    e = BatchedEngine(1, players, seed=seed)
    rng = random.Random(seed)
    for _ in range(130):
        if e.ended[0]:
            break
        data = snapshot(e, 0)
        legal = e.legal_action_mask()[0].nonzero().flatten().tolist()
        s = single_from_batched(e)
        assert native.legal_actions(data) == legal == s.legal_actions()
        assert astra_reference.legal_actions(data) == legal
        for action in rng.sample(legal, min(3, len(legal))):
            child = e.clone()
            child.step(torch.tensor([action]), finalize_round=False)
            assert native.transition(data, action, False) == snapshot(child, 0)
        action = rng.choice(legal)
        child = e.clone()
        child._prepare_next_round_batch = lambda _: None
        s._prepare_next_round = lambda: None
        child.step(torch.tensor([action]))
        s.step(action)
        expected = snapshot(child, 0)
        is_boundary = not child.factory_tiles.any() and not child.center_tiles.any()
        expected[6] = int(is_boundary or bool(child.ended[0]))
        actual = native.transition(data, action)
        assert actual == expected
        assert astra_reference.transition(data, action) == expected
        for p, board in enumerate(s.players):
            assert board.score == int(child.scores[0, p])
            assert board.wall == child.wall[0, p].tolist()
            assert board.pattern_count == child.pattern_count[0, p].tolist()
            assert board.pattern_color == child.pattern_color[0, p].tolist()
            assert board.floor_count == int(child.floor_count[0, p])
            assert board.floor_has_first == bool(child.floor_first[0, p])
        assert s.ended == bool(child.ended[0])
        assert s.first_player == int(child.first_player[0])
        # Only advance the live game after comparing both deterministic engines.
        e.step(torch.tensor([action]))


def test_native_search_preserves_state_rng_and_handles_batch_index() -> None:
    e = BatchedEngine(3, 4, seed=71)
    original = {k: v.clone() for k, v in vars(e).items() if isinstance(v, torch.Tensor)}
    rng = e._rng.get_state().clone()
    bot = HeuristicAstraBot(seed=3)
    a = bot.analyze(e, 2)
    b = bot.analyze(e, 2)
    assert (a["action"], a["nodes"], a["principal_variation"]) == (b["action"], b["nodes"], b["principal_variation"])
    assert a["nodes"] <= bot.configuration(4).nodes
    assert e.legal_action_mask()[2, a["action"]]
    assert torch.equal(rng, e._rng.get_state())
    for k, v in original.items():
        assert torch.equal(v, getattr(e, k))


def test_boundaries_and_timeout_fallback() -> None:
    e = BatchedEngine(1, 2, seed=12)
    data = snapshot(e, 0)
    for malformed in ([], data[:-1], [2, *data[1:]], [1, 5, *data[2:]]):
        with pytest.raises(ValueError):
            native.legal_actions(malformed)
    with pytest.raises(ValueError):
        native.transition(data, 299)
    with pytest.raises(ValueError):
        native.resolve_round(data)
    with pytest.raises(ValueError):
        native.analyze(data, weights=[float("nan")] * 8)
    for kwargs in ({"nodes": 0}, {"time_ms": 2001}, {"depth": 65}, {"width": -1}):
        with pytest.raises(ValueError):
            AstraConfig(**kwargs)
    with pytest.raises(ValueError):
        AstraWeights(urgency=0)
    for cfg in (AstraConfig(nodes=1), AstraConfig(nodes=100_000_000, time_ms=1)):
        result = HeuristicAstraBot(config=cfg).analyze(e, 0)
        assert result["action"] in native.legal_actions(data)
        assert result["cutoff_reason"] in ("time", "nodes")
        assert result["elapsed_s"] < 0.25


def test_full_floor_marker_and_ordered_scoring() -> None:
    e = BatchedEngine(1, 2, seed=1)
    e.factory_tiles.zero_()
    e.center_tiles.zero_()
    e.center_tiles[0, 0] = 2
    e.floor_count[0, 0] = 7
    e.scores[0, 0] = 20
    e.pattern_count[0, 0, 0] = 1
    e.pattern_color[0, 0, 0] = 0
    e.pattern_count[0, 0, 1] = 2
    e.pattern_color[0, 0, 1] = 4  # Both completed placements in column zero.
    action = e.num_factories * 30 + 5
    result = native.transition(snapshot(e, 0), action)
    assert result[3] == 0  # Marker grants initiative even when it cannot fit.
    assert result[57 + 1] == 9  # 20 + 1 + 2 - 14.
    assert result[57 + 2:57 + 4] == [0, 0]


def test_terminal_shared_win_and_score_clamping() -> None:
    e = BatchedEngine(1, 2, seed=1)
    e.factory_tiles.zero_()
    e.center_tiles.zero_()
    e.center_first.zero_()
    for p in range(2):
        e.wall[0, p, 0, :4] = True
        e.pattern_count[0, p, 0] = 1
        e.pattern_color[0, p, 0] = 4
        e.floor_count[0, p] = 7
    data = native.resolve_round(snapshot(e, 0))
    assert data[5:7] == [1, 1]
    assert data[58] == data[72] == 2  # Score clamps to zero, then +2 row bonus.
    assert native.resolve_round(data) == data  # Bonuses cannot be applied twice.
    assert native.legal_actions(data) == []


def test_config_round_trip() -> None:
    from dataclasses import asdict
    c = replace(AstraConfig(), weights=AstraWeights(adjacency=0.1))
    assert AstraConfig.from_dict(asdict(c)) == c


def test_deliberate_floor_sacrifice_forces_a_win() -> None:
    """Taking 3 blue at -4 forces the opponent to floor red at -8 total.

    Taking red safely would let the opponent place blue and win instead.
    Both top rows are already pending completion, so the end is unavoidable.
    """
    e = BatchedEngine(1, 2, seed=1)
    e.factory_tiles.zero_()
    e.center_tiles.zero_()
    e.center_first.zero_()
    e.center_tiles[0, 0] = e.center_tiles[0, 2] = 3
    e.scores[0, :2] = torch.tensor([20, 22])
    e.floor_count[0, 1] = 2
    for r in range(5):
        e.wall[0, 0, r, r] = True
        e.wall[0, 1, r, (r + 2) % 5] = True
    for p in range(2):
        e.wall[0, p, 0, :4] = True
        e.wall[0, p, 0, 4] = False
        e.pattern_count[0, p, 0] = 1
        e.pattern_color[0, p, 0] = 4
    for r in (1, 2, 3):
        e.pattern_count[0, 0, r] = 1
        e.pattern_color[0, 0, r] = 3
    action = HeuristicAstraBot(config=AstraConfig(nodes=16000)).select_action(e, 0)
    assert action == e.num_factories * 30 + 5  # Blue, directly to floor.
    e.step(torch.tensor([action]))
    e.step(torch.tensor([e.num_factories * 30 + 2 * 6 + 5]))
    assert e.ended[0] and int(e.get_winners()[0]) == 0


@pytest.mark.parametrize("score,should_end", [(10, False), (80, True)])
def test_end_or_extend_uses_terminal_outcome(score: int, should_end: bool) -> None:
    e = BatchedEngine(1, 2, seed=1)
    e.factory_tiles.zero_()
    e.center_tiles.zero_()
    e.center_first.zero_()
    e.center_tiles[0, 4] = 1
    e.wall[0, 0, 0, :4] = True
    e.scores[0, :2] = torch.tensor([score, 60])
    a = HeuristicAstraBot(config=AstraConfig(nodes=16000)).select_action(e, 0)
    assert (a % 6 == 0) == should_end


def test_bag_and_random_generator_are_not_policy_inputs() -> None:
    e = BatchedEngine(1, 3, seed=22)
    bot = HeuristicAstraBot(seed=3)
    before = bot.analyze(e, 0)
    e.bag.zero_()
    e.box_lid.fill_(20)
    e._rng.manual_seed(923432)
    after = bot.analyze(e, 0)
    assert (before["action"], before["principal_variation"], before["values"]) == (
        after["action"], after["principal_variation"], after["values"])


def test_recorded_denial_and_flexibility_positions() -> None:
    import json
    from pathlib import Path
    fixtures = json.loads((Path(__file__).parent / "fixtures/astra_tactics.json").read_text())
    for fixture in fixtures:
        result = native.analyze(fixture["snapshot"], nodes=16000)
        assert result["action"] in fixture["expected_actions"], fixture["name"]
        assert result["action"] in astra_reference.legal_actions(fixture["snapshot"])


def test_forced_terminal_budget_recovers_recorded_multiplayer_win() -> None:
    import json
    from dataclasses import replace
    from pathlib import Path
    from agent.eval.heuristic_astra import native_options, pending_game_end

    fixture = json.loads((Path(__file__).parent / "fixtures/astra_terminal_win.json").read_text())
    state = fixture["snapshot"]
    config = AstraConfig.from_dict(fixture["config"])
    assert pending_game_end(state) and any(state[7:52])
    assert native_options(config, state)["nodes"] == 4000
    options = native_options(replace(config, terminal_nodes=256000), state)
    result = native.analyze(state, seed=fixture["bot_seed"], **options)
    assert result["solved"] and result["action"] == fixture["deeper_analysis"]["action"]
    assert result["values"][state[2]] > 1_000_000
    for action in result["principal_variation"]:
        assert action in astra_reference.legal_actions(state)
        expected = astra_reference.transition(state, action)
        state = native.transition(state, action)
        assert state == expected
    assert state[5] == 1  # The backed-up terminal result has an actual ending PV.
    scores = [state[58 + 14 * p] for p in range(state[1])]
    assert scores[fixture["snapshot"][2]] > max(scores[:2])


def test_terminal_budget_requires_a_completed_ending_line() -> None:
    from dataclasses import asdict, replace
    from agent.eval.heuristic_astra import native_options, pending_game_end

    state = snapshot(BatchedEngine(1, 3, seed=19), 0)
    config = AstraConfig(terminal_nodes=256000)
    assert not pending_game_end(state)
    expected = asdict(replace(config, terminal_nodes=0))
    expected.pop("terminal_nodes")
    expected["weights"] = list(expected["weights"].values())
    assert native_options(config, state) == expected
    state[57] = 15  # Four wall tiles alone do not guarantee an ending.
    assert not pending_game_end(state)
    state[61] = 1
    state[66] = 4
    assert pending_game_end(state)
    assert native_options(config, state)["nodes"] == 256000
    for value in (-1, True, 1.5, 100_000_001):
        with pytest.raises(ValueError, match="terminal_nodes"):
            AstraConfig(terminal_nodes=value)
    assert not pending_game_end([1, 3])


def test_private_per_game_rngs_are_preserved() -> None:
    e = BatchedEngine(3, 3, game_seeds=[7, 23, 29])
    states = [rng.get_state().clone() for rng in e._game_rngs]
    HeuristicAstraBot().select_action(e, 2)
    assert all(torch.equal(a, b.get_state()) for a, b in zip(states, e._game_rngs))


def test_invalid_public_state_invariants_are_rejected() -> None:
    data = snapshot(BatchedEngine(1, 2, seed=11), 0)
    for index, value in [(7, 5), (52, 21), (57, 1 << 25), (58, -1),
                         (59, 8), (61, 2), (66, 5), (5, 1)]:
        invalid = data.copy()
        invalid[index] = value
        with pytest.raises(ValueError):
            native.analyze(invalid)
    invalid = data.copy()
    invalid[60] = 1  # Marker on floor zero, while center still owns it.
    with pytest.raises(ValueError):
        native.legal_actions(invalid)
    with pytest.raises(ValueError):
        HeuristicAstraBot(config={"time_ms": 3000})
    with pytest.raises(ValueError):
        HeuristicAstraBot(config="not a config")


def test_production_defaults_are_per_player_count() -> None:
    from agent.eval.heuristic_astra import production_config
    bot = HeuristicAstraBot()
    for n in (2, 3, 4):
        assert bot.configuration(n) == production_config(n)
        assert bot.configuration(n).time_ms <= 1950
    override = AstraConfig(nodes=1)
    assert HeuristicAstraBot(config=override).configuration(4) == override
    with pytest.raises(ValueError):
        production_config(5)


def test_extra_center_budget_can_complete_a_round() -> None:
    e = BatchedEngine(1, 3, seed=3)
    e.factory_tiles.zero_()
    e.center_tiles.zero_()
    e.center_tiles[0, 0] = 1
    e.center_tiles[0, 1] = 2
    cfg = AstraConfig(nodes=1, center_nodes=16000)
    result = HeuristicAstraBot(config=cfg).analyze(e, 0)
    assert result["solved"]
    assert 1 < result["nodes"] <= 16000
    assert result["cutoff_reason"] == "solved"
    assert result["action"] in native.legal_actions(snapshot(e, 0))


def test_current_rust_play_engine_remains_unchanged_by_astra() -> None:
    from agent.env.engine import GameEngine
    e = GameEngine(3, 4, seed=91)
    before = e.state_dict()
    result = HeuristicAstraBot(config=AstraConfig(nodes=4000)).analyze(e, 2)
    assert e.legal_action_mask()[2, result['action']]
    assert e.state_dict() == before  # Includes full native RNG and hidden state.
