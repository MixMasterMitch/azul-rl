"""Production Rust integration, legacy draw streams, and CPU/CUDA training."""
from __future__ import annotations

import json
from typing import Iterator

import numpy as np
import pytest
import torch

from agent.env.batched_engine import BatchedEngine as TorchEngine, _STATE_TENSOR_ATTRS
from agent.env.engine import GameEngine
from agent.net.encoder import encode_state
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.search.gumbel_mcts import gumbel_root_act
from agent.train.bot_selfplay import run_bot_selfplay
from agent.train.instrumentation import PerfCounters
from agent.train.league import League
from agent.train.league_selfplay import run_league_selfplay
from agent.train.learner import make_optimizer, step_from_buffer
from agent.train.replay_buffer import ReplayBuffer
from agent.train.selfplay import run_selfplay


@pytest.fixture(autouse=True)
def threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request: pytest.FixtureRequest) -> str:
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


@pytest.mark.parametrize('players', [2, 3, 4])
@pytest.mark.parametrize('per_game', [False, True])
def test_legacy_streams_continue_identically_through_full_games(players: int, per_game: bool) -> None:
    reference = TorchEngine(3, players, seed=372, game_seeds=[21, 22, 23] if per_game else None)
    engine = GameEngine.from_batched(reference, preserve_rng=True)
    rng = torch.Generator().manual_seed(421)
    for turn in range(400):
        for name in _STATE_TENSOR_ATTRS:
            assert torch.equal(getattr(engine, name), getattr(reference, name)), (turn, name)
        g, s = encode_state(engine)
        expected_g, expected_s = encode_state(reference)
        torch.testing.assert_close(g, expected_g, rtol=0, atol=0)
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
        assert torch.equal(engine.legal_action_mask(), reference.legal_action_mask())
        assert torch.equal(engine.get_winners(), reference.get_winners())
        if bool(reference.ended.all()):
            break
        if turn % 7 == 0:
            engine = GameEngine.from_state_dict(json.loads(json.dumps(engine.state_dict())))
        legal = reference.legal_action_mask()
        actions = torch.rand(legal.shape, generator=rng).masked_fill(~legal, -1).argmax(1)
        # Also check explicitly deferred scoring/refills used in tree search.
        engine.step(actions, finalize_round=bool(turn % 2))
        reference.step(actions, finalize_round=bool(turn % 2))
        if not turn % 2:
            engine.finalize_round()
            reference.finalize_round()
        assert torch.equal(engine._legacy_rng.get_state(), reference._rng.get_state())
        if per_game:
            assert all(torch.equal(a.get_state(), b.get_state())
                       for a, b in zip(engine._legacy_game_rngs, reference._game_rngs))
    else:
        pytest.fail('Legacy continuation did not finish')


@pytest.mark.parametrize('players', [2, 3, 4])
@pytest.mark.parametrize('backend', ['one_ply', 'gumbel_tree'])
def test_search_on_native_state_preserves_parity_and_live_rng(device: str, players: int, backend: str) -> None:
    torch.manual_seed(39)
    reference = TorchEngine(3, players, seed=291)
    engine = GameEngine.from_batched(reference)
    engine.device = torch.device(device)
    saved = engine.state_dict()
    net = AzulNet(hidden=32, arch='flat').to(device).eval()
    cfg = SearchConfig(backend=backend, num_simulations=8, max_root_candidates=4, seed=64)
    actions, policy = gumbel_root_act(engine, net, search_config=cfg)
    perf = PerfCounters(True, device)
    repeated_actions, repeated_policy = gumbel_root_act(engine, net, search_config=cfg, perf=perf)
    assert torch.equal(actions, repeated_actions)
    torch.testing.assert_close(policy, repeated_policy, rtol=0, atol=0)
    assert engine.state_dict() == saved
    assert engine.legal_action_mask().gather(1, actions[:, None]).all()
    torch.testing.assert_close(policy.sum(1), torch.ones(3, device=device))
    assert (policy[~engine.legal_action_mask()] == 0).all()
    if device == 'cpu':
        # These shallow searches cannot reach a refill from the opening position.
        expected_actions, expected_policy = gumbel_root_act(reference, net, search_config=cfg)
        assert torch.equal(actions, expected_actions)
        torch.testing.assert_close(policy, expected_policy, rtol=1e-5, atol=1e-6)
    selected = engine.index_select(torch.tensor([2, 0], device=device))
    assert selected.native.snapshots() == [saved['snapshots'][2], saved['snapshots'][0]]
    for name in _STATE_TENSOR_ATTRS:
        assert torch.equal(getattr(engine, name).cpu(), getattr(reference, name)), name
    assert engine.cpu_view().public_snapshot(0) == engine.public_snapshot(0)


@pytest.mark.parametrize('mode,players', [('self', 2), ('self', 3), ('self', 4),
                                         ('bot_batched', 3), ('bot_scalar', 2), ('league', 4)])
def test_all_selfplay_modes_train_with_native_default(device: str, mode: str, players: int,
                                                      tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(84)
    net = AzulNet(hidden=32, arch='flat').to(device)
    league = League(tmp_path / 'league')
    league.add_checkpoint(net)
    buffer = ReplayBuffer(12000, 275, 10, 5, 300, 4, device)
    def no_reference(*args: object, **kwargs: object) -> None:
        raise AssertionError('Production must not construct the PyTorch engine')
    monkeypatch.setattr(TorchEngine, '__init__', no_reference)
    kwargs = dict(net=net, buffer=buffer, num_players=players, num_games=6,
                  device=device, max_turns=400, num_sims=2, seed=42)
    if mode == 'self':
        result = run_selfplay(**kwargs)
    elif mode == 'league':
        result = run_league_selfplay(**kwargs, league=league, league_prob=1, opponent_sims=2)
    else:
        result = run_bot_selfplay(**kwargs, bot_policy='batched' if mode == 'bot_batched' else 'scalar',
                                  opus_prob=.5)
    assert result['finished'] > 0 and buffer.size > 0
    assert torch.isfinite(buffer.value_target[:buffer.size]).all()
    policy = buffer.policy_target[:buffer.size]
    assert torch.isfinite(policy).all()
    torch.testing.assert_close(policy.sum(1), torch.ones(buffer.size, device=device), rtol=1e-5, atol=1e-5)
    optimizer = make_optimizer(net, lr=1e-3)
    before = next(net.parameters()).detach().clone()
    metrics = step_from_buffer(net, buffer, optimizer, 16, players)
    assert np.isfinite(metrics['loss']) and not metrics.get('skipped', 0)
    assert not torch.equal(before, next(net.parameters()))


@pytest.mark.parametrize('backend', ['one_ply', 'gumbel_tree'])
@pytest.mark.parametrize('terminal', [False, True])
def test_search_round_boundaries_hide_live_rng(device: str, backend: str, terminal: bool) -> None:
    from agent.env.actions import encode_action
    reference = TorchEngine(1, 2, seed=19)
    reference.factory_tiles.zero_()
    reference.center_tiles.zero_()
    reference.center_tiles[0, 0] = 1
    reference.center_first[0] = False
    reference.bag.fill_(20)
    reference.bag[0, 0] -= 1
    if terminal:
        reference.wall[0, 0, 0, 1:] = True
        reference.bag[0, 1:] -= 1
        reference.scores[0, 0] = 30
    engine = GameEngine.from_batched(reference)
    engine.device = torch.device(device)
    other = engine.clone()
    other.cpu_view()
    other.reseed(9123)
    assert other.cpu_view().state_dict() == other.state_dict()
    net = AzulNet(hidden=32, arch='flat').to(device).eval()
    cfg = SearchConfig(backend=backend, num_simulations=32, max_root_candidates=16, seed=78)
    before = engine.state_dict()
    actions, policy = gumbel_root_act(engine, net, search_config=cfg)
    perf = PerfCounters(True, device)
    alternative = gumbel_root_act(other, net, search_config=cfg, perf=perf)
    assert engine.state_dict() == before
    assert torch.equal(actions, alternative[0])
    torch.testing.assert_close(policy, alternative[1], rtol=0, atol=0)
    if terminal:
        assert actions.item() == encode_action(engine.num_factories, 0, 0)
        engine.step(actions)
        assert engine.ended.all() and engine.get_winners().item() == 0
        assert engine.final_values()[0].tolist() == [1, -1, -1, -1]
    else:
        engine.step(actions)
        assert not engine.ended.any() and engine.factory_tiles.sum() == 20
