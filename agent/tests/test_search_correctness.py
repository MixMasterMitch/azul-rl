from __future__ import annotations

import pytest
import torch

from agent.env import actions as A
from agent.env.batched_engine import BatchedEngine, _STATE_TENSOR_ATTRS
from agent.env.outcomes import final_values
from agent.search.gumbel_mcts import _evaluate_root_children_batched, _finalize_round_for_child_values


class ZeroValue:
    def forward_value(self, g: torch.Tensor, s: torch.Tensor, num_players: int) -> torch.Tensor:
        return torch.zeros((len(g), 4), device=g.device)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('players', [2, 3, 4])
def test_search_finalization_matches_engine(device: str, players: int) -> None:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    engine = BatchedEngine(3, players, device, seed=29)
    engine.factory_tiles[:2] = 0
    engine.center_tiles[:2] = 0
    engine.center_tiles[:2, 0] = 1
    engine.center_first[:2] = False
    engine.floor_count[:2, 0] = 2
    engine.floor_tiles[:2, 0, 1] = 2
    engine.floor_slots[:2, 0, :2] = 1
    # One child ends the game, another refills; a third is mid-round.
    engine.wall[0, 0, 0, 1:] = True
    actions = engine.legal_action_mask().long().argmax(1)
    actions[:2] = A.encode_action(engine.num_factories, 0, 0)
    real, search = engine.clone(), engine.clone()
    real.step(actions)
    search.step(actions, finalize_round=False)
    _finalize_round_for_child_values(search)
    for attr in _STATE_TENSOR_ATTRS:
        assert torch.equal(getattr(real, attr), getattr(search, attr)), attr
    assert torch.equal(real._rng.get_state(), search._rng.get_state())


@pytest.mark.parametrize('reward_mode', ['binary', 'score_scaled'])
def test_terminal_children_use_exact_root_utility(reward_mode: str) -> None:
    e = BatchedEngine(2, 2, 'cpu', seed=1)
    e.factory_tiles.zero_()
    e.center_tiles.zero_()
    e.center_tiles[:, 0] = 1
    e.center_first[:] = False
    e.wall[:, 0, 0, 1:] = True
    e.scores[0, 0] = 30
    e.scores[1, 1] = 70
    actions = torch.full((2, 1), A.encode_action(e.num_factories, 0, 0))
    real = e.clone()
    real.step(actions[:, 0])
    assert real.ended.all()
    expected = final_values(real, 2, reward_mode)[:, 0]
    original_rng = e._rng.get_state().clone()
    q = _evaluate_root_children_batched(e, ZeroValue(), actions, e.legal_action_mask(), 2, reward_mode=reward_mode)
    assert torch.equal(q[:, 0], expected)
    assert torch.equal(e._rng.get_state(), original_rng)


@pytest.mark.parametrize('backend', ['one_ply', 'gumbel_tree'])
def test_search_cannot_read_the_live_refill_stream(backend: str) -> None:
    from agent.net.model import AzulNet
    from agent.search.config import SearchConfig
    from agent.search.gumbel_mcts import gumbel_root_act
    e = BatchedEngine(1, 2, 'cpu', seed=11, game_seeds=[12])
    e.factory_tiles.zero_(); e.center_tiles.zero_(); e.center_tiles[0, 2] = 1
    other = e.clone()
    other._rng.manual_seed(9876)
    other._game_rngs[0].manual_seed(9876)
    net = AzulNet(hidden=32, arch='flat').eval()
    cfg = SearchConfig(backend=backend, num_simulations=16, seed=178)
    a, p = gumbel_root_act(e, net, search_config=cfg)
    b, q = gumbel_root_act(other, net, search_config=cfg)
    assert torch.equal(a, b)
    assert torch.equal(p, q)
