from __future__ import annotations
from dataclasses import replace
import time
import numpy as np
import torch
from agent.env.batched_engine import BatchedEngine
from agent.env import actions as A
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.search.gumbel_mcts import gumbel_root_act
from agent.search.tree import Edge, Node, completed_q, considered_visits
from agent.train.instrumentation import PerfCounters


def test_tree_descends_and_preserves_live_state() -> None:
    e = BatchedEngine(3, 2, 'cpu', seed=902, game_seeds=[1, 2, 3])
    before = e.clone()
    net = AzulNet(hidden=32, arch='flat').eval()
    cfg = SearchConfig(backend='gumbel_tree', num_simulations=32, max_root_candidates=4, seed=789)
    perf = PerfCounters(True)
    actions, policy = gumbel_root_act(e, net, search_config=cfg, perf=perf)
    assert e.legal_action_mask().gather(1, actions[:, None]).all()
    assert torch.allclose(policy.sum(1), torch.ones(3))
    assert (policy[~e.legal_action_mask()] == 0).all()
    assert perf.snapshot()['profile_tree_max_depth'] > 1
    assert torch.equal(e.factory_tiles, before.factory_tiles)
    assert torch.equal(e._rng.get_state(), before._rng.get_state())
    assert all(torch.equal(x.get_state(), y.get_state()) for x, y in zip(e._game_rngs, before._game_rngs))
    actions2, policy2 = gumbel_root_act(e, net, search_config=cfg)
    assert torch.equal(actions, actions2)
    assert torch.equal(policy, policy2)


class ZeroNet:
    def __call__(self, g: torch.Tensor, s: torch.Tensor, legal: torch.Tensor, players: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(legal, dtype=torch.float32).masked_fill(~legal, -1e9), torch.zeros((len(g), 4))


def test_tree_finds_exact_terminal_win() -> None:
    e = BatchedEngine(1, 2, 'cpu', seed=1)
    e.factory_tiles.zero_(); e.center_tiles.zero_(); e.center_tiles[0, 0] = 1
    e.center_first[0] = False; e.wall[0, 0, 0, 1:] = True; e.scores[0, 0] = 30
    cfg = SearchConfig(backend='gumbel_tree', num_simulations=32, max_root_candidates=16, seed=19)
    actions, _ = gumbel_root_act(e, ZeroNet(), search_config=cfg)
    assert actions.item() == A.encode_action(e.num_factories, 0, 0)


def test_absolute_player_q_and_mixed_completion() -> None:
    e = BatchedEngine(1, 3, 'cpu', seed=0)
    node = Node(e, np.zeros(300), np.array([.2, .4, -.2, 0]), np.ones(300, dtype=bool), 1)
    node.edges[7] = Edge(visits=2, total=np.array([-2., 1.5, .5, 0]))
    q, visits = completed_q(node)
    assert q[7] == .75
    assert abs(q[8] - (.4 + 2 * .75) / 3) < 1e-10
    assert visits.sum() == 2


def test_leaf_values_convert_relative_network_seats_to_absolute_players() -> None:
    from agent.search.tree import _nodes
    class RelativeNet(ZeroNet):
        def __call__(self, g, s, legal, players):
            logits, _ = super().__call__(g, s, legal, players)
            return logits, torch.tensor([[.1, .3, .7, 0.]]).repeat(len(g), 1)
    e = BatchedEngine(3, 3, 'cpu', seed=3)
    e.current_player[:] = torch.tensor([0, 1, 2], dtype=torch.int8)
    nodes = _nodes(e, RelativeNet(), SearchConfig())
    expected = [[.1, .3, .7, 0], [.7, .1, .3, 0], [.3, .7, .1, 0]]
    assert np.allclose([n.raw_value for n in nodes], expected)


def test_tree_deadline_and_ended_rows() -> None:
    e = BatchedEngine(2, 2, 'cpu', seed=0); e.ended[0] = True
    cfg = SearchConfig(backend='gumbel_tree', num_simulations=10000, move_deadline_s=.1, seed=4)
    started = time.monotonic(); actions, policies = gumbel_root_act(e, ZeroNet(), search_config=cfg)
    assert time.monotonic() - started < .6
    assert policies[0].sum() == 0
    assert e.legal_action_mask()[1, actions[1]]


def test_sequential_halving_balances_then_eliminates() -> None:
    assert considered_visits(4, 16) == [0]*4 + [1]*4 + [2]*2 + [3]*2 + [4]*2 + [5]*2
    assert considered_visits(1, 4) == [0, 1, 2, 3]
