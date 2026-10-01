from __future__ import annotations

import torch

from agent.env.engine import GameEngine
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.search.gumbel_mcts import gumbel_root_act
from agent.search.parallel import shutdown_parallel_tree_pools


def test_parallel_tree_search_is_legal_deterministic_and_non_mutating() -> None:
    engine = GameEngine(8, 2, "cpu", seed=701)
    before = engine.state_dict()
    net = AzulNet(hidden=32, arch="source_attn").eval()
    config = SearchConfig(
        backend="gumbel_tree",
        num_simulations=4,
        max_root_candidates=4,
        leaf_batch_size=16,
        cpu_workers=2,
        inference_batch_size=32,
        inference_wait_ms=5,
        seed=702,
    )
    try:
        actions, policies = gumbel_root_act(engine, net, search_config=config)
        actions_again, policies_again = gumbel_root_act(
            engine, net, search_config=config
        )
    finally:
        shutdown_parallel_tree_pools()

    legal = engine.legal_action_mask()
    assert legal.gather(1, actions[:, None]).all()
    assert torch.allclose(policies.sum(1), torch.ones(engine.batch_size))
    assert (policies[~legal] == 0).all()
    assert torch.equal(actions, actions_again)
    assert torch.equal(policies, policies_again)
    assert engine.state_dict() == before


def test_parallel_config_validation() -> None:
    for workers in (0, 9, True, 1.5):
        try:
            SearchConfig(cpu_workers=workers)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted invalid cpu_workers={workers!r}")
