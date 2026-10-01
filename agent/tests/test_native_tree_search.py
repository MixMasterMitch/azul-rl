from __future__ import annotations

from dataclasses import replace
import torch

from agent.env import actions as A
from agent.env.batched_engine import BatchedEngine
from agent.env.engine import GameEngine
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.search.gumbel_mcts import gumbel_root_act
from agent.search.parallel import shutdown_parallel_tree_pools
from agent.search.native_tree import NativeInferenceAdapter, _inference_adapter
from agent.net import encoder as ENC
from agent.train.instrumentation import PerfCounters


def _config(workers: int = 1) -> SearchConfig:
    return SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=8,
        max_root_candidates=4,
        chance_samples=2,
        seed=9182,
        cpu_workers=workers,
        inference_batch_size=64,
        inference_wait_ms=1,
    )


def test_native_tree_is_legal_normalized_deterministic_and_nonmutating() -> None:
    torch.manual_seed(3)
    net = AzulNet(hidden=32, dropout=0).eval()
    engine = GameEngine(8, 2, "cpu", seed=41)
    snapshots = engine.state_dict()["snapshots"]

    actions, policies = gumbel_root_act(engine, net, search_config=_config())
    actions_again, policies_again = gumbel_root_act(
        engine, net, search_config=_config()
    )

    assert engine.state_dict()["snapshots"] == snapshots
    assert engine.legal_action_mask().gather(1, actions[:, None]).all()
    assert torch.allclose(policies.sum(1), torch.ones(8), atol=1e-5)
    assert torch.isfinite(policies).all()
    assert torch.equal(actions, actions_again)
    assert torch.equal(policies, policies_again)


def test_native_tree_uses_exact_terminal_values_and_reports_depth() -> None:
    reference = BatchedEngine(1, 2, "cpu", seed=1)
    returned = reference.factory_tiles.sum(1) + reference.center_tiles
    reference.bag += returned.to(reference.bag.dtype)
    reference.factory_tiles.zero_()
    reference.center_tiles.zero_()
    reference.center_tiles[0, 0] = 1
    reference.bag[0, 0] -= 1
    reference.center_first[0] = False
    reference.wall[0, 0, 0, 1:] = True
    reference.bag[0, 1:] -= 1
    reference.scores[0, 0] = 30
    engine = GameEngine.from_batched(reference, seed=1)
    perf = PerfCounters(True)
    actions, _ = gumbel_root_act(
        engine,
        AzulNet(hidden=32, arch="flat").eval(),
        search_config=replace(_config(), num_simulations=32, max_root_candidates=16),
        perf=perf,
    )
    assert actions.item() == A.encode_action(engine.num_factories, 0, 0)
    assert perf.snapshot()["profile_native_tree_max_depth"] >= 1


def test_native_tree_runs_in_parallel_workers() -> None:
    torch.manual_seed(4)
    net = AzulNet(hidden=32, dropout=0).eval()
    engine = GameEngine(8, 2, "cpu", seed=42)
    try:
        actions, policies = gumbel_root_act(engine, net, search_config=_config(2))
        assert engine.legal_action_mask().gather(1, actions[:, None]).all()
        assert torch.allclose(policies.sum(1), torch.ones(8), atol=1e-5)
    finally:
        shutdown_parallel_tree_pools()


def test_native_tree_supports_every_player_count() -> None:
    net = AzulNet(hidden=32, arch="flat").eval()
    for players in (2, 3, 4):
        engine = GameEngine(2, players, "cpu", seed=players)
        actions, policies = gumbel_root_act(engine, net, search_config=_config())
        assert engine.legal_action_mask().gather(1, actions[:, None]).all()
        assert torch.allclose(policies.sum(1), torch.ones(2), atol=1e-5)


def test_tree_core_configuration_is_validated() -> None:
    try:
        SearchConfig(tree_core="unknown")
    except ValueError as exc:
        assert "tree core" in str(exc)
    else:
        raise AssertionError("invalid tree core was accepted")


def test_inference_cache_deduplicates_inputs_and_is_bounded() -> None:
    class CountingModel:
        def __init__(self) -> None:
            self.rows = 0

        def __call__(
            self, g: torch.Tensor, s: torch.Tensor, legal: torch.Tensor, players: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.rows += len(g)
            return legal.float() * g[:, :1], g[:, :4] + players

    model = CountingModel()
    adapter = NativeInferenceAdapter(model, cache_size=2)
    g = torch.zeros(3, ENC.D_GLOBAL)
    g[:, 0] = torch.tensor([1.0, 1.0, 2.0])
    s = torch.zeros(3, ENC.NUM_SOURCES, ENC.D_SOURCE)
    legal = torch.ones(3, 300, dtype=torch.uint8)
    args = (
        bytearray(g.numpy().tobytes()),
        bytearray(s.numpy().tobytes()),
        bytearray(legal.numpy().tobytes()),
        3,
        2,
    )
    first = adapter(*args)
    assert model.rows == 2
    assert adapter(*args) == first
    assert model.rows == 2 and adapter.cache_hits == 4
    # Player-count heads must not collide; total cached entries stay bounded.
    assert adapter(*args[:-1], 3) != first
    assert model.rows == 4 and len(adapter.cache) == 2


def test_cached_tree_preserves_search_and_never_mutates_live_draws() -> None:
    torch.manual_seed(33)
    net = AzulNet(hidden=32, arch="flat", dropout=0).eval()
    engine = GameEngine(4, 2, game_seeds=[12] * 4)
    before = engine.state_dict()
    actions, policy = gumbel_root_act(engine, net, search_config=_config())
    perf = PerfCounters(True)
    cached_actions, cached_policy = gumbel_root_act(
        engine,
        net,
        search_config=replace(_config(), inference_cache_size=128),
        perf=perf,
    )
    assert torch.equal(actions, cached_actions)
    torch.testing.assert_close(policy, cached_policy, atol=1e-5, rtol=1e-5)
    assert perf.snapshot()["profile_native_tree_inference_cache_hits"] > 0
    assert before == engine.state_dict()


def test_search_cache_reuses_moves_and_invalidates_after_weight_updates() -> None:
    net = AzulNet(hidden=32, arch="flat", dropout=0).eval()
    cfg = replace(_config(), inference_cache_size=128)
    adapter = _inference_adapter(net, cfg)
    assert _inference_adapter(net, cfg) is adapter
    with torch.no_grad():
        next(net.parameters()).add_(0.1)
    after_update = _inference_adapter(net, cfg)
    assert after_update is not adapter
    net.load_state_dict(net.state_dict())
    assert _inference_adapter(net, cfg) is not after_update
    other_net = AzulNet(hidden=32, arch="flat", dropout=0).eval()
    assert _inference_adapter(other_net, cfg) is not _inference_adapter(net, cfg)
