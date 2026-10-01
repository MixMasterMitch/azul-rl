from dataclasses import replace

import pytest
import torch

from agent.env.engine import GameEngine
from agent.net.encoder import encode_state
from agent.net.model import AzulNet
from agent.tests.test_training_enhancements import make_buffer
from agent.train.policy_surprise import policy_surprise
from agent.train.loop import LoopConfig


def test_surprise_metadata_ring_legacy_and_invalid() -> None:
    b = make_buffer(4)
    g, s = encode_state(GameEngine(7, 2, seed=12))
    legal = torch.ones(7, 300, dtype=torch.bool)
    policy = legal.float() / 300
    value = torch.zeros(7, 4)
    b.add(g[:3], s[:3], legal[:3], policy[:3], value[:3], policy_surprise=torch.ones(3))
    b.add(
        g, s, legal, policy, value, policy_sims=256, policy_surprise=torch.arange(7.0)
    )
    assert b.policy_surprise.tolist() == [5.0, 6.0, 3.0, 4.0]
    copy = make_buffer(4)
    copy.load_state_dict(b.state_dict())
    torch.testing.assert_close(copy.policy_surprise, b.policy_surprise)
    legacy = b.state_dict()
    legacy.pop("policy_surprise")
    legacy.pop("policy_surprise_valid")
    copy.load_state_dict(legacy)
    assert copy.policy_surprise.isnan().all()
    for bad in [torch.full((7,), -1.0), torch.full((7,), float("inf")), torch.ones(3)]:
        with pytest.raises(ValueError, match="policy_surprise"):
            b.add(g, s, legal, policy, value, policy_surprise=bad)


def test_capped_sampling_preserves_unknown_fast_and_uniform_control() -> None:
    b = make_buffer(10)
    b.size = 10
    b.policy_sims[:] = 256
    # Mean KL of the eligible eight is one. Weights: seven .5, one 4 (capped), two 1.
    b.policy_surprise[:] = torch.tensor([0.0] * 7 + [8.0, float("nan"), 1000.0])
    b.policy_sims[-1] = 64
    torch.manual_seed(5)
    expected = torch.randint(0, 10, (100,))
    torch.manual_seed(5)
    actual = b._sample_indices(100, 0.0, 256, 4.0)
    assert torch.equal(expected, actual)
    torch.manual_seed(19)
    sampled = b._sample_indices(95000, 0.5, 256, 4.0)
    probabilities = torch.bincount(sampled, minlength=10).float() / len(sampled)
    torch.testing.assert_close(
        probabilities,
        torch.tensor([0.5] * 7 + [4.0, 1.0, 1.0]) / 9.5,
        atol=0.004,
        rtol=0,
    )
    state = torch.get_rng_state()
    first = b._sample_indices(100, 0.5, 256, 4.0)
    copy = make_buffer(10)
    copy.load_state_dict(b.state_dict())
    torch.set_rng_state(state)
    assert torch.equal(first, copy._sample_indices(100, 0.5, 256, 4.0))


def test_kl_uses_raw_prior_no_rng_and_restores_mode() -> None:
    b = make_buffer(8)
    b.policy_surprise_min_sims = 256
    engine = GameEngine(8, 2, seed=9)
    g, s = encode_state(engine)
    legal = engine.legal_action_mask()
    net = AzulNet(hidden=32, arch="flat", dropout=0).eval()
    with torch.no_grad():
        logits, _ = net(g, s, legal, 2)
        prior = logits.softmax(-1)
    state = torch.get_rng_state()
    net.train()
    kl = policy_surprise(net, b, g, s, legal, prior, 256, 2)
    assert net.training and torch.equal(state, torch.get_rng_state())
    assert kl.max() < 1e-6
    target = legal.float() / legal.sum(-1, keepdim=True)
    kl = policy_surprise(net, b, g, s, legal, target, 256, 2)
    expected = (target * (target.clamp_min(1e-30).log() - logits.log_softmax(-1))).sum(
        -1
    )
    torch.testing.assert_close(kl, expected.clamp_min(0), atol=1e-6, rtol=1e-4)
    assert policy_surprise(net, b, g, s, legal, target, 64, 2) is None


@pytest.mark.parametrize("kind", ["self", "league", "bot"])
def test_generators_capture_only_finished_full_search_targets(tmp_path, kind) -> None:
    from agent.train.selfplay import run_selfplay
    from agent.train.league_selfplay import run_league_selfplay
    from agent.train.bot_selfplay import run_bot_selfplay
    from agent.train.league import League

    b = make_buffer(1000, 0)
    b.policy_surprise_min_sims = 2
    net = AzulNet(hidden=32, arch="flat", dropout=0)
    args = dict(
        num_games=2,
        num_players=2,
        num_sims=2,
        max_turns=200,
        seed=91,
        search_backend="gumbel_tree",
        search_tree_core="rust",
        reward_mode="binary",
    )
    if kind == "self":
        result = run_selfplay(net, b, **args)
    elif kind == "bot":
        result = run_bot_selfplay(net, b, **args)
    else:
        league = League(tmp_path / "league")
        league.add_checkpoint(net, "opponent")
        result = run_league_selfplay(net, b, league, **args)
    assert result["finished"] == 2 and b.size > 0
    assert torch.isfinite(b.policy_surprise[: b.size]).all()
    assert (b.policy_surprise[: b.size] >= 0).all()


def test_reanalysis_refreshes_kl_and_invalidates_sampler() -> None:
    from agent.tests.test_training_enhancements import add_positions
    from agent.train.reanalysis import reanalyse_policy_targets
    from agent.search.config import SearchConfig

    b = make_buffer()
    b.policy_surprise_min_sims = 8
    add_positions(b, GameEngine(8, 2, seed=2))
    b.policy_surprise[:8] = 99.0
    b.policy_sims[:8] = 64
    b._surprise_mean_cache = (8, 99.0)
    before = b.value_target.clone()
    cfg = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=8,
        max_root_candidates=4,
    )
    reanalyse_policy_targets(
        AzulNet(hidden=32, arch="flat", dropout=0), b, 2, cfg, 3, seed=4
    )
    assert (b.policy_surprise[:8] != 99).sum() == 3
    assert b._surprise_mean_cache is None
    assert torch.equal(before, b.value_target)


def test_config_and_cli() -> None:
    from agent.scripts.train import parse_config

    cfg, explicit = parse_config(
        ["--policy-surprise-fraction", ".5", "--policy-surprise-record"]
    )
    assert cfg.policy_surprise_fraction == 0.5 and cfg.policy_surprise_record
    for fields in [
        dict(policy_surprise_fraction=0.6),
        dict(policy_surprise_min_sims=0),
        dict(policy_surprise_max_weight=float("nan")),
    ]:
        with pytest.raises(ValueError, match="policy_surprise"):
            replace(LoopConfig(), **fields)


def test_training_resume_is_deterministic(tmp_path) -> None:
    from agent.obs.run import Run
    from agent.tests.test_loop_recovery import small_config
    from agent.train.loop import run_loop
    from agent.train.checkpointing import load_checkpoint_payload
    from agent.tests.test_finetune_campaign import assert_same

    cfg = replace(
        small_config(str(tmp_path), "resumed"),
        policy_fast_weight=0.25,
        policy_surprise_fraction=0.5,
        policy_surprise_min_sims=4,
        policy_surprise_record=True,
        search_backend="gumbel_tree",
        search_tree_core="rust",
        selfplay_full_sims=4,
        selfplay_full_fraction=0.5,
        reanalysis_positions=8,
        reanalysis_every=1,
        reanalysis_sims=8,
        reanalysis_snapshot_capacity=128,
    )

    def train(c):
        run = Run(c.run_id, runs_root=str(tmp_path))
        try:
            run_loop(run, c)
        finally:
            run.close()

    train(cfg)
    train(cfg)
    train(
        replace(
            cfg, run_id="full", league_root=str(tmp_path / "full/league"), max_iters=2
        )
    )
    a = load_checkpoint_payload(tmp_path / "resumed/checkpoints/latest_resume.pt")
    b = load_checkpoint_payload(tmp_path / "full/checkpoints/latest_resume.pt")
    for k in ["model_state_dict", "optimizer_state_dict", "rng_state"]:
        assert_same(a[k], b[k])
    for k in a["buffer"]:
        if k == "policy_surprise":
            torch.testing.assert_close(
                a["buffer"][k], b["buffer"][k], equal_nan=True, rtol=0, atol=0
            )
        else:
            assert_same(a["buffer"][k], b["buffer"][k])
    with pytest.raises(ValueError, match="policy_surprise_fraction"):
        train(replace(cfg, policy_surprise_fraction=0.0))
