from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from agent.env.engine import GameEngine
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.tests.test_training_enhancements import make_buffer, add_positions
from agent.train.learner import step, step_from_buffer
from agent.train.reanalysis import reanalyse_policy_targets
from agent.train.loop import LoopConfig
from agent.scripts.train import parse_config


def test_budget_metadata_survives_ring_wrap_oversize_resume_and_legacy() -> None:
    b = make_buffer(4)
    e = GameEngine(7, 2, seed=12)
    from agent.net.encoder import encode_state

    g, s = encode_state(e)
    legal = e.legal_action_mask()
    p = legal.float() / legal.sum(-1, keepdim=True)
    v = torch.zeros(7, 4)
    b.add(g[:3], s[:3], legal[:3], p[:3], v[:3], policy_sims=64)
    b.add(g, s, legal, p, v, policy_sims=torch.tensor([1, 2, 3, 4, 5, 6, 7]))
    assert b.pos == 2
    assert b.policy_sims.tolist() == [6, 7, 4, 5]
    copy = make_buffer(4)
    copy.load_state_dict(b.state_dict())
    assert torch.equal(copy.policy_sims, b.policy_sims)
    torch.manual_seed(3)
    batch = copy.sample(30, include_policy_sims=True)
    for features, budget in zip(batch[0], batch[-1]):
        assert any(
            torch.equal(features, g[j]) and int(budget) == j + 1 for j in range(3, 7)
        )
    legacy = b.state_dict()
    legacy.pop("policy_sims")
    copy.load_state_dict(legacy)
    assert not copy.policy_sims.any()
    for bad in [-1, 1.5, float("nan"), torch.tensor([1, 2])]:
        with pytest.raises(ValueError, match="policy_sims"):
            b.add(g, s, legal, p, v, policy_sims=bad)


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(
            torch.tensor([[1.0, -1.0], [-0.3, 0.3], [0.2, -0.2]])
        )
        self.values = nn.Parameter(
            torch.tensor([[0.2, -0.2], [0.4, -0.4], [0.1, -0.1]])
        )

    def forward(self, g, s, legal, num_players):
        indices = g[:, 0].long()
        return self.logits[indices].masked_fill(~legal, -1e9), self.values[indices]


def test_weighted_kl_normalizes_after_filtering_and_leaves_value_gradient_unchanged() -> (
    None
):
    plain = TinyNet()
    weighted = TinyNet()
    g = torch.arange(3.0).reshape(3, 1)
    s = torch.zeros(3, 1, 1)
    legal = torch.tensor([[True, True], [True, True], [False, False]])
    p = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    v = torch.tensor([[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]])
    base = step(
        plain,
        torch.optim.SGD(plain.parameters(), lr=0),
        g,
        s,
        legal,
        p,
        v,
        entropy_bonus=0,
        max_grad_norm=1e9,
    )
    result = step(
        weighted,
        torch.optim.SGD(weighted.parameters(), lr=0),
        g,
        s,
        legal,
        p,
        v,
        entropy_bonus=0,
        max_grad_norm=1e9,
        policy_weights=torch.tensor([0.25, 1.0, 999.0]),
    )
    loss = F.kl_div(F.log_softmax(plain.logits[:2], -1), p[:2], reduction="none").sum(
        -1
    )
    assert result["policy_loss"] == pytest.approx(
        float((loss.detach() * torch.tensor([0.25, 1.0])).sum() / 1.25)
    )
    assert result["value_loss"] == base["value_loss"]
    torch.testing.assert_close(plain.values.grad, weighted.values.grad)
    torch.testing.assert_close(weighted.logits.grad[0], plain.logits.grad[0] * 0.4)
    torch.testing.assert_close(weighted.logits.grad[1], plain.logits.grad[1] * 1.6)


def test_weight_one_is_exact_control_and_unknown_budgets_are_neutral() -> None:
    from agent.train import learner

    b = make_buffer(8)
    add_positions(b, GameEngine(8, 2, seed=6))
    b.policy_sims[:8] = torch.tensor([0, 64, 256, 0, 64, 256, 64, 256])
    net = AzulNet(hidden=32, arch="flat", dropout=0)
    copy = AzulNet(hidden=32, arch="flat", dropout=0)
    copy.load_state_dict(net.state_dict())
    optim = torch.optim.SGD(net.parameters(), lr=0.01)
    other = torch.optim.SGD(copy.parameters(), lr=0.01)
    torch.manual_seed(8)
    batch = b.sample(8)
    step(net, optim, *batch)
    torch.manual_seed(8)
    step_from_buffer(copy, b, other, 8, policy_fast_weight=1.0)
    for a, c in zip(net.parameters(), copy.parameters()):
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    from unittest.mock import patch

    torch.manual_seed(9)
    with patch.object(learner, "step", return_value={}) as mocked:
        metrics = step_from_buffer(copy, b, other, 128, policy_fast_weight=0.25)
    weights = mocked.call_args.kwargs["policy_weights"]
    assert set(weights.tolist()) == {0.25, 1.0}
    assert 0 < metrics["policy_budget_known_fraction"] < 1
    b.policy_sims.zero_()
    with patch.object(learner, "step", return_value={}) as mocked:
        step_from_buffer(copy, b, other, 8, policy_fast_weight=0.25)
    assert mocked.call_args.kwargs["policy_weights"].tolist() == [1.0] * 8


def test_reanalysis_changes_budget_only_for_refreshed_positions() -> None:
    b = make_buffer()
    add_positions(b, GameEngine(8, 2, seed=2))
    b.policy_sims[:8] = 64
    before = b.value_target.clone()
    cfg = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=8,
        max_root_candidates=4,
    )
    reanalyse_policy_targets(
        AzulNet(hidden=32, arch="flat", dropout=0), b, 2, cfg, 3, batch_size=2, seed=4
    )
    assert (b.policy_sims[:8] == 8).sum() == 3
    assert (b.policy_sims[:8] == 64).sum() == 5
    assert torch.equal(before, b.value_target)


def test_cli_and_config_validate_weight() -> None:
    cfg, explicit = parse_config(
        [
            "--preset",
            "enhanced-2p",
            "--run-id",
            "weight-test",
            "--policy-fast-weight",
            ".25",
        ]
    )
    assert cfg.policy_fast_weight == 0.25 and "policy_fast_weight" in explicit
    for w in [0, -1, 1.01, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="policy_fast_weight"):
            replace(cfg, policy_fast_weight=w)
    assert LoopConfig().policy_fast_weight == 1.0


def test_weighted_training_resume_is_deterministic(tmp_path) -> None:
    from agent.obs.run import Run
    from agent.tests.test_loop_recovery import small_config
    from agent.train.loop import run_loop
    from agent.train.checkpointing import load_checkpoint_payload
    from agent.tests.test_finetune_campaign import assert_same

    cfg = replace(
        small_config(str(tmp_path), "resumed"),
        policy_fast_weight=0.25,
        search_backend="gumbel_tree",
        search_tree_core="rust",
        selfplay_full_sims=4,
        selfplay_full_fraction=0.5,
        reanalysis_positions=8,
        reanalysis_every=1,
        reanalysis_sims=8,
        reanalysis_snapshot_capacity=128,
    )
    for _ in range(2):
        run = Run(cfg.run_id, runs_root=str(tmp_path))
        try:
            run_loop(run, cfg)
        finally:
            run.close()
    full = replace(
        cfg, run_id="full", league_root=str(tmp_path / "full/league"), max_iters=2
    )
    run = Run(full.run_id, runs_root=str(tmp_path))
    try:
        run_loop(run, full)
    finally:
        run.close()
    a = load_checkpoint_payload(tmp_path / "resumed/checkpoints/latest_resume.pt")
    b = load_checkpoint_payload(tmp_path / "full/checkpoints/latest_resume.pt")
    for k in ["model_state_dict", "optimizer_state_dict", "buffer", "rng_state"]:
        assert_same(a[k], b[k])
    run = Run(cfg.run_id, runs_root=str(tmp_path))
    try:
        with pytest.raises(ValueError, match="policy_fast_weight"):
            run_loop(run, replace(cfg, policy_fast_weight=1.0))
    finally:
        run.close()


@pytest.mark.parametrize("kind", ["self", "league", "bot"])
def test_every_generator_records_actual_learner_search_budget(tmp_path, kind) -> None:
    from agent.train.selfplay import run_selfplay
    from agent.train.league_selfplay import run_league_selfplay
    from agent.train.bot_selfplay import run_bot_selfplay
    from agent.train.league import League

    b = make_buffer(1000, 0)
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
        r = run_selfplay(net, b, **args)
    elif kind == "bot":
        r = run_bot_selfplay(net, b, **args)
    else:
        league = League(tmp_path / "league")
        league.add_checkpoint(net, "opponent")
        r = run_league_selfplay(net, b, league, **args)
    assert r["finished"] == 2 and b.size > 0
    assert b.policy_sims[: b.size].unique().tolist() == [2]
