from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from agent.env.engine import GameEngine
from agent.net.encoder import encode_state
from agent.net.model import AzulNet
from agent.scripts.aux_score_campaign import (
    augment_training_state,
    candidate_rank,
    clear_regression,
)
from agent.tests.test_finetune_campaign import assert_same
from agent.tests.test_loop_recovery import small_config
from agent.tests.test_training_enhancements import make_buffer, add_positions
from agent.train.checkpointing import (
    load_checkpoint_payload,
    load_net_from_checkpoint,
    save_checkpoint,
)
from agent.train.learner import make_optimizer, step, step_from_buffer
from agent.train.loop import run_loop
from agent.train.score_targets import final_score_margins


def test_final_score_perspective_and_shared_score() -> None:
    scores = torch.tensor([[75, 52, 0, 0], [60, 60, 0, 0]])
    idx = torch.tensor([0, 1, 0, 1])
    cp = torch.tensor([0, 1, 1, 0])
    assert final_score_margins(scores, idx, cp, 2).tolist() == [23.0, 0.0, -23.0, 0.0]
    assert torch.isnan(final_score_margins(scores, idx, cp, 3)).all()


def test_score_labels_wrap_sample_resume_and_legacy() -> None:
    buffer = make_buffer(4)
    g, s = encode_state(GameEngine(7, 2, seed=5))
    legal = torch.ones(7, 300, dtype=torch.bool)
    policy = legal.float() / 300
    values = torch.zeros(7, 4)
    buffer.add(g[:3], s[:3], legal[:3], policy[:3], values[:3])
    margins = torch.arange(7.0) - 3
    buffer.add(g, s, legal, policy, values, score_margin=margins)
    assert buffer.score_margin.tolist() == [2.0, 3.0, 0.0, 1.0]
    restored = make_buffer(4)
    restored.load_state_dict(buffer.state_dict())
    batch = restored.sample(40, include_score_margin=True)
    for features, margin in zip(batch[0], batch[-1]):
        assert any(
            torch.equal(features, g[i]) and margin == margins[i] for i in range(3, 7)
        )
    # An unlabelled overwrite invalidates the old label, including exact zero margins.
    restored.add(g[:2], s[:2], legal[:2], policy[:2], values[:2])
    assert restored.score_margin_valid.tolist() == [True, True, False, False]
    state = restored.state_dict()
    state.pop("score_margin")
    state.pop("score_margin_valid")
    restored.load_state_dict(state)
    assert not restored.score_margin_valid.any()
    assert torch.isnan(restored.sample(8, include_score_margin=True)[-1]).all()
    with pytest.raises(ValueError, match="score_margin"):
        restored.add(
            g, s, legal, policy, values, score_margin=torch.full((7,), float("inf"))
        )


def test_auxiliary_head_is_skipped_in_inference_and_zero_weight_is_exact_control(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    base = AzulNet(hidden=16, arch="source_attn", dropout=0)
    torch.manual_seed(7)
    aux = AzulNet(hidden=16, arch="source_attn", dropout=0, aux_score=True)
    for k, v in base.state_dict().items():
        assert torch.equal(v, aux.state_dict()[k])
    buffer = make_buffer(8)
    add_positions(buffer, GameEngine(8, 2, seed=71))
    batch = buffer.sample(8)
    base.eval()
    aux.eval()
    calls = []
    handle = aux.score_head.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        expected = base(*batch[:3], 2)
        actual = aux(*batch[:3], 2)
        assert_same(expected, actual)
        with_score = aux.forward_with_score(*batch[:3], 2)
        assert_same(expected, with_score[:2])
    assert len(calls) == 1
    handle.remove()
    a = make_optimizer(base)
    b = make_optimizer(aux)
    step(base, a, *batch)
    step(aux, b, *batch, score_margin=torch.full((8,), 25.0), aux_score_weight=0)
    for k, v in base.state_dict().items():
        assert torch.equal(v, aux.state_dict()[k])
    path = tmp_path / "aux.pt"
    save_checkpoint(path, aux, config={"num_players": 2})
    loaded, payload = load_net_from_checkpoint(path)
    assert loaded.aux_score and payload["aux_score"]
    assert_same(loaded.state_dict(), aux.state_dict())


def test_score_loss_masks_legacy_and_illegal_rows_without_changing_value_targets() -> (
    None
):
    net = AzulNet(hidden=16, arch="flat", aux_score=True)
    buffer = make_buffer(4)
    add_positions(buffer, GameEngine(4, 2, seed=71))
    g, s, legal, policy, value = buffer.sample(4)
    legal[-1] = False
    original_value = value.clone()
    result = step(
        net,
        torch.optim.SGD(net.parameters(), lr=0),
        g,
        s,
        legal,
        policy,
        value,
        score_margin=torch.tensor([50.0, -50.0, float("nan"), 10000.0]),
        aux_score_weight=0.1,
    )
    assert result["skipped"] == 0
    assert result["aux_score_loss"] == pytest.approx(1 / 3)
    assert result["aux_score_label_fraction"] == pytest.approx(2 / 3)
    assert torch.equal(original_value, value)
    # No labelled rows still permits a valid ordinary update.
    result = step(
        net,
        make_optimizer(net),
        g,
        s,
        legal,
        policy,
        value,
        score_margin=torch.full((4,), float("nan")),
        aux_score_weight=0.1,
    )
    assert result["aux_score_loss"] == result["aux_score_label_fraction"] == 0
    assert result["skipped"] == 0


@pytest.mark.parametrize("kind", ["self", "league", "bot"])
def test_all_generators_record_scores_only_for_finished_games(
    tmp_path: Path, kind: str
) -> None:
    from agent.train.selfplay import run_selfplay
    from agent.train.bot_selfplay import run_bot_selfplay
    from agent.train.league_selfplay import run_league_selfplay
    from agent.train.league import League

    net = AzulNet(hidden=16, arch="flat")
    args = dict(
        num_games=4,
        num_players=2,
        num_sims=2,
        max_turns=200,
        seed=17,
        search_backend="gumbel_tree",
        search_tree_core="rust",
        reward_mode="binary",
    )
    buffer = make_buffer(1000)
    league = League(tmp_path / "league")
    league.add_checkpoint(net, "opponent")

    def generate(target, **kwargs):
        if kind == "self":
            return run_selfplay(net, target, **kwargs)
        if kind == "bot":
            return run_bot_selfplay(net, target, **kwargs)
        return run_league_selfplay(net, target, league, **kwargs)

    result = generate(buffer, **args)
    assert result["finished"] == 4 and buffer.size > 0
    assert buffer.score_margin_valid[: buffer.size].all()
    margins = buffer.score_margin[: buffer.size]
    values = buffer.value_target[: buffer.size, 0]
    assert (values[margins > 0] > 0).all() and (values[margins < 0] < 0).all()
    unfinished = make_buffer(1000)
    generate(unfinished, **{**args, "max_turns": 1})
    assert unfinished.size == 0 and not unfinished.score_margin_valid.any()


def test_augmented_forks_retain_optimizer_replay_and_identical_initial_state(
    tmp_path: Path,
) -> None:
    net = AzulNet(hidden=16, arch="flat")
    buffer = make_buffer(8)
    add_positions(buffer, GameEngine(8, 2, seed=17))
    optim = make_optimizer(net)
    step_from_buffer(net, buffer, optim, 8)
    path = tmp_path / "source.pt"
    save_checkpoint(path, net, optim, 12, {"num_players": 2}, buffer)
    original = load_checkpoint_payload(path)
    cfg = replace(
        small_config(str(tmp_path), "score"),
        hidden=16,
        aux_score_head=True,
        aux_score_weight=0.1,
    )
    treatment = augment_training_state(copy.deepcopy(original), cfg)
    control = augment_training_state(
        copy.deepcopy(original), replace(cfg, aux_score_weight=0)
    )
    for key in ("model_state_dict", "optimizer_state_dict", "buffer", "rng_state"):
        assert_same(control[key], treatment[key])
    assert_same(
        original["optimizer_state_dict"]["state"],
        treatment["optimizer_state_dict"]["state"],
    )
    assert_same(original["buffer"], treatment["buffer"])
    assert (
        treatment["iteration"] == 12 and treatment["progress"]["training_wall_s"] == 0
    )
    loaded = AzulNet(hidden=16, arch="flat", aux_score=True)
    loaded.load_state_dict(treatment["model_state_dict"])
    resumed_optim = make_optimizer(loaded)
    resumed_optim.load_state_dict(treatment["optimizer_state_dict"])
    assert (
        step_from_buffer(loaded, buffer, resumed_optim, 8, aux_score_weight=0.1)[
            "skipped"
        ]
        == 0
    )


def test_score_training_resume_matches_uninterrupted_and_rejects_weight_change(
    tmp_path: Path,
) -> None:
    from agent.obs.run import Run

    cfg = replace(
        small_config(str(tmp_path), "resumed"),
        aux_score_head=True,
        aux_score_weight=0.1,
        training_cycle_length=4,
        max_wall_minutes=5,
        search_backend="gumbel_tree",
        search_tree_core="rust",
    )

    def execute(config):
        run = Run(config.run_id, runs_root=config.runs_root)
        try:
            return run_loop(run, config, explicit_fields=set(config.__dict__))
        finally:
            run.close()

    execute(cfg)
    execute(cfg)
    execute(
        replace(
            cfg, run_id="full", league_root=str(tmp_path / "full/league"), max_iters=2
        )
    )
    a = load_checkpoint_payload(tmp_path / "resumed/checkpoints/latest_resume.pt")
    b = load_checkpoint_payload(tmp_path / "full/checkpoints/latest_resume.pt")
    for key in ("model_state_dict", "optimizer_state_dict", "buffer", "rng_state"):
        assert_same(a[key], b[key])
    assert a["buffer"]["score_margin_valid"].any()
    rows = [
        json.loads(line)
        for line in (tmp_path / "full/events.log").read_text().splitlines()
    ]
    metrics = [r["fields"] for r in rows if r.get("event") == "learner_done"]
    assert all(
        0 < r["aux_score_label_fraction"] <= 1 and r["aux_score_loss"] > 0
        for r in metrics
    )
    with pytest.raises(ValueError, match="aux_score_weight"):
        execute(replace(cfg, aux_score_weight=0.2))


def test_selection_tolerates_uncertain_astra_drop_but_vetoes_resolved_regression() -> (
    None
):
    def candidate(point, interval):
        return {
            "head": {"match_score": point},
            "differences": {
                "astra": {"match_score_difference": -0.05, "paired_ci95": interval}
            },
        }

    assert candidate_rank(candidate(0.514, [-0.11, 0.01])) > candidate_rank(
        candidate(0.46, [-0.02, 0.02])
    )
    assert candidate_rank(candidate(0.514, [-0.11, -0.04])) < candidate_rank(
        candidate(0.46, [-0.02, 0.02])
    )
    assert clear_regression({"match_score_ci95": [0.25, 0.4]})
    assert not clear_regression({"match_score_ci95": [0.44, 0.55]})
