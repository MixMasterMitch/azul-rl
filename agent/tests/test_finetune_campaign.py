from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from agent.eval.arena import ArenaConfig, write_report
from agent.net import encoder as E
from agent.net.model import AzulNet
from agent.scripts import finetune_campaign as FC
from agent.scripts.competitive import Campaign
from agent.tests.test_enhancement_campaign import report
from agent.train.checkpointing import (
    load_checkpoint,
    load_checkpoint_payload,
    save_checkpoint,
)
from agent.train.league import League
from agent.train.loop import teacher_simulations
from agent.train.presets import enhanced_2p_config
from agent.train.replay_buffer import ReplayBuffer


def assert_same(a: object, b: object) -> None:
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("arm", FC.ARMS)
def test_fork_preserves_learning_state_and_applies_only_declared_change(
    tmp_path: Path, arm: str
) -> None:
    net = AzulNet(hidden=32, arch="flat")
    config = replace(enhanced_2p_config(), hidden=32, arch="flat")
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr)
    sum(p.sum() for p in net.parameters()).backward()
    optimizer.step()
    replay = ReplayBuffer(4, E.D_GLOBAL, E.NUM_SOURCES, E.D_SOURCE, 300, 4)
    replay.add(
        torch.rand(4, E.D_GLOBAL),
        torch.rand(4, E.NUM_SOURCES, E.D_SOURCE),
        torch.ones(4, 300, dtype=torch.bool),
        torch.ones(4, 300) / 300,
        torch.randn(4, 4),
    )
    source = tmp_path / "original.pt"
    save_checkpoint(
        source, net, optimizer, 420, asdict(config), replay, {"training_wall_s": 1234.0}
    )
    original = load_checkpoint_payload(source)
    original_hash = FC.checkpoint_hash(source)
    league = League(tmp_path / "league")
    league.add_checkpoint(net, "original", metadata={"pinned": True})
    directory = tmp_path / arm
    changed = replace(
        config,
        **FC.COMMON_MITIGATION,
        **FC.VARIATIONS[arm],
        run_id=arm,
        league_root=str(directory / "league"),
    )
    record = FC.fork_variant(
        source, league.root, directory, changed, FC.VARIATIONS[arm]
    )
    clone = load_checkpoint_payload(directory / "checkpoints/latest_resume.pt")
    for key in ["model_state_dict", "buffer", "rng_state", "iteration", "progress"]:
        assert_same(original[key], clone[key])
    assert_same(
        original["optimizer_state_dict"]["state"],
        clone["optimizer_state_dict"]["state"],
    )
    loaded = AzulNet(hidden=32, arch="flat")
    restored_optim = torch.optim.AdamW(loaded.parameters(), lr=999.0)
    load_checkpoint(directory / "checkpoints/latest_resume.pt", loaded, restored_optim)
    assert all(group["lr"] == changed.lr for group in restored_optim.param_groups)
    assert FC.checkpoint_hash(source) == original_hash
    assert (
        FC.fork_variant(source, league.root, directory, changed, FC.VARIATIONS[arm])
        == record
    )
    if arm == "deeper_teacher":
        assert {teacher_simulations(changed, i) for i in range(421, 450)} == {256}
    with pytest.raises(ValueError, match="changed"):
        FC.fork_variant(
            source, league.root, directory, replace(changed, lr=0.1), {"lr": 0.1}
        )
    with pytest.raises(ValueError, match="Undeclared"):
        FC.fork_variant(
            source,
            league.root,
            tmp_path / "bad",
            replace(changed, entropy_bonus=0.5),
            FC.VARIATIONS[arm],
        )


def panel(score: float, cfg: ArenaConfig) -> dict:
    return {name: report(score, cfg, name) for name in FC.PANEL}


def test_exploratory_selection_does_not_require_a_resolved_pilot_gain() -> None:
    cfg = ArenaConfig(num_games=16)
    baseline = panel(0.5, cfg)
    candidate = panel(0.5, cfg)
    # A small positive mean with a wide paired interval; research can continue it.
    for r in candidate.values():
        r["summary"]["pair_scores"] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5, 1.0]
        r["summary"]["match_score"] = 0.5625
    decision = FC.choose_continuation(
        {"current": baseline, "lower_lr": candidate, "deeper_teacher": baseline},
        baseline,
    )
    assert decision["selected_arm"] == "lower_lr"
    assert decision["versus_control"]["lower_lr"]["paired_ci95"][0] < 0
    candidate["astra"] = report(0.25, cfg, "astra")
    assert (
        FC.choose_continuation(
            {"current": baseline, "lower_lr": candidate, "deeper_teacher": baseline},
            baseline,
        )["selected_arm"]
        == "current"
    )


@pytest.mark.parametrize("outcome", ["pass", "fail", "start"])
def test_campaign_keeps_start_on_failed_confirmation_skips_duplicates_and_resumes(
    tmp_path: Path, monkeypatch, outcome: str
) -> None:
    campaign = Campaign(tmp_path / "campaign", "cpu", 8, campaign_hours=24)
    write_report(campaign.root / "reliability.json", {"cleared_for_training": True})
    source = tmp_path / "source.pt"
    source.write_text("source")
    frozen = {}
    for name in ("start", "previous"):
        p = tmp_path / f"{name}.pt"
        p.write_text(name)
        frozen[name] = str(p)
    prepared = {
        "arms": {
            arm: asdict(
                replace(
                    enhanced_2p_config(),
                    run_id=arm,
                    runs_root=str(campaign.root / "experiments"),
                    league_root=str(campaign.root / arm),
                )
            )
            for arm in FC.ARMS
        },
        "frozen": frozen,
        "base_training_wall_s": 600.0,
        "source_resume_sha256": FC.checkpoint_hash(source),
    }
    monkeypatch.setattr(FC, "prepare", lambda *args: prepared)
    now = time.time()
    write_report(
        campaign.root / "preflight_budget.json",
        {"started_at": now - 100, "deadline": now + 86000},
    )
    calls, matches = [], []

    def train(campaign, config, minutes, label, *, evaluation_reserve_minutes):
        assert evaluation_reserve_minutes == 195
        assert campaign.deadline <= time.monotonic() + 86000
        calls.append((config.run_id, minutes, label))
        directory = Path(config.runs_root) / config.run_id
        path = directory / "milestones" / f"{label}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config.run_id + label)
        resume = directory / "checkpoints/latest_resume.pt"
        resume.parent.mkdir(exist_ok=True)
        resume.write_text(label)
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": FC.checkpoint_hash(path),
            "iteration": int(minutes),
            "training_wall_s": minutes * 60,
            "target_minutes": minutes,
            "executed_target_minutes": minutes,
            "replay_retained": True,
        }

    monkeypatch.setattr(FC, "train_segment", train)

    def match(label, candidate, opponent, cfg, search):
        matches.append(label)
        score = 0.5
        if label.startswith("pilot_lower_lr") and outcome != "start":
            score = 0.75
        if label.startswith("minutes_0780") and outcome != "start":
            score = 1.0
        if label.startswith(("confirmation_", "greedy_")):
            assert (campaign.root / "final_selection.json").exists()
            assert cfg.seed == (150_000_008 if cfg.greedy else 140_000_008)
            assert cfg.num_games == (512 if cfg.greedy else 2048)
            if "candidate" in label:
                score = 0.75 if outcome == "pass" else 0.25
        else:
            assert cfg.seed == 130_000_008 and cfg.num_games == 512
        r = report(score, cfg, opponent)
        write_report(campaign.root / "evaluations" / f"{label}.json", r)
        return r

    monkeypatch.setattr(campaign, "_match", match)
    result = FC.run_finetune_campaign(campaign, str(source))
    assert calls[:3] == [(arm, 190.0, "pilot") for arm in FC.ARMS]
    assert calls[-1][1] == 790
    assert result["improvement_resolved"] == (outcome == "pass")
    assert result["selected_checkpoint"] == (
        result["candidate_checkpoint"] if outcome == "pass" else frozen["start"]
    )
    assert Path(result["resume_checkpoint"]).exists()
    assert result["confirmation_reused_start"] == (outcome == "start")
    assert bool(
        [label for label in matches if label.startswith("confirmation_candidate")]
    ) == (outcome != "start")
    assert (result["panel_difference"] is None) == (outcome == "start")
    count = len(calls), len(matches)
    FC.run_finetune_campaign(campaign, str(source))
    assert count == (len(calls), len(matches))
    assert (
        json.loads((campaign.root / "status.json").read_text())["stage"]
        == "finetune_campaign_complete"
    )
    assert (
        json.loads((campaign.root / "finetune_campaign_budget.json").read_text())[
            "deadline"
        ]
        == now + 86000
    )


def test_unresolved_reliability_blocks_training(tmp_path: Path, monkeypatch) -> None:
    campaign = Campaign(tmp_path, "cpu", 1, campaign_hours=24)
    write_report(tmp_path / "reliability.json", {"cleared_for_training": False})
    monkeypatch.setattr(FC, "prepare", lambda *args: pytest.fail("must not start"))
    with pytest.raises(RuntimeError, match="memory-corruption"):
        FC.run_finetune_campaign(campaign, "unused.pt")


def test_production_clearance_requires_the_debug_allocator(
    tmp_path: Path, monkeypatch
) -> None:
    campaign = Campaign(tmp_path, "cpu", 1, campaign_hours=24)
    write_report(
        tmp_path / "reliability.json",
        {"cleared_for_training": True, "require_debug_allocator": True},
    )
    monkeypatch.setenv("PYTHONMALLOC", "malloc")
    monkeypatch.setattr(FC, "prepare", lambda *args: pytest.fail("must not start"))
    with pytest.raises(RuntimeError, match="diagnostics"):
        FC.run_finetune_campaign(campaign, "unused.pt")
