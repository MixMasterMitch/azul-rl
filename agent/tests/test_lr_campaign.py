from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import pytest

from agent.eval.arena import ArenaConfig, summarize_games
from agent.net.model import AzulNet
from agent.scripts import lr_campaign
from agent.scripts.competitive import Campaign, training_config
from agent.train.checkpointing import load_checkpoint_payload, save_checkpoint


def _report(score: float, config: ArenaConfig, opponent: str) -> dict:
    # Every seed pair has the same score, making the selection tests unambiguous.
    pair = {
        0.0: [0.0, 0.0],
        0.25: [0.0, 0.5],
        0.5: [1.0, 0.0],
        0.75: [1.0, 0.5],
        1.0: [1.0, 1.0],
    }[score]
    records = [
        {
            "pair_seed": config.seed + i // 2,
            "candidate_seat": i % 2,
            "outcome": {0.0: "loss", 0.5: "shared", 1.0: "win"}[pair[i % 2]],
            "match_score": pair[i % 2],
        }
        for i in range(config.num_games)
    ]
    return {
        "config": asdict(config),
        "opponent": opponent,
        "opponent_sha256": None,
        "opponent_identity": None,
        "opponent_search": asdict(config.search),
        "records": records,
        "summary": summarize_games(records),
    }


@pytest.mark.parametrize(
    "lower_primary,lower_astra,expected",
    [
        (0.75, 0.5, "lower"),
        (0.5, 0.75, "current"),
        (0.75, 0.25, "current"),
        (0.25, 0.5, "current"),
    ],
)
def test_rate_selection_requires_resolved_improvement_and_astra_guard(
    lower_primary: float, lower_astra: float, expected: str
) -> None:
    cfg = ArenaConfig(num_games=8)
    reports = {
        "current": {
            "tree64_frozen": _report(0.5, cfg, "frozen"),
            "tree64_astra": _report(0.5, cfg, "astra"),
        },
        "lower": {
            "tree64_frozen": _report(lower_primary, cfg, "frozen"),
            "tree64_astra": _report(lower_astra, cfg, "astra"),
        },
    }
    assert lr_campaign.choose_learning_rate(reports)["selected_arm"] == expected


def test_segments_restore_optimizer_replay_and_rng_without_retraining_cached_work(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    save_checkpoint(source, AzulNet(hidden=32, arch="flat"), config={"num_players": 2})
    campaign = Campaign(tmp_path / "campaign", "cpu", 22, campaign_hours=10)
    cfg = replace(
        training_config(campaign.root, "tiny", str(source), seed=22, device="cpu"),
        arch="flat",
        hidden=32,
        selfplay_games=4,
        selfplay_sims=2,
        replay_capacity=1000,
        learner_steps_per_iter=2,
        learner_batch=16,
        training_cycle_length=0,
        eval_games=0,
        checkpoint_every=1000,
        profile_every=0,
    )
    first = lr_campaign.train_segment(campaign, cfg, 0.003, "first")
    directory = Path(cfg.runs_root) / cfg.run_id
    resume = directory / "checkpoints/latest_resume.pt"
    payload1 = load_checkpoint_payload(resume)
    assert payload1["buffer"]["total_added"] > 0
    assert payload1["optimizer_state_dict"]["state"] and payload1["rng_state"]
    assert "buffer" not in load_checkpoint_payload(first["checkpoint"])
    before = resume.stat().st_mtime_ns
    assert lr_campaign.train_segment(campaign, cfg, 0.003, "first") == first
    assert resume.stat().st_mtime_ns == before
    campaign.deadline = time.monotonic() + 65 * 60 + 0.2
    requested_minutes = payload1["progress"]["training_wall_s"] / 60 + 1
    second = lr_campaign.train_segment(campaign, cfg, requested_minutes, "second")
    assert second["executed_target_minutes"] < requested_minutes
    payload2 = load_checkpoint_payload(resume)
    assert second["iteration"] > first["iteration"]
    assert payload2["buffer"]["total_added"] > payload1["buffer"]["total_added"]
    assert payload2["buffer"]["total_sampled"] > payload1["buffer"]["total_sampled"]
    steps1 = max(
        v["step"].item() for v in payload1["optimizer_state_dict"]["state"].values()
    )
    steps2 = max(
        v["step"].item() for v in payload2["optimizer_state_dict"]["state"].values()
    )
    assert steps2 > steps1
    events = [
        json.loads(line) for line in (directory / "events.log").read_text().splitlines()
    ]
    assert [r["fields"]["iter"] for r in events if r["event"] == "loop_resumed"] == [
        first["iteration"]
    ]
    with pytest.raises(ValueError, match="configuration changed"):
        lr_campaign.train_segment(campaign, replace(cfg, lr=0.1), 0.003, "first")
    lr_campaign.retire_replay(cfg, second)
    assert not resume.exists() and Path(second["checkpoint"]).exists()
    with pytest.raises(ValueError, match="retained optimizer and replay"):
        lr_campaign.train_segment(campaign, cfg, 1, "third")


def test_campaign_continues_selected_arm_and_freezes_selection_before_held_out_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    save_checkpoint(
        source, AzulNet(hidden=256, arch="source_attn"), config={"num_players": 2}
    )
    campaign = Campaign(
        tmp_path / "campaign", "cpu", 44, bot_workers=8, campaign_hours=10
    )
    stages, matches, retired = [], [], []

    def train(campaign: Campaign, config, minutes: float, label: str) -> dict:
        stages.append((config, minutes, label))
        path = campaign.root / f"{config.run_id}_{label}.pt"
        path.write_text(label)
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": lr_campaign.checkpoint_hash(path),
            "iteration": int(minutes),
            "target_minutes": minutes,
            "replay_retained": True,
        }

    def match(
        label: str, candidate: str, opponent: str, cfg: ArenaConfig, opponent_search
    ) -> dict:
        matches.append((label, candidate, opponent, cfg))
        score = 0.5
        if "pilot_lower" in label or "minutes_0180" in label:
            score = 0.75
        if "minutes_0300" in label:
            score = 1.0
        if label.startswith("confirmation_"):
            selected = json.loads((campaign.root / "final_selection.json").read_text())
            assert selected["selected_milestone"] == "minutes_0300"
            assert cfg.seed == 50_000_044 and cfg.split == "confirmation"
        else:
            assert cfg.seed == 40_000_044 and cfg.split == "development"
        assert cfg.search.cpu_workers == 1 and cfg.search.tree_core == "rust"
        assert cfg.search.dirichlet_mix == 0 and cfg.bot_workers == 8
        assert cfg.search == opponent_search
        return _report(score, cfg, opponent)

    monkeypatch.setattr(lr_campaign, "train_segment", train)
    monkeypatch.setattr(
        lr_campaign, "retire_replay", lambda cfg, milestone: retired.append(cfg.run_id)
    )
    monkeypatch.setattr(campaign, "_match", match)
    result = lr_campaign.run_lr_campaign(campaign, str(source))
    assert [minutes for _, minutes, _ in stages] == [60, 60, 180, 300, 450]
    first, second = [asdict(stages[i][0]) for i in range(2)]
    assert {k for k in first if first[k] != second[k]} == {
        "lr",
        "run_id",
        "league_root",
    }
    assert first["dirichlet_mix"] == 0 and first["search_cpu_workers"] == 1
    assert first["selfplay_sims"] == 64 and first["learner_steps_per_iter"] == 72
    assert all(cfg.lr == 0.0003 for cfg, _, _ in stages[1:])
    assert result["selected_arm"] == "lower"
    assert result["selected_milestone"] == "minutes_0300"
    assert not result["automatic_promotion"]
    assert [
        cfg.num_games
        for label, _, _, cfg in matches
        if label.startswith("confirmation_")
    ] == [1024, 1024, 1024, 256, 256]
    assert (
        len(matches) == 20
    )  # Eight pilot, baseline, six progress, five confirmation screens.
    budget = json.loads((campaign.root / "lr_campaign_budget.json").read_text())
    before = len(stages), len(matches)
    assert (
        lr_campaign.run_lr_campaign(campaign, str(source))["selected_checkpoint"]
        == result["selected_checkpoint"]
    )
    assert (len(stages), len(matches)) == before
    assert json.loads((campaign.root / "lr_campaign_budget.json").read_text()) == budget
    assert (
        json.loads(campaign.status_path.read_text())["stage"] == "lr_campaign_complete"
    )
