from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import pytest

from agent.eval.arena import ArenaConfig, summarize_games, write_report
from agent.net.model import AzulNet
from agent.scripts import enhancement_campaign as EC
from agent.scripts.competitive import Campaign
from agent.train.checkpointing import save_checkpoint


def report(score: float, config: ArenaConfig, opponent: str) -> dict:
    pairs = {
        0.0: [0.0, 0.0],
        0.25: [0.0, 0.5],
        0.5: [1.0, 0.0],
        0.75: [1.0, 0.5],
        1.0: [1.0, 1.0],
    }
    pair = pairs[score]
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
    "primary,astra,expected",
    [
        (0.75, 0.5, "enhanced"),
        (0.5, 0.75, "control"),
        (0.75, 0.25, "control"),
        (0.25, 0.5, "control"),
    ],
)
def test_selection_requires_resolved_primary_gain_and_astra_guard(
    primary: float, astra: float, expected: str
) -> None:
    cfg = ArenaConfig(num_games=8)
    reports = {
        "control": {
            "tree64_frozen": report(0.5, cfg, "frozen"),
            "tree64_astra": report(0.5, cfg, "astra"),
        },
        "enhanced": {
            "tree64_frozen": report(primary, cfg, "frozen"),
            "tree64_astra": report(astra, cfg, "astra"),
        },
    }
    assert EC.choose_arm(reports)["selected_arm"] == expected


def test_bundle_keeps_common_learning_settings_and_latest_control(
    tmp_path: Path,
) -> None:
    campaign = Campaign(tmp_path, "cpu", 20260920, campaign_hours=8)
    a, b = EC.campaign_configs(campaign, "/frozen.pt").values()
    for key in (
        "init_from",
        "arch",
        "hidden",
        "seed",
        "reward_mode",
        "lr",
        "weight_decay",
        "learner_steps_per_iter",
        "selfplay_games",
        "dirichlet_mix",
        "time_discount",
        "search_cpu_workers",
        "search_tree_core",
        "replay_capacity",
    ):
        assert getattr(a, key) == getattr(b, key), key
    assert a.dirichlet_mix == 0 and a.reward_mode == "binary" and a.selfplay_sims == 64
    assert a.reanalysis_positions == 0 and a.league_opponent_sims == 4
    assert b.reanalysis_positions == 256 and b.selfplay_full_fraction == 0.25
    assert b.league_opponent_sims == 64 and b.bot_selfplay_astra_prob == 0.5
    assert a.league_root != b.league_root


def test_end_to_end_protocol_selection_confirmation_and_idempotent_restart(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.pt"
    save_checkpoint(
        source, AzulNet(hidden=256, arch="source_attn"), config={"num_players": 2}
    )
    campaign = Campaign(
        tmp_path / "campaign", "cpu", 44, bot_workers=8, campaign_hours=8
    )
    started = time.time() - 300
    write_report(
        campaign.root / "preflight_budget.json",
        {"started_at": started, "deadline": started + 8 * 3600},
    )
    stages, matches, retired = [], [], []

    def train(
        campaign: Campaign,
        config,
        minutes: float,
        label: str,
        *,
        evaluation_reserve_minutes: float,
    ) -> dict:
        assert evaluation_reserve_minutes == 100
        stages.append((config, minutes, label))
        path = campaign.root / f"{config.run_id}_{label}.pt"
        path.write_text(label)
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": EC.checkpoint_hash(path),
            "iteration": int(minutes),
            "target_minutes": minutes,
            "replay_retained": True,
        }

    def match(
        label: str, candidate: str, opponent: str, cfg: ArenaConfig, opponent_search
    ) -> dict:
        matches.append((label, cfg))
        score = 0.75 if "pilot_enhanced" in label else 0.5
        if "minutes_0175" in label:
            score = 1.0
        if label.startswith("confirmation_"):
            selected = json.loads((campaign.root / "final_selection.json").read_text())
            assert selected["selected_milestone"] == "minutes_0175"
            assert cfg.seed == 70_000_044 and cfg.split == "confirmation"
        else:
            assert cfg.seed == 60_000_044 and cfg.split == "development"
        assert cfg.search == opponent_search and cfg.search.num_simulations == 64
        assert cfg.search.root_noise_scale == 1 and cfg.search.cpu_workers == 1
        result = report(score, cfg, opponent)
        write_report(campaign.root / "evaluations" / f"{label}.json", result)
        return result

    monkeypatch.setattr(EC, "train_segment", train)
    monkeypatch.setattr(
        EC, "retire_replay", lambda cfg, milestone: retired.append(cfg.run_id)
    )
    monkeypatch.setattr(campaign, "_match", match)
    result = EC.run_enhancement_campaign(campaign, str(source))
    assert [minutes for _, minutes, _ in stages] == [60, 60, 175, 290]
    assert (
        result["selected_arm"] == "enhanced"
        and result["selected_milestone"] == "minutes_0175"
    )
    assert all(cfg.reanalysis_positions == 256 for cfg, _, _ in stages[1:])
    assert len(matches) == 18 and not result["automatic_promotion"]
    assert [
        cfg.num_games for label, cfg in matches if label.startswith("confirmation_")
    ] == [1024, 1024, 1024, 256, 256]
    budget = json.loads(
        (campaign.root / "enhancement_campaign_budget.json").read_text()
    )
    assert budget["started_at"] == started and budget["deadline"] == started + 8 * 3600
    assert (campaign.root / "REPORT.md").exists()
    verification = json.loads((campaign.root / "verification.json").read_text())
    assert (
        verification["evaluation_games"] == 6912
        and verification["unfinished_evaluations"] == 0
    )
    before = len(stages), len(matches)
    assert (
        EC.run_enhancement_campaign(campaign, str(source))["selected_checkpoint"]
        == result["selected_checkpoint"]
    )
    assert before == (len(stages), len(matches))
    assert (
        json.loads(campaign.status_path.read_text())["stage"] == "enhancements_complete"
    )


def test_reanalysis_arms_only_change_refresh_count_and_run_paths(
    tmp_path: Path,
) -> None:
    campaign = Campaign(tmp_path, "cpu", 20260921, campaign_hours=12)
    a, b = (
        asdict(cfg) for cfg in EC.reanalysis_configs(campaign, "/frozen.pt").values()
    )
    assert {key for key in a if a[key] != b[key]} == {
        "run_id",
        "league_root",
        "reanalysis_positions",
    }
    assert a["reanalysis_positions"] == 256 and b["reanalysis_positions"] == 1024
    assert a["reanalysis_every"] == 4 and a["reanalysis_sims"] == 256
    assert a["hidden"] == 256 and a["search_tree_core"] == "rust"


@pytest.mark.parametrize(
    "experimental_gain,selected_arm", [(True, "reanalysis4x"), (False, "current")]
)
def test_reanalysis_protocol_retains_selected_resume_and_respects_deadline(
    tmp_path: Path,
    monkeypatch,
    experimental_gain: bool,
    selected_arm: str,
) -> None:
    source = tmp_path / "source.pt"
    save_checkpoint(
        source, AzulNet(hidden=256, arch="source_attn"), config={"num_players": 2}
    )
    campaign = Campaign(
        tmp_path / "campaign", "cpu", 45, bot_workers=8, campaign_hours=12
    )
    started = time.time() - 300
    write_report(
        campaign.root / "preflight_budget.json",
        {"started_at": started, "deadline": started + 12 * 3600},
    )
    stages, matches, retired = [], [], []

    def train(
        campaign: Campaign,
        config,
        minutes: float,
        label: str,
        *,
        evaluation_reserve_minutes: float,
    ) -> dict:
        assert evaluation_reserve_minutes == 100
        stages.append((config, minutes, label))
        directory = Path(config.runs_root) / config.run_id
        path = directory / "milestones" / f"{label}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(label)
        resume = directory / "checkpoints/latest_resume.pt"
        resume.parent.mkdir(exist_ok=True)
        resume.write_text(f"full-state:{label}")
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": EC.checkpoint_hash(path),
            "iteration": int(minutes),
            "target_minutes": minutes,
            "replay_retained": True,
        }

    def match(
        label: str, candidate: str, opponent: str, cfg: ArenaConfig, opponent_search
    ) -> dict:
        matches.append((label, cfg))
        confirmation = label.startswith("confirmation_")
        assert cfg.seed == (90_000_045 if confirmation else 80_000_045)
        assert cfg.search == opponent_search and cfg.search.num_simulations == 64
        if confirmation:
            assert (campaign.root / "final_selection.json").exists()
        else:
            assert cfg.num_games == 512 and not cfg.greedy
        score = 0.75 if experimental_gain and "pilot_reanalysis4x" in label else 0.5
        if "minutes_0330" in label:
            score = 1.0
        result = report(score, cfg, opponent)
        write_report(campaign.root / "evaluations" / f"{label}.json", result)
        return result

    real_retire = EC.retire_replay

    def retire(config, milestone: dict) -> None:
        retired.append(config.run_id)
        real_retire(config, milestone)

    monkeypatch.setattr(EC, "train_segment", train)
    monkeypatch.setattr(EC, "retire_replay", retire)
    monkeypatch.setattr(campaign, "_match", match)
    result = EC.run_enhancement_campaign(campaign, str(source), reanalysis=True)
    assert [minutes for _, minutes, _ in stages] == [90, 90, 210, 330, 450, 495]
    assert (
        result["selected_arm"] == selected_arm
        and result["selected_milestone"] == "minutes_0330"
    )
    assert all(cfg.run_id.startswith(selected_arm) for cfg, _, _ in stages[2:])
    resume = Path(result["resume_checkpoint"])
    assert resume.read_text() == "full-state:minutes_0495"
    assert not any(name.startswith(selected_arm) for name in retired)
    assert len(matches) == 18 and not result["automatic_promotion"]
    budget = json.loads((campaign.root / "reanalysis_campaign_budget.json").read_text())
    assert budget == {"started_at": started, "deadline": started + 12 * 3600}
    verification = json.loads((campaign.root / "verification.json").read_text())
    assert verification["within_budget"] and verification["evaluation_games"] == 10240
    before = len(stages), len(matches)
    EC.run_enhancement_campaign(campaign, str(source), reanalysis=True)
    assert before == (len(stages), len(matches)) and resume.exists()
    assert (
        json.loads(campaign.status_path.read_text())["stage"] == "reanalysis_complete"
    )
