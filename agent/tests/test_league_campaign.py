from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time
import zipfile

import pytest
import torch

from agent.env.engine import GameEngine
from agent.net import encoder as ENC
from agent.net.model import AzulNet
from agent.eval.arena import ArenaConfig, write_report
from agent.scripts import league_campaign as LC
from agent.scripts.competitive import Campaign
from agent.tests.test_enhancement_campaign import report
from agent.train.checkpointing import (
    checkpoint_launch_headroom,
    load_checkpoint_payload,
    save_checkpoint,
)
from agent.train.league import League
from agent.train.presets import enhanced_2p_config
from agent.train.replay_buffer import ReplayBuffer


def test_fork_keeps_complete_state_and_isolates_atomic_updates(tmp_path: Path) -> None:
    net = AzulNet(hidden=32, arch="flat")
    optimizer = torch.optim.AdamW(net.parameters())
    sum(p.sum() for p in net.parameters()).backward()
    optimizer.step()
    buffer = ReplayBuffer(
        4, ENC.D_GLOBAL, ENC.NUM_SOURCES, ENC.D_SOURCE, 300, 4, snapshot_capacity=4
    )
    engine = GameEngine(4, 2, seed=1)
    g, s = ENC.encode_state(engine)
    legal = engine.legal_action_mask()
    buffer.add(
        g,
        s,
        legal,
        legal.float() / legal.sum(-1, keepdim=True),
        torch.ones(4, 4),
        dict(enumerate(engine.state_dict()["snapshots"])),
    )
    source = tmp_path / "source.pt"
    save_checkpoint(
        source,
        net,
        optimizer,
        420,
        {"num_players": 2},
        buffer,
        {"training_wall_s": 1000.0},
    )
    league = League(tmp_path / "original_league")
    league.add_checkpoint(net, "frozen_baseline", metadata={"pinned": True})
    source_hash = LC.checkpoint_hash(source)
    league_bytes = league.manifest_path.read_bytes()
    for name in ("a", "b"):
        LC.fork_training_state(source, league.root, tmp_path / name)
        LC.fork_training_state(source, league.root, tmp_path / name)
        copied = load_checkpoint_payload(
            tmp_path / name / "checkpoints/latest_resume.pt"
        )
        assert (
            copied["iteration"] == 420 and copied["progress"]["training_wall_s"] == 1000
        )
        assert copied["optimizer_state_dict"]["state"] and copied["rng_state"]
        assert copied["buffer"]["snapshots"] == buffer.snapshots
        assert torch.equal(copied["buffer"]["global_feat"], buffer.global_feat)
        assert (
            LC.checkpoint_hash(tmp_path / name / "checkpoints/latest_resume.pt")
            == source_hash
        )
    changed = League(tmp_path / "a/league")
    changed.manifest["measured_strong_min_games"] = 128
    changed._save_manifest()
    assert league.manifest_path.read_bytes() == league_bytes
    assert json.loads((tmp_path / "b/league/league.json").read_text()) == json.loads(
        league_bytes
    )
    save_checkpoint(
        tmp_path / "a/checkpoints/latest_resume.pt",
        net,
        optimizer,
        421,
        {"num_players": 2},
        buffer,
        {"training_wall_s": 1100.0},
    )
    assert LC.checkpoint_hash(source) == source_hash
    assert (
        LC.checkpoint_hash(tmp_path / "b/checkpoints/latest_resume.pt") == source_hash
    )
    assert (
        LC.checkpoint_hash(tmp_path / "a/checkpoints/latest_resume.pt") != source_hash
    )


def test_compressed_resume_headroom_preserves_raw_checks_for_new_runs(
    tmp_path: Path,
) -> None:
    raw = 3 * 1024**3
    for compression in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
        path = tmp_path / f"{compression}.pt"
        with zipfile.ZipFile(path, "w", compression=compression) as archive:
            archive.writestr("data", bytes(1000))
        assert checkpoint_launch_headroom(path, raw) == (
            1024**3 if compression == zipfile.ZIP_DEFLATED else raw
        )
    assert checkpoint_launch_headroom(None, raw) == raw
    assert checkpoint_launch_headroom(path, 0) == 0


def test_measured_strong_pool_excludes_unrated_and_breaks_rating_ties(
    tmp_path: Path,
) -> None:
    league = League(tmp_path)
    for i in range(10):
        (tmp_path / f"{i}.pt").touch()
        league.manifest["entries"].append(
            {
                "idx": i,
                "path": f"{i}.pt",
                "rating_2p": 9999 if i == 0 else 2000,
                "games": 0 if i == 0 else 128,
            }
        )
    league.manifest["measured_strong_min_games"] = 128

    class StrongRandom(random.Random):
        def randrange(self, *args, **kwargs):
            return 2

    rng = StrongRandom(5)
    chosen = {league.sample_opponent(rng, "mixed")["idx"] for _ in range(200)}
    assert chosen == set(range(1, 10))


def test_panel_gate_uses_all_opponents_and_guards_astra() -> None:
    cfg = ArenaConfig(num_games=16)
    baseline = {name: report(0.5, cfg, name) for name in LC.PANEL}
    improved = {name: report(0.75, cfg, name) for name in LC.PANEL}
    assert (
        LC.choose_arm({"current": baseline, "rated": improved})["selected_arm"]
        == "rated"
    )
    improved["astra"] = report(0.25, cfg, "astra")
    assert (
        LC.choose_arm({"current": baseline, "rated": improved})["selected_arm"]
        == "current"
    )
    assert (
        LC.choose_arm({"current": baseline, "rated": baseline})["selected_arm"]
        == "current"
    )
    assert (
        LC.choose_checkpoint({"start": baseline, "regression": improved}, 0.5)
        == "start"
    )
    improved["astra"] = report(0.75, replace(cfg, seed=cfg.seed + 1), "astra")
    with pytest.raises(ValueError, match="Unmatched"):
        LC.panel_difference(improved, baseline)


def test_rating_records_are_applied_once_and_preserve_manifest_options(
    tmp_path: Path, monkeypatch
) -> None:
    campaign = Campaign(tmp_path / "campaign", "cpu", 12, campaign_hours=10)
    config = replace(enhanced_2p_config(), league_root=str(tmp_path / "league"))
    league = League(config.league_root)
    for i in range(5):
        p = league.root / f"{i}.pt"
        p.write_text(str(i))
        league.manifest["entries"].append(
            {
                "idx": i,
                "path": p.name,
                "tag": "frozen_baseline" if i == 0 else f"i{i}",
                "iteration": i,
                "rating": 1500.0,
                "games": 0,
                "active": True,
            }
        )
    league._save_manifest()
    frozen = {}
    for name in ("start", "historical", "alternative"):
        p = tmp_path / f"{name}.pt"
        p.write_text(name)
        frozen[name] = str(p)
    prepared = {"frozen": frozen, "initial_iteration": 4}
    matches = []

    def match(label, candidate, opponent, cfg, search):
        matches.append(label)
        return report(0.75, cfg, opponent)

    monkeypatch.setattr(campaign, "_match", match)
    result = LC.rate_league(campaign, config, prepared, "initial")
    assert result["games"] == 16 * 128
    fresh = League(config.league_root)
    assert fresh.manifest["measured_strong_min_games"] == 128
    assert len(fresh.manifest["campaign_rating_reports"]) == 16
    assert (
        len({e.get("rating_2p") for e in fresh.list_entries() if e.get("games", 0)}) > 1
    )
    before = fresh.manifest_path.read_bytes()
    assert LC.rate_league(campaign, config, prepared, "initial") == result
    assert len(matches) == 16 and fresh.manifest_path.read_bytes() == before
    # Interrupted after durable match application but before writing the stage marker.
    (campaign.root / "league_ratings/initial.json").unlink()
    LC.rate_league(campaign, config, prepared, "initial")
    assert len(matches) == 16
    assert League(config.league_root).manifest["results"] == fresh.manifest["results"]


@pytest.mark.parametrize("rated_wins", [True, False])
def test_full_campaign_resume_panel_and_equal_wall_accounting(
    tmp_path: Path, monkeypatch, rated_wins: bool
) -> None:
    campaign = Campaign(tmp_path / "campaign", "cpu", 7, campaign_hours=10)
    source = tmp_path / "source.pt"
    source.write_text("source")
    frozen = {}
    for name in ("start", "previous", "historical", "alternative"):
        p = tmp_path / f"{name}.pt"
        p.write_text(name)
        frozen[name] = str(p)
    configs = {
        arm: asdict(
            replace(
                enhanced_2p_config(),
                run_id=arm,
                runs_root=str(campaign.root / "experiments"),
                league_root=str(campaign.root / "experiments" / arm / "league"),
            )
        )
        for arm in ("current", "rated")
    }
    prepared = {
        "arms": configs,
        "frozen": frozen,
        "initial_iteration": 420,
        "base_training_wall_s": 600.0,
        "source_resume_sha256": LC.checkpoint_hash(source),
    }
    monkeypatch.setattr(LC, "prepare", lambda *args: prepared)
    start = time.time() - 120
    write_report(
        campaign.root / "preflight_budget.json",
        {"started_at": start, "deadline": start + 36000},
    )

    def rate(campaign, config, prepared, stage):
        path = campaign.root / "league_ratings" / f"{stage}.json"
        if not path.exists():
            write_report(path, {"elapsed_s": 60.0})
        return json.loads(path.read_text())

    monkeypatch.setattr(LC, "rate_league", rate)
    stages, matches = [], []

    def train(campaign, config, minutes, label, *, evaluation_reserve_minutes):
        stages.append((config.run_id, minutes, label))
        assert evaluation_reserve_minutes == 105
        directory = Path(config.runs_root) / config.run_id
        path = directory / "milestones" / f"{label}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(label)
        resume = directory / "checkpoints/latest_resume.pt"
        resume.parent.mkdir(exist_ok=True)
        resume.write_text(label)
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": LC.checkpoint_hash(path),
            "iteration": 420 + int(minutes),
            "target_minutes": minutes,
            "executed_target_minutes": minutes,
            "training_wall_s": minutes * 60,
            "replay_retained": True,
        }

    monkeypatch.setattr(LC, "train_segment", train)

    def match(label, candidate, opponent, cfg, search):
        matches.append(label)
        if label.startswith("confirmation_"):
            assert cfg.seed == 110_000_007 and cfg.num_games == 1024
            assert (campaign.root / "final_selection.json").exists()
        else:
            assert cfg.seed == 100_000_007 and cfg.num_games == 512
        assert cfg.search == search == LC.SEARCH
        score = 0.75 if rated_wins and label.startswith("pilot_rated") else 0.5
        if label.startswith("minutes_0330"):
            score = 1.0
        r = report(score, cfg, opponent)
        write_report(campaign.root / "evaluations" / f"{label}.json", r)
        return r

    monkeypatch.setattr(campaign, "_match", match)
    result = LC.run_league_campaign(campaign, str(source))
    selected = "rated" if rated_wins else "current"
    assert result["selected_arm"] == selected
    assert stages[:4] == [
        ("current", 70.0, "pilot_half"),
        ("current", 130.0, "pilot"),
        ("rated", 69.0, "pilot_half"),
        ("rated", 128.0, "pilot"),
    ]
    assert stages[-1][1] == (336.0 if rated_wins else 340.0)
    assert result["selected_milestone"] == "minutes_0330"
    assert len(matches) == 21
    assert Path(result["resume_checkpoint"]).exists()
    assert LC.checkpoint_hash(source) == prepared["source_resume_sha256"]
    budget = json.loads((campaign.root / "league_campaign_budget.json").read_text())
    assert budget == {"started_at": start, "deadline": start + 36000}
    counts = len(stages), len(matches)
    LC.run_league_campaign(campaign, str(source))
    assert counts == (len(stages), len(matches))
    assert (
        json.loads(campaign.status_path.read_text())["stage"]
        == "league_campaign_complete"
    )
