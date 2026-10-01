from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import pytest
import torch

from agent.eval.arena import write_report
from agent.env.engine import GameEngine
from agent.net.model import AzulNet
from agent.scripts import weight_refine_campaign as C
from agent.scripts.competitive import Campaign
from agent.scripts.supervise_training import recovery_reason
from agent.tests.test_enhancement_campaign import report
from agent.tests.test_finetune_campaign import assert_same
from agent.tests.test_training_enhancements import make_buffer, add_positions
from agent.train.checkpointing import (
    checkpoint_launch_headroom,
    load_checkpoint_payload,
    save_checkpoint,
)
from agent.train.league import League
from agent.train.presets import enhanced_2p_config


def inputs(tmp_path: Path) -> tuple[Path, dict]:
    net = AzulNet(hidden=256, arch="source_attn")
    cfg = replace(enhanced_2p_config(), replay_capacity=8)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    sum(p.sum() for p in net.parameters()).backward()
    optimizer.step()
    replay = make_buffer(8)
    replay.iteration = 10
    add_positions(replay, GameEngine(8, 2, seed=7))
    replay.policy_sims[:4] = 64
    replay.policy_sims[4:] = 256
    replay.iteration = 12
    source = tmp_path / "source"
    source.mkdir()
    save_checkpoint(
        source / "replay.pt",
        net,
        optimizer,
        12,
        asdict(cfg),
        replay,
        {"training_wall_s": 7200.0},
    )
    with torch.no_grad():
        next(net.parameters()).add_(0.01)
    save_checkpoint(source / "start.pt", net, iteration=6, config=asdict(cfg))
    league = League(source / "league")
    league.add_checkpoint(net, "frozen")
    manifest = {
        "start": str(source / "start.pt"),
        "previous": str(source / "start.pt"),
        "replay": str(source / "replay.pt"),
        "league": str(league.root),
    }
    path = tmp_path / "inputs.json"
    write_report(path, manifest)
    return path, manifest


def test_forks_share_known_replay_and_fresh_state_without_borrowing_optimizer(
    tmp_path,
) -> None:
    path, source = inputs(tmp_path)
    campaign = Campaign(tmp_path / "campaign", "cpu", 41)
    prepared = C.prepare(campaign, path)
    old = load_checkpoint_payload(source["replay"])
    states = []
    for arm in C.ARMS:
        record = C.fork_arm(prepared, arm)
        cfg = prepared["arms"][arm]
        payload = load_checkpoint_payload(
            Path(cfg["runs_root"]) / cfg["run_id"] / "checkpoints/latest_resume.pt"
        )
        states.append(payload)
        assert payload["optimizer_state_dict"]["state"] == {}
        assert payload["iteration"] == 0
        assert payload["progress"]["training_wall_s"] == 0
        assert payload["config"]["policy_fast_weight"] == C.ARMS[arm]
        assert torch.equal(
            payload["buffer"]["inserted_at"], old["buffer"]["inserted_at"] - 12
        )
        assert C.fork_arm(prepared, arm) == record
    for field in (
        "model_state_dict",
        "optimizer_state_dict",
        "buffer",
        "rng_state",
        "progress",
    ):
        assert_same(states[0][field], states[1][field])
    assert_same(
        states[0]["model_state_dict"],
        load_checkpoint_payload(source["start"])["model_state_dict"],
    )
    assert old["optimizer_state_dict"]["state"]
    assert_same(old, load_checkpoint_payload(source["replay"]))
    assert C.prepare(campaign, path) == prepared


def test_unknown_replay_budgets_rejected(tmp_path) -> None:
    path, source = inputs(tmp_path)
    payload = load_checkpoint_payload(source["replay"])
    payload["buffer"]["policy_sims"][0] = 0
    C.save_checkpoint_payload(source["replay"], payload)
    with pytest.raises(ValueError, match="known search budget"):
        C.prepare(Campaign(tmp_path / "campaign", "cpu", 42), path)


def test_compacted_full_state_is_lossless_and_retains_matching_league(tmp_path) -> None:
    path, _ = inputs(tmp_path)
    campaign = Campaign(tmp_path / "campaign", "cpu", 43)
    prepared = C.prepare(campaign, path)
    C.fork_arm(prepared, "quarter")
    cfg = C.LoopConfig(**prepared["arms"]["quarter"])
    directory = Path(cfg.runs_root) / cfg.run_id
    source = directory / "checkpoints/latest_resume.pt"
    payload = load_checkpoint_payload(source)
    milestone = {"checkpoint": prepared["frozen"]["start"], "iteration": 0}
    best = C.retain_best(
        campaign, "quarter", milestone, {"start": 0.6, "astra": 0.7}, 0.7, cfg
    )
    assert_same(payload, load_checkpoint_payload(best["resume"]))
    assert source.stat().st_ino == Path(best["resume"]).stat().st_ino
    assert checkpoint_launch_headroom(Path(best["resume"]), 2 * 1024**3) == 1024**3
    assert (
        checkpoint_launch_headroom(
            Path(best["resume"]), 2 * 1024**3, bounded_storage=True
        )
        == 256 * 1024**2
    )
    assert json.loads((Path(best["league"]) / "league.json").read_text()) == json.loads(
        (directory / "league/league.json").read_text()
    )
    assert (
        C.retain_best(
            campaign, "quarter", milestone, {"start": 0.55, "astra": 0.7}, 0.7, cfg
        )
        == best
    )
    C.retire_completed_arm(campaign, cfg, best)
    assert not source.exists()
    assert Path(best["resume"]).exists()
    assert Path(prepared["frozen"]["replay"]).exists()


@pytest.mark.parametrize(
    "outcome", ["confirmed", "provisional", "regressed", "initializer"]
)
def test_campaign_freezes_choice_uses_fresh_confirmation_and_reuses_completed_work(
    tmp_path, monkeypatch, outcome
) -> None:
    campaign = Campaign(tmp_path, "cpu", 50)
    for k, v in C.DIAGNOSTIC_ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    now = time.time()
    write_report(
        tmp_path / "preflight_budget.json", {"started_at": now, "deadline": now + 28800}
    )
    write_report(tmp_path / "ready.json", {"passed": True, "code": campaign.code})
    frozen = {}
    for name in ("start", "previous"):
        p = tmp_path / f"{name}.pt"
        p.write_text(name)
        frozen[name] = str(p)
    prepared = {
        "initialization": "test",
        "frozen": frozen,
        "input_hashes": {},
        "arms": {
            arm: asdict(
                replace(enhanced_2p_config(), run_id=arm, policy_fast_weight=weight)
            )
            for arm, weight in C.ARMS.items()
        },
    }
    monkeypatch.setattr(C, "prepare", lambda *args: prepared)
    monkeypatch.setattr(C, "fork_arm", lambda *args: {})
    monkeypatch.setattr(C, "retire_completed_arm", lambda *args: None)
    monkeypatch.setattr(
        C,
        "retain_best",
        lambda c, a, m, s, b, cfg: {"resume": m["checkpoint"], "scores": s},
    )
    calls = []

    def train(c, cfg, minutes, label, *, evaluation_reserve_minutes):
        calls.append(label)
        p = tmp_path / f"{label}.pt"
        p.write_text(label)
        return {
            "checkpoint": str(p),
            "iteration": int(minutes),
            "executed_target_minutes": minutes,
        }

    monkeypatch.setattr(C, "train_segment", train)

    def match(label, candidate, opponent, cfg, search):
        confirmation = label.startswith("confirmation_")
        assert cfg.seed in ({220000050, 221000050} if confirmation else {210000050})
        if confirmation:
            assert (tmp_path / "final_selection.json").exists()
        score = 0.75 if opponent == "astra" else 0.5
        if (
            not confirmation
            and "half_150m" in label
            and opponent != "astra"
            and outcome != "initializer"
        ):
            score = 0.75
        if label == "confirmation_head_to_head":
            score = 0.75 if outcome != "initializer" else 0.5
        if label == "confirmation_candidate_astra" and outcome == "regressed":
            score = 0.5
        result = report(score, cfg, opponent)
        result["summary"]["match_score_ci95"] = [
            0.49 if outcome == "provisional" else score - 0.01,
            score + 0.01,
        ]
        return result

    monkeypatch.setattr(campaign, "_match", match)
    result = C.run_weight_refine_campaign(campaign, str(tmp_path / "inputs.json"))
    assert len(calls) == 6
    assert result["provisional_improvement"] == (
        outcome in {"confirmed", "provisional"}
    )
    assert result["improvement_resolved"] == (outcome == "confirmed")
    assert result["selected_checkpoint"] == (
        result["candidate_checkpoint"]
        if result["provisional_improvement"]
        else frozen["start"]
    )
    C.run_weight_refine_campaign(campaign, str(tmp_path / "inputs.json"))
    assert len(calls) == 6


def test_disk_threshold_override_is_scoped_to_explicit_campaign_health() -> None:
    health = {"disk_free_gib": 0.5, "activity_age_s": 0}
    assert recovery_reason(health) == "low_disk"
    assert recovery_reason({**health, "disk_floor_gib": 0.3}) is None
