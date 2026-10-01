from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import pytest

from agent.eval.arena import checkpoint_hash, write_report
from agent.scripts import resume_distillation_campaign as R
from agent.scripts.competitive import Campaign
from agent.tests.test_enhancement_campaign import report
from agent.tests.test_finetune_campaign import assert_same
from agent.tests.test_weight_refine_campaign import inputs
from agent.train.checkpointing import load_checkpoint_payload, save_checkpoint_payload
from agent.train.distillation import make_bank
from agent.train.presets import enhanced_2p_config


def test_relocation_preserves_every_training_state_and_rejects_learning_changes(
    tmp_path: Path,
) -> None:
    _, old = inputs(tmp_path)
    payload = load_checkpoint_payload(old["replay"])
    cfg = replace(
        R.LoopConfig(**payload["config"]),
        distillation_teacher=old["start"],
        distillation_teacher_sha256=checkpoint_hash(old["start"]),
        distillation_capacity=8,
    )
    payload["config"] = asdict(cfg)
    payload["distillation"] = {
        "teacher_sha256": cfg.distillation_teacher_sha256,
        "bank": make_bank(8, "cpu").state_dict(),
    }
    save_checkpoint_payload(old["replay"], payload)
    source = {
        "resume": old["replay"],
        "resume_sha256": checkpoint_hash(old["replay"]),
        "league": old["league"],
    }
    cfg = replace(
        cfg,
        runs_root=str(tmp_path / "runs"),
        run_id="resumed",
        league_root=str(tmp_path / "runs/resumed/league"),
    )
    R.relocate_resume(source, cfg)
    new = load_checkpoint_payload(
        tmp_path / "runs/resumed/checkpoints/latest_resume.pt"
    )
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "buffer",
        "rng_state",
        "distillation",
        "progress",
        "iteration",
    ):
        assert_same(new[key], payload[key])
    assert_same(payload, load_checkpoint_payload(old["replay"]))
    R.relocate_resume(source, cfg)
    with pytest.raises(ValueError, match="configuration"):
        R.relocate_resume(source, replace(cfg, run_id="different", lr=0.0001))


def test_continuation_allocation_accounts_for_screens_and_final_reserve() -> None:
    assert R.continuation_targets(180, 180, 20) == []
    assert R.continuation_targets(340, 180, 20) == [60.0, 120.0]
    assert R.continuation_targets(310, 180, 20) == [60.0, 90.0]


@pytest.mark.parametrize("student_regresses", [True, False])
@pytest.mark.parametrize("timing_passed", [True, False])
def test_resume_protocol_stops_bad_student_and_freezes_fresh_finalists(
    tmp_path: Path, monkeypatch, student_regresses, timing_passed
) -> None:
    campaign = Campaign(tmp_path, "cpu", 123, campaign_hours=12)
    for k, v in R.DIAGNOSTIC_ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    write_report(
        tmp_path / "preflight_budget.json",
        {"started_at": time.time(), "deadline": time.time() + 43200},
    )
    write_report(tmp_path / "ready.json", {"passed": True, "code": campaign.code})
    states, profiles, historical = {}, {}, {}
    for name, hidden, sims in [("small", 256, 1024), ("wide", 512, 384)]:
        path = tmp_path / (name + ".pt")
        path.write_text(name)
        states[name] = {
            "checkpoint": str(path),
            "hidden": hidden,
            "config": asdict(replace(enhanced_2p_config(), hidden=hidden)),
        }
        profiles[name] = {
            "search": asdict(replace(R.SEARCH, num_simulations=sims)),
            "serving_search": asdict(
                replace(R.SEARCH, num_simulations=sims, move_deadline_s=1.8)
            ),
        }
        historical["initial_" + name + "_astra"] = {
            "config": {"seed": 111},
            "summary": {"match_score": 0.75},
            "wall_s": 256.0,
        }
    historical["initial_serving_head_to_head"] = {"wall_s": 512.0}
    prior = {
        "milestone": {"checkpoint": states["small"]["checkpoint"]},
        "scores": {"start": 0.25, "astra": 0.5},
    }
    prepared = {
        "states": states,
        "baseline": {"name": "small", "profiles": profiles},
        "historical": historical,
        "student_source": {},
        "student_config": asdict(
            replace(
                enhanced_2p_config(),
                runs_root=str(tmp_path / "experiments"),
                run_id="student_resumed",
            )
        ),
        "previous_best_student": prior,
        "hashes": {},
    }
    monkeypatch.setattr(R, "prepare", lambda *args: prepared)
    monkeypatch.setattr(R, "relocate_resume", lambda *args: {})
    monkeypatch.setattr(R, "fork_arm", lambda *args: {})
    monkeypatch.setattr(R, "retire_arm", lambda *args: None)
    monkeypatch.setattr(
        R, "retain_best", lambda c, a, m, s, b, cfg: {"milestone": m, "scores": s}
    )
    monkeypatch.setattr(
        R,
        "serving_profile",
        lambda c, label, path, hidden, **kw: profiles[
            "small" if hidden == 256 else "wide"
        ],
    )
    calls = []

    def train(c, cfg, minutes, label, **kwargs):
        assert kwargs["evaluation_reserve_minutes"] >= 180
        calls.append(label)
        path = tmp_path / (label + ".pt")
        path.write_text(label)
        return {"checkpoint": str(path), "iteration": int(minutes)}

    monkeypatch.setattr(R, "train_segment", train)

    def match(label, candidate, opponent, cfg, search):
        is_final = label.startswith("confirmation_")
        assert cfg.seed == (
            300000123 + (1000000 if "cpu_audit" in label else 0) if is_final else 111
        )
        if is_final:
            assert (tmp_path / "final_selection.json").exists()
        score = 0.75 if opponent == "astra" else 0.5
        if label.startswith("development_student") and opponent != "astra":
            score = 0.25 if student_regresses else 0.75
        if label == "confirmation_head_to_head":
            score = 0.75
        result = report(score, cfg, opponent)
        result.update(
            opponent_search=asdict(search),
            wall_s=cfg.num_games * 2,
            timed_move_latency={
                "candidate": {"count": 100, "over_2s": 0 if timing_passed else 1}
            },
        )
        return result

    monkeypatch.setattr(campaign, "_match", match)
    result = R.run_resume_distillation(campaign, "inputs.json")
    assert ("student_150m" in calls) == (not student_regresses)
    assert ("replicate_60m" in calls) == (not student_regresses)
    assert "wide_150m" in calls
    assert result["provisional_improvement"] == timing_passed
    assert not result["automatic_deployment"]
    count = len(calls)
    assert R.run_resume_distillation(campaign, "inputs.json") == json.loads(
        json.dumps(result)
    )
    assert len(calls) == count
