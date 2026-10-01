from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import pytest
import torch

from agent.eval.arena import checkpoint_hash, write_report
from agent.scripts import distillation_campaign as D
from agent.scripts.competitive import Campaign
from agent.tests.test_enhancement_campaign import report
from agent.tests.test_finetune_campaign import assert_same
from agent.tests.test_weight_refine_campaign import inputs
from agent.train.checkpointing import load_checkpoint_payload
from agent.train.presets import enhanced_2p_config


def test_new_seed_fork_preserves_optimizer_and_replay_and_is_idempotent(
    tmp_path: Path,
) -> None:
    _, old = inputs(tmp_path)
    payload = load_checkpoint_payload(old["replay"])
    source = {
        "resume": old["replay"],
        "resume_sha256": checkpoint_hash(old["replay"]),
        "league": old["league"],
    }
    cfg = replace(
        D.LoopConfig(**payload["config"]),
        runs_root=str(tmp_path / "runs"),
        run_id="fork",
        league_root=str(tmp_path / "runs/fork/league"),
        seed=12345,
        distillation_teacher=old["start"],
        distillation_teacher_sha256=checkpoint_hash(old["start"]),
        distillation_capacity=8,
    )
    record = D.fork_arm(cfg, source)
    fork = load_checkpoint_payload(tmp_path / "runs/fork/checkpoints/latest_resume.pt")
    for key in ("model_state_dict", "optimizer_state_dict", "buffer", "iteration"):
        assert_same(payload[key], fork[key])
    assert not torch.equal(payload["rng_state"]["torch"], fork["rng_state"]["torch"])
    assert fork["distillation"]["bank"]["size"] == 0
    assert fork["progress"] == {"training_wall_s": 0.0}
    assert D.fork_arm(cfg, source) == record
    assert_same(payload, load_checkpoint_payload(old["replay"]))
    with pytest.raises(ValueError, match="changed"):
        D.fork_arm(replace(cfg, seed=33), source)


@pytest.mark.parametrize(
    "seconds,rate,expected", [(18000, 5, 2048), (10000, 5, 1024), (2000, 5, 0)]
)
def test_confirmation_count_fits_measured_budget(
    seconds: float, rate: float, expected: int
) -> None:
    assert D.confirmation_games(seconds, rate) == expected


@pytest.mark.parametrize("student_wins", [True, False])
@pytest.mark.parametrize("timing_passed", [True, False])
def test_campaign_freezes_serving_comparisons_and_retains_fallback(
    tmp_path: Path, monkeypatch, student_wins, timing_passed
) -> None:
    campaign = Campaign(tmp_path, "cpu", 123, campaign_hours=24)
    for k, v in D.DIAGNOSTIC_ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    write_report(
        tmp_path / "preflight_budget.json",
        {"started_at": time.time(), "deadline": time.time() + 86400},
    )
    write_report(tmp_path / "ready.json", {"passed": True, "code": campaign.code})
    states = {}
    for name, hidden in [("small", 256), ("wide", 512)]:
        path = tmp_path / (name + ".pt")
        path.write_text(name)
        states[name] = {
            "checkpoint": str(path),
            "hidden": hidden,
            "config": asdict(replace(enhanced_2p_config(), hidden=hidden)),
        }
    prepared = {"states": states, "hashes": {}}
    monkeypatch.setattr(D, "prepare", lambda *args: prepared)
    monkeypatch.setattr(D, "fork_arm", lambda *args: {})
    monkeypatch.setattr(D, "retire_arm", lambda *args: None)
    monkeypatch.setattr(
        D, "retain_best", lambda c, a, m, s, b, cfg: {"milestone": m, "scores": s}
    )
    monkeypatch.setattr(
        D,
        "serving_profile",
        lambda c, l, p, h, **kwargs: {
            "search": asdict(
                replace(D.SEARCH, num_simulations=512 if h == 512 else 1024)
            ),
            "serving_search": asdict(
                replace(
                    D.SEARCH,
                    num_simulations=512 if h == 512 else 1024,
                    move_deadline_s=1.8,
                )
            ),
        },
    )
    trains = []

    def train(c, cfg, minutes, label, **kwargs):
        assert kwargs["evaluation_reserve_minutes"] >= 240
        trains.append((cfg, minutes))
        path = tmp_path / (label + ".pt")
        path.write_text(label)
        return {"checkpoint": str(path), "iteration": int(minutes)}

    monkeypatch.setattr(D, "train_segment", train)

    def match(label, candidate, opponent, cfg, opponent_search):
        if label.startswith("confirmation_"):
            assert (tmp_path / "final_selection.json").exists()
            assert cfg.split == "confirmation"
        score = 0.75 if opponent == "astra" else 0.5
        if label.startswith("development_student"):
            score = 0.75 if opponent == "astra" else (0.75 if student_wins else 0.25)
        if label == "confirmation_head_to_head":
            score = 0.75
        result = report(score, cfg, opponent)
        result["opponent_search"] = asdict(opponent_search)
        result["wall_s"] = cfg.num_games * 2
        result["timed_move_latency"] = {
            "candidate": {"count": 100, "over_2s": 0 if timing_passed else 1}
        }
        return result

    monkeypatch.setattr(campaign, "_match", match)
    result = D.run_distillation_campaign(campaign, "inputs.json")
    choice = json.loads((tmp_path / "approach_selection.json").read_text())
    assert choice["arm"] == ("student" if student_wins else "wide")
    assert result["provisional_improvement"] == timing_passed
    assert result["cpu_timing_passed"] == timing_passed
    assert any(cfg.seed == 123 + 1009 for cfg, _ in trains)
    assert any(cfg.seed == 123 + 2017 for cfg, _ in trains)
    assert not result["automatic_deployment"]
    count = len(trains)
    assert D.run_distillation_campaign(campaign, "inputs.json") == json.loads(
        json.dumps(result)
    )
    assert len(trains) == count
