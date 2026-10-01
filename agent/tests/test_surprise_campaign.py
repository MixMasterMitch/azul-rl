from dataclasses import asdict, replace
import json
import time

import pytest

from agent.eval.arena import write_report
from agent.scripts import surprise_campaign as C
from agent.scripts.competitive import Campaign
from agent.tests.test_enhancement_campaign import report
from agent.train.presets import enhanced_2p_config


@pytest.mark.parametrize("score_supported", [False, True])
@pytest.mark.parametrize("cpu_passed", [False, True])
def test_closeout_matched_forks_reserve_resume_and_confirmation(
    tmp_path, monkeypatch, score_supported, cpu_passed
):
    c = Campaign(tmp_path, "cpu", 123, campaign_hours=20)
    for k, v in C.DIAGNOSTIC_ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    write_report(
        tmp_path / "preflight_budget.json",
        {"started_at": time.time(), "deadline": time.time() + 72000},
    )
    write_report(
        tmp_path / "ready.json",
        {
            "passed": True,
            "code": c.code,
            "reliability_passed": True,
            "native_hashes": {},
        },
    )
    paths = {}
    for name in (
        "baseline",
        "resume",
        "wide",
        "historical",
        "score",
        "score_resume",
        "old_control",
    ):
        p = tmp_path / f"{name}.pt"
        p.write_text(name)
        paths[name] = str(p)
    states = {
        name: {
            "checkpoint": paths[name],
            "config": asdict(
                replace(
                    enhanced_2p_config(),
                    aux_score_head=name == "score",
                    aux_score_weight=0.1 if name == "score" else 0.0,
                )
            ),
        }
        for name in ("baseline", "score")
    }
    monkeypatch.setattr(
        C, "prepare", lambda *args: {"paths": paths, "states": states, "hashes": {}}
    )
    monkeypatch.setattr(C, "retire_scratch", lambda *args: None)
    monkeypatch.setattr(
        C,
        "serving_profile",
        lambda c, label, path, hidden, **kw: {
            "search": asdict(
                replace(C.SEARCH, num_simulations=384 if hidden == 512 else 1024)
            )
        },
    )

    def retain(c, arm, config, candidate):
        marker = tmp_path / f"best_{arm}.json"
        old = json.loads(marker.read_text()) if marker.exists() else None
        if old and C.candidate_rank(old["candidate"]) >= C.candidate_rank(candidate):
            return old
        result = {"candidate": candidate, "milestone": candidate["milestone"]}
        write_report(marker, result)
        return result

    monkeypatch.setattr(C, "retain", retain)
    calls, forks = [], []

    def train(c, cfg, minutes, label, **kw):
        assert kw["evaluation_reserve_minutes"] >= 210
        calls.append(label)
        p = tmp_path / f"{label}.pt"
        p.write_text(label)
        return {"checkpoint": str(p), "iteration": int(minutes)}

    monkeypatch.setattr(C, "train_segment", train)
    interrupted = False

    def fork(cfg, source):
        nonlocal interrupted
        forks.append(cfg)
        if cfg.run_id == "continuation" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return {}

    monkeypatch.setattr(C, "fork_arm", fork)

    def match(label, checkpoint, opponent, cfg, search):
        phase = (
            "confirmation"
            if label.startswith("confirmation_")
            else "closeout"
            if (label.startswith("saved_") or label.startswith("closeout_"))
            else "development"
        )
        assert cfg.seed == {
            "closeout": 510000123,
            "development": 520000123,
            "confirmation": 530000123,
        }[phase] + (1000000 if label == "confirmation_cpu" else 0)
        if phase == "confirmation":
            assert (tmp_path / "final_selection.json").exists()
        score = 0.75 if opponent == "astra" else 0.5
        if label == "saved_score_head":
            score = 0.75
        if label == "saved_score_vs_old_control":
            score = 0.75 if score_supported else 0.25
        if label.startswith("surprise_") and opponent == paths["baseline"]:
            score = 1.0
        if label == "confirmation_head_to_head":
            score = 0.75
        r = report(score, cfg, opponent)
        r.update(
            opponent_search=asdict(search),
            wall_s=cfg.num_games * 2,
            timed_move_latency={
                side: {"count": 100, "over_2s": 0 if cpu_passed else 1}
                for side in ("candidate", "opponent")
            },
        )
        return r

    monkeypatch.setattr(c, "_match", match)
    with pytest.raises(KeyboardInterrupt):
        C.run_surprise_campaign(c, "inputs.json")
    completed = list(calls)
    result = C.run_surprise_campaign(c, "inputs.json")
    assert calls[: len(completed)] == completed and len(calls) == len(set(calls))
    assert (
        "control_120m" in calls
        and "surprise_120m" in calls
        and "continuation_360m" in calls
    )
    assert result["training_selection"]["source_name"] == (
        "score" if score_supported else "baseline"
    )
    assert (
        result["provisional_improvement"] == cpu_passed
        and not result["automatic_deployment"]
    )
    a, b = (asdict(cfg) for cfg in forks[:2])
    different = {key for key in a if a[key] != b[key]}
    assert different == {"run_id", "league_root", "policy_surprise_fraction"}
    assert (
        a["policy_surprise_record"]
        and a["policy_surprise_fraction"] == 0
        and b["policy_surprise_fraction"] == 0.5
    )
    count = len(calls)
    C.run_surprise_campaign(c, "inputs.json")
    assert len(calls) == count


def test_reliability_gate_cannot_be_skipped(tmp_path):
    c = Campaign(tmp_path, "cpu", 123, campaign_hours=20)
    write_report(
        tmp_path / "preflight_budget.json",
        {"started_at": time.time(), "deadline": time.time() + 72000},
    )
    write_report(
        tmp_path / "ready.json",
        {"passed": True, "code": c.code, "reliability_passed": False},
    )
    with pytest.raises(ValueError, match="reliability"):
        C.run_surprise_campaign(c, "missing_inputs.json")
