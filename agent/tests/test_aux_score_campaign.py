from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import pytest

from agent.eval.arena import write_report
from agent.scripts import aux_score_campaign as C
from agent.scripts.competitive import Campaign
from agent.tests.test_enhancement_campaign import report
from agent.train.presets import enhanced_2p_config


def test_retirement_keeps_archived_resume_league_and_milestones(tmp_path: Path) -> None:
    campaign = Campaign(tmp_path, "cpu", 123, campaign_hours=16)
    directory = tmp_path / "experiments/control"
    config = replace(
        enhanced_2p_config(),
        run_id="control",
        runs_root=str(directory.parent),
        league_root=str(directory / "league"),
    )
    for path in (
        directory / "checkpoints/latest_resume.pt",
        directory / "milestones/control_30m.pt",
        directory / "league/scratch.pt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"scratch")
    archive = tmp_path / "retained/control/control_30m"
    (archive / "league").mkdir(parents=True)
    (archive / "resume.pt").write_bytes(b"preserved resume")
    (archive / "league/model.pt").write_bytes(b"preserved model")
    write_report(
        archive / "league/league.json",
        {"entries": [{"path": "model.pt", "active": True}]},
    )
    best = {
        "resume": str(archive / "resume.pt"),
        "resume_sha256": C.checkpoint_hash(archive / "resume.pt"),
        "league": str(archive / "league"),
    }
    C.retire_scratch(campaign, config, best)
    C.retire_scratch(campaign, config, best)
    assert (
        not list((directory / "checkpoints").glob("*.pt"))
        and not (directory / "league").exists()
    )
    assert (archive / "resume.pt").read_bytes() == b"preserved resume"
    assert (archive / "league/model.pt").read_bytes() == b"preserved model"
    assert (directory / "milestones/control_30m.pt").exists()


@pytest.mark.parametrize("score_regresses", [True, False])
@pytest.mark.parametrize("timing_passed", [True, False])
def test_protocol_separates_seeds_preserves_reserve_and_resumes_after_pilots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    score_regresses: bool,
    timing_passed: bool,
) -> None:
    campaign = Campaign(tmp_path, "cpu", 123, campaign_hours=16)
    for k, v in C.DIAGNOSTIC_ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    write_report(
        tmp_path / "preflight_budget.json",
        {"started_at": time.time(), "deadline": time.time() + 57600},
    )
    write_report(tmp_path / "ready.json", {"passed": True, "code": campaign.code})
    paths = {}
    for name in ("baseline", "resume", "wide", "historical"):
        path = tmp_path / f"{name}.pt"
        path.write_text(name)
        paths[name] = str(path)
    arms = {
        name: asdict(
            replace(
                enhanced_2p_config(),
                run_id=name,
                runs_root=str(tmp_path / "experiments"),
                league_root=str(tmp_path / "experiments" / name / "league"),
                aux_score_head=True,
                aux_score_weight=weight,
            )
        )
        for name, weight in [("control", 0.0), ("score", 0.1)]
    }
    prepared = {"paths": paths, "arms": arms, "hashes": {}}
    monkeypatch.setattr(C, "prepare", lambda *args: prepared)
    monkeypatch.setattr(C, "fork_pilot", lambda *args: {})
    monkeypatch.setattr(C, "retire_completed_arm", lambda *args: None)
    monkeypatch.setattr(
        C,
        "serving_profile",
        lambda c, label, path, hidden, **kw: {
            "search": asdict(
                replace(C.SEARCH, num_simulations=384 if hidden == 512 else 1024)
            )
        },
    )

    def retain(c: Campaign, arm: str, config: C.LoopConfig, candidate: dict) -> dict:
        marker = tmp_path / f"best_{arm}.json"
        old = json.loads(marker.read_text()) if marker.exists() else None
        if old and C.candidate_rank(old["candidate"]) >= C.candidate_rank(candidate):
            return old
        result = {"candidate": candidate, "milestone": candidate["milestone"]}
        write_report(marker, result)
        return result

    monkeypatch.setattr(C, "retain", retain)
    calls = []

    def train(
        c: Campaign, cfg: C.LoopConfig, minutes: float, label: str, **kwargs: object
    ) -> dict:
        assert kwargs["evaluation_reserve_minutes"] >= 210
        calls.append(label)
        path = tmp_path / f"{label}.pt"
        path.write_text(label)
        return {"checkpoint": str(path), "iteration": int(minutes)}

    monkeypatch.setattr(C, "train_segment", train)
    interrupted = False

    def fork(*args: object) -> dict:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return {}

    monkeypatch.setattr(C, "fork_continuation", fork)

    def match(
        label: str, candidate: str, opponent: str, cfg: object, search: object
    ) -> dict:
        final = label.startswith("confirmation_")
        assert cfg.seed == (
            420000123 + (1000000 if label == "confirmation_cpu" else 0)
            if final
            else 410000123
        )
        if final:
            assert (tmp_path / "final_selection.json").exists()
        if label.startswith("saved_wide_180m") and opponent != paths["historical"]:
            assert cfg.num_games == 1024
        score = 0.75 if opponent == "astra" else 0.5
        if opponent == paths["baseline"] and label.startswith("score_"):
            score = 0.25 if score_regresses else 0.75
        if label == "confirmation_head_to_head":
            score = 0.75
        result = report(score, cfg, opponent)
        result.update(
            opponent_search=asdict(search),
            wall_s=cfg.num_games * 2,
            timed_move_latency={
                side: {"count": 100, "over_2s": 0 if timing_passed else 1}
                for side in ("candidate", "opponent")
            },
        )
        return result

    monkeypatch.setattr(campaign, "_match", match)
    with pytest.raises(KeyboardInterrupt):
        C.run_aux_score_campaign(campaign, "inputs.json")
    calls_at_interrupt = list(calls)
    result = C.run_aux_score_campaign(campaign, "inputs.json")
    assert calls[: len(calls_at_interrupt)] == calls_at_interrupt
    assert len(calls) == len(set(calls))  # completed pilots were not retrained
    assert ("score_120m" in calls) == (not score_regresses)
    assert (
        "control_60m" in calls
        and "control_120m" in calls
        and "continuation_240m" in calls
    )
    selection = json.loads((tmp_path / "continuation_selection.json").read_text())
    assert selection["arm"] == ("control" if score_regresses else "score")
    assert result["provisional_improvement"] == timing_passed
    assert not result["automatic_deployment"]
    count = len(calls)
    assert C.run_aux_score_campaign(campaign, "inputs.json") == json.loads(
        json.dumps(result)
    )
    assert len(calls) == count
