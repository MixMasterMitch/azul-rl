from dataclasses import asdict, replace
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from agent.eval.arena import ArenaConfig, write_report, checkpoint_hash
from agent.net.widen import widen_source_attention
from agent.scripts import capacity_campaign as C
from agent.scripts.competitive import Campaign
from agent.tests.test_enhancement_campaign import report
from agent.tests.test_finetune_campaign import assert_same
from agent.tests.test_weight_refine_campaign import inputs
from agent.train.checkpointing import (
    load_checkpoint_payload,
    load_net_from_checkpoint,
    save_checkpoint,
)
from agent.train.presets import enhanced_2p_config


def capacity_inputs(tmp_path: Path) -> tuple[Campaign, Path]:
    _, old = inputs(tmp_path)
    root = tmp_path / "capacity"
    campaign = Campaign(root, "cpu", 123)
    teacher, payload = load_net_from_checkpoint(old["replay"])
    wide = widen_source_attention(teacher, 512)
    save_checkpoint(
        root / "initializers/wide512.pt",
        wide,
        config={**payload["config"], "hidden": 512},
    )
    path = root / "inputs.json"
    write_report(
        path,
        {
            "teacher": old["replay"],
            "source_resume": old["replay"],
            "previous": old["start"],
            "league": old["league"],
        },
    )
    write_report(
        root / "transfer_validation.json",
        {"cpu": {"argmax_agreement": 1.0, "policy_kl_mean": 0.0, "value_mse": 0.0}},
    )
    return campaign, path


def test_width_forks_keep_identical_replay_with_new_optimizers_and_immutable_inputs(
    tmp_path: Path,
) -> None:
    campaign, path = capacity_inputs(tmp_path)
    prepared = C.prepare(campaign, path)
    states = []
    for arm in ("control", "wide_current"):
        cfg = C.LoopConfig(**prepared["arms"][arm])
        record = C.fork_arm(prepared, cfg)
        assert record["fresh_optimizer"]
        assert C.fork_arm(prepared, cfg) == record
        states.append(
            load_checkpoint_payload(
                Path(cfg.runs_root) / cfg.run_id / "checkpoints/latest_resume.pt"
            )
        )
    for key in ("buffer", "rng_state", "progress"):
        assert_same(states[0][key], states[1][key])
    assert [s["hidden"] for s in states] == [256, 512]
    assert states[0]["buffer"]["inserted_at"].tolist() == [-2] * 8
    assert C.prepare(campaign, path) == prepared
    best = C.retain_best(
        campaign,
        "wide_current",
        {"checkpoint": cfg.init_from, "iteration": 0},
        {"start": 0.6, "astra": 0.75},
        0.75,
        cfg,
    )
    C.retire_arm(campaign, cfg, best)
    assert Path(best["resume"]).is_file()
    assert list(Path(best["league"]).glob("*.pt"))
    assert not list(Path(cfg.league_root).glob("*.pt"))
    assert Path(prepared["frozen"]["source_resume"]).is_file()


def test_continuation_keeps_optimizer_replay_iteration_and_rng(tmp_path: Path) -> None:
    campaign, path = capacity_inputs(tmp_path)
    prepared = C.prepare(campaign, path)
    cfg = replace(C.LoopConfig(**prepared["arms"]["control"]), run_id="continued")
    cfg = replace(cfg, league_root=str(Path(cfg.runs_root) / cfg.run_id / "league"))
    source = prepared["frozen"]["source_resume"]
    old = load_checkpoint_payload(source)
    state = {
        "resume": source,
        "resume_sha256": checkpoint_hash(source),
        "league": str(Path(source).parent / "league"),
    }
    record = C.fork_arm(prepared, cfg, continuation=state)
    new = load_checkpoint_payload(
        Path(cfg.runs_root) / cfg.run_id / "checkpoints/latest_resume.pt"
    )
    assert not record["fresh_optimizer"]
    assert new["progress"] == {"training_wall_s": 0.0}
    assert new["config"]["run_id"] == "continued"
    for key in (
        "model_state_dict",
        "buffer",
        "optimizer_state_dict",
        "iteration",
        "rng_state",
    ):
        assert_same(new[key], old[key])


def test_transfer_validation_rejects_policy_loss(tmp_path: Path) -> None:
    campaign, path = capacity_inputs(tmp_path)
    write_report(
        campaign.root / "transfer_validation.json",
        {"cpu": {"argmax_agreement": 0.8, "policy_kl_mean": 0.1, "value_mse": 0.0}},
    )
    with pytest.raises(ValueError, match="transfer failed"):
        C.prepare(campaign, path)


def test_cpu_profile_excludes_incomplete_budget_and_reuses_frozen_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = Campaign(tmp_path, "cpu", 123)
    path = tmp_path / "weights.pt"
    path.write_bytes(b"weights")
    calls = []

    def benchmark(checkpoint, search, **kwargs):
        assert kwargs["deadline_s"] == 1.8 and kwargs["games"] == 8
        calls.append(search.num_simulations)
        return {"qualified": search.num_simulations <= 256}

    monkeypatch.setattr(C, "benchmark_latency", benchmark)
    result = C.serving_profile(campaign, "test", str(path))
    assert result["search"]["num_simulations"] == 256
    assert result["serving_search"]["move_deadline_s"] == 1.8
    assert C.serving_profile(campaign, "test", str(path)) == result
    assert calls == [64, 128, 256, 512]


@pytest.mark.parametrize("wide_wins", [True, False])
@pytest.mark.parametrize("control_selected", [True, False])
@pytest.mark.parametrize("low_disk", [True, False])
def test_campaign_reserves_confirmation_freezes_choice_and_reports_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wide_wins: bool,
    control_selected: bool,
    low_disk: bool,
) -> None:
    campaign = Campaign(tmp_path, "cpu", 123, campaign_hours=20)
    for k, v in C.DIAGNOSTIC_ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    write_report(
        tmp_path / "preflight_budget.json",
        {"started_at": time.time(), "deadline": time.time() + 72000},
    )
    write_report(tmp_path / "ready.json", {"passed": True, "code": campaign.code})
    frozen = {}
    for name in ("teacher", "wide512", "source_resume", "previous"):
        p = tmp_path / f"{name}.pt"
        p.write_text(name)
        frozen[name] = str(p)
    prepared = {
        "frozen": frozen,
        "input_hashes": {},
        "arms": {
            a: asdict(
                replace(
                    enhanced_2p_config(),
                    run_id=a,
                    hidden=256 if a == "control" else 512,
                )
            )
            for a in ("control", "wide_current", "wide_lower")
        },
    }
    monkeypatch.setattr(C, "prepare", lambda *args: prepared)
    if low_disk:
        monkeypatch.setattr(
            C.shutil, "disk_usage", lambda *args: SimpleNamespace(free=512 * 1024**2)
        )
    monkeypatch.setattr(C, "fork_arm", lambda *args, **kwargs: {})
    monkeypatch.setattr(C, "retire_arm", lambda *args: None)
    monkeypatch.setattr(
        C, "retain_best", lambda c, a, m, s, b, cfg: {"milestone": m, "scores": s}
    )
    calls = []

    def train(c, cfg, minutes, label, *, evaluation_reserve_minutes):
        assert evaluation_reserve_minutes == 360
        calls.append(label)
        p = tmp_path / f"{label}.pt"
        p.write_text(label)
        return {
            "checkpoint": str(p),
            "iteration": 1,
            "executed_target_minutes": minutes,
        }

    monkeypatch.setattr(C, "train_segment", train)
    monkeypatch.setattr(
        C,
        "serving_profile",
        lambda *args: {
            "search": asdict(C.SEARCH),
            "serving_search": asdict(replace(C.SEARCH, move_deadline_s=1.8)),
        },
    )

    def match(label, candidate, opponent, cfg, search):
        confirmation = label.startswith("confirmation_")
        assert cfg.seed == (
            252000123
            if "cpu_timed" in label
            else 250000123
            if confirmation
            else 240000123
        )
        if confirmation:
            assert (tmp_path / "final_selection.json").exists()
        score = 0.75 if opponent == "astra" else 0.5
        if not confirmation and "wide_" in label and opponent != "astra":
            score = 0.75
        if (
            not confirmation
            and "control_" in label
            and opponent != "astra"
            and control_selected
        ):
            score = 0.6
        if label == "confirmation_small_control_vs_teacher":
            score = 0.53
        if label in {
            "confirmation_serving_head_to_head",
            "confirmation_wide_vs_teacher",
        }:
            score = 0.53 if wide_wins else 0.47
        result = report(0.75 if opponent == "astra" else 0.5, cfg, opponent)
        result["summary"]["match_score"] = score
        result["opponent_search"] = asdict(search)
        result["summary"]["match_score_ci95"] = [0.49, 0.57]
        return result

    monkeypatch.setattr(campaign, "_match", match)
    result = C.run_capacity_campaign(campaign, "inputs.json")
    assert result["provisional_larger_improvement"] == wide_wins
    assert not result["resolved_larger_improvement"]
    assert result["selected_label"] == (
        "wide_current_30m"
        if wide_wins
        else "control_30m"
        if control_selected
        else "teacher"
    )
    count = len(calls)
    assert count == (8 if low_disk else 12)
    C.run_capacity_campaign(campaign, "inputs.json")
    assert len(calls) == count


def test_serving_difference_allows_different_search_and_uses_complete_common_pairs() -> (
    None
):
    a = report(
        0.75,
        ArenaConfig(num_games=8, seed=2, search=replace(C.SEARCH, num_simulations=512)),
        "astra",
    )
    b = report(
        0.5,
        ArenaConfig(
            num_games=4, seed=2, search=replace(C.SEARCH, num_simulations=1024)
        ),
        "astra",
    )
    a["opponent_search"] = b["opponent_search"] = asdict(C.SEARCH)
    delta = C.paired_serving_difference(a, b)
    assert delta["match_score_difference"] == 0.25
    assert delta["seed_pairs"] == 2
    b["records"][0]["pair_seed"] += 1
    with pytest.raises(ValueError, match="seed pairs"):
        C.paired_serving_difference(a, b)
