from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import pytest
import torch

from agent.scripts import policy_weight_campaign as C
from agent.scripts.competitive import Campaign
from agent.train.presets import enhanced_2p_config
from agent.train.checkpointing import save_checkpoint, load_checkpoint_payload
from agent.train.league import League
from agent.tests.test_finetune_campaign import assert_same
from agent.tests.test_training_enhancements import make_buffer, add_positions
from agent.tests.test_enhancement_campaign import report
from agent.net.model import AzulNet
from agent.env.engine import GameEngine
from agent.eval.arena import write_report


@pytest.mark.parametrize("weight", [1.0, 0.25])
def test_weight_forks_preserve_all_learning_state(tmp_path, weight) -> None:
    net = AzulNet(hidden=32, arch="flat")
    cfg = replace(
        enhanced_2p_config(), hidden=32, arch="flat", search_inference_cache_size=0
    )
    optim = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    sum(p.sum() for p in net.parameters()).backward()
    optim.step()
    b = make_buffer(8)
    add_positions(b, GameEngine(8, 2, seed=6))
    b.policy_sims[:8] = 64
    source = tmp_path / "source.pt"
    save_checkpoint(source, net, optim, 12, asdict(cfg), b, {"training_wall_s": 60.0})
    before = load_checkpoint_payload(source)
    league = League(tmp_path / "league")
    league.add_checkpoint(net, "frozen")
    directory = tmp_path / "fork"
    changed = replace(
        cfg,
        policy_fast_weight=weight,
        run_id="fork",
        league_root=str(directory / "league"),
    )
    record = C.fork_weight(source, league.root, directory, changed)
    after = load_checkpoint_payload(directory / "checkpoints/latest_resume.pt")
    for k in [
        "model_state_dict",
        "optimizer_state_dict",
        "buffer",
        "rng_state",
        "iteration",
        "progress",
    ]:
        assert_same(before[k], after[k])
    assert after["config"]["policy_fast_weight"] == weight
    assert C.fork_weight(source, league.root, directory, changed) == record
    with pytest.raises(ValueError, match="changed"):
        C.fork_weight(
            source, league.root, directory, replace(changed, policy_fast_weight=0.1)
        )
    with pytest.raises(ValueError, match="Undeclared"):
        C.fork_weight(source, league.root, tmp_path / "bad", replace(changed, lr=0.01))


@pytest.mark.parametrize("outcome", ["pass", "fail", "start"])
@pytest.mark.parametrize("setup_hours", [0, 4])
def test_policy_campaign_freezes_selection_and_reuses_completed_work(
    tmp_path, monkeypatch, outcome, setup_hours
) -> None:
    c = Campaign(tmp_path, "cpu", 27, campaign_hours=10)
    for k, v in C.DIAGNOSTIC_ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    now = time.time() - setup_hours * 3600
    write_report(
        tmp_path / "preflight_budget.json", {"started_at": now, "deadline": now + 36000}
    )
    write_report(tmp_path / "ready.json", {"passed": True, "code": c.code})
    closeout = tmp_path / "closeout.json"
    write_report(closeout, {})
    frozen = {}
    for n in ["start", "previous"]:
        p = tmp_path / f"{n}.pt"
        p.write_text(n)
        frozen[n] = str(p)
    prepared = {
        "initialization": "test",
        "base_training_wall_s": 600.0,
        "input_hashes": {},
        "frozen": frozen,
        "arms": {
            n: asdict(
                replace(
                    enhanced_2p_config(),
                    policy_fast_weight=w,
                    run_id=n,
                    runs_root=str(tmp_path / "experiments"),
                    league_root=str(tmp_path / n / "league"),
                )
            )
            for n, w in C.ARMS.items()
        },
    }
    monkeypatch.setattr(C, "prepare", lambda *args: prepared)
    calls = []
    matches = []

    def train(c, cfg, minutes, label, *, evaluation_reserve_minutes):
        calls.append((cfg.run_id, minutes, evaluation_reserve_minutes))
        p = tmp_path / "experiments" / cfg.run_id / f"{label}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(label)
        return {
            "checkpoint": str(p),
            "target_minutes": minutes,
            "executed_target_minutes": minutes,
        }

    monkeypatch.setattr(C, "train_segment", train)

    def match(label, path, opponent, cfg, search):
        matches.append(label)
        score = 0.5
        if label.startswith(("confirmation_", "greedy_")):
            assert (tmp_path / "final_selection.json").exists()
            assert cfg.seed in {190000027, 200000027}
            if "candidate" in label:
                score = 0.75 if outcome == "pass" else 0.25
        else:
            assert cfg.seed == 180000027
            if label.startswith("weighted_2") and outcome != "start":
                score = 0.75
        return report(score, cfg, opponent)

    monkeypatch.setattr(c, "_match", match)
    result = C.run_policy_weight_campaign(c, str(closeout))
    minutes = result["allocation"]["minutes_per_arm"]
    assert minutes == pytest.approx(120.0 if setup_hours == 0 else 85.0, abs=0.1)
    assert calls == [
        ("control", 10 + minutes / 2, 180 + minutes),
        ("control", 10 + minutes, 180 + minutes),
        ("weighted", 10 + minutes / 2, 180),
        ("weighted", 10 + minutes, 180),
    ]
    assert result["improvement_resolved"] == (outcome == "pass")
    assert result["selected_checkpoint"] == (
        result["candidate_checkpoint"] if outcome == "pass" else frozen["start"]
    )
    if outcome == "start":
        assert result["panel_difference"] is None
        assert not any("confirmation_candidate" in m for m in matches)
    before = len(calls), len(matches)
    C.run_policy_weight_campaign(c, str(closeout))
    assert before == (len(calls), len(matches))
    assert (
        json.loads((tmp_path / "preflight_budget.json").read_text())["deadline"]
        == now + 36000
    )


def test_missing_validation_or_diagnostics_prevents_training(
    tmp_path, monkeypatch
) -> None:
    c = Campaign(tmp_path, "cpu", 1, campaign_hours=10)
    write_report(
        tmp_path / "preflight_budget.json",
        {"started_at": time.time(), "deadline": time.time() + 36000},
    )
    monkeypatch.delenv("PYTHONMALLOC", raising=False)
    with pytest.raises(RuntimeError, match="diagnostics"):
        C.run_policy_weight_campaign(c, str(tmp_path / "missing"))
    for k, v in C.DIAGNOSTIC_ENVIRONMENT.items():
        monkeypatch.setenv(k, v)
    write_report(tmp_path / "ready.json", {"passed": False, "code": c.code})
    with pytest.raises(ValueError, match="validation"):
        C.run_policy_weight_campaign(c, str(tmp_path / "missing"))


@pytest.mark.parametrize("label", ["initializer", "paused", "pilot_current"])
def test_prepare_uses_matching_full_state_or_matched_fresh_state(
    tmp_path, label
) -> None:
    old = tmp_path / "old"
    old.mkdir()
    source = old / "full.pt"
    selected = old / "selected.pt"
    previous = old / "previous.pt"
    net = AzulNet(hidden=256, arch="source_attn")
    net.trained_player_counts = [2]
    cfg = replace(enhanced_2p_config(), replay_capacity=8)
    b = make_buffer(8)
    add_positions(b, GameEngine(8, 2, seed=6))
    optim = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    save_checkpoint(source, net, optim, 420, asdict(cfg), b, {"training_wall_s": 123.0})
    save_checkpoint(previous, net, iteration=1, config=asdict(cfg))
    if label == "pilot_current":
        with torch.no_grad():
            next(net.parameters()).add_(0.01)
    save_checkpoint(selected, net, iteration=420, config=asdict(cfg))
    league = League(old / "league")
    league.add_checkpoint(net, "start")
    write_report(
        old / "finetune_prepared.json", {"frozen": {"previous": str(previous)}}
    )
    closeout = tmp_path / "closeout"
    closeout.mkdir()
    write_report(
        closeout / "plan.json",
        {"old_root": str(old), "source_pause": {"checkpoint": str(source)}},
    )
    write_report(
        closeout / "result.json",
        {
            "selected_checkpoint": str(selected),
            "selected_sha256": C.checkpoint_hash(selected),
            "candidate_label": label,
            "improvement_resolved": label != "initializer",
            "source_resume": str(source),
            "original_league_snapshot": str(league.root),
        },
    )
    c = Campaign(tmp_path / "new", "cpu", 1, campaign_hours=10)
    result = C.prepare(c, closeout / "result.json")
    states = []
    for arm in C.ARMS:
        conf = result["arms"][arm]
        state = load_checkpoint_payload(
            Path(conf["runs_root"]) / conf["run_id"] / "checkpoints/latest_resume.pt"
        )
        states.append(state)
        assert state["config"]["policy_fast_weight"] == C.ARMS[arm]
        assert state["iteration"] == (0 if label == "pilot_current" else 420)
        assert state["buffer"]["size"] == (0 if label == "pilot_current" else 8)
    for k in [
        "model_state_dict",
        "optimizer_state_dict",
        "buffer",
        "rng_state",
        "progress",
    ]:
        assert_same(states[0][k], states[1][k])
    assert C.prepare(c, closeout / "result.json") == result
    # A crash after preparing one or both forks must not replace common RNG
    # state or make its completed fork identities change on restart.
    (c.root / "prepared.json").unlink()
    assert C.prepare(c, closeout / "result.json") == result
