from __future__ import annotations
from dataclasses import replace
import json
import os
import signal
import torch
from agent.obs.run import Run
from agent.train.loop import LoopConfig, run_loop, _apply_eval_results
from agent.train.checkpointing import load_checkpoint_payload
from agent.train.reproducibility import provenance


def small_config(root: str, run_id: str) -> LoopConfig:
    return LoopConfig(
        device="cpu",
        hidden=32,
        arch="flat",
        seed=191,
        selfplay_games=4,
        selfplay_sims=2,
        selfplay_max_turns=150,
        selfplay_turns_per_player=0,
        replay_capacity=1000,
        learner_batch=16,
        learner_steps_per_iter=2,
        checkpoint_every=1,
        eval_games=0,
        training_cycle_length=0,
        league_selfplay_every=0,
        max_iters=1,
        max_wall_minutes=1,
        run_id=run_id,
        runs_root=root,
        league_root=f"{root}/{run_id}/league",
    )


def test_resume_matches_uninterrupted_training(tmp_path) -> None:
    cfg = small_config(str(tmp_path), "resumed")
    run = Run("resumed", runs_root=str(tmp_path))
    run_loop(run, cfg)
    run.close()
    run = Run("resumed", runs_root=str(tmp_path))
    run_loop(run, cfg)
    run.close()
    full = small_config(str(tmp_path), "full")
    run = Run("full", runs_root=str(tmp_path))
    run_loop(run, replace(full, max_iters=2))
    run.close()
    a = load_checkpoint_payload(tmp_path / "resumed/checkpoints/latest_resume.pt")
    b = load_checkpoint_payload(tmp_path / "full/checkpoints/latest_resume.pt")
    assert a["iteration"] == b["iteration"] == 2
    for key, value in a["model_state_dict"].items():
        assert torch.equal(value, b["model_state_dict"][key]), key
    assert a["buffer"]["total_sampled"] == b["buffer"]["total_sampled"]


def test_reanalysis_and_mixed_budget_resume_matches_uninterrupted(tmp_path) -> None:
    cfg = replace(
        small_config(str(tmp_path), "reanalysis_resumed"),
        search_backend="gumbel_tree",
        search_tree_core="rust",
        search_inference_cache_size=128,
        selfplay_full_sims=4,
        selfplay_full_fraction=0.5,
        reanalysis_positions=8,
        reanalysis_every=1,
        reanalysis_sims=8,
        reanalysis_snapshot_capacity=128,
        reanalysis_batch_size=4,
    )
    for _ in range(2):
        run = Run(cfg.run_id, runs_root=str(tmp_path))
        try:
            run_loop(run, cfg)
        finally:
            run.close()
    full = replace(
        cfg,
        run_id="reanalysis_full",
        league_root=str(tmp_path / "reanalysis_full/league"),
        max_iters=2,
    )
    run = Run(full.run_id, runs_root=str(tmp_path))
    try:
        run_loop(run, full)
    finally:
        run.close()
    a = load_checkpoint_payload(tmp_path / cfg.run_id / "checkpoints/latest_resume.pt")
    b = load_checkpoint_payload(tmp_path / full.run_id / "checkpoints/latest_resume.pt")
    assert (
        a["buffer"]["snapshots"]
        and a["buffer"]["snapshots"] == b["buffer"]["snapshots"]
    )
    assert torch.equal(a["buffer"]["policy_target"], b["buffer"]["policy_target"])
    for key, value in a["model_state_dict"].items():
        assert torch.equal(value, b["model_state_dict"][key]), key


def test_sigterm_finishes_iteration_and_updates_resume_state(
    tmp_path, monkeypatch
) -> None:
    cfg = small_config(str(tmp_path), "interrupt")
    run = Run("interrupt", runs_root=str(tmp_path))
    event = run.event

    def interrupt(name, fields=None, level="INFO"):
        event(name, fields, level)
        if name == "iter_started":
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(run, "event", interrupt)
    result = run_loop(run, replace(cfg, max_iters=20))
    run.close()
    assert result["stopped"] and result["iter"] == 1
    assert json.loads((run.root / "state.json").read_text())["iter"] == 1
    assert json.loads((run.root / "heartbeat.json").read_text())["phase"] == "stopped"
    assert load_checkpoint_payload(run.ckpt_dir / "latest_resume.pt")["iteration"] == 1


def test_delayed_evaluation_uses_its_job_identity() -> None:
    class League:
        def __init__(self):
            self.results = []

        def record_result(self, *args, **kwargs):
            self.results.append(args[:2])

        def recompute_ratings(self):
            return {}

    league = League()
    results = {
        "job_context": {"entity": "ckpt:10", "league_map": {"league_0": 8}},
        "pairwise": [{"winner": "eval_agent", "loser": "league_0", "weight": 1.0}],
    }
    _apply_eval_results(league, results, "ckpt:11", {"league_0": 9})
    assert league.results == [("ckpt:10", "ckpt:8")]


def test_provenance_is_yaml_serializable() -> None:
    import yaml

    assert yaml.safe_load(yaml.safe_dump(provenance()))["torch"] == str(
        torch.__version__
    )
