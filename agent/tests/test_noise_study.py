from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from agent.eval.arena import ArenaConfig, checkpoint_hash, summarize_games
from agent.net.model import AzulNet
from agent.scripts.competitive import Campaign
from agent.scripts.noise_study import paired_difference, run_noise_study
from agent.search.config import SearchConfig
from agent.train.checkpointing import save_checkpoint


def report(
    scores: list[float], cfg: ArenaConfig | None = None, opponent: str = "astra"
) -> dict:
    cfg = cfg or ArenaConfig(num_games=len(scores), seed=123)
    records = [
        {
            "pair_seed": cfg.seed + i // 2,
            "candidate_seat": i % 2,
            "match_score": score,
            "outcome": {0.0: "loss", 0.5: "shared", 1.0: "win"}[score],
        }
        for i, score in enumerate(scores)
    ]
    return {
        "config": asdict(cfg),
        "opponent": opponent,
        "opponent_sha256": "frozen_hash" if opponent != "astra" else None,
        "opponent_identity": {"name": opponent},
        "opponent_search": asdict(cfg.search),
        "summary": summarize_games(records),
        "records": records,
    }


def test_difference_bootstraps_seed_pairs_and_retains_negative_deltas() -> None:
    candidate = report([1.0, 0.5, 1.0, 0.5])
    control = report([0.0, 0.5, 0.5, 0.0])
    assert paired_difference(candidate, control) == {
        "match_score_difference": 0.5,
        "paired_ci95": (0.5, 0.5),
        "seed_pairs": 2,
    }
    assert paired_difference(control, candidate) == {
        "match_score_difference": -0.5,
        "paired_ci95": (-0.5, -0.5),
        "seed_pairs": 2,
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "seed",
        "search",
        "opponent",
        "opponent_hash",
        "unfinished",
        "missing",
        "reordered",
        "invalid_score",
    ],
)
def test_difference_rejects_mismatched_or_incomplete_evidence(corruption: str) -> None:
    candidate = report([1.0, 0.0, 0.5, 1.0])
    control = deepcopy(candidate)
    if corruption == "seed":
        control["config"]["seed"] += 1
    elif corruption == "search":
        control["config"]["search"]["num_simulations"] = 8
    elif corruption == "opponent":
        control["opponent_identity"] = {"name": "different"}
    elif corruption == "opponent_hash":
        control["opponent_sha256"] = "different"
    elif corruption == "unfinished":
        control["records"][0].update(outcome="unfinished", match_score=None)
    elif corruption == "missing":
        control["records"].pop()
    elif corruption == "reordered":
        control["records"].reverse()
    elif corruption == "invalid_score":
        control["records"][0]["match_score"] = float("nan")
    with pytest.raises(ValueError):
        paired_difference(candidate, control)


@pytest.mark.parametrize(
    "astra_regression, expected", [(False, "no_dirichlet"), (True, "control")]
)
def test_study_isolates_noise_and_freezes_matching_evaluations(
    tmp_path: Path, monkeypatch, astra_regression: bool, expected: str
) -> None:
    source = tmp_path / "initializer.pt"
    save_checkpoint(
        source, AzulNet(hidden=256, arch="source_attn"), config={"num_players": 2}
    )
    campaign = Campaign(
        tmp_path / "study", "cpu", 20260919, bot_workers=8, campaign_hours=3
    )
    configs, screens = [], []

    def train(cfg) -> str:
        configs.append(cfg)
        return f"{cfg.run_id}.pt"

    def match(
        label: str,
        candidate: str,
        opponent: str,
        cfg: ArenaConfig,
        opponent_search: SearchConfig,
    ) -> dict:
        screens.append((label, candidate, opponent, cfg, opponent_search))
        wins = 128
        if "no_dirichlet" in candidate:
            wins = 112 if astra_regression and opponent == "astra" else 144
        return report([1.0] * wins + [0.0] * (cfg.num_games - wins), cfg, opponent)

    monkeypatch.setattr(campaign, "_train", train)
    monkeypatch.setattr(campaign, "_match", match)
    result = run_noise_study(campaign, str(source), minutes=60)
    assert result["recommended_arm"] == expected and not result["automatic_promotion"]
    assert not (campaign.root / "models/registry.json").exists()
    assert len(configs) == 2 and len(screens) == 8
    control, candidate = map(asdict, configs)
    assert {key for key in control if control[key] != candidate[key]} == {
        "run_id",
        "league_root",
        "dirichlet_mix",
    }
    assert (
        control["dirichlet_mix"] == 0.5332343378344558
        and candidate["dirichlet_mix"] == 0.0
    )
    assert control["max_wall_minutes"] == 60 and control["seed"] == 20260919
    assert control["selfplay_sims"] == 64 and control["search_tree_core"] == "rust"
    assert control["learner_steps_per_iter"] == 72 and control["eval_games"] == 0
    frozen = campaign.root / "initializers/frozen_finalist.pt"
    assert (
        control["init_from"] == str(frozen)
        and frozen.read_bytes() == source.read_bytes()
    )
    for label, _, opponent, cfg, opponent_search in screens:
        assert cfg.seed == 50260919 and cfg.num_games == 256 and cfg.bot_workers == 8
        assert cfg.greedy == ("greedy" in label)
        assert cfg.opponent_greedy == (cfg.greedy and opponent != "astra")
        assert cfg.search == opponent_search and cfg.search.tree_core == "rust"
        assert cfg.search.num_simulations == 64 and cfg.search.dirichlet_mix == 0.0
        assert cfg.search.root_noise_scale == 1.0
    assert (
        json.loads((campaign.root / "status.json").read_text())["stage"]
        == "noise_ablation_complete"
    )
    # Persisted JSON must compare cleanly after a supervisor restart.
    assert (
        run_noise_study(campaign, str(source), minutes=60)["recommended_arm"]
        == expected
    )
    with pytest.raises(ValueError, match="protocol changed"):
        run_noise_study(campaign, str(source), minutes=30)


def test_evaluation_heartbeats_keep_active_batches_alive_and_cache_finished_results(
    tmp_path: Path, monkeypatch
) -> None:
    from agent.scripts import competitive

    source = tmp_path / "net.pt"
    save_checkpoint(source, AzulNet(hidden=32, arch="flat"), config={"num_players": 2})
    campaign = Campaign(tmp_path / "study", "cpu", 1)
    cfg = ArenaConfig(num_games=2, game_batch_size=2)
    calls = []

    def evaluate(
        candidate, opponent, config, opponent_search, *, progress, activity
    ) -> dict:
        calls.append(candidate)
        activity({"completed": 0, "turn": 80})
        status = json.loads(campaign.status_path.read_text())
        assert status["turn"] == 80 and status["completed_games_total"] == 0
        assert status["run_id"] == "evaluation:test"
        progress({"completed": 2, "unfinished": 0})
        assert (
            json.loads(campaign.status_path.read_text())["completed_games_total"] == 2
        )
        result = report([1.0, 0.0], config, opponent)
        result.update(
            candidate_sha256=checkpoint_hash(candidate),
            provenance=campaign.code,
            opponent_identity=None,
            opponent_sha256=None,
        )
        return result

    monkeypatch.setattr(competitive, "evaluate_match", evaluate)
    campaign._match("test", str(source), "random", cfg)
    campaign._match("test", str(source), "random", cfg)
    assert len(calls) == 1
