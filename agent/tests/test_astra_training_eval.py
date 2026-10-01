"""Astra evaluation integration, outcome accounting, and frozen-run compatibility."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from agent.eval import arena, builtin_opponents
from agent.eval.heuristic_astra import (
    AstraUnavailableError,
    HeuristicAstraBot,
    production_config,
)
from agent.net.model import AzulNet
from agent.scripts import competitive
from agent.train import unified_eval as UE
from agent.train.checkpointing import save_checkpoint
from agent.train.league import League
from agent.train.loop import LoopConfig, _apply_eval_results, _record_eval_completion


@pytest.mark.parametrize("fraction", [-0.1, 1.01, float("inf"), float("nan")])
def test_rejects_invalid_astra_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="astra_opponent_fraction"):
        UE.UnifiedEvalConfig(astra_opponent_fraction=fraction)
    with pytest.raises(ValueError, match="eval_astra_fraction"):
        LoopConfig(eval_astra_fraction=fraction)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bot_selfplay_astra_prob": -0.1},
        {"bot_selfplay_opus_prob": 0.8, "bot_selfplay_astra_prob": 0.3},
        {"bot_selfplay_workers": 0},
    ],
)
def test_rejects_invalid_astra_training_mix(kwargs: dict) -> None:
    with pytest.raises(
        ValueError, match="bot self-play probabilities|bot_selfplay_workers"
    ):
        LoopConfig(**kwargs)


def test_production_astra_registration_and_disabled_native_import(monkeypatch) -> None:
    seen = []

    def games(**kwargs):
        policies = kwargs["policies"]
        bot = policies["astra"].bot if "astra" in policies else None
        seen.append((kwargs["num_players"], kwargs["astra_opponent_fraction"], bot))
        return [], {}

    monkeypatch.setattr(UE, "_run_multiplayer_games", games)
    net = AzulNet(hidden=32, arch="flat")
    cfg = UE.UnifiedEvalConfig(total_games=3, weight_2p=1, weight_3p=1, weight_4p=1)
    result = UE.run_unified_eval(net.state_dict(), [], cfg, 31, hidden=32, arch="flat")
    assert LoopConfig().eval_astra_fraction == cfg.astra_opponent_fraction == 0.125
    assert [n for n, _, _ in seen] == [2, 3, 4]
    for n, fraction, bot in seen:
        assert fraction == 0.125 and isinstance(bot, HeuristicAstraBot)
        assert result["astra_identity"][str(n)]["config"] == asdict(
            production_config(n)
        )
        assert result["astra_identity"][str(n)]["native_sha256"]

    def unavailable():
        raise AstraUnavailableError("missing test extension")

    monkeypatch.setattr(builtin_opponents, "native_module", unavailable)
    seen.clear()
    disabled = UE.run_unified_eval(
        net.state_dict(),
        [],
        replace(cfg, astra_opponent_fraction=0),
        31,
        hidden=32,
        arch="flat",
    )
    assert disabled["astra_identity"] == {}
    assert all(bot is None for _, _, bot in seen)
    with pytest.raises(AstraUnavailableError, match="missing test extension"):
        UE.run_unified_eval(net.state_dict(), [], cfg, 31, hidden=32, arch="flat")


class SharedEngine:
    """Ended synthetic games: all seats share victory, without drafting."""

    def __init__(self, batch_size: int, num_players: int, **kwargs) -> None:
        self.ended = torch.ones(batch_size, dtype=torch.bool)
        self.scores = torch.ones((batch_size, num_players), dtype=torch.long)
        self.wall = torch.zeros((batch_size, num_players, 5, 5), dtype=torch.bool)

    def get_winners(self) -> torch.Tensor:
        return torch.full_like(self.ended, UE.BE.SHARED_VICTORY, dtype=torch.long)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_astra_sampling_and_shared_win_share(monkeypatch, players: int) -> None:
    monkeypatch.setattr(UE.BE, "BatchedEngine", SharedEngine)
    policies = {"eval_agent": None, "random": None, "astra": None}
    suffix = f"{players}p_{'vs' if players == 2 else 'with'}_astra"
    for fraction in (0.0, 0.125, 1.0):
        args = dict(
            num_players=players,
            num_games=128,
            policies=policies,
            eval_agent_name="eval_agent",
            seed=51,
            max_turns=0,
            astra_opponent_fraction=fraction,
        )
        pairwise, metrics = UE._run_multiplayer_games(**args)
        assert UE._run_multiplayer_games(**args) == (pairwise, metrics)
        assert metrics[f"eval_win_share_{players}p"] == pytest.approx(1 / players)
        games = metrics[f"games_{suffix}"]
        if fraction == 0:
            assert games == 0
            assert all("astra" not in (r.winner, r.loser) for r in pairwise)
        else:
            assert metrics[f"eval_win_share_{suffix}"] == pytest.approx(1 / players)
            assert (games == 128) if fraction == 1 else (3 < games < 70)
        assert all(r.is_tie for r in pairwise)


def test_multiplayer_shared_winners_beat_losers_and_exclude_unfinished() -> None:
    pairs = UE.extract_pairwise_results(
        [["a", "b", "c"], ["a", "b", "c"]],
        torch.tensor([UE.BE.SHARED_VICTORY, 0]),
        torch.tensor([True, False]),
        3,
        torch.tensor([[True, True, False], [True, False, False]]),
    )
    assert [(r.winner, r.loser, r.weight, r.is_tie) for r in pairs] == [
        ("a", "b", 0.5, True),
        ("a", "c", 1.0, False),
        ("b", "a", 0.5, True),
        ("b", "c", 1.0, False),
    ]


def test_worker_merge_weights_astra_subgroups_and_rejects_identity_mismatch() -> None:
    parts = [
        {
            "astra_identity": {"2": "same"},
            "metrics": {
                "games_2p": games,
                "finished_2p": games,
                "wins_2p": credit,
                "games_2p_vs_astra": games,
                "finished_2p_vs_astra": finished,
                "unfinished_2p_vs_astra": games - finished,
                "win_share_sum_2p_vs_astra": credit,
                "eval_win_share_2p_vs_astra": credit / finished,
            },
        }
        for games, finished, credit in [(10, 8, 4), (2, 2, 2)]
    ]
    metrics = UE._merge_eval_results(parts)["metrics"]
    assert metrics["eval_win_share_2p_vs_astra"] == 0.6
    assert metrics["unfinished_2p_vs_astra"] == 2
    parts[1]["astra_identity"] = {"2": "changed"}
    assert "error" in UE._merge_eval_results(parts)


@pytest.mark.parametrize("legacy", [False, True])
def test_astra_ties_survive_league_ingestion_and_remain_floating(
    tmp_path: Path, legacy: bool
) -> None:
    league = League(tmp_path)
    league.manifest["entries"].append(
        {
            "idx": 0,
            "tag": "test",
            "path": "dummy.pt",
            "rating": 1500,
            "games": 0,
            "hidden": 32,
            "arch": "flat",
            "active": True,
        }
    )
    pairwise = [
        asdict(r)
        for r in UE.extract_pairwise_results(
            [["eval_agent", "astra"]],
            torch.tensor([UE.BE.SHARED_VICTORY]),
            torch.tensor([True]),
            2,
        )
    ]
    if legacy:
        for row in pairwise:
            row.pop("is_tie")
    result = {"pairwise": pairwise, "job_context": {"entity": "ckpt:0"}}
    _apply_eval_results(league, result, "wrong_checkpoint", {})
    score = league.entry_by_idx(0)
    assert score["games_2p_vs_astra"] == 1
    assert score["winrate_2p_vs_astra"] == 0.5
    assert "astra" not in league.manifest["anchors"]
    row = next(
        r for r in league.manifest["results"] if {r["a"], r["b"]} == {"astra", "ckpt:0"}
    )
    assert row["ties_2p"] == 1
    assert row["wins_a_2p"] == row["wins_b_2p"] == 0


def test_completion_persists_astra_metrics_identity_and_errors(monkeypatch) -> None:
    records, events = [], []

    class Run:
        def metric(self, row: dict) -> None:
            records.append(row)

        def event(self, name: str, row: dict) -> None:
            events.append((name, row))

    monkeypatch.setattr(
        "agent.train.loop._apply_eval_results", lambda *args: {"ckpt:2": 1500}
    )
    _record_eval_completion(
        Run(),
        None,
        25,
        {
            "metrics": {"games_2p_vs_astra": 8},
            "astra_identity": {"2": "id"},
            "job_context": {"entity": "ckpt:2"},
        },
        "stale",
        {},
    )
    assert records == [{"iter": 25, "rating": 1500, "games_2p_vs_astra": 8}]
    assert events[0][1]["astra_identity"] == {"2": "id"}
    _record_eval_completion(
        Run(), None, 26, {"error": "missing extension"}, "stale", {}
    )
    assert len(records) == 1 and events[1][0] == "unified_eval_failed"


def test_arena_supports_real_production_astra(tmp_path: Path) -> None:
    path = tmp_path / "net.pt"
    save_checkpoint(path, AzulNet(hidden=32, arch="flat"), config={"num_players": 2})
    cfg = arena.ArenaConfig(num_games=2, game_batch_size=2, max_turns=2, greedy=True)
    result = arena.evaluate_match(str(path), "astra", cfg)
    assert result["opponent_identity"] == builtin_opponents.builtin_identity("astra", 2)
    assert result["opponent_sha256"] is None
    assert result["summary"]["unfinished"] == 2
    assert [r["candidate_seat"] for r in result["records"]] == [0, 1]


def test_competitive_screen_contains_astra_and_cache_tracks_native_config(
    tmp_path: Path, monkeypatch
) -> None:
    campaign = competitive.Campaign(tmp_path, "cpu", 51)
    seen = []

    def match(label, candidate, opponent, *args):
        seen.append(opponent)
        return {"summary": {"unfinished": 0}}

    monkeypatch.setattr(campaign, "_match", match)
    screen = campaign._screen(
        "screen", "candidate.pt", {"champion": "champ.pt", "historical": "old.pt"}
    )
    assert "astra" in seen and "astra" in screen["scores"]

    campaign = competitive.Campaign(tmp_path, "cpu", 51)
    identity, calls = {"config": 1}, []
    monkeypatch.setattr(competitive, "checkpoint_hash", lambda path: "checkpoint")
    monkeypatch.setattr(competitive, "builtin_identity", lambda name: dict(identity))

    def evaluate(
        candidate, opponent, cfg, opponent_search, *, progress=None, activity=None
    ):
        calls.append(opponent)
        return {
            "candidate_sha256": "checkpoint",
            "opponent": opponent,
            "opponent_sha256": None,
            "opponent_identity": dict(identity),
            "config": asdict(cfg),
            "opponent_search": asdict(opponent_search or cfg.search),
            "provenance": campaign.code,
            "summary": {},
        }

    monkeypatch.setattr(competitive, "evaluate_match", evaluate)
    cfg = campaign.arena(2)
    campaign._match("astra_test", "candidate.pt", "astra", cfg)
    campaign._match("astra_test", "candidate.pt", "astra", cfg)
    assert calls == ["astra"]
    identity["config"] = 2
    campaign._match("astra_test", "candidate.pt", "astra", cfg)
    assert calls == ["astra", "astra"]


def test_old_campaign_preserves_frozen_opponents_and_strict_resume(
    tmp_path: Path,
) -> None:
    cfg = competitive.training_config(tmp_path, "old", "init.pt", device="cpu")
    assert cfg.eval_astra_fraction == 0.125
    directory = tmp_path / "experiments" / "old"
    directory.mkdir(parents=True)
    old = asdict(cfg)
    old.pop("eval_astra_fraction")
    (directory / "experiment_complete.json").write_text(
        json.dumps({"config": old, "checkpoint": "final.pt"})
    )
    resumed = competitive.training_config(tmp_path, "old", "init.pt", device="cpu")
    assert resumed.eval_astra_fraction == 0
    campaign = competitive.Campaign(tmp_path, "cpu", 51)
    assert campaign._train(resumed) == "final.pt"
    with pytest.raises(ValueError, match="configuration differs"):
        campaign._train(replace(resumed, eval_astra_fraction=0.125))
    with pytest.raises(ValueError, match="configuration differs"):
        campaign._train(replace(resumed, learner_steps_per_iter=1))
