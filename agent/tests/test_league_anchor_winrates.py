"""Tests for per-PC winrate-vs-anchor tracking in league manifests."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent.train import ranking as R
from agent.train.league import League


def _make_row(a: str, b: str, **kwargs) -> dict:
    return {"a": a, "b": b, **kwargs}


def test_winrate_vs_anchor_canonical_order() -> None:
    results = [
        _make_row("ckpt:42", "random", wins_a_2p=10, wins_b_2p=2, ties_2p=0),
    ]
    score, total = R.winrate_vs_anchor(results, "ckpt:42", "random", 2)
    assert total == 12
    assert score == pytest.approx(10.0)


def test_winrate_vs_anchor_reverse_order() -> None:
    results = [
        _make_row("random", "zz_ckpt", wins_a_2p=3, wins_b_2p=7, ties_2p=0),
    ]
    score, total = R.winrate_vs_anchor(results, "zz_ckpt", "random", 2)
    assert total == 10
    assert score == pytest.approx(7.0)


def test_winrate_vs_anchor_ties_count_as_half() -> None:
    results = [
        _make_row("ckpt:1", "random", wins_a_2p=4, wins_b_2p=4, ties_2p=2),
    ]
    score, total = R.winrate_vs_anchor(results, "ckpt:1", "random", 2)
    assert total == 10
    assert score == pytest.approx(5.0)


def test_winrate_vs_anchor_per_pc_isolated() -> None:
    results = [
        _make_row(
            "ckpt:1",
            "random",
            wins_a_2p=8,
            wins_b_2p=2,
            wins_a_3p=1,
            wins_b_3p=9,
        ),
    ]
    s2, t2 = R.winrate_vs_anchor(results, "ckpt:1", "random", 2)
    s3, t3 = R.winrate_vs_anchor(results, "ckpt:1", "random", 3)
    assert (s2, t2) == (pytest.approx(8.0), 10)
    assert (s3, t3) == (pytest.approx(1.0), 10)


def test_league_recompute_populates_anchor_winrates(tmp_path: pathlib.Path) -> None:
    league = League(root=tmp_path)
    league.manifest["entries"].append(
        {
            "idx": 0,
            "tag": "test",
            "path": "dummy.pt",
            "rating": 1500,
            "games": 0,
            "hidden": 256,
            "arch": "attn",
            "active": True,
        }
    )

    league.record_result("ckpt:0", "random", wins_a=90, wins_b=10, num_players=2)
    league.record_result("ckpt:0", "random", wins_a=45, wins_b=5, num_players=3)
    league.record_result("ckpt:0", "random", wins_a=40, wins_b=10, num_players=4)
    league.record_result("ckpt:0", "heuristic", wins_a=60, wins_b=40, num_players=2)
    league.record_result("ckpt:0", "heuristic_opus", wins_a=30, wins_b=70, num_players=3)

    league.recompute_ratings()
    entry = league.entry_by_idx(0)
    assert entry is not None
    assert entry["winrate_2p_vs_random"] == pytest.approx(0.90, abs=1e-4)
    assert entry["games_2p_vs_random"] == 100
    assert entry["winrate_3p_vs_random"] == pytest.approx(0.90, abs=1e-4)
    assert entry["winrate_4p_vs_random"] == pytest.approx(0.80, abs=1e-4)
    assert entry["winrate_2p_vs_heuristic"] == pytest.approx(0.60, abs=1e-4)
    assert entry["winrate_3p_vs_heuristic_opus"] == pytest.approx(0.30, abs=1e-4)
    assert "winrate_4p_vs_heuristic" not in entry


def test_league_migrates_legacy_results(tmp_path: pathlib.Path) -> None:
    league_root = tmp_path / "league"
    league_root.mkdir(parents=True)
    manifest = {
        "entries": [],
        "results": [
            {
                "a": "ckpt:0",
                "b": "random",
                "wins_a": 10,
                "wins_b": 2,
                "num_players": 2,
            }
        ],
        "anchors": {"random": 1000.0},
    }
    (league_root / "league.json").write_text(json.dumps(manifest))

    league = League(league_root)
    row = league.manifest["results"][0]
    assert row["wins_a_2p"] == 10
    assert row["wins_b_2p"] == 2
    assert "wins_a" not in row
