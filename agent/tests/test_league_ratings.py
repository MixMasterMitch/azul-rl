"""Tests for league rating recompute and legacy manifest migration."""

from __future__ import annotations

import pathlib

import pytest
import torch

from agent.net import model as M
from agent.train.league import League


def test_league_recompute_ratings_updates_entry_fields(tmp_path: pathlib.Path) -> None:
    torch.manual_seed(23)
    league = League(tmp_path / "league")
    league.add_checkpoint(M.AzulNet(hidden=32, arch="flat"), tag="a")
    league.add_checkpoint(M.AzulNet(hidden=32, arch="flat"), tag="b")

    row_a = {
        "rank_winrate_vs_random": 63.0 / 64.0,
        "rank_ties_vs_random": 0.0,
        "rank_winrate_vs_heuristic": 40.0 / 64.0,
        "rank_ties_vs_heuristic": 0.0,
    }
    row_b = {
        "rank_winrate_vs_random": 52.0 / 64.0,
        "rank_ties_vs_random": 0.0,
        "rank_winrate_vs_heuristic": 18.0 / 64.0,
        "rank_ties_vs_heuristic": 4.0 / 64.0,
    }
    league.record_checkpoint_baselines(0, row_a, rank_games=64, eval_games=16)
    league.record_checkpoint_baselines(1, row_b, rank_games=64, eval_games=16)
    league.record_result("ckpt:0", "ckpt:1", 46.0, 14.0, 4.0)

    ratings = league.recompute_ratings()
    entries = {int(entry["idx"]): entry for entry in league.list_entries()}

    assert ratings["random"] == pytest.approx(1000.0)
    assert ratings["heuristic"] > ratings["random"]
    assert float(entries[0]["rating"]) > float(entries[1]["rating"])
    assert league.manifest["rating_system"] == "anchored_bt_per_pc"
    assert int(entries[0]["games"]) > 0


def test_league_migrates_legacy_entries_into_results(tmp_path: pathlib.Path) -> None:
    league_root = tmp_path / "league"
    league_root.mkdir(parents=True, exist_ok=True)
    legacy_path = league_root / "ckpt_00000_i5.pt"
    torch.save(
        {
            "model_state_dict": M.AzulNet(hidden=32, arch="flat").state_dict(),
            "hidden": 32,
            "arch": "flat",
            "iteration": 5,
        },
        legacy_path,
    )
    manifest_path = league_root / "league.json"
    manifest_path.write_text(
        """
{
  "entries": [
    {
      "idx": 0,
      "tag": "i5",
      "path": "%s",
      "rating": 0.0,
      "games": 0,
      "hidden": 256,
      "arch": "attn",
      "rank_winrate_vs_random": 0.95703125,
      "rank_ties_vs_random": 0.0,
      "rank_winrate_vs_heuristic": 0.544921875,
      "rank_ties_vs_heuristic": 0.00390625
    }
  ],
  "anchors": {"random": 1000.0}
}
""".strip()
        % legacy_path.name
    )

    league = League(league_root)
    entry = league.list_entries()[0]

    assert league.manifest["anchors"]["random"] == pytest.approx(1000.0)
    assert len(league.manifest["results"]) >= 2
    assert float(entry["rating"]) > 1500.0
    assert int(entry["games"]) > 0
