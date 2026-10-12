"""Fixed play ladder, head-to-head updates, and profile migration."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from agent.train import ranking as R
from agent.train.model_registry import ModelRegistry
from play import fixed_ratings as F
from play.models import ModelCatalog
from play.ratings import public_profile, record_result


def game(opponent: str, won: bool, pc: int = 2) -> dict:
    return {"user_sub": "tester", "num_players": pc, "human_seat": 0,
            "winner_seats": [0] if won else [1],
            "scores": [80, 60] if won else [60, 80],
            "opponent_info": [{"id": opponent}]}


def test_fixed_opponents_and_balanced_human_rating() -> None:
    user = None
    for i in range(10):
        user = record_result(user, game("opus", i % 2 == 0), F.references())
    assert user is not None
    assert user["rating_display"]["version"] == F.VERSION
    assert abs(user["rating_2p"] - 2250) < 1
    assert public_profile(user, F.references())["rating_2p"] == 2250


def test_catalog_and_human_use_the_same_fixed_anchor(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "models": {},
                                "rating_scheme": F.VERSION,
                                "reference_anchors_per_pc": F.references()}))
    catalog = ModelCatalog(ModelRegistry(path))
    assert {row["id"]: row["rating_2p"] for row in catalog.list_opponents(2)
            if row["id"] != "astra"} == {
                "random": 1000, "heuristic": 2000, "opus": 2250}
    match = game("opus", True)
    match["opponent_info"] = [catalog.resolve("opus", 2)]
    user = record_result(None, match, catalog.reference_anchors)
    assert user["anchors_per_pc"]["2"]["heuristic_opus"] == 2250


def test_perfect_record_against_agent_caps_at_3500() -> None:
    user = None
    for _ in range(5):
        user = record_result(user, game(F.AGENT_ID, True), F.references())
    assert user is not None
    assert public_profile(user, F.references())["rating_2p"] == 3500


def test_beating_only_weaker_bots_does_not_skip_ladder_rungs() -> None:
    for opponent, expected_cap in (("random", 2000), ("heuristic", 2250),
                                   ("opus", 2500), ("astra", 3000)):
        user = None
        for _ in range(5):
            user = record_result(user, game(opponent, True), F.references())
        assert user is not None
        assert public_profile(user, F.references())["rating_2p"] == expected_cap


def test_multiplayer_second_place_records_each_bot_comparison() -> None:
    match = {"user_sub": "tester", "num_players": 3, "human_seat": 0,
             "winner_seats": [1], "scores": [80, 90, 70],
             "opponent_info": [{"id": "astra"}, {"id": "opus"}]}
    user = record_result(None, match, F.references())
    assert user["wins"] == 0
    assert R.winrate_vs_anchor(user["results"], "human", "astra", 3) == (0, 1)
    assert R.winrate_vs_anchor(user["results"], "human", "heuristic_opus", 3) == (1, 1)


def test_legacy_profile_is_refitted_without_mutating_history() -> None:
    user = None
    for i in range(10):
        match = game("heuristic", i % 2 == 0)
        match["opponent_info"][0]["raw_ratings"] = {"2": 3000.0}
        user = record_result(user, match,
                             {pc: {"random": 1000.0, "heuristic": 3000.0}
                              for pc in (2, 3, 4)})
    assert user is not None
    old = deepcopy(user)
    profile = public_profile(user, F.references())
    assert user == old
    assert profile["rating_2p"] == 2000
    updated = record_result(user, game("heuristic", True), F.references())
    assert updated["rating_display"]["version"] == F.VERSION
    assert old["rating_display"]["version"] != F.VERSION
