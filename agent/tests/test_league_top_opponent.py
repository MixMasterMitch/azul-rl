"""League top-opponent selection."""

from agent.train.league import League


def test_top_opponent_picks_highest_rating(tmp_path) -> None:
    league = League(tmp_path)
    league.manifest["entries"] = [
        {"idx": 0, "path": "a.pt", "rating": 1200, "active": True},
        {"idx": 1, "path": "b.pt", "rating": 1900, "active": True},
    ]
    (tmp_path / "a.pt").write_bytes(b"x")
    (tmp_path / "b.pt").write_bytes(b"y")
    top = league.top_opponent_entry()
    assert top is not None
    assert int(top["idx"]) == 1
