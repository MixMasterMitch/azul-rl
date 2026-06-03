from __future__ import annotations

import pytest

from agent.train import ranking as R
from play.human_rating import HumanRatingStore


def _opus_opponents() -> list[dict[str, object]]:
    opus_rating = R.DEFAULT_REFERENCE_ANCHORS_PER_PC[4]["heuristic_opus"]
    return [
        {"seat": 1, "entity_id": "opus", "rating": opus_rating},
        {"seat": 2, "entity_id": "opus", "rating": opus_rating},
        {"seat": 3, "entity_id": "opus", "rating": opus_rating},
    ]


def test_4p_human_wins_expand_to_full_bt_wins_per_loser(tmp_path) -> None:
    store = HumanRatingStore("u", data_dir=str(tmp_path))

    for _ in range(4):
        store.record_multiplayer_game(
            opponents=_opus_opponents(),
            human_rank=0,
            ranks=[0, 1, 2, 3],
        )
    for _ in range(6):
        store.record_multiplayer_game(
            opponents=_opus_opponents(),
            human_rank=1,
            ranks=[1, 0, 2, 3],
        )

    assert store.data["games_4p"] == 10
    assert store.data["wins"] == 4
    assert store.data["rating_4p"] == pytest.approx(3917.0, abs=5.0)
    assert store.data["rating_4p"] > R.calibrate_rating(
        R.DEFAULT_REFERENCE_ANCHORS_PER_PC[4]["heuristic_opus"], 4
    )


def test_record_game_remains_2p_wrapper(tmp_path) -> None:
    store = HumanRatingStore("u", data_dir=str(tmp_path))

    store.record_game("random", R.RANDOM_ANCHOR_RATING, won=True)
    store.record_game("random", R.RANDOM_ANCHOR_RATING, won=True)
    store.record_game("random", R.RANDOM_ANCHOR_RATING, won=False)

    assert store.data["games"] == 3
    assert store.data["games_2p"] == 3
    assert store.data["wins"] == 2
    assert store.rating is not None
