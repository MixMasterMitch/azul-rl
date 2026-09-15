from __future__ import annotations

import math
import pytest

from agent.train import ranking as R
from agent.train import rating_display as D
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
    # Four first places produce twelve pairwise wins; six losses give 2:1 odds.
    expected_raw = R.DEFAULT_REFERENCE_ANCHORS_PER_PC[4]['heuristic_opus'] + 1000 * math.log10(2)
    assert store.data['raw_rating_4p'] == pytest.approx(expected_raw, abs=1.0)
    assert store.data['rating_4p'] == pytest.approx(D.to_display(
        expected_raw, 4, D.scales_for(R.DEFAULT_REFERENCE_ANCHORS_PER_PC)), abs=5.0)


def test_record_game_remains_2p_wrapper(tmp_path) -> None:
    store = HumanRatingStore("u", data_dir=str(tmp_path))

    store.record_game("random", R.RANDOM_ANCHOR_RATING, won=True)
    store.record_game("random", R.RANDOM_ANCHOR_RATING, won=True)
    store.record_game("random", R.RANDOM_ANCHOR_RATING, won=False)

    assert store.data["games"] == 3
    assert store.data["games_2p"] == 3
    assert store.data["wins"] == 2
    assert store.rating is not None
