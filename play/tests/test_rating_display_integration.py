"""Use one rating presentation for bots, profiles, and legacy stored users."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from agent.train import ranking as R
from agent.train import rating_display as D
from agent.train.model_registry import ModelRegistry
from play.models import ModelCatalog
from play.ratings import public_profile, record_result


@pytest.mark.parametrize('references', [R.DEFAULT_REFERENCE_ANCHORS_PER_PC,
    {n: {'random': 1000, 'heuristic': value, 'heuristic_opus': value + 200}
     for n, value in D.CAMPAIGN_HEURISTIC.items()}])
def test_catalog_and_profiles_agree_for_each_raw_reference_basis(tmp_path: Path, references: dict) -> None:
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps({'schema_version': 1, 'models': {}, 'reference_anchors_per_pc': references}))
    before = path.read_bytes()
    catalog = ModelCatalog(ModelRegistry(path))
    opponents = {info['id']: info for info in catalog.list_opponents()}
    assert path.read_bytes() == before
    assert opponents['random']['rating'] == 1000
    assert opponents['heuristic']['rating'] == 2500
    for n in D.PLAYER_COUNTS:
        assert opponents['heuristic'][f'rating_{n}p'] == 2500
        assert catalog.resolve('heuristic', n)['raw_ratings'][str(n)] == references[n]['heuristic']
    # A legacy user rated exactly at heuristic must display at the same level.
    user = {'username': 'test', 'games': 30, 'wins': 5, 'rating': 99999,
            **{f'games_{n}p': 10 for n in D.PLAYER_COUNTS},
            **{f'rating_{n}p': R.calibrate_rating(references[n]['heuristic'], n, R.calibration_scales_for(references))
               for n in D.PLAYER_COUNTS}}
    original = deepcopy(user)
    profile = public_profile(user, references)
    assert user == original
    assert profile['rating'] == 2500
    assert [profile[f'rating_{n}p'] for n in D.PLAYER_COUNTS] == [2500] * 3
    assert public_profile(user, references) == profile


def test_recorded_profile_persists_raw_ratings_and_survives_json_reload() -> None:
    references = R.DEFAULT_REFERENCE_ANCHORS_PER_PC
    user = None
    for index in range(10):
        game = {'user_sub': 'test', 'num_players': 2, 'human_seat': 0,
                'winner_seats': [0] if index % 2 else [1],
                'opponent_info': [{'id': 'heuristic', 'raw_ratings': {'2': references[2]['heuristic']}}]}
        user = record_result(user, game, references)
    assert user['rating_display']['version'] == D.VERSION
    assert user['raw_rating_2p'] == pytest.approx(references[2]['heuristic'], abs=1)
    assert user['rating_2p'] == pytest.approx(2500, abs=1)
    profile = public_profile(json.loads(json.dumps(user)))
    assert profile['rating'] == profile['rating_2p'] == 2500
    assert profile['games'] == 10 and profile['wins'] == 5 and profile['placed']
    # Saved source references take precedence over a later catalog's references.
    other = {n: {'random': 1000, 'heuristic': value} for n, value in D.CAMPAIGN_HEURISTIC.items()}
    assert public_profile(user, other) == profile


def test_profile_combines_converted_values_by_games_and_hides_unplaced_ratings() -> None:
    user = {'username': 'test', 'games': 12, 'wins': 5, 'rating': 99999,
            'games_2p': 3, 'games_4p': 9, 'raw_rating_2p': 3000, 'raw_rating_4p': 6000,
            'rating_reference_anchors_per_pc': {str(n): {'random': 1000, 'heuristic': raw}
                                              for n, raw in D.CAMPAIGN_HEURISTIC.items()}}
    expected = (3 * D.to_display(3000, 2) + 9 * D.to_display(6000, 4)) / 12
    assert public_profile(user)['rating'] == round(expected)
    user['wins'] = 4
    assert public_profile(user)['rating'] is None
    assert 'rating_2p' not in public_profile(user)
