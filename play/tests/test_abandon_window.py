from __future__ import annotations

from copy import deepcopy

import pytest

from play.server import create_app
from play.service import PlayService
from play.state import GameSession
from play.tests.test_hosted_play import store, registry  # Reuse both persistence backends.


@pytest.mark.parametrize('players', [2, 3, 4])
def test_abandon_closes_after_each_players_fourth_turn(store, registry, players: int) -> None:
    service = PlayService(store, registry)
    game = service.create_game(players, opponents=['random'] * (players - 1), username='Alice')
    counts = [0] * players
    last_allowed = None
    for _ in range(40):
        assert game['can_abandon']
        last_allowed = deepcopy(store.load_game(game['game_id']))
        seat = game['current_player']
        counts[seat] += 1
        if seat == game['human_seat']:
            game, error = service.apply_action(game['game_id'], game['legal_actions'][0]['index'],
                username='Alice', expected_revision=game['revision'])
            assert error is None
        else:
            game = service.step_ai(game['game_id'], username='Alice', expected_revision=game['revision'])
        # A separate service must restore both the board and the turn cutoff.
        service = PlayService(store, registry)
        assert service.get_state(game['game_id'], username='Alice') == game
        assert store.load_game(game['game_id'])['turn_counts'] == counts
        if all(turns >= 4 for turns in counts):
            break
    else:
        pytest.fail('Players did not reach four turns')
    assert game['status'] == 'active' and not game['can_abandon']
    client = create_app(store, registry).test_client()
    headers = {'X-Azul-Username': 'Alice'}
    response = client.post(f"/api/game/{game['game_id']}/abandon",
        json={'expected_revision': game['revision']}, headers=headers)
    assert response.status_code == 400
    assert 'four turns' in response.get_json()['error']
    assert service.get_state(game['game_id'], username='Alice') == game
    assert store.load_user('Alice') is None
    # The same endpoint still permits abandonment immediately before the cutoff.
    last_allowed['game_id'] = 'before-cutoff'
    store.create_game(last_allowed)
    response = client.post('/api/game/before-cutoff/abandon',
        json={'expected_revision': last_allowed['revision']}, headers=headers)
    assert response.status_code == 200
    assert response.get_json()['status'] == 'abandoned'
    assert not response.get_json()['can_abandon']


def test_round_starters_do_not_replace_individual_turn_counts() -> None:
    game = GameSession('uneven', 3, 0, ['random', 'random'], move_number=12, turn_counts=[5, 4, 3])
    game = GameSession.from_record(game.to_record())
    assert game.can_abandon
    game.record_turn(2)
    assert not GameSession.from_record(game.to_record()).can_abandon


@pytest.mark.parametrize('players', [2, 3, 4])
def test_older_saves_keep_total_move_cutoff(players: int) -> None:
    game = GameSession('legacy', players, 0, ['random'] * (players - 1))
    record = game.to_record()
    record.pop('turn_counts')
    record['move_number'] = players * 4 - 1
    restored = GameSession.from_record(record)
    assert restored.can_abandon
    restored.record_turn(0)
    assert not GameSession.from_record(restored.to_record()).can_abandon
