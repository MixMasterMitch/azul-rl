from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import boto3
from moto import mock_aws
import pytest
import torch

from agent.env.batched_engine import _STATE_TENSOR_ATTRS
from agent.env.engine import GameEngine as BatchedEngine
from agent.eval.bots import HeuristicBot
from agent.train.model_registry import ModelRegistry
from play.dynamo_store import DynamoPlayStore
from play.ratings import fresh_user, record_result, public_profile
from play.server import create_app
from play.service import PlayService, ModelUnavailableError
from play.state import GameSession
from play.store import JsonPlayStore, ConflictError


@pytest.fixture(params=['json', 'dynamo'])
def store(request, tmp_path, monkeypatch):
    if request.param == 'json':
        yield JsonPlayStore(str(tmp_path / 'data'))
        return
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-west-2')
    with mock_aws():
        db = boto3.client('dynamodb', region_name='us-west-2')
        db.create_table(TableName='games', BillingMode='PAY_PER_REQUEST',
            KeySchema=[{'AttributeName': 'game_id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': k, 'AttributeType': 'S'} for k in ['game_id', 'user_sub', 'updated_at']],
            GlobalSecondaryIndexes=[{'IndexName': 'user_sub-updated_at-index',
                'KeySchema': [{'AttributeName':'user_sub','KeyType':'HASH'}, {'AttributeName':'updated_at','KeyType':'RANGE'}],
                'Projection': {'ProjectionType':'ALL'}}])
        db.create_table(TableName='users', BillingMode='PAY_PER_REQUEST',
            KeySchema=[{'AttributeName':'username','KeyType':'HASH'}],
            AttributeDefinitions=[{'AttributeName':'username','AttributeType':'S'}])
        yield DynamoPlayStore('games', 'users', 'us-west-2')


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / 'empty-registry.json')


def test_fresh_service_restores_and_rejects_stale_mutations(store, registry):
    service = PlayService(store, registry)
    game = service.create_game(username='Alice')
    second = PlayService(store, registry)
    assert second.get_state(game['game_id'], username='Alice') == game
    action = game['legal_actions'][0]['index']
    next_state, error = second.apply_action(game['game_id'], action, username='Alice', expected_revision=0)
    assert error is None and next_state['revision'] == 1
    with pytest.raises(ConflictError):
        service.apply_action(game['game_id'], action, username='Alice', expected_revision=0)
    with pytest.raises(PermissionError):
        second.get_state(game['game_id'], username='alice')
    third = PlayService(store, registry)
    stepped = third.step_ai(game['game_id'], username='Alice', expected_revision=1)
    assert stepped['revision'] == 2 and stepped['move_number'] == 2
    assert third.step_ai(game['game_id'], username='Alice', expected_revision=2) == stepped


@pytest.mark.parametrize('players', [2, 3, 4])
def test_snapshot_full_game_reproduces_rng_and_refills(players):
    original = GameSession('test', players, 0, ['heuristic'] * (players - 1))
    original.engine = BatchedEngine(1, players, seed=123, game_seeds=[789])
    bot = HeuristicBot(seed=100)
    for turn in range(400):
        restored = GameSession.from_record(json.loads(json.dumps(original.to_record())))
        for attr in _STATE_TENSOR_ATTRS:
            assert torch.equal(getattr(original.engine, attr), getattr(restored.engine, attr))
        assert original.to_record()['engine'] == restored.to_record()['engine']
        if original.engine.ended[0]:
            assert original.winner_seats() == restored.winner_seats()
            assert turn > 10
            break
        action = torch.tensor([bot.select_action(original.engine, 0)])
        original.engine.step(action)
        restored.engine.step(action)
        assert original.to_record() == restored.to_record()
    else:
        pytest.fail('Game did not finish')


def finish(service, game_id, winners=(0,)):
    session = GameSession.from_record(service.store.load_game(game_id))
    reference = session.engine.native.to_batched()
    reference.ended[0] = True
    reference.scores.zero_()
    reference.bag += reference.factory_tiles.sum(1).to(torch.int8)
    reference.factory_tiles.zero_()
    for seat in winners:
        reference.scores[0, seat] = 30
        reference.wall[0, seat, 0] = True
        reference.bag[0] -= 1
    session.engine = BatchedEngine.from_batched(reference)
    return session


def test_completion_is_atomic_and_exactly_once(store, registry):
    service = PlayService(store, registry)
    game = service.create_game(username='Alice')
    final = finish(service, game['game_id'])
    duplicate = deepcopy(final)
    result = service._commit(final)
    assert result['status'] == 'completed'
    assert store.load_user('Alice')['games'] == 1
    assert store.load_game(game['game_id'])['rating_recorded'] is True
    with pytest.raises(ConflictError):
        service._commit(duplicate)
    assert store.load_user('Alice')['games'] == 1
    service.step_ai(game['game_id'], username='Alice', expected_revision=result['revision'])
    assert store.load_user('Alice')['games'] == 1


def test_failed_user_condition_never_commits_game(store, registry):
    service = PlayService(store, registry)
    first = service.create_game(username='Alice')
    service._commit(finish(service, first['game_id']))
    second = service.create_game(username='Alice')
    record = store.load_game(second['game_id'])
    record['revision'] += 1
    with pytest.raises(ConflictError):
        store.commit_game(record, 0, {**fresh_user('Alice'), 'revision': 1}, expected_user_revision=0)
    assert store.load_game(second['game_id'])['revision'] == 0
    assert store.load_user('Alice')['games'] == 1


def test_interleaved_finishes_refit_latest_user(store, registry, monkeypatch):
    service = PlayService(store, registry)
    first = service.create_game(username='Alice')
    second = service.create_game(username='Alice')
    commit = store.commit_game
    interleaved = False
    def racing_commit(data, expected_revision, user=None, expected_user_revision=None):
        nonlocal interleaved
        if not interleaved:
            interleaved = True
            service._commit(finish(service, second['game_id']))
        commit(data, expected_revision, user, expected_user_revision)
    monkeypatch.setattr(store, 'commit_game', racing_commit)
    service._commit(finish(service, first['game_id']))
    assert store.load_user('Alice')['games'] == 2


def test_history_pagination_and_abandonment(store, registry):
    service = PlayService(store, registry)
    games = [service.create_game(username='Alice') for _ in range(5)]
    service.create_game(username='Bob')
    service.abandon(games[0]['game_id'], username='Alice', expected_revision=0)
    assert store.load_user('Alice') is None
    ids, cursor = [], None
    for _ in range(10):
        page = service.list_games('Alice', status='active', limit=2, cursor=cursor)
        ids.extend(row['game_id'] for row in page['games'])
        cursor = page['next_cursor']
        if cursor is None: break
    assert len(ids) == len(set(ids)) == 4
    assert games[0]['game_id'] not in ids
    page = service.list_games('Alice', limit=1)
    with pytest.raises(ValueError):
        service.list_games('Bob', cursor=page['next_cursor'])
    with pytest.raises(ValueError):
        service.list_games('Alice', cursor='bad!')


def test_api_identity_revisions_and_private_state(store, registry):
    client = create_app(store, registry).test_client()
    assert client.post('/api/game', json={}).status_code == 400
    for name in ['../bad', 'bad name', '', 'a'*33]:
        assert client.post('/api/game', json={}, headers={'X-Azul-Username': name}).status_code == 400
    headers = {'X-Azul-Username': ' Alice '}
    response = client.post('/api/game', json={}, headers=headers)
    assert response.status_code == 201
    game = response.get_json()
    assert not {'engine', 'rng', 'seed', 'bag', 'game_rngs'}.intersection(game)
    path = f"/api/game/{game['game_id']}/action"
    assert client.post(path, json={'action': 0}, headers=headers).status_code == 400
    move = {'action': game['legal_actions'][0]['index'], 'expected_revision': game['revision']}
    assert client.post(path, json=move, headers=headers).status_code == 200
    assert client.post(path, json=move, headers=headers).status_code == 409
    assert client.get(f"/api/game/{game['game_id']}", headers={'X-Azul-Username':'Bob'}).status_code == 403
    assert client.get('/api/health').headers['Cache-Control'] == 'no-store'


def test_shared_victory_and_placement(store, registry):
    service = PlayService(store, registry)
    for _ in range(5):
        game = service.create_game(3, opponents=['random', 'random'], username='Alice')
        result = service._commit(finish(service, game['game_id'], winners=(0, 1)))
        assert result['winner_seats'] == [0, 1] and result['winner'] is None
    user = store.load_user('Alice')
    row = user['results'][0]
    assert row['ties_3p'] == 5
    assert row.get('wins_a_3p', 0) + row.get('wins_b_3p', 0) == 5
    profile = service.profile('Alice')
    assert profile['wins'] == 5 and profile['placed'] and profile['rating'] is not None
    assert any(row['label'] == 'Alice' for row in service.leaderboard()['entities'])


def test_horizontal_rows_break_score_tie():
    game = GameSession('tie', 2, 0, ['random'])
    reference = game.engine.native.to_batched()
    reference.ended[0] = True
    reference.bag += reference.factory_tiles.sum(1).to(torch.int8)
    reference.factory_tiles.zero_()
    reference.scores[0, :2] = 50
    reference.wall[0, 1, 0] = True
    reference.bag[0] -= 1  # Place one tile of each color without changing inventory.
    game.engine = BatchedEngine.from_batched(reference)
    assert game.winner_seats() == [1]


def test_real_release_model_search_and_player_count(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'artifacts/registry.json'
    if not path.exists():
        pytest.skip('Run python -m play.scripts.prepare_release for artifact integration tests')
    service = PlayService(JsonPlayStore(str(tmp_path)), ModelRegistry(path))
    game = service.create_game(2, 1, ['net:league:294'])
    result = service.step_ai(game['game_id'], expected_revision=0)
    assert result['revision'] == 1 and result['current_player'] == 1
    assert 'net:league:294' not in {o['id'] for o in service.list_opponents(3)}
    with pytest.raises(ValueError):
        service.create_game(3, 0, ['net:league:294', 'random'])
    with pytest.raises(ValueError):
        service.create_game(opponents=['net:/tmp/file.pt'])
    assert service.readiness(True)['models'] == ['net:league:294']


@pytest.mark.parametrize('players', [2, 3, 4])
@pytest.mark.parametrize('per_game', [False, True])
def test_legacy_saved_session_migrates_and_keeps_future_draws(players: int, per_game: bool) -> None:
    import base64
    from agent.env.batched_engine import BatchedEngine as TorchEngine
    game = GameSession('legacy', players, 0, ['heuristic'] * (players - 1), seed=984)
    reference = TorchEngine(1, players, seed=984, game_seeds=[593] if per_game else None)
    bot = HeuristicBot(seed=12)
    for _ in range(13):
        reference.step(torch.tensor([bot.select_action(reference, 0)]))
    record = game.to_record()
    record['schema_version'] = 1
    record['engine'] = {name: getattr(reference, name).tolist() for name in _STATE_TENSOR_ATTRS}
    def pack(rng: torch.Generator) -> str:
        return base64.b64encode(bytes(rng.get_state().tolist())).decode()
    record['rng'] = pack(reference._rng)
    record['game_rngs'] = [pack(rng) for rng in reference._game_rngs] if per_game else None
    game = GameSession.from_record(json.loads(json.dumps(record)))
    assert isinstance(game.engine, BatchedEngine)
    assert game.to_record()['schema_version'] == 2
    for turn in range(400):
        for name in _STATE_TENSOR_ATTRS:
            assert torch.equal(getattr(game.engine, name), getattr(reference, name)), (turn, name)
        if reference.ended[0]:
            break
        action = torch.tensor([bot.select_action(reference, 0)])
        game.engine.step(action)
        reference.step(action)
        # Save/reload repeatedly, including across refills and completion.
        game = GameSession.from_record(json.loads(json.dumps(game.to_record())))
    else:
        pytest.fail('Migrated game did not finish')
    assert 'legacy_rng' not in game.to_dict()
