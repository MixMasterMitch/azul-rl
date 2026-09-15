from __future__ import annotations
from dataclasses import asdict
import json
import pytest
import torch
from agent.net.model import AzulNet
from agent.train.checkpointing import save_checkpoint
from agent.train.model_registry import ModelRegistry
from agent.search.config import SearchConfig
from play.service import PlayService
from play.server import create_app
from play.store import JsonPlayStore


def test_registry_serves_search_and_enforces_player_count(tmp_path, monkeypatch) -> None:
    source = tmp_path/'source.pt'
    save_checkpoint(source, AzulNet(hidden=32, arch='source_attn'), config={'num_players': 2})
    registry = ModelRegistry(tmp_path/'models/registry.json')
    registry.register('trained_2p', str(source), SearchConfig(num_simulations=2, move_deadline_s=5))
    service = PlayService(store=JsonPlayStore(str(tmp_path/'games')), registry=registry)
    assert service.catalog.default_model_id == 'trained_2p'
    state = service.create_game(2, 1, ['trained_2p'])
    seen = []
    def search(engine, net, search_config):
        seen.append(search_config)
        return engine.legal_action_mask().long().argmax(1), torch.zeros(1, 300)
    monkeypatch.setattr('play.service.gumbel_root_act', search)
    service.step_ai(state['game_id'])
    assert seen and seen[0].num_simulations == 2
    with pytest.raises(ValueError): service.create_game(3, 0, ['trained_2p', 'random'])
    with pytest.raises(ValueError): service.create_game(2, 0, ['net:/arbitrary/path'])
    assert service.list_opponents()[-1]['supported_players'] == [2]


@pytest.mark.parametrize('payload', [[], {'num_players': 1}, {'num_players': True},
    {'num_players': 2, 'human_seat': -1}, {'num_players': 2, 'opponents': []},
    {'num_players': 2, 'opponents': ['unknown']}])
def test_invalid_game_requests_return_400(payload, tmp_path) -> None:
    client = create_app(store=JsonPlayStore(str(tmp_path))).test_client()
    assert client.post('/api/game', json=payload, headers={'X-Azul-Username': 'validation_test'}).status_code == 400


@pytest.mark.parametrize('action', [-1, 300, '4', 1.5, True])
def test_invalid_action_requests_return_400(action, tmp_path) -> None:
    client = create_app(store=JsonPlayStore(str(tmp_path))).test_client()
    headers = {'X-Azul-Username': 'validation_test'}
    created = client.post('/api/game', json={}, headers=headers)
    assert created.status_code == 201
    state = created.get_json()
    response = client.post(f"/api/game/{state['game_id']}/action", headers=headers,
                           json={'action': action, 'expected_revision': state['revision']})
    assert response.status_code == 400
    assert 'Action must be an integer' in response.get_json()['error']
