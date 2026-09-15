from __future__ import annotations

import pytest

pytest.importorskip("azul_astra")

from play.service import PlayService
from play.store import JsonPlayStore


@pytest.mark.parametrize("players", [2, 3, 4])
def test_astra_can_be_discovered_created_and_played(tmp_path, players: int) -> None:
    service = PlayService(store=JsonPlayStore(str(tmp_path)))
    opponent = next(o for o in service.list_opponents() if o["id"] == "astra")
    assert opponent["name"] == "Astra Heuristic"
    assert opponent["supported_players"] == [2, 3, 4]
    data = service.create_game(players, human_seat=players - 1, opponents=["astra"] * (players - 1))
    game_id = data["game_id"]
    for _ in range(players - 1):
        result = service.step_ai(game_id)
        assert result is not None
    engine = service.sessions[game_id].engine
    assert int(engine.current_player[0]) == players - 1
    assert not engine.ended[0]


def test_missing_extension_is_hidden_and_explained(tmp_path, monkeypatch) -> None:
    import play.models as models
    original = models.importlib.util.find_spec
    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: None if name == "azul_astra" else original(name))
    service = PlayService(store=JsonPlayStore(str(tmp_path)))
    assert "astra" not in {o["id"] for o in service.list_opponents()}
    with pytest.raises(ValueError, match="native extension"):
        service.create_game(2, opponents=["astra"])


def test_broken_native_installation_returns_explicit_service_unavailable(tmp_path, monkeypatch) -> None:
    from agent.eval.heuristic_astra import AstraUnavailableError, HeuristicAstraBot
    from play.server import create_app
    client = create_app(JsonPlayStore(str(tmp_path))).test_client()
    headers = {"X-Azul-Username": "astra-install-test"}
    state = client.post('/api/game', headers=headers,
                        json={"num_players": 2, "human_seat": 1, "opponents": ["astra"]}).get_json()
    def unavailable(self: HeuristicAstraBot, engine: object, game_idx: int) -> int:
        raise AstraUnavailableError("Astra native extension is unavailable")
    monkeypatch.setattr(HeuristicAstraBot, "select_action", unavailable)
    response = client.post(f"/api/game/{state['game_id']}/step-ai", headers=headers,
                           json={"expected_revision": 0})
    assert response.status_code == 503
    assert 'native extension' in response.get_json()['error']
