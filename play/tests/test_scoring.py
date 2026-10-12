from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from agent.eval.bots import HeuristicBot
from play.scoring import endgame_bonuses, placement_points
from play.state import GameSession


def test_bonus_categories_and_adjacency() -> None:
    assert endgame_bonuses([[True] * 5 for _ in range(5)]) == {
        "rows": 10, "columns": 35, "colors": 50}
    wall = [[False] * 5 for _ in range(5)]
    wall[2][1:4] = [True] * 3
    wall[1][2] = True
    wall[3][2] = True
    assert placement_points(wall, 2, 2) == 6


def test_completed_column_counts_in_running_score_once() -> None:
    session = GameSession('column-in-progress', 2, 0, ['heuristic'])
    wall = torch.zeros((1, 2, 5, 5), dtype=torch.bool)
    wall[0, 0, :, 1] = True
    session.scoring = {'placement': [23, 3], 'floor': [-7, -3]}
    session.engine = SimpleNamespace(wall=wall, scores=torch.tensor([[16, 0]]),
                                     ended=torch.tensor([False]))
    active = session.score_breakdown()[0]
    assert active == {'placement': 23, 'floor': -7, 'rows': 0, 'columns': 7,
                      'colors': 0, 'earlier_rounds': 0, 'total': 23}
    session.engine.scores[0, 0] = 23
    session.engine.ended[0] = True
    finished = session.score_breakdown()[0]
    assert finished['columns'] == 7
    assert finished['total'] == 23


@pytest.mark.parametrize("players", [2, 3, 4])
def test_score_breakdown_reconciles_each_turn_and_survives_resume(players: int) -> None:
    session = GameSession('scoring-test', players, 0, ['heuristic'] * (players - 1), seed=13579)
    bot = HeuristicBot(seed=24680)
    for turn in range(300):
        seat = int(session.engine.current_player[0])
        session.step(bot.select_action(session.engine, 0), seat)
        view = session.to_dict()
        for player, item in enumerate(session.score_breakdown()):
            assert item['earlier_rounds'] == 0
            assert item['floor'] <= 0
            bonuses = endgame_bonuses(session.engine.wall[0, player].tolist())
            assert all(item[key] == bonuses[key] for key in bonuses)
            assert item['total'] == (item['placement'] + item['floor'] + item['earlier_rounds']
                                     + sum(bonuses.values()))
            assert view['players'][player]['score'] == item['total']
        if turn == 20:
            session = GameSession.from_record(json.loads(json.dumps(session.to_record())))
        if bool(session.engine.ended[0]):
            assert any(item['rows'] > 0 for item in session.score_breakdown())
            legacy = session.to_record()
            legacy.pop('scoring')
            restored = GameSession.from_record(legacy)
            for item in restored.score_breakdown():
                assert item['total'] == (item['placement'] + item['floor'] + item['earlier_rounds']
                                         + item['rows'] + item['columns'] + item['colors'])
            break
    else:
        pytest.fail('Game did not finish')
