from __future__ import annotations

import json

import pytest

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


@pytest.mark.parametrize("players", [2, 3, 4])
def test_score_breakdown_reconciles_each_turn_and_survives_resume(players: int) -> None:
    session = GameSession('scoring-test', players, 0, ['heuristic'] * (players - 1), seed=13579)
    bot = HeuristicBot(seed=24680)
    for turn in range(300):
        seat = int(session.engine.current_player[0])
        session.step(bot.select_action(session.engine, 0), seat)
        for item in session.score_breakdown():
            assert item['earlier_rounds'] == 0
            assert item['floor'] <= 0
            assert item['total'] == sum(value for key, value in item.items() if key != 'total')
        if turn == 20:
            session = GameSession.from_record(json.loads(json.dumps(session.to_record())))
        if bool(session.engine.ended[0]):
            assert any(item['rows'] > 0 for item in session.score_breakdown())
            legacy = session.to_record()
            legacy.pop('scoring')
            restored = GameSession.from_record(legacy)
            for item in restored.score_breakdown():
                assert item['total'] == sum(value for key, value in item.items() if key != 'total')
            break
    else:
        pytest.fail('Game did not finish')
