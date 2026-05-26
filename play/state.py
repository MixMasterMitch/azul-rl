"""Game session state for the play server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from agent.env import actions as A
from agent.env.batched_engine import BatchedEngine


@dataclass
class GameSession:
    game_id: str
    num_players: int
    human_seat: int
    opponents: list[str]
    engine: BatchedEngine = field(init=False)

    def __post_init__(self):
        self.engine = BatchedEngine(
            batch_size=1,
            num_players=self.num_players,
            device="cpu",
        )

    def to_dict(self) -> dict:
        """Serialize game state for the frontend."""
        engine = self.engine
        b = 0

        # Build factory state
        factories = []
        for f in range(engine.num_factories):
            tiles = engine.factory_tiles[b, f].tolist()
            factories.append(tiles)

        center = engine.center_tiles[b].tolist()
        center_has_first = engine.center_first[b].item()

        # Build player states
        players = []
        for p in range(self.num_players):
            slots = engine.floor_slots[b, p].tolist()
            player_data = {
                "pattern_count": engine.pattern_count[b, p].tolist(),
                "pattern_color": engine.pattern_color[b, p].tolist(),
                "wall": engine.wall[b, p].tolist(),
                "floor_count": engine.floor_count[b, p].item(),
                "floor_slots": [
                    None if v == -1 else ("first" if v == A.FLOOR_MARKER else int(v))
                    for v in slots
                ],
                "score": engine.scores[b, p].item(),
            }
            players.append(player_data)

        # Legal actions for current player
        legal_mask = engine.legal_action_mask()
        legal_actions = legal_mask[b].nonzero(as_tuple=True)[0].tolist()
        legal_actions_named = [
            {"index": idx, "name": A.action_name(idx)}
            for idx in legal_actions
        ]

        return {
            "game_id": self.game_id,
            "num_players": self.num_players,
            "human_seat": self.human_seat,
            "opponents": self.opponents,
            "current_player": engine.current_player[b].item(),
            "factories": factories,
            "center": center,
            "center_has_first": center_has_first,
            "center_source": engine.num_factories,
            "players": players,
            "legal_actions": legal_actions_named,
            "ended": engine.ended[b].item(),
            "winner": engine.get_winners()[b].item() if engine.ended[b] else None,
        }
