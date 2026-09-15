"""Game session state for the play server."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import base64
import secrets
from typing import Optional

import torch

from agent.env import actions as A
from agent.env.engine import GameEngine as BatchedEngine
from agent.env.batched_engine import BatchedEngine as TorchEngine, _STATE_TENSOR_ATTRS


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _winner_for_api(engine: BatchedEngine, batch_idx: int) -> Optional[int]:
    if not engine.ended[batch_idx]:
        return None
    w = int(engine.get_winners()[batch_idx].item())
    return w if w >= 0 else None


@dataclass
class GameSession:
    game_id: str
    num_players: int
    human_seat: int
    opponents: list[str]
    user_sub: str = "local"
    seed: int = field(default_factory=lambda: secrets.randbits(63))
    revision: int = 0
    move_number: int = 0
    turn_counts: list[int] | None = field(default_factory=list)
    status: str = "active"
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    opponent_info: list[dict] = field(default_factory=list)
    rating_recorded: bool = False
    engine: BatchedEngine = field(init=False)

    def __post_init__(self) -> None:
        if self.turn_counts == []:
            self.turn_counts = [0] * self.num_players
        self.engine = BatchedEngine(
            batch_size=1,
            num_players=self.num_players,
            device="cpu",
            seed=self.seed,
        )

    def record_turn(self, seat: int) -> None:
        self.move_number += 1
        if self.turn_counts is not None:
            self.turn_counts[seat] += 1

    @property
    def can_abandon(self) -> bool:
        if self.status != "active":
            return False
        if self.turn_counts is None:
            # Older saves have only a total move count. Apply the cutoff without
            # giving an established game a new abandonment window.
            return self.move_number < 4 * self.num_players
        return any(turns < 4 for turns in self.turn_counts)

    def winner_seats(self) -> list[int]:
        if not bool(self.engine.ended[0]):
            return []
        boards = [(int(self.engine.scores[0, p]), int(self.engine.wall[0, p].all(-1).sum()))
                  for p in range(self.num_players)]
        best = max(boards)
        return [p for p, score in enumerate(boards) if score == best]

    def to_record(self) -> dict:
        """Full internal snapshot, including private draw state; never sent to clients."""
        return {
            "schema_version": 2,
            **{k: getattr(self, k) for k in (
                "game_id", "num_players", "human_seat", "opponents", "user_sub", "seed", "revision",
                "move_number", "status", "created_at", "updated_at", "opponent_info", "rating_recorded")},
            "turn_counts": self.turn_counts,
            "winner_seats": self.winner_seats(),
            "scores": self.engine.scores[0, :self.num_players].tolist(),
            "engine": self.engine.state_dict(),
        }

    @classmethod
    def from_record(cls, record: dict) -> GameSession:
        if record.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported saved game schema")
        session = cls(**{k: record[k] for k in (
            "game_id", "num_players", "human_seat", "opponents", "user_sub", "seed", "revision",
            "move_number", "status", "created_at", "updated_at", "opponent_info", "rating_recorded")},
            turn_counts=record.get("turn_counts"))
        if record["schema_version"] == 2:
            session.engine = BatchedEngine.from_state_dict(record["engine"])
            if session.engine.batch_size != 1 or session.engine.num_players != session.num_players:
                raise ValueError("Saved game dimensions do not match session")
            return session

        # Existing hosted saves retain their exact future draws when migrated.
        legacy = TorchEngine(1, session.num_players, device="cpu", seed=session.seed)
        for name in _STATE_TENSOR_ATTRS:
            existing = getattr(legacy, name)
            restored = torch.tensor(record["engine"][name], dtype=existing.dtype)
            if restored.shape != existing.shape:
                raise ValueError(f"Invalid saved engine shape: {name}")
            setattr(legacy, name, restored)

        def restore_rng(value: str) -> torch.Generator:
            generator = torch.Generator(device="cpu")
            generator.set_state(torch.tensor(list(base64.b64decode(value, validate=True)), dtype=torch.uint8))
            return generator

        legacy._rng = restore_rng(record["rng"])
        legacy._game_rngs = ([restore_rng(value) for value in record["game_rngs"]]
                             if record.get("game_rngs") is not None else None)
        if legacy._game_rngs is not None and len(legacy._game_rngs) != 1:
            raise ValueError("Invalid saved per-game RNG count")
        session.engine = BatchedEngine.from_batched(legacy, preserve_rng=True)
        return session

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
            "revision": self.revision,
            "status": self.status,
            "move_number": self.move_number,
            "can_abandon": self.can_abandon,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "winner_seats": self.winner_seats(),
            "opponent_names": [entry["name"] for entry in self.opponent_info],
            "num_players": self.num_players,
            "human_seat": self.human_seat,
            "opponents": self.opponents,
            "current_player": engine.current_player[b].item(),
            "factories": factories,
            "center": center,
            "center_has_first": center_has_first,
            "center_source": engine.num_factories,
            "players": players,
            "legal_actions": legal_actions_named if self.status == "active" else [],
            "ended": engine.ended[b].item(),
            "winner": _winner_for_api(engine, b),
        }
