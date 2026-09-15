"""Authoritative game orchestration shared by local Flask and AWS Lambda."""
from __future__ import annotations

import logging
import json
import time
import uuid
from typing import Optional

import torch
from agent.env import actions as A
from agent.env.engine import BatchedEngine
from agent.eval.bots import RandomBot, HeuristicBot
from agent.eval.heuristic_opus import HeuristicOpusBot
from agent.eval.heuristic_astra import AstraUnavailableError, HeuristicAstraBot
from agent.net.model import AzulNet
from agent.train.checkpointing import load_net_from_checkpoint
from agent.train.model_registry import ModelRegistry
from agent.eval.arena import checkpoint_hash
from agent.search.config import SearchConfig
from agent.search.gumbel_mcts import gumbel_root_act
from .auth import normalize_username
from .models import ModelCatalog, serving_registry
from .ratings import fresh_user, public_profile, record_result
from .state import GameSession, now
from .store import ConflictError, PlayStore, JsonPlayStore

logger = logging.getLogger(__name__)


class ModelUnavailableError(Exception):
    """A pinned model cannot be served by this release."""


class PlayService:
    def __init__(self, store: Optional[PlayStore] = None, registry: ModelRegistry | None = None) -> None:
        self.store = store if store is not None else JsonPlayStore()
        self.registry = registry if registry is not None else serving_registry()
        self.catalog = ModelCatalog(self.registry)
        self._policy_cache: dict[str, AzulNet] = {}
        # Diagnostic last-loaded sessions only; requests always load the store.
        self.sessions: dict[str, GameSession] = {}

    def create_game(self, num_players: int = 2, human_seat: int = 0,
                    opponents: list[str] | None = None, *, username: str = "local") -> dict:
        username = normalize_username(username)
        if type(num_players) is not int or num_players not in (2, 3, 4):
            raise ValueError("num_players must be 2, 3, or 4")
        if type(human_seat) is not int or not 0 <= human_seat < num_players:
            raise ValueError("Invalid human seat")
        if opponents is None:
            opponents = ["heuristic"] * (num_players - 1)
        if not isinstance(opponents, list) or len(opponents) != num_players - 1:
            raise ValueError("One opponent is required for each non-human seat")
        infos = []
        for opponent in opponents:
            if not isinstance(opponent, str):
                raise ValueError("Invalid opponent identifier")
            info = self.catalog.resolve(opponent, num_players)
            if not info.get("available_for_new_games", True):
                raise ValueError("This opponent is available only for saved games")
            if "sha256" in info:
                self._get_or_load_net(info)
            infos.append({k: v for k, v in info.items() if k != "checkpoint"})
        session = GameSession(uuid.uuid4().hex, num_players, human_seat, opponents,
                              user_sub=username, opponent_info=infos)
        self.store.create_game(session.to_record())
        return self._view(session)

    def _view(self, session: GameSession) -> dict:
        # Bound this diagnostic cache independently of request correctness.
        if len(self.sessions) >= 32:
            self.sessions.pop(next(iter(self.sessions)))
        self.sessions[session.game_id] = session
        return session.to_dict()

    def _load(self, game_id: str, username: str, expected_revision: int | None = None) -> GameSession:
        username = normalize_username(username)
        record = self.store.load_game(game_id)
        if record is None:
            raise KeyError("Game not found")
        if record["user_sub"] != username:
            raise PermissionError("This game belongs to another screen name")
        if expected_revision is not None and record["revision"] != expected_revision:
            raise ConflictError("Game changed; reload before continuing")
        return GameSession.from_record(record)

    def get_state(self, game_id: str, *, username: str = "local") -> dict | None:
        try:
            return self._view(self._load(game_id, username))
        except KeyError:
            return None

    def _commit(self, session: GameSession) -> dict:
        expected = session.revision
        session.revision += 1
        session.updated_at = now()
        if bool(session.engine.ended[0]):
            session.status = "completed"
        if session.status == "completed" and not session.rating_recorded:
            session.rating_recorded = True
            record = session.to_record()
            # Other games under the same screen name may finish concurrently.
            # Retry only the rating fit; never reapply a move or overwrite a game.
            for _ in range(3):
                previous = self.store.load_user(session.user_sub)
                user = record_result(previous, record, self.catalog.reference_anchors)
                try:
                    self.store.commit_game(record, expected, user, previous["revision"] if previous else 0)
                    return self._view(session)
                except ConflictError:
                    current = self.store.load_game(session.game_id)
                    if current is None or current["revision"] != expected:
                        raise
            raise ConflictError("Player rating changed; reload and retry")
        self.store.commit_game(session.to_record(), expected)
        return self._view(session)

    def apply_action(self, game_id: str, action: int, *, username: str = "local",
                     expected_revision: int | None = None) -> tuple[dict | None, str | None]:
        try:
            session = self._load(game_id, username, expected_revision)
        except KeyError:
            return None, "Game not found"
        if session.status != "active":
            return None, "Game is no longer active"
        engine = session.engine
        if int(engine.current_player[0]) != session.human_seat:
            return None, "Not your turn"
        if type(action) is not int or not 0 <= action < A.NUM_ACTIONS:
            return None, "Action must be an integer from 0 to 299"
        if not bool(engine.legal_action_mask()[0, action]):
            return None, "Illegal action"
        engine.step(torch.tensor([action], dtype=torch.long))
        session.record_turn(session.human_seat)
        return self._commit(session), None

    def step_ai(self, game_id: str, *, username: str = "local",
                expected_revision: int | None = None) -> dict | None:
        try:
            session = self._load(game_id, username, expected_revision)
        except KeyError:
            return None
        if session.status != "active" or int(session.engine.current_player[0]) == session.human_seat:
            return self._view(session)
        seat = int(session.engine.current_player[0])
        index = [p for p in range(session.num_players) if p != session.human_seat].index(seat)
        info = session.opponent_info[index]
        seed = (session.seed + session.move_number * 100003 + seat) % (2**63 - 1)
        action = self._get_ai_action(session.engine, info["id"], info=info, seed=seed)
        if not bool(session.engine.legal_action_mask()[0, action]):
            raise RuntimeError("Opponent returned an illegal action")
        session.engine.step(torch.tensor([action], dtype=torch.long))
        session.record_turn(seat)
        return self._commit(session)

    def abandon(self, game_id: str, *, username: str, expected_revision: int) -> dict:
        session = self._load(game_id, username, expected_revision)
        if session.status != "active":
            raise ValueError("Only active games can be abandoned")
        if not session.can_abandon:
            raise ValueError("Games cannot be abandoned after every player has taken four turns")
        session.status = "abandoned"
        return self._commit(session)

    def _get_ai_action(self, engine: BatchedEngine, opponent_type: str,
                       *, info: dict | None = None, seed: int = 0) -> int:
        factories = {"random": RandomBot, "heuristic": HeuristicBot, "opus": HeuristicOpusBot, "astra": HeuristicAstraBot}
        started = time.perf_counter()
        if opponent_type in factories:
            try:
                action = factories[opponent_type](seed=seed).select_action(engine, 0)
            except AstraUnavailableError as exc:
                raise ModelUnavailableError(str(exc)) from exc
        else:
            entry = self.catalog.resolve(opponent_type, engine.num_players)
            if info and entry["sha256"] != info["sha256"]:
                raise ModelUnavailableError("The saved game's model differs from the bundled artifact")
            net = self._get_or_load_net(entry)
            settings = dict((info or entry)["search"])
            settings.update(seed=seed, move_deadline_s=min(settings.get("move_deadline_s") or 5.0, 5.0))
            with torch.inference_mode():
                actions, _ = gumbel_root_act(engine, net, search_config=SearchConfig(**settings))
            action = int(actions[0])
        logger.info(json.dumps({"event": "model_move", "model": opponent_type, "players": engine.num_players,
                                "duration_ms": round((time.perf_counter() - started) * 1000, 3)}))
        return action

    def _get_or_load_net(self, entry: dict) -> AzulNet:
        key = entry["sha256"]
        if key not in self._policy_cache:
            started = time.perf_counter()
            logger.info(json.dumps({"event": "model_load", "stage": "started", "sha256": key}))
            try:
                if checkpoint_hash(entry["checkpoint"]) != key:
                    raise ValueError("Registered model checksum mismatch")
                net, _ = load_net_from_checkpoint(entry["checkpoint"], map_location="cpu")
                net.eval()
                if len(self._policy_cache) >= 4:
                    self._policy_cache.pop(next(iter(self._policy_cache)))
                self._policy_cache[key] = net
                logger.info(json.dumps({"event": "model_load", "stage": "complete", "sha256": key,
                                        "duration_ms": round((time.perf_counter() - started) * 1000, 3)}))
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                raise ModelUnavailableError(f"Model artifact unavailable: {entry.get('id', entry.get('name'))}") from exc
        return self._policy_cache[key]

    def list_opponents(self, num_players: int | None = None) -> list[dict]:
        return self.catalog.list_opponents(num_players)

    def profile(self, username: str) -> dict:
        username = normalize_username(username)
        return public_profile(self.store.load_user(username) or fresh_user(username), self.catalog.reference_anchors)

    def list_games(self, username: str, status: str | None = None, limit: int = 20,
                   cursor: str | None = None) -> dict:
        if status not in {None, "active", "completed", "abandoned"}:
            raise ValueError("Invalid game status")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        records, token = self.store.list_games(normalize_username(username), status, limit, cursor)
        keys = ("game_id", "num_players", "human_seat", "status", "revision", "updated_at", "created_at", "scores", "winner_seats")
        games = [{**{k: record[k] for k in keys}, "opponent_names": [i["name"] for i in record["opponent_info"]]}
                 for record in records]
        return {"games": games, "next_cursor": token}

    def leaderboard(self) -> dict:
        entities = [{**bot, "label": bot["name"], "entity_id": bot["id"], "kind": "agent"}
                    for bot in self.list_opponents()]
        for user in self.store.list_users():
            profile = public_profile(user, self.catalog.reference_anchors)
            if profile["placed"] and profile["rating"] is not None:
                entities.append({**profile, "entity_id": f"human:{user['username']}", "label": user["username"], "kind": "human"})
        entities.sort(key=lambda row: (-(row.get("rating") or 0), row["label"]))
        return {"entities": entities}

    def readiness(self, require_model: bool = False) -> dict:
        if require_model and not self.catalog.default_model_id:
            raise ModelUnavailableError("The release has no default trained opponent")
        for model_id, entry in self.catalog.models.items():
            self._get_or_load_net(self.catalog.resolve(model_id, entry["trained_player_counts"][0]))
        self.store.load_user("_readiness")
        return {"status": "ready", "release_id": self.catalog.release_id, "models": list(self.catalog.models)}
