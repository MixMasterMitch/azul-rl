"""Play service: orchestrates game creation, actions, and AI stepping."""

from __future__ import annotations

import uuid
from typing import Optional

import torch

from agent.env import actions as A
from agent.env.batched_engine import BatchedEngine
from agent.eval.bots import RandomBot, HeuristicBot
from agent.eval.heuristic_opus import HeuristicOpusBot
from agent.net.encoder import encode_state
from agent.net.model import AzulNet
from agent.train.checkpointing import load_net_from_checkpoint

from .state import GameSession
from .store import PlayStore, JsonPlayStore


class PlayService:
    """Manages active game sessions and AI opponents."""

    def __init__(self, store: Optional[PlayStore] = None):
        self.store = store or JsonPlayStore()
        self.sessions: dict[str, GameSession] = {}
        self._policy_cache: dict[str, AzulNet] = {}
        self._bots = {
            "random": RandomBot(),
            "heuristic": HeuristicBot(),
            "opus": HeuristicOpusBot(),
        }

    def create_game(
        self,
        num_players: int = 2,
        human_seat: int = 0,
        opponents: list[str] | None = None,
    ) -> dict:
        """Create a new game session."""
        game_id = uuid.uuid4().hex[:12]
        opponents = opponents or ["heuristic"] * (num_players - 1)

        session = GameSession(
            game_id=game_id,
            num_players=num_players,
            human_seat=human_seat,
            opponents=opponents,
        )
        self.sessions[game_id] = session
        return session.to_dict()

    def get_state(self, game_id: str) -> Optional[dict]:
        session = self.sessions.get(game_id)
        if session is None:
            return None
        return session.to_dict()

    def apply_action(self, game_id: str, action: int) -> tuple[Optional[dict], Optional[str]]:
        """Apply a human player's action. Returns (state, error_message)."""
        session = self.sessions.get(game_id)
        if session is None:
            return None, "Game not found"

        engine = session.engine
        if engine.ended[0]:
            return session.to_dict(), None

        # Verify it's the human's turn
        cp = engine.current_player[0].item()
        if cp != session.human_seat:
            return None, "Not your turn"

        # Verify action is legal
        legal = engine.legal_action_mask()
        if not legal[0, action]:
            return None, "Illegal action"

        actions = torch.tensor([action], dtype=torch.long, device=engine.device)
        engine.step(actions)

        return session.to_dict(), None

    def step_ai(self, game_id: str) -> Optional[dict]:
        """Step all AI players until it's the human's turn again."""
        session = self.sessions.get(game_id)
        if session is None:
            return None

        engine = session.engine
        max_steps = 100  # safety limit

        for _ in range(max_steps):
            if engine.ended[0]:
                break
            cp = engine.current_player[0].item()
            if cp == session.human_seat:
                break

            # Determine which bot to use
            opponent_idx = self._get_opponent_index(session, cp)
            opponent_type = session.opponents[opponent_idx]

            action = self._get_ai_action(engine, opponent_type)
            actions = torch.tensor([action], dtype=torch.long, device=engine.device)
            engine.step(actions)

        return session.to_dict()

    def _get_opponent_index(self, session: GameSession, seat: int) -> int:
        """Map a seat index to the opponent list index."""
        opponent_seats = [s for s in range(session.num_players) if s != session.human_seat]
        return opponent_seats.index(seat)

    def _get_ai_action(self, engine: BatchedEngine, opponent_type: str) -> int:
        """Get an action for an AI opponent."""
        if opponent_type in self._bots:
            return self._bots[opponent_type].select_action(engine, 0)

        # ML bot: load checkpoint and use greedy policy
        if opponent_type.startswith("net:"):
            net = self._get_or_load_net(opponent_type)
            global_feat, source_feat = encode_state(engine)
            legal_mask = engine.legal_action_mask()
            with torch.no_grad():
                logits, _ = net(global_feat, source_feat, legal_mask, engine.num_players)
            return logits[0].argmax().item()

        # Default: random
        return self._bots["random"].select_action(engine, 0)

    def _get_or_load_net(self, key: str) -> AzulNet:
        if key not in self._policy_cache:
            path = key.replace("net:", "")
            net, _ = load_net_from_checkpoint(path, map_location="cpu")
            net.eval()
            self._policy_cache[key] = net
        return self._policy_cache[key]
