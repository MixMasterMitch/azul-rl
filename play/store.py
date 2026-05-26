"""Game storage backends."""

from __future__ import annotations

import json
import pathlib
from typing import Optional, Protocol


class PlayStore(Protocol):
    def save_game(self, game_id: str, data: dict) -> None: ...
    def load_game(self, game_id: str) -> Optional[dict]: ...
    def save_user(self, username: str, data: dict) -> None: ...
    def load_user(self, username: str) -> Optional[dict]: ...


class JsonPlayStore:
    """File-based storage for local development."""

    def __init__(self, base_dir: str = "play/play_data"):
        self.base_dir = pathlib.Path(base_dir)
        self.games_dir = self.base_dir / "games"
        self.users_dir = self.base_dir / "users"
        self.games_dir.mkdir(parents=True, exist_ok=True)
        self.users_dir.mkdir(parents=True, exist_ok=True)

    def save_game(self, game_id: str, data: dict) -> None:
        path = self.games_dir / f"{game_id}.json"
        path.write_text(json.dumps(data, indent=2))

    def load_game(self, game_id: str) -> Optional[dict]:
        path = self.games_dir / f"{game_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def save_user(self, username: str, data: dict) -> None:
        path = self.users_dir / f"{username}.json"
        path.write_text(json.dumps(data, indent=2))

    def load_user(self, username: str) -> Optional[dict]:
        path = self.users_dir / f"{username}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())
