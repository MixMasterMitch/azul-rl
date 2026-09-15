"""Authoritative stores with compare-and-swap game and rating transactions."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from typing import Iterator, Protocol


class ConflictError(Exception):
    """The submitted game or user revision is no longer current."""


def encode_cursor(value: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode()


def decode_cursor(value: str | None) -> dict | None:
    if not value:
        return None
    try:
        result = json.loads(base64.b64decode(value, altchars=b"-_", validate=True))
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("Invalid pagination cursor") from exc


def game_cursor(value: str | None, username: str) -> dict | None:
    after = decode_cursor(value)
    if after and (set(after) != {"user_sub", "updated_at", "game_id"}
                  or after["user_sub"] != username or not all(isinstance(v, str) for v in after.values())):
        raise ValueError("Invalid pagination cursor")
    return after


class PlayStore(Protocol):
    def create_game(self, data: dict) -> None: ...
    def load_game(self, game_id: str) -> dict | None: ...
    def commit_game(self, data: dict, expected_revision: int, user: dict | None = None,
                    expected_user_revision: int | None = None) -> None: ...
    def load_user(self, username: str) -> dict | None: ...
    def list_games(self, username: str, status: str | None, limit: int,
                   cursor: str | None = None) -> tuple[list[dict], str | None]: ...
    def list_users(self) -> list[dict]: ...


class JsonPlayStore:
    """Atomically replace one JSON document under a cross-process file lock.

    A single replacement commits both a finished game and its rating, so local
    processes cannot observe partial two-file transactions.
    """
    def __init__(self, base_dir: str = "play/play_data") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "store.json"

    @contextmanager
    def _locked(self, write: bool = False) -> Iterator[dict]:
        with (self.base_dir / ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {"games": {}, "users": {}}
            try:
                yield data
                if write:
                    temporary = self.path.with_suffix(".tmp")
                    with temporary.open("w") as target:
                        json.dump(data, target, separators=(",", ":"))
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temporary, self.path)
                    directory = os.open(self.base_dir, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def create_game(self, data: dict) -> None:
        with self._locked(write=True) as database:
            if data["game_id"] in database["games"]:
                raise ConflictError("Game already exists")
            database["games"][data["game_id"]] = data

    def load_game(self, game_id: str) -> dict | None:
        with self._locked() as database:
            return database["games"].get(game_id)

    def commit_game(self, data: dict, expected_revision: int, user: dict | None = None,
                    expected_user_revision: int | None = None) -> None:
        with self._locked(write=True) as database:
            previous = database["games"].get(data["game_id"])
            if previous is None or previous["revision"] != expected_revision:
                raise ConflictError("Game changed; reload before continuing")
            if user is not None:
                previous_user = database["users"].get(user["username"], {})
                if previous_user.get("revision", 0) != expected_user_revision:
                    raise ConflictError("Player rating changed; retry after reloading")
                database["users"][user["username"]] = user
            database["games"][data["game_id"]] = data

    def load_user(self, username: str) -> dict | None:
        with self._locked() as database:
            return database["users"].get(username)

    def list_games(self, username: str, status: str | None, limit: int,
                   cursor: str | None = None) -> tuple[list[dict], str | None]:
        after = game_cursor(cursor, username)
        with self._locked() as database:
            rows = [g for g in database["games"].values()
                    if g["user_sub"] == username and (status is None or g["status"] == status)]
        rows.sort(key=lambda g: (g["updated_at"], g["game_id"]), reverse=True)
        if after:
            rows = [g for g in rows if (g["updated_at"], g["game_id"]) < (after["updated_at"], after["game_id"])]
        page = rows[:limit]
        token = (encode_cursor({k: page[-1][k] for k in ("user_sub", "updated_at", "game_id")})
                 if len(rows) > limit else None)
        return page, token

    def list_users(self) -> list[dict]:
        with self._locked() as database:
            return list(database["users"].values())

    def save_game(self, game_id: str, data: dict) -> None:
        """Unconditional import helper; live mutations use commit_game."""
        with self._locked(write=True) as database:
            database["games"][game_id] = data

    def save_user(self, username: str, data: dict) -> None:
        with self._locked(write=True) as database:
            database["users"][username] = data
