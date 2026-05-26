"""DynamoDB-based storage for AWS Lambda deployment."""

from __future__ import annotations

import json
import os
from typing import Optional

import boto3

from .store import PlayStore


class DynamoPlayStore:
    """DynamoDB storage backend for production (Lambda)."""

    def __init__(self):
        self.dynamodb = boto3.resource("dynamodb")
        self.games_table = self.dynamodb.Table(os.environ.get("GAMES_TABLE", "AzulGames"))
        self.users_table = self.dynamodb.Table(os.environ.get("USERS_TABLE", "AzulUsers"))

    def save_game(self, game_id: str, data: dict) -> None:
        item = {"game_id": game_id, "data": json.dumps(data)}
        if "user_sub" in data:
            item["user_sub"] = data["user_sub"]
        self.games_table.put_item(Item=item)

    def load_game(self, game_id: str) -> Optional[dict]:
        response = self.games_table.get_item(Key={"game_id": game_id})
        item = response.get("Item")
        if item is None:
            return None
        return json.loads(item["data"])

    def save_user(self, username: str, data: dict) -> None:
        self.users_table.put_item(Item={
            "username": username,
            "data": json.dumps(data),
        })

    def load_user(self, username: str) -> Optional[dict]:
        response = self.users_table.get_item(Key={"username": username})
        item = response.get("Item")
        if item is None:
            return None
        return json.loads(item["data"])
