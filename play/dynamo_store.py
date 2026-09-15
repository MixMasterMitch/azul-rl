"""DynamoDB persistence with atomic game/rating finalization."""
from __future__ import annotations

import json
import os
import boto3
from boto3.dynamodb.conditions import Key, Attr
from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError
from .store import ConflictError, game_cursor, encode_cursor


class DynamoPlayStore:
    def __init__(self, games_table: str | None = None, users_table: str | None = None,
                 region: str | None = None) -> None:
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.games_table = self.dynamodb.Table(games_table or os.environ["GAMES_TABLE"])
        self.users_table = self.dynamodb.Table(users_table or os.environ["USERS_TABLE"])
        # Resource clients have document marshalling hooks; use a separate
        # low-level client for the explicitly serialized transaction items.
        self.client = boto3.client("dynamodb", region_name=region)

    @staticmethod
    def _game_item(data: dict) -> dict:
        return {**{k: data[k] for k in ("game_id", "user_sub", "updated_at", "status", "revision")},
                "data": json.dumps(data, separators=(",", ":"))}

    @staticmethod
    def _user_item(data: dict) -> dict:
        return {"username": data["username"], "revision": data["revision"],
                "data": json.dumps(data, separators=(",", ":"))}

    def create_game(self, data: dict) -> None:
        try:
            self.games_table.put_item(Item=self._game_item(data), ConditionExpression="attribute_not_exists(game_id)")
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise ConflictError("Game already exists") from exc
            raise

    def load_game(self, game_id: str) -> dict | None:
        item = self.games_table.get_item(Key={"game_id": game_id}, ConsistentRead=True).get("Item")
        return json.loads(item["data"]) if item else None

    def load_user(self, username: str) -> dict | None:
        item = self.users_table.get_item(Key={"username": username}, ConsistentRead=True).get("Item")
        return json.loads(item["data"]) if item else None

    def commit_game(self, data: dict, expected_revision: int, user: dict | None = None,
                    expected_user_revision: int | None = None) -> None:
        serializer = TypeSerializer()

        def put(table: str, item: dict, revision: int, new: bool = False) -> dict:
            return {"Put": {"TableName": table,
                            "Item": {k: serializer.serialize(v) for k, v in item.items()},
                            "ConditionExpression": "attribute_not_exists(#rev)" if new else "#rev = :revision",
                            "ExpressionAttributeNames": {"#rev": "revision"},
                            **({} if new else {"ExpressionAttributeValues": {":revision": {"N": str(revision)}}})}}

        transaction = [put(self.games_table.name, self._game_item(data), expected_revision)]
        if user is not None:
            if expected_user_revision is None:
                raise ValueError("Expected user revision is required")
            transaction.append(put(self.users_table.name, self._user_item(user), expected_user_revision,
                                   new=expected_user_revision == 0))
        try:
            self.client.transact_write_items(TransactItems=transaction)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "TransactionCanceledException" and any(
                reason.get("Code") in {"ConditionalCheckFailed", "TransactionConflict"}
                for reason in exc.response.get("CancellationReasons", [])
            ):
                raise ConflictError("Game or player rating changed; reload before continuing") from exc
            raise

    def list_games(self, username: str, status: str | None, limit: int,
                   cursor: str | None = None) -> tuple[list[dict], str | None]:
        after = game_cursor(cursor, username)
        args = {"IndexName": "user_sub-updated_at-index", "KeyConditionExpression": Key("user_sub").eq(username),
                "ScanIndexForward": False, "Limit": limit}
        if status:
            args["FilterExpression"] = Attr("status").eq(status)
        if after:
            args["ExclusiveStartKey"] = after
        response = self.games_table.query(**args)
        return ([json.loads(item["data"]) for item in response["Items"]],
                encode_cursor(response["LastEvaluatedKey"]) if response.get("LastEvaluatedKey") else None)

    def list_users(self) -> list[dict]:
        items: list[dict] = []
        args: dict = {"ConsistentRead": True}
        while True:
            response = self.users_table.scan(**args)
            items.extend(json.loads(item["data"]) for item in response["Items"])
            if not response.get("LastEvaluatedKey"):
                return items
            args["ExclusiveStartKey"] = response["LastEvaluatedKey"]
