"""Flask application factory shared by local development and AWS Lambda."""
from __future__ import annotations

import logging
import os
import torch
from flask import Flask
from agent.train.model_registry import ModelRegistry
from .service import PlayService
from .store import PlayStore, JsonPlayStore
from .views import register_routes


def create_app(store: PlayStore | None = None, registry: ModelRegistry | None = None) -> Flask:
    torch.set_num_threads(int(os.environ.get("AZUL_TORCH_THREADS", "1")))
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # Another application in this interpreter already started Torch work.
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16384
    if store is None:
        if os.environ.get("GAMES_TABLE") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            from .dynamo_store import DynamoPlayStore
            store = DynamoPlayStore()
        else:
            store = JsonPlayStore(os.environ.get("AZUL_PLAY_DATA", "play/play_data"))
    service = PlayService(store=store, registry=registry)
    app.extensions["play_service"] = service
    register_routes(app, service)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = create_app()
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()
