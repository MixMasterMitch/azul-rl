"""Flask-based play server for Azul."""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

from .service import PlayService
from .views import register_routes


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")

    service = PlayService()
    register_routes(app, service)

    return app


def main() -> None:
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
