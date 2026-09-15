"""Validated JSON API routes used by both Flask and Lambda."""
from __future__ import annotations

import json
import os
import time
from flask import Flask, Response, jsonify, request, g
from werkzeug.exceptions import HTTPException
from .auth import normalize_username
from .service import PlayService, ModelUnavailableError
from .store import ConflictError


def register_routes(app: Flask, service: PlayService) -> None:
    def identity() -> str:
        return normalize_username(request.headers.get("X-Azul-Username"))

    def body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def revision(data: dict) -> int:
        value = data.get("expected_revision")
        if type(value) is not int or value < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        return value

    @app.before_request
    def start_timing() -> None:
        g.started = time.perf_counter()

    @app.after_request
    def finish_timing(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        app.logger.info(json.dumps({"event": "request", "method": request.method,
                                    "route": str(request.url_rule), "status": response.status_code,
                                    "duration_ms": round((time.perf_counter() - g.started) * 1000, 3)}))
        return response

    @app.errorhandler(Exception)
    def api_error(exc: Exception) -> tuple[Response, int]:
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code or 500
        if isinstance(exc, (ValueError, ConflictError, PermissionError, KeyError, ModelUnavailableError)):
            code = (409 if isinstance(exc, ConflictError) else 403 if isinstance(exc, PermissionError)
                    else 404 if isinstance(exc, KeyError) else 503 if isinstance(exc, ModelUnavailableError) else 400)
            return jsonify(error=str(exc).strip("'")), code
        app.logger.exception("Unhandled API failure")
        return jsonify(error="An internal error occurred; reload and try again"), 500

    @app.get("/api/health")
    def health() -> Response:
        return jsonify(status="ok", release_id=os.environ.get("AZUL_RELEASE_ID", service.catalog.release_id))

    @app.get("/api/ready")
    def ready() -> Response:
        return jsonify(service.readiness(require_model=os.environ.get("AZUL_REQUIRE_MODEL") == "1"))

    @app.get("/api/me")
    def me() -> Response:
        return jsonify(service.profile(identity()))

    @app.get("/api/games")
    def games() -> Response:
        try:
            limit = int(request.args.get("limit", "20"))
        except ValueError as exc:
            raise ValueError("Invalid page limit") from exc
        return jsonify(service.list_games(identity(), request.args.get("status"), limit, request.args.get("cursor")))

    @app.get("/api/leaderboard")
    def leaderboard() -> Response:
        return jsonify(service.leaderboard())

    @app.post("/api/game")
    def create_game() -> tuple[Response, int]:
        data = body()
        state = service.create_game(data.get("num_players", 2), data.get("human_seat", 0),
                                    data.get("opponents"), username=identity())
        return jsonify(state), 201

    @app.get("/api/game/<game_id>")
    def get_game(game_id: str) -> Response:
        state = service.get_state(game_id, username=identity())
        if state is None:
            raise KeyError("Game not found")
        return jsonify(state)

    @app.post("/api/game/<game_id>/action")
    def apply_action(game_id: str) -> Response:
        data = body()
        state, error = service.apply_action(game_id, data.get("action"), username=identity(), expected_revision=revision(data))
        if error == "Game not found":
            raise KeyError(error)
        if error:
            raise ValueError(error)
        return jsonify(state)

    @app.post("/api/game/<game_id>/step-ai")
    def step_ai(game_id: str) -> Response:
        state = service.step_ai(game_id, username=identity(), expected_revision=revision(body()))
        if state is None:
            raise KeyError("Game not found")
        return jsonify(state)

    @app.post("/api/game/<game_id>/abandon")
    def abandon(game_id: str) -> Response:
        return jsonify(service.abandon(game_id, username=identity(), expected_revision=revision(body())))

    @app.get("/api/opponents")
    def opponents() -> Response:
        value = request.args.get("num_players")
        pc = int(value) if value is not None else None
        if pc is not None and pc not in (2, 3, 4):
            raise ValueError("num_players must be 2, 3, or 4")
        return jsonify(opponents=service.list_opponents(pc))
