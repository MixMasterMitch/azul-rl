"""Flask route handlers."""

from __future__ import annotations

from flask import Flask, jsonify, request

from .service import PlayService


def register_routes(app: Flask, service: PlayService) -> None:
    """Register all API routes."""

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/game", methods=["POST"])
    def create_game():
        data = request.get_json() or {}
        num_players = data.get("num_players", 2)
        human_seat = data.get("human_seat", 0)
        opponents = data.get("opponents", None)
        state = service.create_game(num_players, human_seat, opponents)
        return jsonify(state)

    @app.route("/api/game/<game_id>", methods=["GET"])
    def get_game(game_id: str):
        state = service.get_state(game_id)
        if state is None:
            return jsonify({"error": "Game not found"}), 404
        return jsonify(state)

    @app.route("/api/game/<game_id>/action", methods=["POST"])
    def apply_action(game_id: str):
        data = request.get_json() or {}
        action = data.get("action")
        if action is None:
            return jsonify({"error": "Missing action"}), 400
        state, err = service.apply_action(game_id, int(action))
        if err == "Game not found":
            return jsonify({"error": err}), 404
        if err is not None:
            return jsonify({"error": err}), 400
        return jsonify(state)

    @app.route("/api/game/<game_id>/step-ai", methods=["POST"])
    def step_ai(game_id: str):
        state = service.step_ai(game_id)
        if state is None:
            return jsonify({"error": "Game not found"}), 404
        return jsonify(state)

    @app.route("/api/opponents", methods=["GET"])
    def list_opponents():
        opponents = [
            {"id": "random", "name": "Random Bot", "rating": 1000},
            {"id": "heuristic", "name": "Heuristic Bot", "rating": 1500},
            {"id": "opus", "name": "Opus Bot", "rating": 1800},
        ]
        return jsonify({"opponents": opponents})
