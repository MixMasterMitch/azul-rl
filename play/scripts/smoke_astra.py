"""Local-only complete human/Astra games through the actual Lambda handler.

Use inside the built runtime image. This script cannot invoke live AWS.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import tempfile
import time
from typing import Any


def local_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """RIE-only adapter: invoke the real handler with an isolated local store.

    Run the image with CMD play.scripts.smoke_astra.local_handler. The production
    CMD remains play.lambda_handler.handler and selects the production store.
    """
    from play import lambda_handler
    if lambda_handler._app is None:
        from play.server import create_app
        from play.store import JsonPlayStore
        lambda_handler._app = create_app(JsonPlayStore("/tmp/astra-rie-store"))
    return lambda_handler.handler(event, context)


def run() -> dict:
    initialization_started = time.perf_counter()
    from play import lambda_handler
    from play.scripts.smoke import event
    from play.server import create_app
    from play.store import JsonPlayStore

    old_app = lambda_handler._app
    reports = []
    with tempfile.TemporaryDirectory(prefix="astra-smoke-") as directory:
        lambda_handler._app = create_app(JsonPlayStore(directory))
        initialization = time.perf_counter() - initialization_started
        def call(method: str, path: str, body: dict | None = None, status: int = 200) -> tuple[dict, float]:
            t = time.perf_counter()
            response = lambda_handler.handler(event(method, path, body, "astra-local-smoke"), None)
            elapsed = time.perf_counter() - t
            assert response["statusCode"] == status, response
            return json.loads(response["body"]), elapsed
        try:
            for n in (2, 3, 4):
                catalog, _ = call("GET", f"/opponents?num_players={n}")
                assert any(o["id"] == "astra" for o in catalog["opponents"])
                state, create_s = call("POST", "/game", {"num_players": n, "human_seat": n - 1,
                                         "opponents": ["astra"] * (n - 1)}, 201)
                rng = random.Random(101 + n)
                ai_times: list[float] = []
                consecutive: list[float] = []
                chain = 0.0
                for turn in range(400):
                    if state["status"] == "completed":
                        break
                    ai = state["current_player"] != state["human_seat"]
                    body = {"expected_revision": state["revision"]}
                    if not ai:
                        if chain:
                            consecutive.append(chain)
                            chain = 0.0
                        body["action"] = rng.choice(state["legal_actions"])["index"]
                    state, elapsed = call("POST", f"/game/{state['game_id']}/{'step-ai' if ai else 'action'}", body)
                    if ai:
                        ai_times.append(elapsed)
                        chain += elapsed
                assert state["status"] == "completed", f"Stalled {n}-player game"
                if chain:
                    consecutive.append(chain)
                ordered = sorted(ai_times)
                reports.append({"players": n, "turns": turn, "status": state["status"],
                                "create_s": create_s, "ai_requests": len(ai_times),
                                "ai_mean_s": sum(ai_times) / len(ai_times),
                                "ai_p95_s": ordered[math.ceil(len(ordered) * .95) - 1],
                                "ai_max_s": max(ai_times),
                                "consecutive_ai_chain_max_s": max(consecutive)})
                assert max(ai_times) < 2.0, reports[-1]
        finally:
            lambda_handler._app = old_app
    return {"application_initialization_s": initialization, "games": reports,
            "scope": "Local handler with a temporary JSON store; includes serialization and all consecutive AI requests; excludes AWS cold starts/network/DynamoDB."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
