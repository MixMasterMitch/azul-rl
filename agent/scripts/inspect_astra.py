"""Review recorded Astra losses with deeper public-information analysis.

Requires a tournament run made with --trace. This is a diagnostic, not an
oracle: a deeper search can still have a worse strategic estimate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from typing import Any

from agent.eval.heuristic_astra import native_module, native_options, pending_game_end, production_config


def describe(action: int, players: int) -> str:
    source, remainder = divmod(action, 30)
    color, target = divmod(remainder, 6)
    origin = "center" if source == 2 * players + 1 else f"factory {source + 1}"
    destination = "floor" if target == 5 else f"line {target + 1}"
    return f"{origin}: {('blue', 'yellow', 'red', 'black', 'white')[color]} → {destination}"


def inspect(records: list[dict[str, Any]], limit: int, nodes: int, *,
            terminal_round_only: bool = False, centers_only: bool = False) -> dict[str, Any]:
    native = native_module()
    disagreements = []
    inspected = 0
    losses = 0
    for game in records:
        if not game["finished"] or game["win_share"] != 0:
            continue
        losses += 1
        rng = random.Random(game["bot_seed"])
        bot_seed = [rng.randrange(1 << 32) for _ in range(game["n"])][game["seat"]]
        for turn, move in enumerate(game.get("trajectory", [])):
            if move["seat"] != game["seat"] or inspected >= limit:
                continue
            if terminal_round_only and not pending_game_end(move["state"]):
                continue
            if centers_only and any(move["state"][7:52]):
                continue
            cfg = replace(production_config(game["n"]), nodes=nodes)
            options = native_options(cfg, move["state"])
            result = native.analyze(move["state"], seed=bot_seed, **options)
            inspected += 1
            if result["action"] != move["action"]:
                disagreements.append({
                    "game": game["key"], "players": game["n"], "turn": turn,
                    "bot_seed": bot_seed,
                    "final_scores": game["scores"], "snapshot": move["state"],
                    "played_action": move["action"],
                    "played_description": describe(move["action"], game["n"]),
                    "deeper_description": describe(result["action"], game["n"]),
                    "original_analysis": move.get("analysis"), "deeper_analysis": result,
                })
        if inspected >= limit:
            break
    return {"inspected_positions": inspected, "losses_visited": losses,
            "terminal_round_only": terminal_round_only, "centers_only": centers_only,
            "deeper_nodes": nodes, "disagreements": disagreements,
            "configs_per_pc": {str(n): asdict(replace(production_config(n), nodes=nodes)) for n in (2, 3, 4)},
            "limitations": "Deeper choices use current production weights and the original bot seed; they are hypotheses, not proven corrections. Only resolved current-round searches certify a tactical result; future-round values remain heuristic."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=256000)
    parser.add_argument("--players", type=int, nargs="+", choices=(2, 3, 4), default=[2, 3, 4])
    parser.add_argument("--terminal-round-only", action="store_true")
    parser.add_argument("--centers-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    records = [g for g in records if g["n"] in args.players]
    if not any(g.get("trajectory") for g in records):
        parser.error("Input has no trajectories; run eval_astra with --trace")
    report = inspect(records, args.limit, args.nodes,
                     terminal_round_only=args.terminal_round_only, centers_only=args.centers_only)
    report["source_records"] = str(args.records.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"positions": report["inspected_positions"],
                      "disagreements": len(report["disagreements"])}))


if __name__ == "__main__":
    main()
