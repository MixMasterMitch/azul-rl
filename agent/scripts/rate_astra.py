"""Fit fresh per-player-count Bradley–Terry diagnostics from Astra raw records.

These use pairwise final (score, completed rows) placements, half credit for a
shared placement, and a weak Gaussian prior to keep separated results finite.
They are secondary diagnostics, not historical league ratings or win shares.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.eval.astra_tournament import atomic_json
from agent.train.ranking import add_match_result, fit_ratings_for_pc


def ratings(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    skipped = 0
    for g in records:
        if not g["finished"] or g["failure"]:
            skipped += 1
            continue
        for p in range(g["n"]):
            for q in range(p + 1, g["n"]):
                a, b = g["names"][p], g["names"][q]
                x = (g["scores"][p], g["rows"][p])
                y = (g["scores"][q], g["rows"][q])
                add_match_result(pairs, a, b, float(x > y), float(y > x),
                                 ties=float(x == y), num_players=g["n"])
    result: dict[str, Any] = {}
    for n in (2, 3, 4):
        graph: dict[str, set[str]] = {}
        for pair in pairs:
            if sum(pair.get(f"{k}_{n}p", 0) for k in ("wins_a", "wins_b", "ties")):
                a, b = pair["a"], pair["b"]
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
        connected = {"random"} if "random" in graph else set()
        while True:
            expanded = connected | {q for p in connected for q in graph[p]}
            if expanded == connected:
                break
            connected = expanded
        usable = [p for p in pairs if p["a"] in connected and p["b"] in connected]
        result[str(n)] = {
            "ratings": fit_ratings_for_pc(usable, n, anchors={"random": 1000.0},
                                           use_reference_anchors=False, prior_sigma=10000.0)
                       if connected else {},
            "unanchored_participants": sorted(set(graph) - connected),
        }
    return {"per_player_count": result, "pairwise_records": pairs,
            "unfinished_or_failed_games_skipped": skipped,
            "method": "Bradley–Terry L-BFGS; random=1000; base-10 rating scale=1000; Gaussian prior mean=1500 sigma=10000; pairwise score/row placements; ties=0.5",
            "limitations": "Correlated multiplayer pairs and near-separated opponents make these secondary diagnostics. Do not compare directly with historical league ratings."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    games = [json.loads(line) for path in args.records for line in path.read_text().splitlines()]
    report = ratings(games)
    atomic_json(args.output, report)
    print(json.dumps(report["per_player_count"], indent=2))


if __name__ == "__main__":
    main()
