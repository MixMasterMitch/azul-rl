"""Fixed opponent ladder and capped human head-to-head ratings for play."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from agent.train import ranking as R
from .human_rating import _normalize_entity

VERSION = "fixed-opponents-v1"
FLOOR = 1000.0
CEILING = 3500.0
PLACEMENT_WINS = 5
AGENT_ID = "net:shared:338"
POINTS = {"random": 1000.0, "heuristic": 2000.0,
          "heuristic_opus": 2250.0, "astra": 2500.0, AGENT_ID: 3000.0}
RUNGS = (1000.0, 2000.0, 2250.0, 2500.0, 3000.0, 3500.0)


def references() -> dict[int, dict[str, float]]:
    return {pc: dict(POINTS) for pc in R.PLAYER_COUNTS}


def is_fixed(values: Mapping | None) -> bool:
    if not values:
        return False
    try:
        return all({str(name): float(score) for name, score in
                    values.get(pc, values.get(str(pc))).items()} == POINTS
                   for pc in R.PLAYER_COUNTS)
    except (AttributeError, TypeError, ValueError):
        return False


def clamp(value: float) -> float:
    return max(FLOOR, min(CEILING, value))


def fit(results: list[dict], pc: int) -> float | None:
    relevant = [row for row in results if "human" in (row.get("a"), row.get("b"))
                and row.get("a") in (*POINTS, "human")
                and row.get("b") in (*POINTS, "human")
                and sum(row.get(f"{key}_{pc}p", 0) for key in
                        ("wins_a", "wins_b", "ties")) > 0]
    if not relevant:
        return None
    rating = R.fit_ratings_for_pc(relevant, pc, anchors=POINTS,
                                  use_reference_anchors=False).get("human")
    strongest = max(POINTS[row["b"] if row["a"] == "human" else row["a"]]
                    for row in relevant)
    ceiling = RUNGS[RUNGS.index(strongest) + 1]
    return None if rating is None else min(ceiling, clamp(rating))


def _head_to_head(game: dict, human: int, bot: int) -> int:
    scores = game.get("scores")
    if scores and len(scores) == game["num_players"] and scores[human] != scores[bot]:
        return (scores[human] > scores[bot]) - (scores[human] < scores[bot])
    winners = game["winner_seats"]
    return (human in winners) - (bot in winners)


def record_result(previous: dict | None, game: dict) -> dict:
    user = deepcopy(previous) if previous else {
        "username": game["user_sub"], "revision": 0, "games": 0,
        "wins": 0, "results": [], "anchors_per_pc": {}, "rating": None,
    }
    pc = game["num_players"]
    human = game["human_seat"]
    winners = game["winner_seats"]
    user["revision"] += 1
    user["games"] += 1
    user["wins"] += int(human in winners)
    user[f"games_{pc}p"] = user.get(f"games_{pc}p", 0) + 1
    user[f"wins_{pc}p"] = user.get(f"wins_{pc}p", 0) + int(human in winners)
    seats = [seat for seat in range(pc) if seat != human]
    for seat, info in zip(seats, game["opponent_info"], strict=True):
        entity = _normalize_entity(info["id"])
        if entity not in POINTS:
            continue
        result = _head_to_head(game, human, seat)
        R.add_match_result(user["results"], "human", entity,
                           int(result > 0), int(result < 0),
                           ties=int(result == 0), num_players=pc)
    user["anchors_per_pc"] = {str(n): dict(POINTS) for n in R.PLAYER_COUNTS}
    user["rating_reference_anchors_per_pc"] = references()
    user["rating_display"] = {"version": VERSION, "floor": FLOOR,
                              "ceiling": CEILING, "anchors": dict(POINTS)}
    total = weight = 0.0
    for n in R.PLAYER_COUNTS:
        value = fit(user["results"], n)
        if value is None:
            user.pop(f"raw_rating_{n}p", None)
            user.pop(f"rating_{n}p", None)
            continue
        user[f"raw_rating_{n}p"] = value
        user[f"rating_{n}p"] = value
        games = user.get(f"games_{n}p", 0)
        total += value * games
        weight += games
    user["rating"] = total / weight if weight else None
    return user


def public_profile(user: dict) -> dict:
    # Refit old profiles from their saved match records; never mutate stored data.
    same_version = (user.get("rating_display", {}).get("version") == VERSION
                    and is_fixed(user.get("rating_reference_anchors_per_pc")))
    values = {}
    total = weight = 0.0
    for pc in R.PLAYER_COUNTS:
        value = user.get(f"rating_{pc}p") if same_version else fit(user.get("results", []), pc)
        if value is None:
            continue
        value = clamp(float(value))
        values[pc] = value
        games = user.get(f"games_{pc}p", 0)
        total += value * games
        weight += games
    placed = user["wins"] >= PLACEMENT_WINS
    result = {key: user[key] for key in ("username", "games", "wins")}
    result.update(placed=placed, placement_wins_required=PLACEMENT_WINS,
                  rating=round(total / weight) if placed and weight else None)
    for pc in R.PLAYER_COUNTS:
        for key in (f"games_{pc}p", f"wins_{pc}p"):
            result[key] = user.get(key, 0)
        if placed and pc in values:
            result[f"rating_{pc}p"] = round(values[pc])
    return result
