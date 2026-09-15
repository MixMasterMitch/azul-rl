"""Pure game-result updates; persistence and exactly-once commits belong to the store."""
from __future__ import annotations

from copy import deepcopy
from agent.train import ranking as R
from agent.train import rating_display as D
from .human_rating import _normalize_entity

PLACEMENT_WINS = 5


def fresh_user(username: str) -> dict:
    return {"username": username, "revision": 0, "games": 0, "wins": 0, "results": [],
            "anchors_per_pc": {}, "rating": None}


def record_result(previous: dict | None, game: dict, reference_anchors: dict[int, dict[str, float]]) -> dict:
    user = deepcopy(previous or fresh_user(game["user_sub"]))
    pc = game["num_players"]
    human = game["human_seat"]
    winners = game["winner_seats"]
    user["revision"] += 1
    user["games"] += 1
    user["wins"] += int(human in winners)
    user[f"games_{pc}p"] = user.get(f"games_{pc}p", 0) + 1
    user[f"wins_{pc}p"] = user.get(f"wins_{pc}p", 0) + int(human in winners)
    anchors = user["anchors_per_pc"].setdefault(str(pc), {"random": R.RANDOM_ANCHOR_RATING})
    seats = [seat for seat in range(pc) if seat != human]
    for seat, info in zip(seats, game["opponent_info"], strict=True):
        entity = _normalize_entity(info["id"])
        rating = info.get("raw_ratings", {}).get(str(pc))
        # Unrated opponents contribute to game/win counts, but cannot anchor a fit.
        if rating is None:
            continue
        anchors[entity] = float(rating)
        if human in winners or seat in winners:
            R.add_match_result(user["results"], "human", entity,
                               wins_a=int(human in winners and seat not in winners),
                               wins_b=int(human not in winners and seat in winners),
                               ties=int(human in winners and seat in winners), num_players=pc)
    scales = D.scales_for(reference_anchors)
    user['rating_display'] = D.metadata(reference_anchors)
    user['rating_reference_anchors_per_pc'] = deepcopy(reference_anchors)
    total, weight = 0.0, 0
    for count in R.PLAYER_COUNTS:
        fitted = R.fit_ratings_for_pc(user["results"], count,
                                     anchors=user["anchors_per_pc"].get(str(count), {"random": 1000.0}),
                                     reference_anchors_per_pc=reference_anchors)
        raw = fitted.get("human")
        if raw is not None:
            user[f"raw_rating_{count}p"] = raw
            user[f"rating_{count}p"] = D.to_display(raw, count, scales)
            games = user.get(f"games_{count}p", 0)
            total += user[f"rating_{count}p"] * games
            weight += games
    user["rating"] = total / weight if weight else None
    return user


def public_profile(user: dict, reference_anchors: dict | None = None) -> dict:
    # Convert existing profiles on read, without rewriting game history or counts.
    references = R.reference_anchors_from_manifest({"reference_anchors_per_pc":
        user.get('rating_reference_anchors_per_pc') or reference_anchors})
    scales = D.scales_for(references)
    values = {}
    total = weight = 0.0
    for pc in R.PLAYER_COUNTS:
        raw = user.get(f'raw_rating_{pc}p')
        old = user.get(f'rating_{pc}p')
        if raw is not None:
            value = D.to_display(raw, pc, scales)
        elif old is None:
            continue
        elif user.get('rating_display', {}).get('version') == D.VERSION:
            value = old
        elif 'rating_display' not in user:
            value = D.from_legacy_calibrated(old, pc, references)
        else:
            raise ValueError('Unknown stored rating display version without raw ratings')
        values[f'rating_{pc}p'] = value
        games = user.get(f'games_{pc}p', 0)
        total += value * games
        weight += games
    rating = total / weight if weight else None
    placed = user["wins"] >= PLACEMENT_WINS
    result = {k: user[k] for k in ("username", "games", "wins")}
    result.update(placed=placed, placement_wins_required=PLACEMENT_WINS,
                  rating=round(rating) if placed and rating is not None else None)
    for pc in R.PLAYER_COUNTS:
        for key in (f"games_{pc}p", f"wins_{pc}p"):
            result[key] = user.get(key, 0)
        rating = values.get(f"rating_{pc}p")
        if placed and rating is not None:
            result[f"rating_{pc}p"] = round(rating)
    return result
