"""Human player rating tracking using anchored Bradley-Terry."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Optional

from agent.train import ranking as R


HUMAN_ENTITY = "human"


def _normalize_entity(entity_id: str) -> str:
    """Map play-server opponent IDs onto league rating entity IDs."""
    if entity_id == "opus":
        return "heuristic_opus"
    if entity_id.startswith("net:league:"):
        return f"ckpt:{entity_id.split(':')[-1]}"
    return entity_id


class HumanRatingStore:
    """Tracks a human player's rating from their full game history."""

    def __init__(self, username: str, data_dir: str = "play/play_data/users"):
        self.username = username
        self.path = pathlib.Path(data_dir) / f"{username}.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {
                "rating": None,
                "results": [],
                "games": 0,
                "wins": 0,
                "anchors": {"random": R.RANDOM_ANCHOR_RATING},
            }
        self.data.setdefault("results", [])
        self.data.setdefault("games", 0)
        self.data.setdefault("wins", 0)
        self.data.setdefault("anchors", {"random": R.RANDOM_ANCHOR_RATING})
        self.data["anchors"] = {
            _normalize_entity(str(entity)): float(rating)
            for entity, rating in self.data["anchors"].items()
        }
        self._migrate_legacy_results()

    def record_game(
        self,
        opponent_id: str,
        opponent_rating: float,
        won: bool,
    ) -> None:
        """Record a 2-player game result and refit rating."""
        self.record_multiplayer_game(
            opponents=[
                {
                    "seat": 1,
                    "entity_id": opponent_id,
                    "rating": opponent_rating,
                }
            ],
            human_rank=0 if won else 1,
            ranks=[0, 1] if won else [1, 0],
        )

    def record_multiplayer_game(
        self,
        opponents: list[dict[str, Any]],
        human_rank: int,
        ranks: list[int],
    ) -> None:
        """Record a 2p/3p/4p table result and refit rating.

        The BT encoding matches the league evaluator:
        - table winner records one win against each losing opponent
        - a human non-winner records one loss against the table winner only

        This makes a 40% first-place rate against three identical opponents
        imply roughly a 2x strength ratio over each single opponent.
        """
        num_players = len(ranks) if ranks else len(opponents) + 1
        if num_players not in R.PLAYER_COUNTS:
            raise ValueError(f"num_players must be one of {R.PLAYER_COUNTS}; got {num_players}")

        self.data["games"] += 1
        self.data[f"games_{num_players}p"] = int(self.data.get(f"games_{num_players}p", 0)) + 1
        if human_rank == 0:
            self.data["wins"] += 1

        for opp in opponents:
            entity = _normalize_entity(str(opp["entity_id"]))
            rating = opp.get("rating")
            if rating is not None:
                self.data["anchors"][entity] = float(rating)

        if human_rank == 0:
            for opp in opponents:
                entity = _normalize_entity(str(opp["entity_id"]))
                R.add_match_result(
                    self.data["results"],
                    HUMAN_ENTITY,
                    entity,
                    wins_a=1.0,
                    wins_b=0.0,
                    num_players=num_players,
                )
        else:
            winner = self._winner_opponent(opponents, ranks)
            if winner is not None:
                entity = _normalize_entity(str(winner["entity_id"]))
                R.add_match_result(
                    self.data["results"],
                    HUMAN_ENTITY,
                    entity,
                    wins_a=0.0,
                    wins_b=1.0,
                    num_players=num_players,
                )

        self._refit_rating()

        self.save()

    def _winner_opponent(
        self,
        opponents: list[dict[str, Any]],
        ranks: list[int],
    ) -> dict[str, Any] | None:
        if not ranks:
            return opponents[0] if opponents else None
        for opp in opponents:
            seat = int(opp["seat"])
            if ranks[seat] == 0:
                return opp
        return None

    def _migrate_legacy_results(self) -> None:
        """Convert old untagged 2p result rows to per-player-count rows."""
        results = self.data.get("results", [])
        if not results:
            return
        if any(any(k.startswith("wins_a_") and k.endswith("p") for k in row) for row in results):
            return

        migrated: list[dict] = []
        for row in results:
            a = _normalize_entity(str(row.get("a", HUMAN_ENTITY)))
            b = _normalize_entity(str(row.get("b", "")))
            if not b:
                continue
            R.add_match_result(
                migrated,
                a,
                b,
                wins_a=float(row.get("wins_a", 0.0)),
                wins_b=float(row.get("wins_b", 0.0)),
                ties=float(row.get("ties", 0.0)),
                num_players=2,
            )
        self.data["results"] = migrated
        if "games_2p" not in self.data and int(self.data.get("games", 0)) > 0:
            self.data["games_2p"] = int(self.data["games"])

    def _refit_rating(self) -> None:
        if int(self.data["games"]) < 3:
            return

        anchors = {
            str(entity): float(rating)
            for entity, rating in self.data.get("anchors", {}).items()
        }
        anchors.setdefault("random", R.RANDOM_ANCHOR_RATING)

        calibrated_sum = 0.0
        weight_sum = 0.0
        for pc in R.PLAYER_COUNTS:
            pc_ratings = R.fit_ratings_for_pc(
                self.data["results"],
                pc,
                anchors=anchors,
            )
            raw = pc_ratings.get(HUMAN_ENTITY)
            if raw is None:
                self.data.pop(f"rating_{pc}p", None)
                continue

            calibrated = R.calibrate_rating(raw, pc)
            games_at_pc = float(self.data.get(f"games_{pc}p", 0.0))
            if games_at_pc <= 0:
                games_at_pc = 1.0

            self.data[f"rating_{pc}p"] = calibrated
            calibrated_sum += calibrated * games_at_pc
            weight_sum += games_at_pc

        if weight_sum > 0:
            self.data["rating"] = calibrated_sum / weight_sum

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    @property
    def rating(self) -> Optional[float]:
        return self.data.get("rating")
