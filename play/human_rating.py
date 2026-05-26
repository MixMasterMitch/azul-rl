"""Human player rating tracking using anchored Bradley-Terry."""

from __future__ import annotations

import json
import pathlib
from typing import Optional

from agent.train.ranking import fit_ratings


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
            }

    def record_game(
        self,
        opponent_id: str,
        opponent_rating: float,
        won: bool,
    ) -> None:
        """Record a game result and refit rating."""
        self.data["games"] += 1
        if won:
            self.data["wins"] += 1

        # Find or create result entry
        for r in self.data["results"]:
            if r["b"] == opponent_id:
                if won:
                    r["wins_a"] += 1
                else:
                    r["wins_b"] += 1
                break
        else:
            self.data["results"].append({
                "a": "human",
                "b": opponent_id,
                "wins_a": 1 if won else 0,
                "wins_b": 0 if won else 1,
            })

        # Refit rating
        if self.data["games"] >= 3:
            anchors = {"random": 1000.0}
            # Add opponent ratings as anchors
            for r in self.data["results"]:
                if r["b"] != "human":
                    anchors[r["b"]] = opponent_rating

            ratings = fit_ratings(self.data["results"], anchors=anchors)
            self.data["rating"] = ratings.get("human")

        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    @property
    def rating(self) -> Optional[float]:
        return self.data.get("rating")
