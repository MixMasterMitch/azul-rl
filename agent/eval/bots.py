"""Reference bots for evaluation."""

from __future__ import annotations

import random
from typing import Protocol

from ..env import actions as A
from ..env import engine as BE


class Bot(Protocol):
    def select_action(self, engine: BE.BatchedEngine, game_idx: int) -> int:
        ...


class RandomBot:
    """Uniform random over legal actions."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, engine: BE.BatchedEngine, game_idx: int) -> int:
        if isinstance(engine, BE.GameEngine):
            engine = engine.cpu_view()
        legal = engine.legal_action_mask()[game_idx].nonzero(as_tuple=True)[0].tolist()
        if not legal:
            return 0
        return self.rng.choice(legal)


class HeuristicBot:
    """Greedy heuristic: prefer filling pattern lines that are close to completion."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, engine: BE.BatchedEngine, game_idx: int) -> int:
        if isinstance(engine, BE.GameEngine):
            engine = engine.cpu_view()
        player = int(engine.current_player[game_idx].item())
        legal = engine.legal_action_mask()[game_idx].nonzero(as_tuple=True)[0].tolist()
        if not legal:
            return 0

        best_score = -1000
        best_actions = []

        for action_idx in legal:
            source, color, target = A.decode_action(action_idx)

            # Score this action
            score = 0

            if target == A.FLOOR_TARGET:
                score = -10  # Avoid floor
            else:
                row = target
                capacity = row + 1
                current = engine.pattern_count[game_idx, player, row].item()

                # How many tiles are we picking?
                if source < engine.num_factories:
                    num_tiles = engine.factory_tiles[game_idx, source, color].item()
                else:
                    num_tiles = engine.center_tiles[game_idx, color].item()

                space = capacity - current
                placed = min(num_tiles, space)
                excess = num_tiles - placed

                # Prefer actions that complete a line
                if current + placed == capacity:
                    score += 20 + capacity  # Bigger lines = more potential points
                else:
                    # Prefer filling lines that are closer to completion
                    fill_ratio = (current + placed) / capacity
                    score += int(fill_ratio * 10)

                # Penalize excess going to floor
                score -= excess * 3

                # Bonus for targeting rows where adjacent wall tiles exist
                wall_col = A.wall_column_for_color(row, color)
                wall = engine.wall[game_idx, player]
                adj = 0
                for c in range(max(0, wall_col - 1), min(5, wall_col + 2)):
                    if c != wall_col and wall[row, c]:
                        adj += 1
                for r in range(max(0, row - 1), min(5, row + 2)):
                    if r != row and wall[r, wall_col]:
                        adj += 1
                score += adj * 3

            if score > best_score:
                best_score = score
                best_actions = [action_idx]
            elif score == best_score:
                best_actions.append(action_idx)

        return self.rng.choice(best_actions)
