"""Single-game reference Azul engine for testing and human play.

This is a non-batched implementation that mirrors the batched engine logic
exactly, but operates on Python data structures for clarity and testability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from . import actions as A
from . import tiles as T


@dataclass
class PlayerBoard:
    pattern_count: list[int] = field(default_factory=lambda: [0] * 5)
    pattern_color: list[int] = field(default_factory=lambda: [-1] * 5)
    wall: list[list[bool]] = field(
        default_factory=lambda: [[False] * 5 for _ in range(5)]
    )
    floor_count: int = 0
    floor_tiles: list[int] = field(default_factory=lambda: [0] * A.NUM_COLORS)
    floor_has_first: bool = False
    score: int = 0


class SingleEngine:
    """Non-batched Azul engine for a single game."""

    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        assert 2 <= num_players <= 4
        self.num_players = num_players
        self.num_factories = A.num_factories_for_players(num_players)
        self.rng = random.Random(seed)

        self.players = [PlayerBoard() for _ in range(num_players)]
        self.factory_tiles: list[list[int]] = [
            [0] * A.NUM_COLORS for _ in range(self.num_factories)
        ]
        self.center_tiles: list[int] = [0] * A.NUM_COLORS
        self.center_has_first: bool = True
        self.bag: list[int] = [T.TILES_PER_COLOR] * A.NUM_COLORS
        self.box_lid: list[int] = [0] * A.NUM_COLORS
        self.current_player: int = 0
        self.first_player: int = 0
        self.ended: bool = False

        self._fill_factories()

    def _fill_factories(self) -> None:
        for f in range(self.num_factories):
            for _ in range(T.TILES_PER_FACTORY):
                color = self._draw_tile()
                if color is None:
                    return
                self.factory_tiles[f][color] += 1

    def _draw_tile(self) -> Optional[int]:
        total = sum(self.bag)
        if total == 0:
            self.bag = self.box_lid[:]
            self.box_lid = [0] * A.NUM_COLORS
            total = sum(self.bag)
            if total == 0:
                return None
        # Weighted random choice
        r = self.rng.randint(0, total - 1)
        cumulative = 0
        for c in range(A.NUM_COLORS):
            cumulative += self.bag[c]
            if r < cumulative:
                self.bag[c] -= 1
                return c
        return A.NUM_COLORS - 1  # shouldn't reach here

    def legal_actions(self) -> list[int]:
        actions = []
        player = self.players[self.current_player]

        for source in range(self.num_factories + 1):
            if source < self.num_factories:
                tiles = self.factory_tiles[source]
            else:
                tiles = self.center_tiles

            for color in range(A.NUM_COLORS):
                if tiles[color] <= 0:
                    continue

                for target in range(A.NUM_TARGETS):
                    if target == A.FLOOR_TARGET:
                        actions.append(A.encode_action(source, color, target))
                    else:
                        row = target
                        line_color = player.pattern_color[row]
                        line_count = player.pattern_count[row]
                        capacity = row + 1

                        if line_color != -1 and line_color != color:
                            continue
                        if line_count >= capacity:
                            continue
                        wall_col = A.wall_column_for_color(row, color)
                        if player.wall[row][wall_col]:
                            continue

                        actions.append(A.encode_action(source, color, target))

        return actions

    def step(self, action: int) -> None:
        assert not self.ended
        source, color, target = A.decode_action(action)
        player = self.players[self.current_player]

        # Pick tiles
        if source < self.num_factories:
            num_picked = self.factory_tiles[source][color]
            for c in range(A.NUM_COLORS):
                if c != color:
                    self.center_tiles[c] += self.factory_tiles[source][c]
            self.factory_tiles[source] = [0] * A.NUM_COLORS
        else:
            num_picked = self.center_tiles[color]
            self.center_tiles[color] = 0
            if self.center_has_first:
                self.center_has_first = False
                player.floor_has_first = True
                player.floor_count = min(player.floor_count + 1, A.FLOOR_SIZE)

        # Place tiles
        if target == A.FLOOR_TARGET:
            can_place = max(0, A.FLOOR_SIZE - player.floor_count)
            placed = min(num_picked, can_place)
            overflow = num_picked - placed
            player.floor_count += placed
            player.floor_tiles[color] += placed
            self.box_lid[color] += overflow
        else:
            row = target
            capacity = row + 1
            space = capacity - player.pattern_count[row]
            placed_in_line = min(num_picked, space)
            excess = num_picked - placed_in_line
            player.pattern_count[row] += placed_in_line
            player.pattern_color[row] = color
            if excess > 0:
                can_place = max(0, A.FLOOR_SIZE - player.floor_count)
                placed = min(excess, can_place)
                overflow = excess - placed
                player.floor_count += placed
                player.floor_tiles[color] += placed
                self.box_lid[color] += overflow

        # Advance player
        self.current_player = (self.current_player + 1) % self.num_players

        # Check round end
        factories_empty = all(
            sum(f) == 0 for f in self.factory_tiles[:self.num_factories]
        )
        center_empty = sum(self.center_tiles) == 0
        if factories_empty and center_empty:
            self._do_wall_tiling()

    def _do_wall_tiling(self) -> None:
        game_over = False

        for player in self.players:
            for row in range(5):
                count = player.pattern_count[row]
                capacity = row + 1
                if count < capacity:
                    continue
                color = player.pattern_color[row]
                col = A.wall_column_for_color(row, color)
                player.wall[row][col] = True
                points = self._score_placement(player, row, col)
                player.score += points
                # Return excess tiles
                self.box_lid[color] += count - 1
                player.pattern_count[row] = 0
                player.pattern_color[row] = -1

            # Floor penalties
            penalty = 0
            for i in range(min(player.floor_count, A.FLOOR_SIZE)):
                penalty += A.FLOOR_PENALTIES[i]
            player.score = max(0, player.score + penalty)

            # Return floor tiles
            for c in range(A.NUM_COLORS):
                self.box_lid[c] += player.floor_tiles[c]
            player.floor_tiles = [0] * A.NUM_COLORS
            player.floor_count = 0

            if player.floor_has_first:
                self.first_player = self.players.index(player)
            player.floor_has_first = False

            # Check game end
            for row in range(5):
                if all(player.wall[row]):
                    game_over = True

        if game_over:
            self._end_game()
        else:
            self._prepare_next_round()

    def _score_placement(self, player: PlayerBoard, row: int, col: int) -> int:
        wall = player.wall
        h_count = 1
        v_count = 1

        for c in range(col - 1, -1, -1):
            if wall[row][c]:
                h_count += 1
            else:
                break
        for c in range(col + 1, 5):
            if wall[row][c]:
                h_count += 1
            else:
                break
        for r in range(row - 1, -1, -1):
            if wall[r][col]:
                v_count += 1
            else:
                break
        for r in range(row + 1, 5):
            if wall[r][col]:
                v_count += 1
            else:
                break

        if h_count == 1 and v_count == 1:
            return 1
        score = 0
        if h_count > 1:
            score += h_count
        if v_count > 1:
            score += v_count
        return score

    def _end_game(self) -> None:
        for player in self.players:
            bonus = 0
            # +2 per complete row
            for row in range(5):
                if all(player.wall[row]):
                    bonus += 2
            # +7 per complete column
            for col in range(5):
                if all(player.wall[r][col] for r in range(5)):
                    bonus += 7
            # +10 per color with all 5 placed
            for color in range(A.NUM_COLORS):
                all_placed = all(
                    player.wall[row][A.wall_column_for_color(row, color)]
                    for row in range(5)
                )
                if all_placed:
                    bonus += 10
            player.score += bonus
        self.ended = True

    def _prepare_next_round(self) -> None:
        self.center_has_first = True
        self.current_player = self.first_player
        self._fill_factories()

    def get_winner(self) -> int:
        """Return index of the winning player (-1 if game not ended)."""
        if not self.ended:
            return -1
        best_score = -1
        best_rows = -1
        best_player = 0
        for i, p in enumerate(self.players):
            rows = sum(1 for r in range(5) if all(p.wall[r]))
            if p.score > best_score or (p.score == best_score and rows > best_rows):
                best_score = p.score
                best_rows = rows
                best_player = i
        return best_player
