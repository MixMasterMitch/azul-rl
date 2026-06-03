"""Heuristic-opus Azul bot.

Production bot: ``HeuristicOpusBot`` dispatches V13 at 2p, V19 at 3p/4p.
Developed iteratively across 20 candidates (see heuristic_opus_RESULTS.md).

Final ratings (192 games/matchup, random=1000):
  2p: 1735 | 3p: 1828 | 4p: 1861
"""

from __future__ import annotations

import random

from ..env import actions as A
from ..env import batched_engine as BE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _GameView:
    """Pure-Python snapshot of one game in the batched engine."""

    __slots__ = (
        "num_players", "num_factories", "current_player",
        "factory_tiles", "center_tiles", "center_has_first",
        "pattern_count", "pattern_color", "wall", "floor_count",
        "scores",
    )

    def __init__(self, engine: BE.BatchedEngine, game_idx: int):
        b = game_idx
        self.num_players = engine.num_players
        self.num_factories = engine.num_factories
        self.current_player = engine.current_player[b].item()

        self.factory_tiles: list[list[int]] = [
            engine.factory_tiles[b, f].tolist()
            for f in range(engine.num_factories)
        ]
        self.center_tiles: list[int] = engine.center_tiles[b].tolist()
        self.center_has_first: bool = bool(engine.center_first[b].item())

        self.pattern_count: list[list[int]] = [
            engine.pattern_count[b, p].tolist()
            for p in range(engine.num_players)
        ]
        self.pattern_color: list[list[int]] = [
            engine.pattern_color[b, p].tolist()
            for p in range(engine.num_players)
        ]
        self.wall: list[list[list[bool]]] = [
            [[bool(engine.wall[b, p, r, c].item()) for c in range(5)] for r in range(5)]
            for p in range(engine.num_players)
        ]
        self.floor_count: list[int] = [
            engine.floor_count[b, p].item()
            for p in range(engine.num_players)
        ]
        self.scores: list[int] = [
            engine.scores[b, p].item()
            for p in range(engine.num_players)
        ]


def _wall_score(wall: list[list[bool]], row: int, col: int) -> int:
    """Simulate scoring for placing a tile at wall[row][col]."""
    h = 1
    v = 1
    for c in range(col - 1, -1, -1):
        if wall[row][c]:
            h += 1
        else:
            break
    for c in range(col + 1, 5):
        if wall[row][c]:
            h += 1
        else:
            break
    for r in range(row - 1, -1, -1):
        if wall[r][col]:
            v += 1
        else:
            break
    for r in range(row + 1, 5):
        if wall[r][col]:
            v += 1
        else:
            break
    if h == 1 and v == 1:
        return 1
    score = 0
    if h > 1:
        score += h
    if v > 1:
        score += v
    return score


def _floor_penalty(floor_before: int, tiles_added: int) -> int:
    """Compute incremental floor penalty for adding tiles_added to a floor."""
    new_floor = min(floor_before + tiles_added, A.FLOOR_SIZE)
    return sum(A.FLOOR_PENALTIES[i] for i in range(floor_before, new_floor))


def _tile_count_at_source(gv: _GameView, source: int, color: int) -> int:
    if source < gv.num_factories:
        return gv.factory_tiles[source][color]
    return gv.center_tiles[color]


def _total_color_available(gv: _GameView, color: int) -> int:
    """Count total tiles of a color across all sources."""
    total = 0
    for f in range(gv.num_factories):
        total += gv.factory_tiles[f][color]
    total += gv.center_tiles[color]
    return total


def _bonus_progress(wall: list[list[bool]]) -> dict:
    """Analyze progress toward end-game bonuses."""
    rows = [sum(1 for c in range(5) if wall[r][c]) for r in range(5)]
    cols = [sum(1 for r in range(5) if wall[r][c]) for c in range(5)]
    colors = [
        sum(1 for r in range(5) if wall[r][A.wall_column_for_color(r, color)])
        for color in range(A.NUM_COLORS)
    ]
    return {"rows": rows, "cols": cols, "colors": colors}


# ---------------------------------------------------------------------------
# V13: Holistic fusion (best at 2p)
# ---------------------------------------------------------------------------

class _V13:
    """Fusion of wall-clustering with a holistic value function.

    Computes a single score per action from: immediate wall score,
    floor penalty, bonus contribution, clustering/positional value,
    selective denial, center timing, and late-game urgency.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, engine: BE.BatchedEngine, game_idx: int) -> int:
        legal = engine.legal_action_mask()[game_idx].nonzero(as_tuple=True)[0].tolist()
        if not legal:
            return 0

        gv = _GameView(engine, game_idx)
        cp = gv.current_player

        best_val = -9999.0
        best_actions: list[int] = []

        for action_idx in legal:
            val = self._value(gv, cp, action_idx)
            if val > best_val:
                best_val = val
                best_actions = [action_idx]
            elif val == best_val:
                best_actions.append(action_idx)

        return self.rng.choice(best_actions)

    def _value(self, gv: _GameView, cp: int, action_idx: int) -> float:
        source, color, target = A.decode_action(action_idx)
        num_tiles = _tile_count_at_source(gv, source, color)

        fp_cost = 0.0
        floor_before = gv.floor_count[cp]
        if source >= gv.num_factories and gv.center_has_first:
            fp_cost = _floor_penalty(floor_before, 1)
            floor_before = min(floor_before + 1, A.FLOOR_SIZE)

        if target == A.FLOOR_TARGET:
            penalty = _floor_penalty(floor_before, num_tiles)
            return -30.0 + penalty + fp_cost

        row = target
        capacity = row + 1
        current = gv.pattern_count[cp][row]
        space = capacity - current
        placed = min(num_tiles, space)
        excess = num_tiles - placed
        fill_after = current + placed
        completes = fill_after == capacity
        wall_col = A.wall_column_for_color(row, color)

        val = 0.0

        # Immediate wall score
        if completes:
            ws = _wall_score(gv.wall[cp], row, wall_col)
            val += ws * 10.0
            val += 12.0 + capacity * 1.5

        # Floor penalty (exponentially worse when floor is full)
        if excess > 0:
            fp = _floor_penalty(floor_before, excess)
            val += fp
            if floor_before >= 4:
                val += fp * 0.5
        val += fp_cost

        # Bonus contribution
        if completes:
            bp = _bonus_progress(gv.wall[cp])

            row_prog = bp["rows"][row] + 1
            if row_prog == 5:
                val += 20.0
            elif row_prog >= 3:
                val += row_prog * 2.5

            col_prog = bp["cols"][wall_col] + 1
            if col_prog == 5:
                val += 70.0
            elif col_prog >= 3:
                val += col_prog * 5.0

            color_prog = bp["colors"][color] + 1
            if color_prog == 5:
                val += 100.0
            elif color_prog >= 3:
                val += color_prog * 6.0

        # Clustering / positional
        if completes:
            wall = gv.wall[cp]
            empty_adj = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, wall_col + dc
                if 0 <= nr < 5 and 0 <= nc < 5 and not wall[nr][nc]:
                    empty_adj += 1
            val += empty_adj * 1.5

            total_wall_tiles = sum(
                1 for r in range(5) for c in range(5) if wall[r][c]
            )
            if total_wall_tiles < 6:
                center_dist = abs(row - 2) + abs(wall_col - 2)
                val += max(0, 3 - center_dist) * 1.5
        else:
            future_ws = _wall_score(gv.wall[cp], row, wall_col)
            completion_ratio = fill_after / capacity
            val += future_ws * completion_ratio * 2.5

        # Fill quality
        if not completes:
            val += (fill_after / capacity) * 5.0

        if num_tiles == space:
            val += 3.0
        elif excess >= 3:
            val -= 2.0

        # Line viability
        if not completes:
            remaining = capacity - fill_after
            total_avail = _total_color_available(gv, color) - num_tiles
            if total_avail >= remaining:
                val += 3.0
            elif total_avail == 0 and remaining > 1:
                val -= 3.0

        # Selective denial
        val += self._denial_value(gv, cp, color)

        # Center timing
        if source >= gv.num_factories and gv.center_has_first:
            factories_remaining = sum(
                1 for f in range(gv.num_factories)
                if sum(gv.factory_tiles[f]) > 0
            )
            if factories_remaining > gv.num_factories // 2:
                val -= 3.0
            elif factories_remaining <= 1:
                val += 1.0

        # Late-game urgency
        max_wall = 0
        for p in range(gv.num_players):
            wt = sum(1 for r in range(5) for c in range(5) if gv.wall[p][r][c])
            max_wall = max(max_wall, wt)
        if max_wall >= 15:
            if completes:
                val += 8.0
            elif current == 0 and capacity >= 3:
                total_avail = _total_color_available(gv, color)
                if total_avail < capacity:
                    val -= 5.0

        return val

    def _denial_value(self, gv: _GameView, cp: int, color: int) -> float:
        denial = 0.0
        scale = 1.0 / max(1, gv.num_players - 1)

        my_score = gv.scores[cp]
        for opp in range(gv.num_players):
            if opp == cp:
                continue

            leader_bonus = 1.0
            if gv.scores[opp] > my_score + 5:
                leader_bonus = 1.5

            for opp_row in range(5):
                opp_count = gv.pattern_count[opp][opp_row]
                opp_color = gv.pattern_color[opp][opp_row]
                if opp_color != color or opp_count == 0:
                    continue
                remaining = (opp_row + 1) - opp_count
                if remaining == 1:
                    wc = A.wall_column_for_color(opp_row, color)
                    ws = _wall_score(gv.wall[opp], opp_row, wc)
                    if ws >= 3:
                        denial += 5.0 * scale * leader_bonus
                    else:
                        denial += 2.0 * scale * leader_bonus

            bp = _bonus_progress(gv.wall[opp])
            for opp_row in range(5):
                if bp["rows"][opp_row] >= 4:
                    for c in range(5):
                        if not gv.wall[opp][opp_row][c]:
                            missing_color = (c - opp_row) % 5
                            if color == missing_color:
                                denial += 10.0 * scale

        return denial


# ---------------------------------------------------------------------------
# V19: V13 + color monopoly + opponent scoring denial (best at 3p/4p)
# ---------------------------------------------------------------------------

class _V19(_V13):
    """V13 extended with color monopoly awareness and opponent scoring
    denial. Strongest at 3p/4p where multi-player interactions matter.
    """

    def _value(self, gv: _GameView, cp: int, action_idx: int) -> float:
        val = super()._value(gv, cp, action_idx)
        source, color, target = A.decode_action(action_idx)
        num_tiles = _tile_count_at_source(gv, source, color)

        if target == A.FLOOR_TARGET:
            return val

        row = target
        capacity = row + 1
        current = gv.pattern_count[cp][row]

        # Color monopoly: prefer colors where we hold most remaining tiles
        total_avail = _total_color_available(gv, color)
        if total_avail > 0:
            monopoly_ratio = num_tiles / total_avail
            if monopoly_ratio >= 0.7 and current + min(num_tiles, capacity - current) < capacity:
                val += 2.0
            elif monopoly_ratio >= 0.5:
                val += 1.0

        # Opponent scoring denial: reward starving opponents of tiles they
        # need to complete high-scoring pattern lines
        for opp in range(gv.num_players):
            if opp == cp:
                continue
            for opp_row in range(5):
                opp_count = gv.pattern_count[opp][opp_row]
                opp_color = gv.pattern_color[opp][opp_row]
                if opp_count == 0 or opp_color != color:
                    continue
                remaining = (opp_row + 1) - opp_count
                after_us = total_avail - num_tiles
                if after_us < remaining:
                    wc = A.wall_column_for_color(opp_row, opp_color)
                    ws = _wall_score(gv.wall[opp], opp_row, wc)
                    if remaining == 1:
                        val += ws * 1.5 / max(1, gv.num_players - 1)
                    elif remaining == 2:
                        val += ws * 0.5 / max(1, gv.num_players - 1)

        # Line viability for new lines
        if current == 0:
            placed = min(num_tiles, capacity)
            remaining = capacity - placed
            if remaining > 0:
                other_sources = total_avail - num_tiles
                if other_sources >= remaining:
                    val += 2.0
                elif remaining == 1 and other_sources >= 1:
                    val += 1.5
                elif remaining > 2 and other_sources == 0:
                    val -= 2.0

        return val


# ---------------------------------------------------------------------------
# Production bot
# ---------------------------------------------------------------------------

class HeuristicOpusBot:
    """Production heuristic bot: V13 at 2p, V19 at 3p/4p."""

    def __init__(self, seed: int | None = None):
        self._v13 = _V13(seed=seed)
        self._v19 = _V19(seed=seed)

    def select_action(self, engine: BE.BatchedEngine, game_idx: int) -> int:
        if engine.num_players == 2:
            return self._v13.select_action(engine, game_idx)
        return self._v19.select_action(engine, game_idx)


# Keep the old name as an alias for backward compat
HeuristicOpusV20 = HeuristicOpusBot
