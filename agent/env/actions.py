"""Flat action encoding for the Azul engine.

Action space:
Each action represents: pick all tiles of one color from one source, place on
one target (pattern line or floor).

Layout:
  action_index = source * (NUM_COLORS * NUM_TARGETS) + color * NUM_TARGETS + target

Where:
- source ∈ [0, MAX_FACTORIES)  → factory displays
- source = MAX_FACTORIES       → center of table
- color  ∈ [0, NUM_COLORS)     → tile color to pick
- target ∈ [0, 5)              → pattern line row (0=size-1, 4=size-5)
- target = 5                   → floor (discard all picked tiles directly)

For a 4-player game there are 9 factories + 1 center = 10 sources.
Total action space: 10 * 5 * 6 = 300.

Legality is controlled by a mask. An action is legal iff:
1. The chosen color exists at the chosen source.
2. If target is a pattern line:
   a. The pattern line is empty OR already holds tiles of the same color.
   b. The corresponding wall row does NOT already have that color placed.
3. If target is floor: always legal (as long as condition 1 holds).
"""

from __future__ import annotations

NUM_COLORS: int = 5
MAX_FACTORIES: int = 9
NUM_SOURCES: int = MAX_FACTORIES + 1  # factories + center
NUM_PATTERN_LINES: int = 5
NUM_TARGETS: int = NUM_PATTERN_LINES + 1  # 5 pattern lines + floor
FLOOR_TARGET: int = 5

NUM_ACTIONS: int = NUM_SOURCES * NUM_COLORS * NUM_TARGETS  # 300

MAX_PLAYERS: int = 4
FLOOR_SIZE: int = 7
FLOOR_MARKER: int = 5  # floor_slots value for the first-player token
FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]

COLOR_NAMES = ["Blue", "Yellow", "Red", "Black", "White"]
COLOR_ABBREV = "BYRKW"

# Standard wall pattern: wall[row][col] = color
# Each row is shifted by 1 from the previous (Latin square).
# Color at position (row, col) = (col - row) % 5
WALL_PATTERN = [
    [(c - r) % NUM_COLORS for c in range(5)]
    for r in range(5)
]


def wall_column_for_color(row: int, color: int) -> int:
    """Return the column index where a given color goes in a wall row."""
    return (color + row) % 5


def encode_action(source: int, color: int, target: int) -> int:
    assert 0 <= source < NUM_SOURCES
    assert 0 <= color < NUM_COLORS
    assert 0 <= target < NUM_TARGETS
    return source * (NUM_COLORS * NUM_TARGETS) + color * NUM_TARGETS + target


def decode_action(action: int) -> tuple[int, int, int]:
    """Returns (source, color, target)."""
    assert 0 <= action < NUM_ACTIONS
    source = action // (NUM_COLORS * NUM_TARGETS)
    remainder = action % (NUM_COLORS * NUM_TARGETS)
    color = remainder // NUM_TARGETS
    target = remainder % NUM_TARGETS
    return source, color, target


def action_name(action: int) -> str:
    source, color, target = decode_action(action)
    src_str = f"factory{source}" if source < MAX_FACTORIES else "center"
    color_str = COLOR_ABBREV[color]
    tgt_str = f"line{target}" if target < NUM_PATTERN_LINES else "floor"
    return f"pick({src_str},{color_str})→{tgt_str}"


def num_factories_for_players(num_players: int) -> int:
    """Number of factory displays based on player count."""
    return 2 * num_players + 1
