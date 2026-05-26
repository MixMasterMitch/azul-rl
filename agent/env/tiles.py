"""Tile constants and utilities for Azul.

The game has 100 tiles total: 20 of each of the 5 colors.
Colors: Blue(0), Yellow(1), Red(2), Black(3), White(4)
"""

from __future__ import annotations

NUM_COLORS: int = 5
TILES_PER_COLOR: int = 20
TOTAL_TILES: int = NUM_COLORS * TILES_PER_COLOR  # 100

TILES_PER_FACTORY: int = 4
