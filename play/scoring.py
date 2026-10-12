"""Player-facing score accounting for the native Azul engine."""
from __future__ import annotations


def placement_points(wall: list[list[bool]], row: int, col: int) -> int:
    horizontal = vertical = 1
    for step in (-1, 1):
        x = col + step
        while 0 <= x < 5 and wall[row][x]:
            horizontal += 1
            x += step
        y = row + step
        while 0 <= y < 5 and wall[y][col]:
            vertical += 1
            y += step
    if horizontal == vertical == 1:
        return 1
    return (horizontal if horizontal > 1 else 0) + (vertical if vertical > 1 else 0)


def endgame_bonuses(wall: list[list[bool]]) -> dict[str, int]:
    rows = sum(all(wall[row]) for row in range(5))
    columns = sum(all(wall[row][col] for row in range(5)) for col in range(5))
    colors = sum(all(wall[row][(row + color) % 5] for row in range(5)) for color in range(5))
    return {"rows": 2 * rows, "columns": 7 * columns, "colors": 10 * colors}


def round_change(before: list[list[bool]], after: list[list[bool]],
                 score_before: int, score_after: int, ended: bool) -> tuple[int, int]:
    """Return placement points and the floor deduction actually applied this step."""
    wall = [row.copy() for row in before]
    placement = 0
    for row in range(5):
        for col in range(5):
            if after[row][col] and not wall[row][col]:
                wall[row][col] = True
                placement += placement_points(wall, row, col)
    bonuses = sum(endgame_bonuses(after).values()) if ended else 0
    floor = score_after - score_before - placement - bonuses
    return placement, floor
