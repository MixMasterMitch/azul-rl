"""Small independent public-state oracle for Astra tests and fair microbenchmarks.

This is not a production bot. Its list snapshot has exactly the native state
contents, so transition timing comparisons do not include RNG/deep-copy work.
"""
from __future__ import annotations

FLOOR = [0, -1, -2, -4, -6, -8, -11, -14]


def wall_score(wall: int, row: int, col: int) -> int:
    h = v = 1
    for dc in (-1, 1):
        c = col + dc
        while 0 <= c < 5 and wall & (1 << (row * 5 + c)):
            h += 1
            c += dc
    for dr in (-1, 1):
        r = row + dr
        while 0 <= r < 5 and wall & (1 << (r * 5 + col)):
            v += 1
            r += dr
    return 1 if h == v == 1 else (h if h > 1 else 0) + (v if v > 1 else 0)


def legal_actions(data: list[int]) -> list[int]:
    if data[5] or data[6]:
        return []
    n, cp = data[1:3]
    nf = 2 * n + 1
    b = 57 + 14 * cp
    legal = []
    for source in range(nf + 1):
        start = 52 if source == nf else 7 + 5 * source
        for color in range(5):
            if data[start + color] == 0:
                continue
            for row in range(6):
                if row == 5 or (data[b + 4 + row] < row + 1
                        and data[b + 9 + row] in (-1, color)
                        and not data[b] & (1 << (row * 5 + (row + color) % 5))):
                    legal.append(source * 30 + color * 6 + row)
    return legal


def resolve_round(data: list[int]) -> list[int]:
    d = data.copy()
    if d[5] or d[6]:
        return d
    for p in range(d[1]):
        b = 57 + 14 * p
        for r in range(5):
            if d[b + 4 + r] == r + 1:
                c = (r + d[b + 9 + r]) % 5
                d[b + 1] += wall_score(d[b], r, c)
                d[b] |= 1 << (r * 5 + c)
                d[b + 4 + r], d[b + 9 + r] = 0, -1
        d[b + 1] = max(0, d[b + 1] + FLOOR[d[b + 2]])
        d[b + 2] = 0
        if d[b + 3]:
            d[3] = p
        d[b + 3] = 0
    d[5] = int(any((d[57 + 14 * p] >> (5 * r)) & 31 == 31 for p in range(d[1]) for r in range(5)))
    if d[5]:
        for p in range(d[1]):
            b = 57 + 14 * p
            w = d[b]
            d[b + 1] += 2 * sum((w >> (5 * r)) & 31 == 31 for r in range(5))
            d[b + 1] += 7 * sum(all(w & (1 << (r * 5 + c)) for r in range(5)) for c in range(5))
            d[b + 1] += 10 * sum(all(w & (1 << (r * 5 + (r + c) % 5)) for r in range(5)) for c in range(5))
    d[6] = 1
    return d


def transition(data: list[int], action: int, resolve: bool = True) -> list[int]:
    if action not in legal_actions(data):
        raise ValueError("Illegal action")
    d = data.copy()
    source, color, row = action // 30, action % 30 // 6, action % 6
    b = 57 + 14 * d[2]
    if source == 2 * d[1] + 1:
        qty = d[52 + color]
        d[52 + color] = 0
        if d[4]:
            d[4] = 0
            d[b + 3] = 1
            d[b + 2] = min(7, d[b + 2] + 1)
    else:
        offset = 7 + source * 5
        qty = d[offset + color]
        for c in range(5):
            if c != color:
                d[52 + c] += d[offset + c]
            d[offset + c] = 0
    placed = min(qty, row + 1 - d[b + 4 + row]) if row < 5 else 0
    if row < 5:
        d[b + 4 + row] += placed
        d[b + 9 + row] = color
    d[b + 2] = min(7, d[b + 2] + qty - placed)
    d[2] = (d[2] + 1) % d[1]
    return resolve_round(d) if resolve and not any(d[7:57]) else d
