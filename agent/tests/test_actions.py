"""Tests for action encoding/decoding."""

import pytest

from agent.env import actions as A


def test_encode_decode_roundtrip():
    """Every valid action encodes and decodes back to the same triple."""
    for source in range(A.NUM_SOURCES):
        for color in range(A.NUM_COLORS):
            for target in range(A.NUM_TARGETS):
                idx = A.encode_action(source, color, target)
                s, c, t = A.decode_action(idx)
                assert s == source
                assert c == color
                assert t == target


def test_action_space_size():
    assert A.NUM_ACTIONS == 300


def test_wall_column_for_color():
    """Verify the wall pattern is a valid Latin square."""
    for row in range(5):
        colors_in_row = set()
        for color in range(5):
            col = A.wall_column_for_color(row, color)
            assert 0 <= col < 5
            colors_in_row.add(col)
        assert len(colors_in_row) == 5  # each color maps to unique column

    # Verify columns also have unique colors
    for col in range(5):
        colors_in_col = set()
        for row in range(5):
            for color in range(5):
                if A.wall_column_for_color(row, color) == col:
                    colors_in_col.add(color)
        assert len(colors_in_col) == 5


def test_num_factories_for_players():
    assert A.num_factories_for_players(2) == 5
    assert A.num_factories_for_players(3) == 7
    assert A.num_factories_for_players(4) == 9


def test_action_name():
    name = A.action_name(A.encode_action(0, 0, 0))
    assert "factory0" in name
    assert "B" in name
    assert "line0" in name

    name = A.action_name(A.encode_action(9, 4, 5))
    assert "center" in name
    assert "W" in name
    assert "floor" in name

    name = A.action_name(A.encode_action(5, 4, 5), num_players=2)
    assert "center" in name
