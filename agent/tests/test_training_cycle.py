"""Training iteration phase cycle."""

from agent.train.loop import _resolve_selfplay_kind, _training_phase


def test_four_step_cycle() -> None:
    assert _training_phase(1, 4) == "selfplay"
    assert _training_phase(2, 4) == "league"
    assert _training_phase(3, 4) == "selfplay"
    assert _training_phase(4, 4) == "bot"
    assert _training_phase(5, 4) == "selfplay"
    assert _training_phase(8, 4) == "bot"


def test_cycle_disabled() -> None:
    assert _training_phase(2, 0) == "selfplay"
    assert _training_phase(4, 0) == "selfplay"


def test_resolve_selfplay_kind() -> None:
    assert _resolve_selfplay_kind("bot", True) == ("bot_selfplay", "bot")
    assert _resolve_selfplay_kind("league", True) == ("league_selfplay", "league")
    assert _resolve_selfplay_kind("league", False) == ("standard_selfplay", "league")
