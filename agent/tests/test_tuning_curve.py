"""Log-curve extrapolation for tuning objective."""

import math

import pytest

from agent.train.tuning_curve import extrapolate_rating, fit_log_rating_curve


def test_log_fit_monotone_growth() -> None:
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ratings = [1200.0, 1250.0, 1300.0, 1340.0, 1370.0, 1390.0, 1400.0]
    fit = fit_log_rating_curve(times, ratings)
    assert fit.slope > 0
    pred = fit.predict(75.0)
    assert pred > ratings[-1]


def test_extrapolate_returns_target_horizon() -> None:
    times = [0.0, 1.0, 2.0, 3.0]
    ratings = [1100.0, 1200.0, 1280.0, 1320.0]
    out = extrapolate_rating(times, ratings, extra_hours=72.0)
    assert out["t_last_hours"] == 3.0
    assert out["t_target_hours"] == pytest.approx(75.0)
    assert out["predicted_rating_2p"] > out["rating_at_last"]


def test_needs_two_points() -> None:
    with pytest.raises(ValueError):
        fit_log_rating_curve([0.0], [1000.0])
