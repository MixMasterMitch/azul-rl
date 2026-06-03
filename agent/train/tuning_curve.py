"""Log-curve fit for tuning: extrapolate 2p rating vs training wall time."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

DEFAULT_LOG_OFFSET_HOURS = 0.05
WINRATE_EPS = 1e-4


@dataclass(frozen=True)
class LogRatingCurveFit:
    """rating ≈ intercept + slope * log(t_hours + log_offset_hours)."""

    intercept: float
    slope: float
    log_offset_hours: float

    def predict(self, t_hours: float) -> float:
        return self.intercept + self.slope * math.log(
            max(t_hours, 0.0) + self.log_offset_hours
        )


def fit_log_rating_curve(
    times_hours: Sequence[float],
    ratings: Sequence[float],
    *,
    log_offset_hours: float = DEFAULT_LOG_OFFSET_HOURS,
) -> LogRatingCurveFit:
    if len(times_hours) != len(ratings):
        raise ValueError("times_hours and ratings must have the same length")
    if len(times_hours) < 2:
        raise ValueError("need at least 2 points to fit a curve")

    x = np.array(
        [math.log(max(float(t), 0.0) + log_offset_hours) for t in times_hours],
        dtype=np.float64,
    )
    y = np.array([float(r) for r in ratings], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return LogRatingCurveFit(
        intercept=float(intercept),
        slope=float(slope),
        log_offset_hours=log_offset_hours,
    )


def extrapolate_rating(
    times_hours: Sequence[float],
    ratings: Sequence[float],
    *,
    extra_hours: float,
    log_offset_hours: float = DEFAULT_LOG_OFFSET_HOURS,
) -> dict[str, float]:
    """Fit log curve on *times_hours* / *ratings* and predict at t_last + *extra_hours*."""
    fit = fit_log_rating_curve(times_hours, ratings, log_offset_hours=log_offset_hours)
    t_last = float(times_hours[-1])
    t_target = t_last + float(extra_hours)
    predicted = fit.predict(t_target)
    return {
        "predicted_rating_2p": predicted,
        "extrapolate_hours": float(extra_hours),
        "fit_intercept": fit.intercept,
        "fit_slope": fit.slope,
        "log_offset_hours": log_offset_hours,
        "t_last_hours": t_last,
        "t_target_hours": t_target,
        "rating_at_last": float(ratings[-1]),
    }


def _clamp_probability(value: float, eps: float = WINRATE_EPS) -> float:
    return min(max(float(value), eps), 1.0 - eps)


def _logit(value: float) -> float:
    p = _clamp_probability(value)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def extrapolate_winrate(
    times_hours: Sequence[float],
    winrates: Sequence[float],
    *,
    extra_hours: float,
    log_offset_hours: float = DEFAULT_LOG_OFFSET_HOURS,
) -> dict[str, float]:
    """Fit log curve in logit space and predict a bounded winrate."""
    logits = [_logit(wr) for wr in winrates]
    curve = extrapolate_rating(
        times_hours,
        logits,
        extra_hours=extra_hours,
        log_offset_hours=log_offset_hours,
    )
    predicted_logit = float(curve["predicted_rating_2p"])
    return {
        "predicted_winrate": _sigmoid(predicted_logit),
        "predicted_winrate_logit": predicted_logit,
        "extrapolate_hours": float(extra_hours),
        "fit_intercept": curve["fit_intercept"],
        "fit_slope": curve["fit_slope"],
        "log_offset_hours": curve["log_offset_hours"],
        "t_last_hours": curve["t_last_hours"],
        "t_target_hours": curve["t_target_hours"],
        "winrate_at_last": float(winrates[-1]),
        "winrate_logit_at_last": float(logits[-1]),
    }
