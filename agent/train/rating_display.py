"""Versioned rating presentation; statistical fits and stored league values stay unchanged.

The default reference is the frozen 19,456-game Astra ranking campaign. Legacy
leagues/registries must supply their own raw reference anchors before conversion.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

VERSION = "heuristic-2500-v1"
RANDOM = 1000.0
HEURISTIC = 2500.0
PLAYER_COUNTS = (2, 3, 4)
CAMPAIGN_HEURISTIC = {
    2: 5010.883103143318,
    3: 5828.328052855152,
    4: 6085.083562651204,
}


def scales_for(reference_anchors: Mapping | None = None) -> dict[int, float]:
    """Freeze reference values, not rounded factors or each new fitted heuristic."""
    try:
        heuristic = (CAMPAIGN_HEURISTIC if reference_anchors is None else
                     {int(pc): values["heuristic"] for pc, values in reference_anchors.items()})
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Display reference requires a heuristic rating for each supplied player count") from exc
    scales = {}
    for pc, value in heuristic.items():
        if pc not in PLAYER_COUNTS or not math.isfinite(value) or value <= RANDOM:
            raise ValueError("Display reference requires a finite heuristic rating above 1000 at 2p/3p/4p")
        if reference_anchors is not None:
            anchors = reference_anchors.get(pc, reference_anchors.get(str(pc)))
            if float(anchors.get("random", RANDOM)) != RANDOM:
                raise ValueError("The rating display reference must anchor random at 1000")
        scales[pc] = (HEURISTIC - RANDOM) / (value - RANDOM)
    return scales


SCALES = scales_for()


def to_display(raw: float, players: int, scales: Mapping[int, float] | None = None) -> float:
    """Apply exactly one positive affine transformation around random=1000."""
    values = SCALES if scales is None else scales
    if players not in PLAYER_COUNTS or players not in values:
        raise ValueError("A rating reference is required for the selected player count")
    scale = values[players]
    if not math.isfinite(raw) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Rating and positive display scale must be finite")
    return RANDOM + (raw - RANDOM) * scale


def from_display(value: float, players: int, scales: Mapping[int, float] | None = None) -> float:
    values = SCALES if scales is None else scales
    # Apply the same validation as the forward conversion.
    to_display(value, players, values)
    return RANDOM + (value - RANDOM) / values[players]


def metadata(reference_anchors: Mapping | None = None) -> dict:
    scales = scales_for(reference_anchors)
    return {"version": VERSION, "random": RANDOM, "heuristic_target": HEURISTIC,
            "reference_heuristic": {str(pc): RANDOM + (HEURISTIC - RANDOM) / scale
                                    for pc, scale in scales.items()},
            "multipliers": {str(pc): scale for pc, scale in scales.items()},
            "formula": "1000 + multiplier[players] * (raw_rating - 1000)"}


def from_legacy_calibrated(value: float, players: int, reference_anchors: Mapping) -> float:
    """Undo the old Opus-based display calibration before applying v1."""
    from .ranking import calibration_scales_for

    references = {int(pc): values for pc, values in reference_anchors.items()}
    old_scale = calibration_scales_for(references)[players]
    raw = RANDOM + (value - RANDOM) / old_scale
    return to_display(raw, players, scales_for(references))
