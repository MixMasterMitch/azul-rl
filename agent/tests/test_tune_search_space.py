"""Tests for Optuna tuning search-space constraints."""

import itertools

import optuna
import pytest

from agent.scripts.tune import (
    FIXED_SELFPLAY_GAMES,
    FIXED_SELFPLAY_SIMS,
    MHA_CHILD_BATCH_LIMIT,
    OBJECTIVE_OPUS_WINRATE,
    OBJECTIVE_RATING_2P,
    OPUS_ONLY_OPPONENTS,
    SELFPLAY_GAMES_CHOICES,
    SELFPLAY_SIMS_CHOICES,
    TUNING_AGENT_ENTITY,
    _baseline_eval_path,
    _rating_entity_for_opponent,
    _winrate_to_match_result,
    ensure_selfplay_mha_limit,
    selfplay_exceeds_mha_limit,
)
from agent.train import ranking as R
from agent.train.ranking import calibrate_rating, calibration_scales_for, fit_ratings_for_pc
from agent.train.tuning_curve import extrapolate_winrate


def test_mha_limit_pairs() -> None:
    invalid = [
        (games, sims)
        for games, sims in itertools.product(SELFPLAY_GAMES_CHOICES, SELFPLAY_SIMS_CHOICES)
        if selfplay_exceeds_mha_limit(games, sims)
    ]
    assert (2047, 64) in invalid
    assert (4095, 32) in invalid
    assert (4095, 64) in invalid
    assert (511, 64) not in invalid
    assert (1023, 64) not in invalid
    assert (2047, 32) not in invalid
    assert (4095, 16) not in invalid


def test_ensure_prunes_over_limit() -> None:
    with pytest.raises(optuna.TrialPruned, match="131008"):
        ensure_selfplay_mha_limit(2047, 64)


def test_ensure_ok_under_limit() -> None:
    ensure_selfplay_mha_limit(1023, 64)
    assert 1023 * 64 <= MHA_CHILD_BATCH_LIMIT


def test_fixed_selfplay_batch_under_mha_limit() -> None:
    ensure_selfplay_mha_limit(FIXED_SELFPLAY_GAMES, FIXED_SELFPLAY_SIMS)
    assert FIXED_SELFPLAY_GAMES * FIXED_SELFPLAY_SIMS == 1023 * 32


def test_trial_rating_2p_from_winrates() -> None:
    results: list[dict] = []
    _winrate_to_match_result(results, TUNING_AGENT_ENTITY, "random", 1.0, 100)
    _winrate_to_match_result(results, TUNING_AGENT_ENTITY, "heuristic", 0.0, 100)
    _winrate_to_match_result(
        results,
        TUNING_AGENT_ENTITY,
        _rating_entity_for_opponent("opus"),
        0.0,
        100,
    )
    ref = R.reference_anchors_from_manifest(None)
    raw = fit_ratings_for_pc(
        results,
        2,
        anchors={"random": R.RANDOM_ANCHOR_RATING},
        use_reference_anchors=True,
        reference_anchors_per_pc=ref,
    )[TUNING_AGENT_ENTITY]
    cal = calibrate_rating(raw, 2, calibration_scales_for(ref))
    assert cal > R.RANDOM_ANCHOR_RATING


def test_tuning_opus_rating_entity_uses_reference_anchor_name() -> None:
    assert _rating_entity_for_opponent("opus") == "heuristic_opus"
    assert _rating_entity_for_opponent("heuristic") == "heuristic"


def test_opus_objective_uses_only_opus_opponent() -> None:
    assert [name for name, _ in OPUS_ONLY_OPPONENTS] == ["opus"]


def test_baseline_cache_path_is_objective_specific() -> None:
    assert _baseline_eval_path("runs", OBJECTIVE_RATING_2P).name == "tune_baseline_eval.json"
    assert (
        _baseline_eval_path("runs", OBJECTIVE_OPUS_WINRATE).name
        == "tune_baseline_eval_opus_winrate.json"
    )


def test_extrapolate_winrate_projects_bounded_curve() -> None:
    curve = extrapolate_winrate(
        [0.0, 0.5, 1.0, 2.0],
        [0.30, 0.35, 0.40, 0.45],
        extra_hours=72.0,
    )
    assert 0.0 < curve["predicted_winrate"] < 1.0
    assert curve["predicted_winrate"] > curve["winrate_at_last"]
    assert curve["t_target_hours"] == 74.0
