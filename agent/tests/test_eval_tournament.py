"""Smoke tests for tournament evaluation."""

from __future__ import annotations

from agent.eval.tournament import combined_winrate, evaluate_checkpoint
from agent.net.model import AzulNet


def test_combined_winrate_averages_opponents() -> None:
    metrics = {
        "vs_random_winrate": 0.5,
        "vs_heuristic_winrate": 0.25,
        "vs_opus_winrate": 0.0,
    }
    assert combined_winrate(metrics) == 0.25


def test_evaluate_checkpoint_smoke_cpu() -> None:
    net = AzulNet(hidden=64, arch="flat")
    metrics = evaluate_checkpoint(
        net,
        num_games=2,
        num_sims=2,
        device="cpu",
        q_scale=5.0,
    )
    assert "vs_random_winrate" in metrics
    assert "vs_heuristic_winrate" in metrics
    assert "vs_opus_winrate" in metrics
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
