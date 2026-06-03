"""Unified eval policy configuration."""

from __future__ import annotations

import torch

from agent.eval.tournament import EvalConfig
from agent.net.model import AzulNet
from agent.train import unified_eval as UE


def test_unified_eval_default_temperature_matches_tournament() -> None:
    assert UE.UnifiedEvalConfig().temperature == EvalConfig().temperature


def test_unified_eval_passes_configured_temperature(monkeypatch) -> None:
    seen: list[float] = []

    def fake_gumbel_root_act(engine, net, num_sims, temperature, q_scale):
        del net, num_sims, q_scale
        seen.append(float(temperature))
        legal = engine.legal_action_mask()
        actions = legal.to(torch.int64).argmax(dim=-1)
        improved = torch.zeros_like(legal, dtype=torch.float32)
        return actions, improved

    monkeypatch.setattr(UE.G, "gumbel_root_act", fake_gumbel_root_act)

    net = AzulNet(hidden=32, arch="flat")
    config = UE.UnifiedEvalConfig(
        total_games=1,
        num_sims=2,
        max_turns=1,
        turns_per_player=0,
        temperature=0.125,
        weight_2p=1.0,
        weight_3p=0.0,
        weight_4p=0.0,
    )
    UE.run_unified_eval(net.state_dict(), [], config, seed=1, hidden=32, arch="flat")

    assert seen == [0.125]
