"""Self-play should not train on positions from unfinished games."""

from __future__ import annotations

import torch

from agent.train.selfplay import _select_finished_samples


def test_select_finished_samples_drops_unfinished_game() -> None:
    ended = torch.tensor([True, False])
    game_idx = torch.tensor([0, 0, 1, 1])
    g = torch.randn(4, 8)
    s = torch.randn(4, 10, 6)
    legal = torch.ones(4, 300, dtype=torch.bool)
    p = torch.softmax(torch.randn(4, 300), dim=-1)
    v = torch.ones(4, 4)

    out = _select_finished_samples(g, s, legal, p, v, game_idx, ended)
    assert out is not None
    assert out[0].shape[0] == 2
