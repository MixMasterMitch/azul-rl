from __future__ import annotations
import torch
from agent.train.replay_buffer import ReplayBuffer


def test_oversized_add_retains_newest_positions_and_counts() -> None:
    b = ReplayBuffer(3, 1, 1, 1, 1, 1)
    values = torch.arange(8).float()[:, None]
    b.iteration = 10
    b.add(values, values[:, None, :], torch.ones(8, 1, dtype=torch.bool), values, values)
    assert b.size == 3 and b.total_added == 8
    assert set(b.global_feat[:, 0].tolist()) == {5., 6., 7.}
    b.iteration = 12
    b.sample(8)
    assert b.total_sampled == 8 and b.last_sample_age == 2
    other = ReplayBuffer(3, 1, 1, 1, 1, 1)
    other.load_state_dict(b.state_dict())
    assert other.total_added == 8 and other.total_sampled == 8
    assert torch.equal(other.inserted_at, b.inserted_at)
