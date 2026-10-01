from __future__ import annotations

from dataclasses import replace
import random

import pytest
import torch

from agent.env.engine import GameEngine
from agent.net import encoder as ENC
from agent.net.model import AzulNet
from agent.scripts.train import parse_config
from agent.search.config import SearchConfig
from agent.train.league import League
from agent.train.loop import LoopConfig, apply_device_defaults, teacher_simulations
from agent.train.reanalysis import SnapshotRecorder, reanalyse_policy_targets
from agent.train.replay_buffer import ReplayBuffer


def make_buffer(capacity: int = 32, snapshots: int = 16) -> ReplayBuffer:
    return ReplayBuffer(
        capacity,
        ENC.D_GLOBAL,
        ENC.NUM_SOURCES,
        ENC.D_SOURCE,
        300,
        4,
        snapshot_capacity=snapshots,
    )


def add_positions(buffer: ReplayBuffer, engine: GameEngine) -> None:
    g, s = ENC.encode_state(engine)
    legal = engine.legal_action_mask()
    policy = legal.float() / legal.sum(-1, keepdim=True)
    values = torch.full((engine.batch_size, 4), -0.5)
    buffer.add(
        g,
        s,
        legal,
        policy,
        values,
        snapshots=dict(enumerate(engine.state_dict()["snapshots"])),
    )


def test_replay_snapshot_wrap_oversize_resume_and_old_rewards() -> None:
    buffer = make_buffer(8, 4)
    add_positions(buffer, GameEngine(6, 2, seed=2))
    add_positions(buffer, GameEngine(4, 2, seed=3))
    assert set(buffer.snapshots) == {6, 7, 0, 1}
    add_positions(buffer, GameEngine(12, 2, seed=4))
    assert len(buffer.snapshots) == 4
    for index, snapshot in buffer.snapshots.items():
        engine = GameEngine.from_state_dict(
            {"backend": "rust", "version": 1, "num_players": 2, "snapshots": [snapshot]}
        )
        torch.testing.assert_close(
            ENC.encode_state(engine)[0][0], buffer.global_feat[index]
        )
    resumed = make_buffer(8, 4)
    resumed.load_state_dict(buffer.state_dict())
    assert resumed.snapshots == buffer.snapshots
    legacy = buffer.state_dict()
    legacy.pop("reward_semantics_version")
    with pytest.raises(ValueError, match="old tie rewards"):
        resumed.load_state_dict(legacy)
    # Overwriting without state capture must evict every stale snapshot.
    buffer.add(
        *(
            t[:8]
            for t in (
                buffer.global_feat.clone(),
                buffer.source_feat.clone(),
                buffer.legal_mask.clone(),
                buffer.policy_target.clone(),
                buffer.value_target.clone(),
            )
        )
    )
    assert not buffer.snapshots


def test_recorder_filters_unfinished_positions_and_does_not_change_torch_rng() -> None:
    engine = GameEngine(128, 2, seed=9)
    recorder = SnapshotRecorder(6, seed=1)
    rng_before = torch.get_rng_state()
    recorder.capture(engine, torch.arange(128))
    assert torch.equal(rng_before, torch.get_rng_state())
    keep = torch.arange(128) % 3 == 0
    selected = recorder.selected(keep)
    assert 0 < len(selected) <= 6
    expected = engine.index_select(keep.nonzero().flatten()).state_dict()["snapshots"]
    assert all(snapshot == expected[index] for index, snapshot in selected.items())


def test_reanalysis_refreshes_policies_without_replacing_outcomes_or_age() -> None:
    torch.manual_seed(4)
    buffer = make_buffer()
    buffer.iteration = 7
    add_positions(buffer, GameEngine(4, 2, seed=4))
    before_policy = buffer.policy_target.clone()
    before_values = buffer.value_target.clone()
    before_age = buffer.inserted_at.clone()
    before_counts = (buffer.total_added, buffer.total_sampled, buffer.pos, buffer.size)
    net = AzulNet(hidden=32, arch="flat", dropout=0).train()
    cfg = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=8,
        max_root_candidates=4,
        inference_cache_size=32,
    )
    metrics = reanalyse_policy_targets(net, buffer, 2, cfg, 4, batch_size=2, seed=22)
    assert metrics["positions"] == 4 and metrics["mean_policy_l1_change"] > 0
    assert not torch.equal(buffer.policy_target, before_policy)
    assert torch.equal(buffer.value_target, before_values)
    assert torch.equal(buffer.inserted_at, before_age)
    assert before_counts == (
        buffer.total_added,
        buffer.total_sampled,
        buffer.pos,
        buffer.size,
    )
    assert net.training
    buffer.snapshots[0] = GameEngine(1, 2, seed=999).state_dict()["snapshots"][0]
    with pytest.raises(ValueError, match="does not match"):
        reanalyse_policy_targets(net, buffer, 2, cfg, 4, seed=22)
    assert net.training


def test_mixed_teacher_budget_and_explicit_cli_overrides() -> None:
    cfg, explicit = parse_config(
        [
            "--preset",
            "enhanced-2p",
            "--run-id",
            "enhanced-test",
            "--selfplay-full-fraction",
            "0.5",
            "--learner-steps",
            "64",
        ]
    )
    assert (
        cfg.arch == "source_attn"
        and cfg.reward_mode == "binary"
        and cfg.dirichlet_mix == 0
    )
    assert cfg.league_search_backend == "gumbel_tree" and cfg.league_opponent_sims == 64
    assert cfg.league_root.endswith("enhanced-test/league")
    assert cfg.bot_selfplay_astra_prob == 0.5 and cfg.reanalysis_positions == 256
    assert apply_device_defaults(cfg, "cuda", explicit).learner_steps_per_iter == 64
    sequence = [teacher_simulations(cfg, i) for i in range(100)]
    assert set(sequence) == {64, 256}
    assert sequence == [teacher_simulations(cfg, i) for i in range(100)]
    assert teacher_simulations(replace(cfg, selfplay_full_fraction=0), 1) == 64
    assert teacher_simulations(replace(cfg, selfplay_full_fraction=1), 1) == 256
    with pytest.raises(ValueError, match="requires tree"):
        LoopConfig(reanalysis_positions=1)


def test_mixed_league_samples_old_strong_and_recent_entries(tmp_path) -> None:
    league = League(tmp_path, max_entries=None, keep_recent=2)
    for i in range(12):
        path = tmp_path / f"{i}.pt"
        path.touch()
        league.manifest["entries"].append(
            {
                "idx": i,
                "path": path.name,
                "active": i != 5,
                "rating_2p": 2000 if i == 0 else 1000,
                "rating": 1000,
            }
        )
    rng = random.Random(3)
    counts = [0] * 12
    for _ in range(1200):
        counts[league.sample_opponent(rng, "mixed", 2)["idx"]] += 1
    assert counts[5] == 0
    assert counts[0] > counts[7] and counts[11] > counts[7] > 0
