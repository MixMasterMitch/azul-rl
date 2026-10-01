from dataclasses import replace
from pathlib import Path

import pytest
import torch

from agent.env.engine import GameEngine
from agent.net.model import AzulNet
from agent.search.config import SearchConfig
from agent.tests.test_training_enhancements import add_positions, make_buffer
from agent.tests.test_finetune_campaign import assert_same
from agent.train import learner
from agent.train.checkpointing import save_checkpoint, load_checkpoint
from agent.train.distillation import PolicyTeacher, file_hash
from agent.train.loop import LoopConfig
from agent.train.reproducibility import capture_rng_state


def teacher_config(tmp_path: Path) -> LoopConfig:
    path = tmp_path / "teacher.pt"
    save_checkpoint(path, AzulNet(hidden=32, arch="flat"), config={"num_players": 2})
    return LoopConfig(
        search_backend="gumbel_tree",
        distillation_teacher=str(path),
        distillation_teacher_sha256=file_hash(str(path)),
        distillation_capacity=8,
    )


def test_teacher_refresh_preserves_outcomes_original_replay_and_frozen_weights(
    tmp_path: Path,
) -> None:
    cfg = teacher_config(tmp_path)
    teacher = PolicyTeacher(cfg, "cpu")
    replay = make_buffer(8, 8)
    add_positions(replay, GameEngine(8, 2, seed=1))
    replay.value_target[:, :2] = torch.tensor([[1.0, -1.0], [0.0, 0.0]] * 4)
    original = {
        key: val.clone()
        for key, val in replay.state_dict().items()
        if torch.is_tensor(val)
    }
    weights = {k: v.clone() for k, v in teacher.net.state_dict().items()}
    search = SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=8,
        max_root_candidates=4,
    )
    result = teacher.refresh(replay, 2, search, 8, 4, 77)
    assert result["positions"] == teacher.bank.size == 8
    for key, val in original.items():
        assert torch.equal(val, replay.state_dict()[key])
    for i in range(8):
        source = (
            (replay.global_feat == teacher.bank.global_feat[i])
            .all(-1)
            .nonzero()
            .flatten()
        )
        assert len(source) == 1
        assert torch.equal(teacher.bank.value_target[i], replay.value_target[source[0]])
    assert (teacher.bank.policy_target[~teacher.bank.legal_mask] == 0).all()
    torch.testing.assert_close(teacher.bank.policy_target.sum(-1), torch.ones(8))
    assert (teacher.bank.policy_sims == 8).all()
    assert_same(weights, teacher.net.state_dict())
    assert not teacher.net.training and all(
        not p.requires_grad for p in teacher.net.parameters()
    )
    assert file_hash(cfg.distillation_teacher) == cfg.distillation_teacher_sha256
    with pytest.raises(ValueError, match="hash mismatch"):
        PolicyTeacher(replace(cfg, distillation_teacher_sha256="0" * 64), "cpu")


def test_mixture_has_declared_policy_share_despite_fast_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay, teacher = make_buffer(4), make_buffer(4)
    add_positions(replay, GameEngine(4, 2, seed=1))
    add_positions(teacher, GameEngine(4, 2, seed=2))
    replay.value_target.fill_(-1)
    teacher.value_target.fill_(1)
    replay.policy_sims.fill_(64)
    teacher.policy_sims.fill_(1024)

    def inspect(net, optim, g, s, legal, policy, value, **kwargs):
        assert (value[:4] == -1).all() and (value[4:] == 1).all()
        weights = kwargs["policy_weights"]
        assert weights[:4].sum() == weights[4:].sum()
        return {"skipped": 0.0}

    monkeypatch.setattr(learner, "step", inspect)
    metrics = learner.step_from_buffer(
        None,
        replay,
        None,
        8,
        policy_fast_weight=0.25,
        teacher_buffer=teacher,
        teacher_fraction=0.5,
    )
    assert metrics["distillation_fraction"] == 0.5
    assert replay.total_sampled == teacher.total_sampled == 4
    with pytest.raises(ValueError, match="populated bank"):
        learner.step_from_buffer(None, replay, None, 8, teacher_fraction=0.5)


def test_atomic_full_resume_reproduces_next_mixed_update(tmp_path: Path) -> None:
    cfg = teacher_config(tmp_path)
    teacher = PolicyTeacher(cfg, "cpu")
    replay = make_buffer(8, 8)
    add_positions(replay, GameEngine(8, 2, seed=1))
    teacher.refresh(
        replay,
        2,
        SearchConfig(backend="gumbel_tree", tree_core="rust", num_simulations=8),
        8,
        8,
        77,
    )
    student = AzulNet(hidden=16, arch="flat")
    optim = learner.make_optimizer(student)
    kwargs = {"teacher_fraction": 0.5, "policy_fast_weight": 0.25}
    learner.step_from_buffer(
        student, replay, optim, 8, teacher_buffer=teacher.bank, **kwargs
    )
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint,
        student,
        optim,
        1,
        {"num_players": 2},
        replay,
        distillation_state=teacher.state_dict(),
    )
    expected_metrics = learner.step_from_buffer(
        student, replay, optim, 8, teacher_buffer=teacher.bank, **kwargs
    )
    expected_rng = capture_rng_state()
    restored = AzulNet(hidden=16, arch="flat")
    restored_optim = learner.make_optimizer(restored)
    restored_teacher = PolicyTeacher(cfg, "cpu")
    restored_replay = make_buffer(8, 8)
    payload = load_checkpoint(checkpoint, restored, restored_optim, restored_replay)
    restored_teacher.load_state_dict(payload["distillation"])
    actual_metrics = learner.step_from_buffer(
        restored,
        restored_replay,
        restored_optim,
        8,
        teacher_buffer=restored_teacher.bank,
        **kwargs,
    )
    assert actual_metrics == expected_metrics
    assert_same(student.state_dict(), restored.state_dict())
    assert_same(optim.state_dict(), restored_optim.state_dict())
    assert_same(expected_rng, capture_rng_state())
    payload["distillation"]["teacher_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="teacher differs"):
        restored_teacher.load_state_dict(payload["distillation"])
