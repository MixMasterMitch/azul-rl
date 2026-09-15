from __future__ import annotations

import torch

from agent.net import encoder as ENC
from agent.net.model import AzulNet
from agent.train.checkpointing import load_model_state_dict_compatible


def test_v2_global_input_weights_migrate_to_richer_encoder() -> None:
    hidden = 8
    target = AzulNet(hidden=hidden, arch="attn")
    state = AzulNet(hidden=hidden, arch="attn").state_dict()

    old_weight = torch.zeros((hidden, 171), dtype=state["g_in.0.weight"].dtype)
    old_row2_fill = 19 + 2 * 2
    old_row2_has_color = old_row2_fill + 1
    old_score = 19 + 36
    old_weight[:, old_row2_fill] = 5.0
    old_weight[:, old_row2_has_color] = 7.0
    old_weight[:, old_score] = 11.0
    state["g_in.0.weight"] = old_weight

    migrated = load_model_state_dict_compatible(target, state)

    assert migrated == ["g_in.0.weight"]
    new_weight = target.state_dict()["g_in.0.weight"]
    new_row2 = 19 + 2 * ENC.D_PATTERN_LINE
    assert torch.all(new_weight[:, new_row2] == 5.0)
    assert torch.all(new_weight[:, new_row2 + 1 : new_row2 + 1 + ENC.NUM_COLORS] == 7.0)
    new_score = 19 + ENC.D_PATTERN + ENC.D_WALL_FLAT + ENC.D_FLOOR
    assert torch.all(new_weight[:, new_score] == 11.0)
    new_floor_tiles = 19 + ENC.D_PATTERN + ENC.D_WALL_FLAT + 1
    assert torch.all(new_weight[:, new_floor_tiles : new_floor_tiles + ENC.NUM_COLORS] == 0.0)


def test_source_attn_warm_start_copies_only_compatible_weights() -> None:
    from agent.train.checkpointing import warm_start_net
    source = AzulNet(hidden=32, arch="attn")
    target = AzulNet(hidden=32, arch="source_attn")
    original_policy = target.policy_heads.state_dict()
    warm_start_net(target, {"hidden": 32, "arch": "attn", "model_state_dict": source.state_dict()})
    for key, value in source.state_dict().items():
        if not key.startswith("policy_heads."):
            assert torch.equal(target.state_dict()[key], value)
    for key, value in original_policy.items():
        assert torch.equal(target.policy_heads.state_dict()[key], value)


def test_checkpoint_restores_all_rngs(tmp_path) -> None:
    import random
    import numpy as np
    from agent.train.checkpointing import save_checkpoint, load_checkpoint
    from agent.train.reproducibility import seed_all
    seed_all(922)
    net = AzulNet(hidden=32, arch="flat")
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, net)
    expected = (random.random(), np.random.rand(), torch.rand(5))
    gpu_expected = torch.rand(5, device="cuda") if torch.cuda.is_available() else None
    load_checkpoint(path, net)
    assert expected[0] == random.random()
    assert expected[1] == np.random.rand()
    assert torch.equal(expected[2], torch.rand(5))
    if gpu_expected is not None:
        assert torch.equal(gpu_expected, torch.rand(5, device="cuda"))


def test_compressed_replay_is_lossless_and_loads_with_pytorch(tmp_path) -> None:
    import zipfile
    from agent.train.checkpointing import save_checkpoint
    from agent.train.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(128, 9, 2, 5, 30, 4)
    buffer.add(torch.rand(73, 9), torch.rand(73, 2, 5), torch.rand(73, 30) > .5,
               torch.rand(73, 30), torch.rand(73, 4))
    path = tmp_path/'resume.pt'
    save_checkpoint(path, AzulNet(hidden=32, arch='flat'), buffer=buffer)
    with zipfile.ZipFile(path) as archive:
        assert all(member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist())
    payload = torch.load(path, weights_only=False)
    assert payload['checkpoint_compression'] == 'deflate'
    for key, value in buffer.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, payload['buffer'][key]), key
        else:
            assert value == payload['buffer'][key]


def test_failed_compressed_replacement_preserves_previous_checkpoint(tmp_path, monkeypatch) -> None:
    import pytest
    from agent.train import checkpointing as CK
    from agent.train.replay_buffer import ReplayBuffer
    net = AzulNet(hidden=32, arch='flat')
    path = tmp_path/'resume.pt'
    CK.save_checkpoint(path, net, iteration=12)
    original = path.read_bytes()
    def fail(payload: dict, temporary) -> None:
        temporary.write_bytes(b'incomplete archive')
        raise OSError('disk full')
    monkeypatch.setattr(CK, '_write_compressed_checkpoint', fail)
    with pytest.raises(OSError, match='disk full'):
        CK.save_checkpoint(path, net, iteration=13, buffer=ReplayBuffer(16, 9, 2, 5, 30, 4))
    assert path.read_bytes() == original
    assert not path.with_suffix('.tmp').exists()
