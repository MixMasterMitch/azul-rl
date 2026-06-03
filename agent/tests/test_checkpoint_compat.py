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
