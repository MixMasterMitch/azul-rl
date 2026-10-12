"""Initialize independent multiplayer specialists without modifying the 2p model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ..eval.arena import checkpoint_hash
from ..net.model import AzulNet
from .checkpointing import load_net_from_checkpoint, save_checkpoint


def initialize_specialist(source: str, destination: str, num_players: int) -> dict:
    """Transfer source-aware policy features; reset the multiplayer value readout.

    The multiplayer weights, optimizer, and replay are independent. A model's
    unused 2p head does not protect 2p once its trunk is trained: bundle routing
    must continue to use the separate frozen checkpoint for two-player games.
    """
    if num_players not in (3, 4):
        raise ValueError("A new specialist must have three or four players")
    target = Path(destination)
    if target.exists() or target.resolve() == Path(source).resolve():
        raise ValueError("Specialist initialization requires a new destination")
    source_hash = checkpoint_hash(source)
    net, payload = load_net_from_checkpoint(source)
    if net.arch != "source_attn" or payload.get("trained_player_counts") != [2]:
        raise ValueError("Expected a source-attention two-player baseline")
    if net.aux_score:
        raise ValueError("Multiplayer initialization does not use the 2p score head")
    key = str(num_players - 2)
    with torch.no_grad():
        net.pc_embed.weight[num_players - 2].copy_(net.pc_embed.weight[0])
        net.policy_heads[key].load_state_dict(net.policy_heads["0"].state_dict())
        net.value_heads[key].load_state_dict(net.value_heads["0"].state_dict())
        # A two-player opponent's negative value is not a multiplayer prediction.
        # Start every active seat at the symmetric binary-reward expectation.
        readout = net.value_heads[key][-1]
        readout.weight.zero_()
        readout.bias.zero_()
        readout.bias[:num_players].fill_(math.atanh(2 / num_players - 1))
    net.trained_player_counts = []  # Transfer initialization is not trained support.
    save_checkpoint(target, net, progress={"initialization_only": True})
    if checkpoint_hash(source) != source_hash:
        raise RuntimeError("Frozen baseline changed during initialization")
    return {
        "num_players": num_players,
        "source": str(Path(source).resolve()),
        "source_sha256": source_hash,
        "checkpoint": str(target.resolve()),
        "checkpoint_sha256": checkpoint_hash(target),
        "policy_initialization": "copy 2p policy head, embedding, and backbone",
        "value_initialization": "copied hidden layer; constant 2/N-1 active outputs",
        "optimizer": "fresh",
        "replay": "empty; exclusively this player count",
    }


def load_specialist(
    manifest: str | Path, num_players: int, device: str = "cpu"
) -> AzulNet:
    """Load an explicitly routed, hash-verified player-count model.

    There is deliberately no fallback from multiplayer weights to 2p. This
    loader does not change the play server's model selection or deploy models.
    """
    if num_players not in (2, 3, 4):
        raise ValueError("num_players must be two, three, or four")
    bundle = json.loads(Path(manifest).read_text())
    entry = bundle["models"][str(num_players)]
    if num_players == 2 and entry != bundle["frozen_2p"]:
        raise ValueError("The two-player route must remain the frozen baseline")
    if checkpoint_hash(entry["checkpoint"]) != entry["checkpoint_sha256"]:
        raise ValueError("Specialist checkpoint identity changed")
    net, payload = load_net_from_checkpoint(entry["checkpoint"], device)
    if num_players not in payload.get("trained_player_counts", []):
        raise ValueError("Checkpoint is not marked trained for this player count")
    return net.eval()
