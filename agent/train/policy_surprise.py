"""Search improvement measured against the generating network's untempered prior."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ..net.model import AzulNet
from .replay_buffer import ReplayBuffer


@torch.no_grad()
def policy_surprise(
    net: AzulNet,
    buffer: ReplayBuffer,
    global_feat: torch.Tensor,
    source_feat: torch.Tensor,
    legal: torch.Tensor,
    target: torch.Tensor,
    simulations: int,
    num_players: int,
) -> torch.Tensor | None:
    """Return KL(search target || raw network prior), or unknown when not collected.

    The generator has not updated the network during the trajectory. Recomputing
    its raw prior in batches avoids changing search/RNG behavior or recording a
    temperature/noise-adjusted search prior. Fast targets remain unknown.
    """
    minimum = buffer.policy_surprise_min_sims
    if not minimum or simulations < minimum:
        return None
    was_training = net.training
    net.eval()
    device = next(net.parameters()).device
    values = []
    try:
        for start in range(0, len(global_feat), 1024):
            part = slice(start, start + 1024)
            mask = legal[part].to(device)
            logits, _ = net(
                global_feat[part].to(device),
                source_feat[part].to(device),
                mask,
                num_players,
            )
            p = target[part].to(device).float() * mask
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
            kl = (
                F.kl_div(F.log_softmax(logits.float(), -1), p, reduction="none")
                .sum(-1)
                .clamp_min(0)
            )
            if not torch.isfinite(kl).all():
                raise ValueError("Nonfinite policy surprise")
            values.append(kl.to(buffer.device))
    finally:
        net.train(was_training)
    return torch.cat(values) if values else torch.empty(0, device=buffer.device)
