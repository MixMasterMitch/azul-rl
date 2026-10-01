"""Explicit integer-width transfer for the source-attention architecture.

Uniformly replicated channels preserve LayerNorm statistics. Attention queries
are additionally scaled to compensate for the new per-head dot-product scale.
This is an inference-preserving initialization, not an optimizer-state resume.
Optional zero-sum outgoing-weight noise breaks replica symmetry without changing
the initial evaluation function, including in heads that do not use dropout.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import AzulNet


def widen_source_attention(
    teacher: AzulNet, hidden: int, *, symmetry_noise: float = 0.0
) -> AzulNet:
    """Return a wider standard AzulNet with the teacher's evaluation function."""
    old = teacher.hidden
    if teacher.arch != "source_attn" or hidden <= old or hidden % old:
        raise ValueError(
            "Widening requires source_attn and an integer width multiple greater than one"
        )
    if not math.isfinite(symmetry_noise) or not 0 <= symmetry_noise <= 0.1:
        raise ValueError("Symmetry noise must be finite and between zero and 0.1")
    heads = teacher.attn.num_heads
    factor = hidden // old
    device = next(teacher.parameters()).device
    dtype = next(teacher.parameters()).dtype
    with torch.random.fork_rng(devices=[]):
        student = AzulNet(
            hidden=hidden,
            arch="source_attn",
            num_heads=heads,
            dropout=teacher.attn.dropout,
        ).to(device=device, dtype=dtype)
    # Keep each old attention head within the same new head.
    channels = (
        torch.arange(old, device=device)
        .reshape(heads, old // heads)
        .repeat(1, factor)
        .flatten()
    )
    feedforward = torch.arange(2 * old, device=device).repeat(factor)

    def concatenated(count: int) -> torch.Tensor:
        return torch.cat([channels + i * old for i in range(count)])

    def linear(
        source: nn.Linear,
        target: nn.Linear,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        counts = torch.bincount(inputs, minlength=source.in_features).to(dtype=dtype)
        target.weight.copy_(
            source.weight[outputs][:, inputs] / counts[inputs].unsqueeze(0)
        )
        if symmetry_noise:
            # All replicas have identical activations at initialization. Each
            # original input's outgoing weights still sum to the old weight.
            noise = torch.randn_like(target.weight) * (
                symmetry_noise * target.weight.std()
            )
            totals = torch.zeros(
                (target.out_features, source.in_features), device=device, dtype=dtype
            )
            totals.index_add_(1, inputs, noise)
            noise.sub_(totals[:, inputs] / counts[inputs].unsqueeze(0))
            target.weight.add_(noise)
        if source.bias is not None:
            target.bias.copy_(source.bias[outputs])

    old_modules = dict(teacher.named_modules())
    with torch.no_grad():
        for name, target in student.named_modules():
            source = old_modules[name]
            if isinstance(target, nn.Embedding):
                target.weight.copy_(source.weight[:, channels])
            elif isinstance(target, nn.LayerNorm):
                target.weight.copy_(source.weight[channels])
                target.bias.copy_(source.bias[channels])
                target.eps = source.eps
            elif isinstance(target, nn.MultiheadAttention):
                counts = torch.bincount(channels, minlength=old).to(dtype=dtype)
                output = concatenated(3)
                target.in_proj_weight.copy_(
                    source.in_proj_weight[output][:, channels]
                    / counts[channels].unsqueeze(0)
                )
                target.in_proj_bias.copy_(source.in_proj_bias[output])
                # Replication multiplies q.k by factor, while the attention
                # denominator only grows by sqrt(factor).
                target.in_proj_weight[:hidden].div_(math.sqrt(factor))
                target.in_proj_bias[:hidden].div_(math.sqrt(factor))
            elif isinstance(target, nn.Linear):
                inputs, outputs = channels, channels
                if name in {"g_in.0", "s_embed.0"}:
                    inputs = torch.arange(source.in_features, device=device)
                elif name.startswith("source_encoder.") and name.endswith(".linear1"):
                    outputs = feedforward
                elif name.startswith("source_encoder.") and name.endswith(".linear2"):
                    inputs = feedforward
                elif name.startswith("policy_heads.") and name.endswith(".0"):
                    inputs = concatenated(3)
                elif name.startswith("value_heads.") and name.endswith(".0"):
                    inputs = concatenated(2)
                elif name.startswith(
                    ("policy_heads.", "value_heads.")
                ) and name.endswith(".2"):
                    outputs = torch.arange(source.out_features, device=device)
                linear(source, target, inputs, outputs)
    student.trained_player_counts = list(getattr(teacher, "trained_player_counts", []))
    return student.eval()


@torch.inference_mode()
def transfer_error(
    teacher: AzulNet,
    student: AzulNet,
    global_feat: torch.Tensor,
    source_feat: torch.Tensor,
    legal: torch.Tensor,
    num_players: int = 2,
) -> dict:
    teacher.eval()
    student.eval()
    p, v = teacher(global_feat, source_feat, legal, num_players)
    q, w = student(global_feat, source_feat, legal, num_players)
    probabilities = p.softmax(-1)
    kl = (probabilities * (p.log_softmax(-1) - q.log_softmax(-1))).sum(-1)
    return {
        "positions": len(p),
        "policy_kl_mean": float(kl.mean()),
        "policy_kl_max": float(kl.max()),
        "max_legal_logit_error": float((p - q)[legal].abs().max()),
        "value_mse": float((v[:, :num_players] - w[:, :num_players]).square().mean()),
        "value_max_error": float((v[:, :num_players] - w[:, :num_players]).abs().max()),
        "argmax_agreement": float((p.argmax(-1) == q.argmax(-1)).float().mean()),
    }
