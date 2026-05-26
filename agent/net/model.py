"""Policy + value network for Azul.

Architecture:
- MLP trunk over `global_feat` with residual blocks → h_g (dim H)
- Learned player-count embedding (3×H) added to h_g
- Per-source MLP embedding of `source_feat` → h_s (B, NUM_SOURCES, H)
- 2-layer self-attention over sources, then cross-attention: query = h_g, KV = h_s
- Per-PC policy heads (3×): MLP over (h_g ‖ h_attn) → NUM_ACTIONS logits
- Per-PC value heads (3×): MLP over (h_g ‖ h_attn) → MAX_PLAYERS scalars
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..env import actions as A
from ..env import batched_engine as BE
from . import encoder as ENC

NUM_ACTIONS = A.NUM_ACTIONS


class _ResidualMLP(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class AzulNet(nn.Module):
    def __init__(
        self,
        hidden: int = 256,
        arch: str = "attn",
        num_heads: int = 4,
        dropout: float = 0.05,
        compile_forward: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        self.arch = arch
        self._compiled = False

        self.pc_embed = nn.Embedding(3, hidden)

        if arch == "attn":
            self.g_in = nn.Sequential(
                nn.Linear(ENC.D_GLOBAL, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.g_blocks = nn.ModuleList(
                [_ResidualMLP(hidden, dropout=dropout) for _ in range(2)]
            )
            self.s_embed = nn.Sequential(
                nn.Linear(ENC.D_SOURCE, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=num_heads,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.source_encoder = nn.TransformerEncoder(
                enc_layer, num_layers=2, enable_nested_tensor=False
            )
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.post_attn = nn.LayerNorm(hidden)
        elif arch == "flat":
            flat_dim = ENC.D_GLOBAL + ENC.NUM_SOURCES * ENC.D_SOURCE
            self.flat_trunk = nn.Sequential(
                nn.Linear(flat_dim, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
            )
        else:
            raise ValueError(f"unsupported AzulNet arch: {arch}")

        self.policy_heads = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, NUM_ACTIONS)
            )
            for i in range(3)
        })
        self.value_heads = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, BE.MAX_PLAYERS)
            )
            for i in range(3)
        })

        if compile_forward:
            self.enable_compile()

    def enable_compile(self) -> None:
        if self._compiled:
            return
        import os

        import torch._inductor.config

        os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
        torch._dynamo.config.capture_scalar_outputs = True
        self._compiled_forward = torch.compile(
            self._forward_impl, mode="reduce-overhead", dynamic=True
        )
        self._compiled_value_forward = torch.compile(
            self._forward_value_impl, mode="reduce-overhead", dynamic=True
        )
        self._compiled = True

    def forward(
        self,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        legal_mask: torch.Tensor,
        num_players: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._compiled:
            policy_logits, value = self._compiled_forward(
                global_feat, source_feat, num_players
            )
            policy_logits = policy_logits.clone()
            value = value.clone()
        else:
            policy_logits, value = self._forward_impl(
                global_feat, source_feat, num_players
            )
        policy_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        return policy_logits, value

    def forward_value(
        self,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        num_players: int = 2,
    ) -> torch.Tensor:
        """Value head only (skips policy) — used for MCTS child evaluation."""
        if self._compiled:
            return self._compiled_value_forward(
                global_feat, source_feat, num_players
            ).clone()
        return self._forward_value_impl(global_feat, source_feat, num_players)

    def _trunk(
        self,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        num_players: int,
    ) -> tuple[torch.Tensor, str]:
        pc_idx = num_players - 2
        pc_key = str(pc_idx)

        if self.arch == "attn":
            h_g = self.g_in(global_feat)
            pc_emb = self.pc_embed(
                torch.full((h_g.shape[0],), pc_idx, dtype=torch.long, device=h_g.device)
            )
            h_g = h_g + pc_emb
            for block in self.g_blocks:
                h_g = block(h_g)

            h_s = self.s_embed(source_feat)
            h_s = self.source_encoder(h_s)

            q = h_g.unsqueeze(1)
            h_attn, _ = self.attn(q, h_s, h_s)
            h_attn = self.post_attn(h_attn.squeeze(1))
            trunk_out = torch.cat([h_g, h_attn], dim=-1)
        else:
            flat_input = torch.cat(
                [global_feat, source_feat.reshape(global_feat.shape[0], -1)], dim=-1
            )
            trunk_out = self.flat_trunk(flat_input)
            pc_emb = self.pc_embed(
                torch.full((trunk_out.shape[0],), pc_idx, dtype=torch.long, device=trunk_out.device)
            )
            trunk_out[:, : self.hidden] = trunk_out[:, : self.hidden] + pc_emb

        return trunk_out, pc_key

    def _forward_impl(
        self,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        num_players: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trunk_out, pc_key = self._trunk(global_feat, source_feat, num_players)
        policy_logits = self.policy_heads[pc_key](trunk_out)
        value = torch.tanh(self.value_heads[pc_key](trunk_out))
        return policy_logits, value

    def _forward_value_impl(
        self,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        num_players: int,
    ) -> torch.Tensor:
        trunk_out, pc_key = self._trunk(global_feat, source_feat, num_players)
        return torch.tanh(self.value_heads[pc_key](trunk_out))
