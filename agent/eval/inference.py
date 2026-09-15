"""Batch network inference on a device independent of game simulation."""
from __future__ import annotations
import torch
from ..net.model import AzulNet


class InferenceModel:
    def __init__(self, net: AzulNet, device: str = 'cpu', batch_size: int = 2048) -> None:
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.batch_size = batch_size

    @torch.inference_mode()
    def __call__(self, g: torch.Tensor, s: torch.Tensor, legal: torch.Tensor, num_players: int) -> tuple[torch.Tensor, torch.Tensor]:
        policies, values = [], []
        for start in range(0, len(g), self.batch_size):
            part = slice(start, start + self.batch_size)
            p, v = self.net(g[part].to(self.device), s[part].to(self.device), legal[part].to(self.device), num_players)
            policies.append(p.to(g.device))
            values.append(v.to(g.device))
        return torch.cat(policies), torch.cat(values)

    @torch.inference_mode()
    def forward_value(self, g: torch.Tensor, s: torch.Tensor, num_players: int) -> torch.Tensor:
        return torch.cat([self.net.forward_value(g[i:i+self.batch_size].to(self.device),
                         s[i:i+self.batch_size].to(self.device), num_players).to(g.device)
                         for i in range(0, len(g), self.batch_size)])
