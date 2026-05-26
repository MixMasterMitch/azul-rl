"""Fixed-capacity ring buffer for training samples."""

from __future__ import annotations

import torch


class ReplayBuffer:
    """Ring buffer storing (global_feat, source_feat, legal_mask, policy_target, value_target)."""

    def __init__(
        self,
        capacity: int,
        d_global: int,
        n_sources: int,
        d_source: int,
        num_actions: int,
        max_players: int,
        device: torch.device | str = "cpu",
    ):
        self.capacity = capacity
        self.device = torch.device(device)
        self.size = 0
        self.pos = 0

        self.global_feat = torch.zeros((capacity, d_global), dtype=torch.float32, device=self.device)
        self.source_feat = torch.zeros((capacity, n_sources, d_source), dtype=torch.float32, device=self.device)
        self.legal_mask = torch.zeros((capacity, num_actions), dtype=torch.bool, device=self.device)
        self.policy_target = torch.zeros((capacity, num_actions), dtype=torch.float32, device=self.device)
        self.value_target = torch.zeros((capacity, max_players), dtype=torch.float32, device=self.device)

    def add(
        self,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        legal_mask: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> None:
        """Add a batch of samples to the buffer."""
        n = global_feat.shape[0]
        if n == 0:
            return

        end = self.pos + n
        if end <= self.capacity:
            self.global_feat[self.pos:end] = global_feat
            self.source_feat[self.pos:end] = source_feat
            self.legal_mask[self.pos:end] = legal_mask
            self.policy_target[self.pos:end] = policy_target
            self.value_target[self.pos:end] = value_target
        else:
            # Wrap around
            first = self.capacity - self.pos
            self.global_feat[self.pos:] = global_feat[:first]
            self.source_feat[self.pos:] = source_feat[:first]
            self.legal_mask[self.pos:] = legal_mask[:first]
            self.policy_target[self.pos:] = policy_target[:first]
            self.value_target[self.pos:] = value_target[:first]

            remainder = n - first
            self.global_feat[:remainder] = global_feat[first:]
            self.source_feat[:remainder] = source_feat[first:]
            self.legal_mask[:remainder] = legal_mask[first:]
            self.policy_target[:remainder] = policy_target[first:]
            self.value_target[:remainder] = value_target[first:]

        self.pos = end % self.capacity
        self.size = min(self.size + n, self.capacity)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "pos": self.pos,
            "global_feat": self.global_feat.cpu(),
            "source_feat": self.source_feat.cpu(),
            "legal_mask": self.legal_mask.cpu(),
            "policy_target": self.policy_target.cpu(),
            "value_target": self.value_target.cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("capacity") != self.capacity:
            raise ValueError("replay buffer capacity mismatch on resume")
        self.size = int(state["size"])
        self.pos = int(state["pos"])
        dev = self.device
        self.global_feat.copy_(state["global_feat"].to(dev))
        self.source_feat.copy_(state["source_feat"].to(dev))
        self.legal_mask.copy_(state["legal_mask"].to(dev))
        self.policy_target.copy_(state["policy_target"].to(dev))
        self.value_target.copy_(state["value_target"].to(dev))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample a random minibatch."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.global_feat[indices],
            self.source_feat[indices],
            self.legal_mask[indices],
            self.policy_target[indices],
            self.value_target[indices],
        )
