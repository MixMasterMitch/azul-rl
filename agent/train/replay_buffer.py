"""Fixed-capacity ring buffer for training samples."""

from __future__ import annotations

import math
import torch
from ..env.outcomes import REWARD_SEMANTICS_VERSION


class ReplayBuffer:
    """Ring buffer of observations/targets, with policy search-budget metadata."""

    def __init__(
        self,
        capacity: int,
        d_global: int,
        n_sources: int,
        d_source: int,
        num_actions: int,
        max_players: int,
        device: torch.device | str = "cpu",
        snapshot_capacity: int = 0,
        policy_surprise_min_sims: int = 0,
    ):
        if snapshot_capacity < 0:
            raise ValueError("snapshot_capacity must be nonnegative")
        if type(policy_surprise_min_sims) is not int or policy_surprise_min_sims < 0:
            raise ValueError("policy_surprise_min_sims must be a nonnegative integer")
        self.capacity = capacity
        self.device = torch.device(device)
        self.size = 0
        self.pos = 0
        self.total_added = 0
        self.total_sampled = 0
        self.iteration = 0
        self.last_sample_age = 0.0
        self.last_sample_surprise_fraction = 0.0
        self.last_sample_surprise_mean = 0.0
        self.snapshot_capacity = min(snapshot_capacity, capacity)
        self.policy_surprise_min_sims = policy_surprise_min_sims
        self.policy_surprise = torch.full((capacity,), float("nan"), device=self.device)
        self._surprise_mean_cache: tuple[int, float] | None = None
        self.snapshots: dict[int, list[int]] = {}
        self.inserted_at = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        # Zero means unknown (legacy replay), never an inferred search budget.
        self.policy_sims = torch.zeros(capacity, dtype=torch.int32, device=self.device)

        self.global_feat = torch.zeros(
            (capacity, d_global), dtype=torch.float32, device=self.device
        )
        self.source_feat = torch.zeros(
            (capacity, n_sources, d_source), dtype=torch.float32, device=self.device
        )
        self.legal_mask = torch.zeros(
            (capacity, num_actions), dtype=torch.bool, device=self.device
        )
        self.policy_target = torch.zeros(
            (capacity, num_actions), dtype=torch.float32, device=self.device
        )
        self.value_target = torch.zeros(
            (capacity, max_players), dtype=torch.float32, device=self.device
        )
        # Explicit availability distinguishes historical replay from a tied score.
        self.score_margin = torch.zeros(capacity, device=self.device)
        self.score_margin_valid = torch.zeros(
            capacity, dtype=torch.bool, device=self.device
        )

    def add(
        self,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        legal_mask: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
        snapshots: dict[int, list[int]] | None = None,
        policy_sims: int | torch.Tensor = 0,
        score_margin: torch.Tensor | None = None,
        policy_surprise: torch.Tensor | None = None,
    ) -> None:
        """Add a batch of samples to the buffer."""
        n = global_feat.shape[0]
        budgets = torch.as_tensor(policy_sims, device=self.device)
        if budgets.ndim > 1 or (budgets.ndim == 1 and budgets.shape[0] != n):
            raise ValueError("policy_sims must be a scalar or one integer per sample")
        if (
            not torch.isfinite(budgets).all()
            or (budgets < 0).any()
            or (budgets != budgets.to(torch.int32)).any()
        ):
            raise ValueError("policy_sims must contain nonnegative integers")
        budgets = budgets.to(torch.int32).expand(n)
        surprise = (
            torch.full((n,), float("nan"), device=self.device)
            if policy_surprise is None
            else policy_surprise.to(device=self.device, dtype=torch.float32)
        )
        if (
            surprise.shape != (n,)
            or torch.isinf(surprise).any()
            or (surprise < 0).any()
        ):
            raise ValueError(
                "policy_surprise must contain nonnegative finite KL or NaN per sample"
            )
        margins = (
            torch.full((n,), float("nan"), device=self.device)
            if score_margin is None
            else score_margin.to(device=self.device, dtype=torch.float32)
        )
        if margins.shape != (n,) or torch.isinf(margins).any():
            raise ValueError(
                "score_margin must contain one finite score or NaN per sample"
            )
        if snapshots and any(i < 0 or i >= n for i in snapshots):
            raise ValueError("snapshot index is outside the added batch")
        if n == 0:
            return

        self.total_added += n
        if n > self.capacity:
            offset = n - self.capacity
            self.pos = (self.pos + offset) % self.capacity
            global_feat, source_feat, legal_mask, policy_target, value_target = (
                t[-self.capacity :]
                for t in (
                    global_feat,
                    source_feat,
                    legal_mask,
                    policy_target,
                    value_target,
                )
            )
            n = self.capacity
            budgets = budgets[-self.capacity :]
            margins = margins[-self.capacity :]
            surprise = surprise[-self.capacity :]
            if snapshots:
                snapshots = {i - offset: s for i, s in snapshots.items() if i >= offset}
        if self.snapshot_capacity:
            # Evict overwritten states even when the replacement has no snapshot.
            for i in range(n):
                self.snapshots.pop((self.pos + i) % self.capacity, None)
            for i, snapshot in (snapshots or {}).items():
                self.snapshots[(self.pos + i) % self.capacity] = list(snapshot)
            while len(self.snapshots) > self.snapshot_capacity:
                del self.snapshots[next(iter(self.snapshots))]
        indices = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.inserted_at[indices] = self.iteration
        self.policy_sims[indices] = budgets
        self.policy_surprise[indices] = surprise
        self._surprise_mean_cache = None
        self.score_margin_valid[indices] = torch.isfinite(margins)
        self.score_margin[indices] = torch.nan_to_num(margins)
        end = self.pos + n
        if end <= self.capacity:
            self.global_feat[self.pos : end] = global_feat
            self.source_feat[self.pos : end] = source_feat
            self.legal_mask[self.pos : end] = legal_mask
            self.policy_target[self.pos : end] = policy_target
            self.value_target[self.pos : end] = value_target
        else:
            # Wrap around
            first = self.capacity - self.pos
            self.global_feat[self.pos :] = global_feat[:first]
            self.source_feat[self.pos :] = source_feat[:first]
            self.legal_mask[self.pos :] = legal_mask[:first]
            self.policy_target[self.pos :] = policy_target[:first]
            self.value_target[self.pos :] = value_target[:first]

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
            "reward_semantics_version": REWARD_SEMANTICS_VERSION,
            "snapshots": self.snapshots,
            "capacity": self.capacity,
            "size": self.size,
            "pos": self.pos,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "iteration": self.iteration,
            "inserted_at": self.inserted_at.cpu(),
            "policy_sims": self.policy_sims.cpu(),
            "policy_surprise": self.policy_surprise.nan_to_num().cpu(),
            "policy_surprise_valid": self.policy_surprise.isfinite().cpu(),
            "score_margin": self.score_margin.cpu(),
            "score_margin_valid": self.score_margin_valid.cpu(),
            "global_feat": self.global_feat.cpu(),
            "source_feat": self.source_feat.cpu(),
            "legal_mask": self.legal_mask.cpu(),
            "policy_target": self.policy_target.cpu(),
            "value_target": self.value_target.cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("reward_semantics_version", 1) != REWARD_SEMANTICS_VERSION:
            raise ValueError(
                "Replay uses old tie rewards; start a new run with --init-from"
            )
        if state.get("capacity") != self.capacity:
            raise ValueError("replay buffer capacity mismatch on resume")
        self.size = int(state["size"])
        self.pos = int(state["pos"])
        self.snapshots = {
            int(i): list(s)
            for i, s in state.get("snapshots", {}).items()
            if 0 <= int(i) < self.size
        }
        while len(self.snapshots) > self.snapshot_capacity:
            del self.snapshots[next(iter(self.snapshots))]
        self.total_added = int(state.get("total_added", self.size))
        self.total_sampled = int(state.get("total_sampled", 0))
        self.iteration = int(state.get("iteration", 0))
        self.policy_sims.zero_()
        self.policy_surprise.fill_(float("nan"))
        self._surprise_mean_cache = None
        if ("policy_surprise" in state) != ("policy_surprise_valid" in state):
            raise ValueError("Incomplete checkpoint policy surprise metadata")
        if "policy_surprise" in state:
            surprise = state["policy_surprise"]
            valid = state["policy_surprise_valid"]
            if (
                surprise.shape != self.policy_surprise.shape
                or not torch.isfinite(surprise).all()
                or (surprise < 0).any()
                or valid.shape != surprise.shape
                or valid.dtype != torch.bool
            ):
                raise ValueError("Invalid checkpoint policy surprise")
            self.policy_surprise.copy_(
                surprise.to(self.device).masked_fill(
                    ~valid.to(self.device), float("nan")
                )
            )
        self.score_margin.zero_()
        self.score_margin_valid.zero_()
        if ("score_margin" in state) != ("score_margin_valid" in state):
            raise ValueError("Incomplete checkpoint score margin metadata")
        if "score_margin" in state:
            margins = state["score_margin"]
            valid = state["score_margin_valid"]
            if (
                margins.shape != self.score_margin.shape
                or not torch.isfinite(margins).all()
                or valid.shape != margins.shape
                or valid.dtype != torch.bool
            ):
                raise ValueError("Invalid checkpoint score_margin metadata")
            self.score_margin.copy_(margins.to(self.device))
            self.score_margin_valid.copy_(valid.to(self.device))
        if "policy_sims" in state:
            budgets = state["policy_sims"]
            if (
                budgets.shape != self.policy_sims.shape
                or not torch.isfinite(budgets).all()
                or (budgets < 0).any()
                or (budgets != budgets.to(torch.int32)).any()
            ):
                raise ValueError("Invalid checkpoint policy_sims metadata")
            self.policy_sims.copy_(budgets.to(self.device))
        if "inserted_at" in state:
            self.inserted_at.copy_(state["inserted_at"].to(self.device))
        dev = self.device
        self.global_feat.copy_(state["global_feat"].to(dev))
        self.source_feat.copy_(state["source_feat"].to(dev))
        self.legal_mask.copy_(state["legal_mask"].to(dev))
        self.policy_target.copy_(state["policy_target"].to(dev))
        self.value_target.copy_(state["value_target"].to(dev))

    def _sample_indices(
        self, batch_size: int, fraction: float, minimum: int, cap: float
    ) -> torch.Tensor:
        """Bounded rejection sampling; no million-entry multinomial per update.

        Full-search weights are (1-f) + f*KL/mean(KL), capped at cap. Unknown
        and fast targets retain weight one. No importance correction: the
        deliberate sampling bias applies to both policy and value learning.
        """
        if (
            not math.isfinite(fraction)
            or not 0 <= fraction <= 0.5
            or minimum < 1
            or not math.isfinite(cap)
            or not 1 <= cap <= 8
        ):
            raise ValueError("Invalid policy surprise sampling configuration")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if fraction == 0:
            return torch.randint(0, self.size, (batch_size,), device=self.device)
        if self._surprise_mean_cache is None or self._surprise_mean_cache[0] != minimum:
            known = torch.isfinite(self.policy_surprise[: self.size]) & (
                self.policy_sims[: self.size] >= minimum
            )
            mean = (
                float(self.policy_surprise[: self.size][known].mean())
                if known.any()
                else 0.0
            )
            self._surprise_mean_cache = (minimum, mean)
        mean = self._surprise_mean_cache[1]
        if mean <= 1e-12:
            return torch.randint(0, self.size, (batch_size,), device=self.device)
        accepted, remaining = [], batch_size
        while remaining:
            indices = torch.randint(
                0, self.size, (max(32, math.ceil(remaining * cap)),), device=self.device
            )
            kl = self.policy_surprise[indices]
            known = torch.isfinite(kl) & (self.policy_sims[indices] >= minimum)
            weights = torch.where(
                known, (1 - fraction) + fraction * kl / mean, 1.0
            ).clamp(max=cap)
            chosen = indices[
                torch.rand(len(indices), device=self.device) < weights / cap
            ][:remaining]
            accepted.append(chosen)
            remaining -= len(chosen)
        return torch.cat(accepted)

    def sample(
        self,
        batch_size: int,
        *,
        include_policy_sims: bool = False,
        include_score_margin: bool = False,
        surprise_fraction: float = 0.0,
        surprise_min_sims: int = 256,
        surprise_max_weight: float = 4.0,
    ) -> tuple[torch.Tensor, ...]:
        """Sample a random minibatch."""
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        indices = self._sample_indices(
            batch_size, surprise_fraction, surprise_min_sims, surprise_max_weight
        )
        if self.policy_surprise_min_sims or surprise_fraction:
            kl = self.policy_surprise[indices]
            known = torch.isfinite(kl) & (
                self.policy_sims[indices] >= surprise_min_sims
            )
            self.last_sample_surprise_fraction = float(known.float().mean())
            self.last_sample_surprise_mean = (
                float(kl[known].mean()) if known.any() else 0.0
            )
        self.total_sampled += batch_size
        self.last_sample_age = float(
            (self.iteration - self.inserted_at[indices]).float().mean()
        )
        batch = (
            self.global_feat[indices],
            self.source_feat[indices],
            self.legal_mask[indices],
            self.policy_target[indices],
            self.value_target[indices],
        )
        if include_policy_sims:
            batch = (*batch, self.policy_sims[indices])
        if include_score_margin:
            batch = (
                *batch,
                self.score_margin[indices].masked_fill(
                    ~self.score_margin_valid[indices], float("nan")
                ),
            )
        return batch
