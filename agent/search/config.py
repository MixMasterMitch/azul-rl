"""Search settings shared by self-play, evaluation, and serving."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SearchConfig:
    backend: str = "one_ply"
    num_simulations: int = 64
    temperature: float = 0.25
    q_scale: float = 28.0
    root_noise_scale: float = 1.0
    dirichlet_alpha: float = 0.0
    dirichlet_mix: float = 0.0
    reward_mode: str = "binary"
    seed: int | None = None
    move_deadline_s: float | None = None
    max_root_candidates: int = 16
    max_depth: int = 120
    leaf_batch_size: int = 256
    chance_samples: int = 4

    def __post_init__(self) -> None:
        if self.backend not in {"one_ply", "gumbel_tree"}:
            raise ValueError(f"Unknown search backend: {self.backend}")
        for key in ("num_simulations", "max_root_candidates", "max_depth", "leaf_batch_size", "chance_samples"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        for key in ("temperature", "q_scale", "root_noise_scale", "dirichlet_alpha", "dirichlet_mix"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"Invalid {key}")
        if self.dirichlet_mix > 1 or (self.dirichlet_mix > 0 and self.dirichlet_alpha == 0):
            raise ValueError("Invalid Dirichlet mixture")
        if self.reward_mode not in {"binary", "score_scaled"}:
            raise ValueError("Invalid reward mode")
        if self.move_deadline_s is not None and (not math.isfinite(self.move_deadline_s) or not 0 < self.move_deadline_s <= 5):
            raise ValueError("Move deadline must be in (0, 5] seconds")
