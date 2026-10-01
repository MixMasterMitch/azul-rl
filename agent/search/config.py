"""Search settings shared by self-play, evaluation, and serving."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SearchConfig:
    backend: str = "one_ply"
    tree_core: str = "python"
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
    cpu_workers: int = 1
    inference_batch_size: int = 2048
    inference_wait_ms: float = 2.0
    inference_cache_size: int = 0

    def __post_init__(self) -> None:
        if type(self.inference_cache_size) is not int or self.inference_cache_size < 0:
            raise ValueError("inference_cache_size must be a nonnegative integer")
        if self.inference_cache_size and (
            self.backend != "gumbel_tree" or self.tree_core != "rust"
        ):
            raise ValueError("inference caching requires the Rust tree core")
        if self.backend not in {"one_ply", "gumbel_tree"}:
            raise ValueError(f"Unknown search backend: {self.backend}")
        if self.tree_core not in {"python", "rust"}:
            raise ValueError(f"Unknown tree core: {self.tree_core}")
        for key in (
            "num_simulations",
            "max_root_candidates",
            "max_depth",
            "leaf_batch_size",
            "chance_samples",
            "cpu_workers",
            "inference_batch_size",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if type(self.cpu_workers) is not int or self.cpu_workers > 8:
            raise ValueError("cpu_workers must be an integer from one to eight")
        if (
            not math.isfinite(self.inference_wait_ms)
            or not 0 <= self.inference_wait_ms <= 100
        ):
            raise ValueError("inference_wait_ms must be in [0, 100]")
        for key in (
            "temperature",
            "q_scale",
            "root_noise_scale",
            "dirichlet_alpha",
            "dirichlet_mix",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"Invalid {key}")
        if self.dirichlet_mix > 1 or (
            self.dirichlet_mix > 0 and self.dirichlet_alpha == 0
        ):
            raise ValueError("Invalid Dirichlet mixture")
        if self.reward_mode not in {"binary", "score_scaled"}:
            raise ValueError("Invalid reward mode")
        if self.move_deadline_s is not None and (
            not math.isfinite(self.move_deadline_s) or not 0 < self.move_deadline_s <= 5
        ):
            raise ValueError("Move deadline must be in (0, 5] seconds")
