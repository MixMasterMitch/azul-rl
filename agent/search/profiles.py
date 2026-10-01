"""Explicit experimental profiles for arena or registry-backed serving."""

from __future__ import annotations

from .config import SearchConfig

SEARCH_PROFILES = ("rust64", "strong128", "strong256")


def search_profile(name: str) -> SearchConfig:
    if name not in SEARCH_PROFILES:
        raise ValueError(f"Unknown search profile: {name}")
    if name == "rust64":
        return SearchConfig(backend="gumbel_tree", tree_core="rust", num_simulations=64)
    return SearchConfig(
        backend="gumbel_tree",
        tree_core="rust",
        num_simulations=128 if name == "strong128" else 256,
        root_noise_scale=0.0,
        max_root_candidates=32,
        chance_samples=8,
        inference_cache_size=8192,
    )
