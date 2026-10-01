"""Experimental presets; improvements must be established by paired evaluation."""

from __future__ import annotations

from .loop import LoopConfig


def enhanced_2p_config() -> LoopConfig:
    """Keep the successful learner settings and spend more on teacher quality."""
    return LoopConfig(
        num_players=2,
        arch="source_attn",
        hidden=256,
        selfplay_games=256,
        selfplay_sims=64,
        selfplay_full_sims=256,
        selfplay_full_fraction=0.25,
        learner_steps_per_iter=72,
        lr=0.0009217470483295072,
        weight_decay=0.0000896240535348826,
        entropy_bonus=0.014274964936918592,
        q_scale=26.747517709474458,
        time_discount=0.999444440976462,
        reward_mode="binary",
        dirichlet_alpha=0.25668356196635866,
        dirichlet_mix=0.0,
        search_backend="gumbel_tree",
        search_tree_core="rust",
        search_max_root_candidates=32,
        search_inference_cache_size=8192,
        bot_selfplay_astra_prob=0.5,
        bot_selfplay_opus_prob=0.25,
        bot_selfplay_workers=8,
        league_selfplay_every=0,
        league_opponent_prob=1.0,
        league_opponent_sims=64,
        league_search_backend="gumbel_tree",
        league_opponent_sampling="mixed",
        league_seed_init=True,
        reanalysis_positions=256,
        reanalysis_every=4,
        reanalysis_sims=256,
        reanalysis_snapshot_capacity=8192,
        eval_games=0,
        eval_sims=128,
        eval_search_backend="gumbel_tree",
        eval_root_noise_scale=0.0,
        checkpoint_every=25,
        max_iters=1_000_000,
        max_wall_minutes=600,
        profile_every=25,
    )
