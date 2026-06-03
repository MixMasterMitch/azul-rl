"""Smoke test: run a tiny training loop to verify everything works end-to-end."""

from __future__ import annotations

import tempfile

from ..obs.run import Run
from ..train.loop import LoopConfig, run_loop


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run = Run("smoke", runs_root=tmpdir, create_ok=True)
        config = LoopConfig(
            num_players=2,
            device="auto",
            hidden=64,
            arch="flat",
            selfplay_games=8,
            selfplay_sims=2,
            selfplay_max_turns=100,
            selfplay_turns_per_player=0,
            replay_capacity=10_000,
            learner_batch=16,
            learner_steps_per_iter=4,
            entropy_bonus=0.01,
            checkpoint_every=2,
            eval_games=0,
            league_selfplay_every=0,
            lr=1e-3,
            max_iters=3,
            max_wall_minutes=5.0,
        )
        run_loop(run, config)
    print("Smoke test passed!")


if __name__ == "__main__":
    main()
