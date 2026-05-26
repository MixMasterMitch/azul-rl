"""Main training entry point."""

from __future__ import annotations

import argparse

from ..obs.run import Run
from ..train.loop import LoopConfig, apply_device_defaults, run_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Azul RL agent")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--arch", type=str, default="attn", choices=["attn", "flat"])
    parser.add_argument("--selfplay-games", type=int, default=512)
    parser.add_argument("--selfplay-sims", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--replay-capacity", type=int, default=600_000)
    parser.add_argument("--learner-batch", type=int, default=256)
    parser.add_argument("--learner-steps", type=int, default=192)
    parser.add_argument("--entropy-bonus", type=float, default=0.015)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--max-wall-minutes", type=float, default=60.0)
    parser.add_argument("--init-from", type=str, default="")
    parser.add_argument("--run-id", type=str, default="default")
    parser.add_argument("--runs-root", type=str, default="")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.15)
    parser.add_argument("--dirichlet-mix", type=float, default=0.40)
    parser.add_argument("--q-scale", type=float, default=22.0)
    parser.add_argument("--time-discount", type=float, default=1.0)
    parser.add_argument("--reward-mode", type=str, default="score_scaled", choices=["binary", "score_scaled"])
    parser.add_argument("--league-selfplay-every", type=int, default=3)
    parser.add_argument("--eval-games", type=int, default=512)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--compile-net", action="store_true")

    args = parser.parse_args()

    explicit = set()
    if args.use_amp:
        explicit.add("use_amp")
    if args.compile_net:
        explicit.add("compile_net")

    config = LoopConfig(
        num_players=args.num_players,
        device=args.device,
        hidden=args.hidden,
        arch=args.arch,
        selfplay_games=args.selfplay_games,
        selfplay_sims=args.selfplay_sims,
        selfplay_max_turns=args.max_turns,
        replay_capacity=args.replay_capacity,
        learner_batch=args.learner_batch,
        learner_steps_per_iter=args.learner_steps,
        entropy_bonus=args.entropy_bonus,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_iters=args.max_iters,
        max_wall_minutes=args.max_wall_minutes,
        init_from=args.init_from,
        run_id=args.run_id,
        runs_root=args.runs_root,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_mix=args.dirichlet_mix,
        q_scale=args.q_scale,
        time_discount=args.time_discount,
        reward_mode=args.reward_mode,
        league_selfplay_every=args.league_selfplay_every,
        eval_games=args.eval_games,
        use_amp=args.use_amp,
        compile_net=args.compile_net,
    )

    from ..train.device import resolve_device

    device = resolve_device(args.device)
    config = apply_device_defaults(config, device, explicit)

    run = Run(args.run_id, runs_root=args.runs_root or None)
    run_loop(run, config, explicit_fields=explicit)


if __name__ == "__main__":
    main()
