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
    parser.add_argument("--arch", type=str, default="attn", choices=["attn", "flat", "source_attn"])
    parser.add_argument("--selfplay-games", type=int, default=1023)
    parser.add_argument("--selfplay-sims", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--turns-per-player",
        type=int,
        default=60,
        help="Self-play turn cap = this × num_players (0 = use --max-turns)",
    )
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--learner-batch", type=int, default=256)
    parser.add_argument("--learner-steps", type=int, default=64)
    parser.add_argument("--entropy-bonus", type=float, default=0.034)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2.75e-3)
    parser.add_argument("--weight-decay", type=float, default=1.2e-5)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--max-wall-minutes", type=float, default=1440.0)
    parser.add_argument("--init-from", type=str, default="")
    parser.add_argument("--run-id", type=str, default="default")
    parser.add_argument("--runs-root", type=str, default="")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.27)
    parser.add_argument("--dirichlet-mix", type=float, default=0.47)
    parser.add_argument("--q-scale", type=float, default=25.0)
    parser.add_argument("--time-discount", type=float, default=1.0)
    parser.add_argument("--reward-mode", type=str, default="score_scaled", choices=["binary", "score_scaled"])
    parser.add_argument(
        "--training-cycle-length",
        type=int,
        default=4,
        help="Iter cycle: selfplay, league, selfplay, bot (0 = all selfplay).",
    )
    parser.add_argument(
        "--bot-opus-prob",
        type=float,
        default=0.5,
        help="Fraction of bot-selfplay games vs heuristic_opus (rest vs heuristic).",
    )
    parser.add_argument(
        "--league-selfplay-every",
        type=int,
        default=0,
        help="Legacy: if --training-cycle-length 0, use hash trigger every N iters.",
    )
    parser.add_argument("--eval-games", type=int, default=512)
    parser.add_argument("--eval-sims", type=int, default=32)
    parser.add_argument("--eval-astra-fraction", type=float, default=0.125,
                        help="Fraction of evaluation opponent seats using production Astra (0 disables).")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use-amp", action="store_true", help="Enable AMP (overrides GPU defaults).")
    amp_group.add_argument("--no-amp", action="store_true", help="Disable AMP (overrides GPU defaults).")

    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile-net", action="store_true", help="Enable torch.compile (overrides GPU defaults)."
    )
    compile_group.add_argument(
        "--no-compile-net", action="store_true", help="Disable torch.compile (overrides GPU defaults)."
    )
    parser.add_argument(
        "--profile-training",
        action="store_true",
        help="Log per-iteration CPU/GPU/resource and stage timing instrumentation.",
    )
    parser.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        help="Synchronize CUDA around timed regions for accurate GPU timings (slower).",
    )
    parser.add_argument(
        "--bot-policy",
        type=str,
        default="batched",
        choices=["batched", "scalar"],
        help="Bot opponent policy for bot self-play (training).",
    )
    parser.add_argument("--eval-workers", type=int, default=1, help="Parallel eval subprocesses.")

    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--search-backend", choices=["one_ply", "gumbel_tree"], default="one_ply")
    parser.add_argument("--eval-search-backend", choices=["one_ply", "gumbel_tree"], default="one_ply")
    parser.add_argument("--eval-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-q-scale", type=float, default=28.0)
    parser.add_argument("--eval-temperature", type=float, default=.25)
    parser.add_argument("--league-root", default="")
    args = parser.parse_args()
    explicit = set()
    if args.use_amp or args.no_amp:
        explicit.add("use_amp")
    if args.compile_net or args.no_compile_net:
        explicit.add("compile_net")

    use_amp = True if args.use_amp else False if args.no_amp else False
    compile_net = True if args.compile_net else False if args.no_compile_net else False

    config = LoopConfig(
        num_players=args.num_players,
        device=args.device,
        hidden=args.hidden,
        arch=args.arch,
        selfplay_games=args.selfplay_games,
        selfplay_sims=args.selfplay_sims,
        selfplay_max_turns=args.max_turns,
        selfplay_turns_per_player=args.turns_per_player,
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
        training_cycle_length=args.training_cycle_length,
        bot_selfplay_opus_prob=args.bot_opus_prob,
        league_selfplay_every=args.league_selfplay_every,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        eval_astra_fraction=args.eval_astra_fraction,
        use_amp=use_amp,
        compile_net=compile_net,
        profile_training=args.profile_training,
        profile_sync_cuda=args.profile_sync_cuda,
        bot_policy=args.bot_policy,
        eval_workers=args.eval_workers,
        seed=args.seed, torch_threads=args.torch_threads, search_backend=args.search_backend,
        eval_search_backend=args.eval_search_backend, eval_device=args.eval_device,
        eval_q_scale=args.eval_q_scale, eval_temperature=args.eval_temperature,
        league_root=args.league_root,
    )

    from ..train.device import resolve_device

    device = resolve_device(args.device)
    config = apply_device_defaults(config, device, explicit)

    run = Run(args.run_id, runs_root=args.runs_root or None)
    try:
        run_loop(run, config, explicit_fields=explicit)
    finally:
        run.close()


if __name__ == "__main__":
    main()
