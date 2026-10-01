"""Main training entry point."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

from ..obs.run import Run, _default_runs_root
from ..train.loop import LoopConfig, apply_device_defaults, run_loop


def parse_config(argv: list[str] | None = None) -> tuple[LoopConfig, set[str]]:
    parser = argparse.ArgumentParser(description="Train Azul RL agent")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--arch", type=str, default="attn", choices=["attn", "flat", "source_attn"]
    )
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
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="score_scaled",
        choices=["binary", "score_scaled"],
    )
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
    parser.add_argument(
        "--eval-astra-fraction",
        type=float,
        default=0.125,
        help="Fraction of evaluation opponent seats using production Astra (0 disables).",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument(
        "--use-amp", action="store_true", help="Enable AMP (overrides GPU defaults)."
    )
    amp_group.add_argument(
        "--no-amp", action="store_true", help="Disable AMP (overrides GPU defaults)."
    )

    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile-net",
        action="store_true",
        help="Enable torch.compile (overrides GPU defaults).",
    )
    compile_group.add_argument(
        "--no-compile-net",
        action="store_true",
        help="Disable torch.compile (overrides GPU defaults).",
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
    parser.add_argument(
        "--eval-workers", type=int, default=1, help="Parallel eval subprocesses."
    )

    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--search-backend", choices=["one_ply", "gumbel_tree"], default="one_ply"
    )
    parser.add_argument(
        "--eval-search-backend", choices=["one_ply", "gumbel_tree"], default="one_ply"
    )
    parser.add_argument("--eval-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-q-scale", type=float, default=28.0)
    parser.add_argument("--eval-temperature", type=float, default=0.25)
    parser.add_argument("--league-root", default="")
    parser.add_argument("--preset", choices=["enhanced-2p"])
    parser.add_argument(
        "--search-tree-core", choices=["python", "rust"], default="python"
    )
    parser.add_argument(
        "--league-search-backend", choices=["one_ply", "gumbel_tree"], default="one_ply"
    )
    parser.add_argument(
        "--league-opponent-sampling", choices=["weighted", "mixed"], default="weighted"
    )
    parser.add_argument(
        "--league-seed-init", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--aux-score-head", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--policy-surprise-record", action=argparse.BooleanOptionalAction, default=False
    )
    defaults = LoopConfig()
    for name in (
        "selfplay_full_sims",
        "search_cpu_workers",
        "search_inference_batch_size",
        "search_max_root_candidates",
        "search_chance_samples",
        "search_inference_cache_size",
        "league_opponent_sims",
        "bot_selfplay_workers",
        "reanalysis_positions",
        "reanalysis_every",
        "reanalysis_sims",
        "reanalysis_batch_size",
        "reanalysis_snapshot_capacity",
        "policy_surprise_min_sims",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    for name in (
        "selfplay_full_fraction",
        "search_inference_wait_ms",
        "bot_selfplay_astra_prob",
        "league_opponent_prob",
        "eval_root_noise_scale",
        "policy_fast_weight",
        "aux_score_weight",
        "aux_score_scale",
        "policy_surprise_fraction",
        "policy_surprise_max_weight",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(defaults, name)
        )
    argv = sys.argv[1:] if argv is None else argv
    aliases = {
        "max_turns": "selfplay_max_turns",
        "turns_per_player": "selfplay_turns_per_player",
        "learner_steps": "learner_steps_per_iter",
        "bot_opus_prob": "bot_selfplay_opus_prob",
    }
    probe = parser.parse_args(argv)
    preset_fields: set[str] = set()
    if probe.preset:
        from ..train.presets import enhanced_2p_config

        preset = dataclasses.asdict(enhanced_2p_config())
        reverse_aliases = {v: k for k, v in aliases.items()}
        parser.set_defaults(**{reverse_aliases.get(k, k): v for k, v in preset.items()})
        preset_fields = set(preset)
    args = parser.parse_args(argv)
    given = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    explicit = preset_fields | {
        aliases.get(action.dest, action.dest)
        for action in parser._actions
        if given.intersection(action.option_strings)
    }
    valid = {field.name for field in dataclasses.fields(LoopConfig)}
    values = {
        aliases.get(k, k): v
        for k, v in vars(args).items()
        if aliases.get(k, k) in valid
    }
    if args.no_amp:
        values["use_amp"] = False
        explicit.add("use_amp")
    if args.no_compile_net:
        values["compile_net"] = False
        explicit.add("compile_net")
    if args.preset:
        if args.num_players != 2:
            parser.error("enhanced-2p requires --num-players 2")
        if args.run_id == "default":
            parser.error("use a new --run-id for the enhanced-2p experiment")
        if not args.league_root:
            values["league_root"] = str(
                pathlib.Path(args.runs_root or _default_runs_root())
                / args.run_id
                / "league"
            )
    return LoopConfig(**values), explicit


def main() -> None:
    config, explicit = parse_config()
    from ..train.device import resolve_device

    device = resolve_device(config.device)
    config = apply_device_defaults(config, device, explicit)

    run = Run(config.run_id, runs_root=config.runs_root or None)
    try:
        run_loop(run, config, explicit_fields=explicit)
    finally:
        run.close()


if __name__ == "__main__":
    main()
