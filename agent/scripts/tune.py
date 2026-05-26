"""Optuna hyperparameter tuning harness for Azul RL."""

from __future__ import annotations

import argparse
import functools
import pathlib
import shutil
import sys

import optuna
import torch

from ..eval.tournament import evaluate_checkpoint
from ..obs.run import Run
from ..train.checkpointing import load_net_from_checkpoint
from ..train.loop import LoopConfig, run_loop
from ..train.ranking import DEFAULT_ANCHORS, add_match_result, fit_anchored_ratings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Azul RL Optuna hyperparameter tuning")
    p.add_argument("--study-name", type=str, default="azul-tune")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--iters-per-trial", type=int, default=10)
    p.add_argument("--minutes-per-trial", type=float, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="agent/runs")
    p.add_argument("--narrow-ranges", action="store_true")
    p.add_argument("--rating-games", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--arch", type=str, default="attn", choices=["attn", "flat"])
    p.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    return p


def define_search_space(trial: optuna.Trial, narrow: bool = False) -> dict:
    if narrow:
        selfplay_games = 512
        selfplay_sims = 16
        learner_batch = 512
        replay_capacity = 600_000
        learner_steps_per_iter = 48
        lr = trial.suggest_float("lr", 1e-4, 1e-3)
        entropy_bonus = trial.suggest_float("entropy_bonus", 0.0, 0.02)
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.15, 0.5)
        dirichlet_mix = trial.suggest_float("dirichlet_mix", 0.15, 0.35)
        q_scale = trial.suggest_float("q_scale", 5.0, 20.0)
        time_discount = trial.suggest_float("time_discount", 0.99, 1.0)
    else:
        selfplay_games = trial.suggest_categorical("selfplay_games", [256, 512, 1024])
        selfplay_sims = trial.suggest_categorical("selfplay_sims", [8, 16, 32])
        learner_batch = trial.suggest_categorical("learner_batch", [128, 256, 512])
        replay_capacity = trial.suggest_categorical("replay_capacity", [400_000, 600_000])
        learner_steps_per_iter = trial.suggest_int("learner_steps_per_iter", 16, 128)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        entropy_bonus = trial.suggest_float("entropy_bonus", 0.0, 0.05)
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.03, 1.0, log=True)
        dirichlet_mix = trial.suggest_float("dirichlet_mix", 0.1, 0.5)
        q_scale = trial.suggest_float("q_scale", 1.0, 30.0)
        time_discount = trial.suggest_float("time_discount", 0.98, 1.0)

    return {
        "selfplay_games": selfplay_games,
        "selfplay_sims": selfplay_sims,
        "learner_batch": learner_batch,
        "replay_capacity": replay_capacity,
        "learner_steps_per_iter": learner_steps_per_iter,
        "lr": lr,
        "entropy_bonus": entropy_bonus,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_mix": dirichlet_mix,
        "q_scale": q_scale,
        "time_discount": time_discount,
    }


def compute_rating_objective(
    net,
    num_players: int,
    num_games: int,
    device: str,
) -> float:
    metrics = evaluate_checkpoint(net, num_games=num_games, num_players=num_players, device=device)
    match_results: list[dict] = []
    agent = "trial_agent"
    for opp in ("random", "heuristic"):
        key = f"vs_{opp}_winrate"
        winrate = metrics.get(key, 0.0)
        total = float(num_games)
        wins_agent = winrate * total
        wins_opp = total - wins_agent
        add_match_result(match_results, agent, opp, wins_agent, wins_opp)
    ratings = fit_anchored_ratings(match_results, anchors=dict(DEFAULT_ANCHORS))
    return ratings.get(agent, 0.0)


def trial_fn(
    trial: optuna.Trial,
    base_cfg: dict,
    output_dir: str,
    iters_per_trial: int,
    minutes_per_trial: float,
    narrow: bool,
    rating_games: int,
    device: str,
) -> float:
    import logging

    logger = logging.getLogger(__name__)
    try:
        sampled = define_search_space(trial, narrow=narrow)
        merged = {**base_cfg, **sampled}

        if minutes_per_trial > 0:
            merged["max_wall_minutes"] = minutes_per_trial
            merged["max_iters"] = 999_999
            merged["checkpoint_every"] = 999_999
            merged["eval_games"] = 0
            merged["league_selfplay_every"] = 0
        else:
            merged["max_iters"] = iters_per_trial
            merged["eval_games"] = 0
            merged["league_selfplay_every"] = 0

        cfg = LoopConfig(**{k: v for k, v in merged.items() if k in LoopConfig.__dataclass_fields__})
        trial_run_id = f"tune_trial_{trial.number:03d}"
        trial_dir = pathlib.Path(output_dir) / trial_run_id
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        run = Run(trial_run_id, runs_root=output_dir)
        run_loop(run, cfg)

        ckpt_path = run.ckpt_dir / "latest_resume.pt"
        net, _ = load_net_from_checkpoint(ckpt_path, map_location="cpu")
        net.eval()
        rating = compute_rating_objective(
            net,
            num_players=cfg.num_players,
            num_games=rating_games,
            device="cpu",
        )
        logger.info("Trial %d finished with rating %.1f", trial.number, rating)
        run.close()
        shutil.rmtree(trial_dir, ignore_errors=True)
        return rating
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logging.getLogger(__name__).error("Trial %d failed: %s", trial.number, exc)
        return float("-inf")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ..train.device import resolve_device

    device = resolve_device(args.device)
    base_cfg: dict = {
        "device": device,
        "num_players": args.num_players,
        "hidden": args.hidden,
        "arch": args.arch,
    }

    storage = args.storage or f"sqlite:///{args.output_dir}/optuna_{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    objective = functools.partial(
        trial_fn,
        base_cfg=base_cfg,
        output_dir=args.output_dir,
        iters_per_trial=args.iters_per_trial,
        minutes_per_trial=args.minutes_per_trial,
        narrow=args.narrow_ranges,
        rating_games=args.rating_games,
        device=device,
    )
    study.optimize(objective, n_trials=args.n_trials)

    completed = [t for t in study.trials if t.value is not None]
    ranked = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    if ranked:
        print("\nTop trials:")
        for t in ranked:
            print(f"  #{t.number}: rating={t.value:.1f} params={t.params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
