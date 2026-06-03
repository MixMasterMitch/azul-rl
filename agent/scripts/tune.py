"""Optuna hyperparameter tuning harness for Azul RL."""

from __future__ import annotations

import argparse
import functools
import json
import logging
import pathlib
import shutil
import sys

import optuna
import torch

from ..eval.tournament import (
    DEFAULT_OPPONENTS,
    EvalConfig,
    evaluate_checkpoint,
    play_vs_net_opponent,
)
from ..obs.run import Run
from ..train import ranking as R
from ..train.checkpointing import load_net_from_checkpoint
from ..train.league import League
from ..train.loop import LoopConfig, run_loop
from ..train.tuning_curve import extrapolate_rating, extrapolate_winrate

# PyTorch efficient attention: MCTS child batch is games × sims (see gumbel_mcts).
MHA_CHILD_BATCH_LIMIT = 65535
# Production default (fixed for phase-2 tuning).
FIXED_SELFPLAY_GAMES = 1023
FIXED_SELFPLAY_SIMS = 32
# Legacy categorical grids (tests / old studies only).
SELFPLAY_GAMES_CHOICES = [511, 1023, 2047, 4095]
SELFPLAY_SIMS_CHOICES = [8, 16, 32, 64]
DEFAULT_REPLAY_CAPACITY = 1_000_000
REWARD_MODES = ("binary", "score_scaled")
TUNING_AGENT_ENTITY = "trial_agent"
TUNING_OPPONENT_ENTITIES = {"opus": "heuristic_opus"}
DEFAULT_BASELINE_Q_SCALE = 25.0
OBJECTIVE_RATING_2P = "rating_2p"
OBJECTIVE_OPUS_WINRATE = "opus_winrate"
OBJECTIVES = (OBJECTIVE_RATING_2P, OBJECTIVE_OPUS_WINRATE)
OPUS_ONLY_OPPONENTS = tuple(
    opponent for opponent in DEFAULT_OPPONENTS if opponent[0] == "opus"
)


def selfplay_child_batch_size(num_games: int, num_sims: int) -> int:
    return num_games * num_sims


def selfplay_exceeds_mha_limit(num_games: int, num_sims: int) -> bool:
    return selfplay_child_batch_size(num_games, num_sims) > MHA_CHILD_BATCH_LIMIT


def ensure_selfplay_mha_limit(num_games: int, num_sims: int) -> None:
    """Prune trial when MCTS child batch would exceed PyTorch attn limit."""
    product = selfplay_child_batch_size(num_games, num_sims)
    if product > MHA_CHILD_BATCH_LIMIT:
        raise optuna.TrialPruned(
            f"selfplay games×sims={num_games}×{num_sims}={product} "
            f"> {MHA_CHILD_BATCH_LIMIT}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Azul RL Optuna hyperparameter tuning")
    p.add_argument("--study-name", type=str, default="azul-tune")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument(
        "--iters-per-trial",
        type=int,
        default=50,
        help="Training iterations per trial when --minutes-per-trial is 0.",
    )
    p.add_argument(
        "--sessions-per-trial",
        type=int,
        default=6,
        help="Training sessions per trial (each followed by a rating eval).",
    )
    p.add_argument(
        "--session-minutes",
        type=float,
        default=30.0,
        help="Wall-clock minutes per training session.",
    )
    p.add_argument(
        "--extrapolate-hours",
        type=float,
        default=72.0,
        help="Extrapolate log rating curve this many hours past last session (default 3 days).",
    )
    p.add_argument(
        "--minutes-per-trial",
        type=float,
        default=0.0,
        help="Legacy: single training block per trial. Set >0 to disable session curve mode.",
    )
    p.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="agent/runs")
    p.add_argument(
        "--narrow-ranges",
        action="store_true",
        help="Tighter ranges for quick/local smoke tuning.",
    )
    p.add_argument(
        "--wide-ranges",
        action="store_true",
        help="Legacy broad search (still fixed 1023×32).",
    )
    p.add_argument(
        "--init-from",
        type=str,
        default="",
        help="Optional checkpoint to warm-start each trial (e.g. trial_010.pt).",
    )
    p.add_argument("--rating-games", type=int, default=256)
    p.add_argument("--rating-sims", type=int, default=32)
    p.add_argument(
        "--objective",
        type=str,
        default=OBJECTIVE_RATING_2P,
        choices=OBJECTIVES,
        help=(
            "Optimization target. rating_2p uses anchored Bradley-Terry rating; "
            "opus_winrate evaluates only vs heuristic-opus and maximizes projected winrate."
        ),
    )
    p.add_argument(
        "--baseline-q-scale",
        type=float,
        default=DEFAULT_BASELINE_Q_SCALE,
        help="q_scale for study baseline eval (init checkpoint).",
    )
    p.add_argument(
        "--force-baseline",
        action="store_true",
        help="Re-run study baseline eval even if tune_baseline_eval.json exists.",
    )
    p.add_argument(
        "--keep-eval-checkpoints",
        action="store_true",
        default=True,
        help="Save net-only checkpoints for re-evaluation (default: on).",
    )
    p.add_argument(
        "--no-keep-eval-checkpoints",
        action="store_false",
        dest="keep_eval_checkpoints",
    )
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--arch", type=str, default="attn", choices=["attn", "flat"])
    p.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    return p


def define_search_space(
    trial: optuna.Trial,
    narrow: bool = False,
    wide: bool = False,
) -> dict:
    """Sample hyperparameters; self-play batch is always FIXED_SELFPLAY_GAMES×SIMS."""
    if wide:
        learner_batch = trial.suggest_categorical("learner_batch", [128, 256, 512])
        learner_steps_per_iter = trial.suggest_int("learner_steps_per_iter", 16, 128)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        entropy_bonus = trial.suggest_float("entropy_bonus", 0.0, 0.05)
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.03, 1.0, log=True)
        dirichlet_mix = trial.suggest_float("dirichlet_mix", 0.1, 0.5)
        q_scale = trial.suggest_float("q_scale", 1.0, 30.0)
        time_discount = trial.suggest_float("time_discount", 0.98, 1.0)
        reward_mode = trial.suggest_categorical("reward_mode", REWARD_MODES)
    elif narrow:
        learner_batch = 256
        learner_steps_per_iter = 48
        lr = trial.suggest_float("lr", 5e-4, 1.5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 3e-4, log=True)
        entropy_bonus = trial.suggest_float("entropy_bonus", 0.01, 0.03)
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.2, 0.35)
        dirichlet_mix = trial.suggest_float("dirichlet_mix", 0.35, 0.5)
        q_scale = trial.suggest_float("q_scale", 20.0, 28.0)
        time_discount = trial.suggest_float("time_discount", 0.998, 1.0)
        reward_mode = trial.suggest_categorical("reward_mode", REWARD_MODES)
    else:
        # Phase-2 default: centered on trial_010 / attn_256_v0 production, deeper search.
        learner_batch = trial.suggest_categorical("learner_batch", [256, 512])
        learner_steps_per_iter = trial.suggest_int("learner_steps_per_iter", 48, 80)
        lr = trial.suggest_float("lr", 8e-4, 4e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 5e-6, 5e-4, log=True)
        entropy_bonus = trial.suggest_float("entropy_bonus", 0.015, 0.045)
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.18, 0.38)
        dirichlet_mix = trial.suggest_float("dirichlet_mix", 0.35, 0.55)
        q_scale = trial.suggest_float("q_scale", 18.0, 30.0)
        time_discount = trial.suggest_float("time_discount", 0.995, 1.0)
        reward_mode = trial.suggest_categorical("reward_mode", REWARD_MODES)

    replay_capacity = DEFAULT_REPLAY_CAPACITY

    return {
        "selfplay_games": FIXED_SELFPLAY_GAMES,
        "selfplay_sims": FIXED_SELFPLAY_SIMS,
        "learner_batch": learner_batch,
        "replay_capacity": replay_capacity,
        "learner_steps_per_iter": learner_steps_per_iter,
        "lr": lr,
        "weight_decay": weight_decay,
        "entropy_bonus": entropy_bonus,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_mix": dirichlet_mix,
        "q_scale": q_scale,
        "time_discount": time_discount,
        "reward_mode": reward_mode,
    }


def _winrate_to_match_result(
    results: list[dict],
    agent: str,
    opponent: str,
    winrate: float,
    num_games: int,
    num_players: int = 2,
) -> None:
    wins = int(round(float(winrate) * num_games))
    wins = max(0, min(wins, num_games))
    losses = num_games - wins
    R.add_match_result(results, agent, opponent, float(wins), float(losses), num_players=num_players)


def _rating_entity_for_opponent(opponent: str) -> str:
    return TUNING_OPPONENT_ENTITIES.get(opponent, opponent)


def compute_trial_rating_2p(
    net,
    *,
    num_games: int,
    device: str,
    num_sims: int = 32,
    q_scale: float = 10.0,
    max_turns: int = 300,
    league: League | None = None,
) -> tuple[float, dict[str, float]]:
    """Anchored 2p Bradley–Terry rating for the trial net from head-to-head evals."""
    num_players = 2
    metrics = evaluate_checkpoint(
        net,
        num_games=num_games,
        num_players=num_players,
        num_sims=num_sims,
        device=device,
        q_scale=q_scale,
        max_turns=max_turns,
    )

    results: list[dict] = []
    for opponent in ("random", "heuristic", "opus"):
        wr_key = f"vs_{opponent}_winrate"
        _winrate_to_match_result(
            results,
            TUNING_AGENT_ENTITY,
            _rating_entity_for_opponent(opponent),
            metrics[wr_key],
            num_games,
            num_players=num_players,
        )

    ref_anchors = R.reference_anchors_from_manifest(
        league.manifest if league is not None else None
    )
    top_entry = league.top_opponent_entry() if league is not None else None
    if top_entry is not None:
        opp_path = league._resolve_path(str(top_entry["path"]))
        opp_net, _ = load_net_from_checkpoint(opp_path, map_location=device)
        opp_net.eval()
        cfg = EvalConfig(
            num_games=num_games,
            num_players=num_players,
            num_sims=num_sims,
            max_turns=max_turns,
            q_scale=q_scale,
            device=str(device),
        )
        top_wr = play_vs_net_opponent(net, opp_net, cfg)
        opp_entity = league._entry_entity_id(int(top_entry["idx"]))
        metrics["vs_top_league_winrate"] = top_wr
        metrics["top_league_opponent"] = float(top_entry.get("iteration", -1))
        metrics["top_league_opponent_rating"] = float(
            top_entry.get("rating_2p", top_entry.get("rating", 0.0))
        )
        _winrate_to_match_result(
            results,
            TUNING_AGENT_ENTITY,
            opp_entity,
            top_wr,
            num_games,
            num_players=num_players,
        )
    else:
        metrics["vs_top_league_winrate"] = float("nan")
        metrics["top_league_opponent"] = -1.0

    per_pc = R.fit_ratings_for_pc(
        results,
        2,
        anchors={"random": R.RANDOM_ANCHOR_RATING},
        use_reference_anchors=True,
        reference_anchors_per_pc=ref_anchors,
    )
    raw_2p = per_pc.get(TUNING_AGENT_ENTITY, R.DEFAULT_INITIAL_RATING)
    cal_scale = R.calibration_scales_for(ref_anchors)
    rating_2p = R.calibrate_rating(raw_2p, 2, cal_scale)
    metrics["rating_2p_raw"] = raw_2p
    metrics["rating_2p"] = rating_2p
    return rating_2p, metrics


def evaluate_objective(
    net,
    *,
    objective_name: str,
    num_games: int,
    device: str,
    num_sims: int = 32,
    q_scale: float = 10.0,
    max_turns: int = 300,
    league: League | None = None,
) -> tuple[float, dict[str, float]]:
    """Evaluate one tuning objective and return (objective_value, metrics)."""
    if objective_name == OBJECTIVE_RATING_2P:
        return compute_trial_rating_2p(
            net,
            num_games=num_games,
            device=device,
            num_sims=num_sims,
            q_scale=q_scale,
            max_turns=max_turns,
            league=league,
        )
    if objective_name == OBJECTIVE_OPUS_WINRATE:
        metrics = evaluate_checkpoint(
            net,
            num_games=num_games,
            num_players=2,
            num_sims=num_sims,
            device=device,
            q_scale=q_scale,
            max_turns=max_turns,
            opponents=OPUS_ONLY_OPPONENTS,
        )
        value = float(metrics["vs_opus_winrate"])
        metrics[OBJECTIVE_OPUS_WINRATE] = value
        return value, metrics
    raise ValueError(f"unknown objective={objective_name!r}; choose from {OBJECTIVES}")


def _baseline_eval_path(
    output_dir: str | pathlib.Path,
    objective_name: str = OBJECTIVE_RATING_2P,
) -> pathlib.Path:
    if objective_name == OBJECTIVE_RATING_2P:
        return pathlib.Path(output_dir) / "tune_baseline_eval.json"
    return pathlib.Path(output_dir) / f"tune_baseline_eval_{objective_name}.json"


def run_study_baseline_eval(
    *,
    init_from: str,
    objective_name: str,
    device: str,
    rating_games: int,
    rating_sims: int,
    q_scale: float,
    max_turns: int,
    league: League | None,
) -> tuple[float, dict[str, float]]:
    net, _ = load_net_from_checkpoint(init_from, map_location=device)
    net.eval()
    return evaluate_objective(
        net,
        objective_name=objective_name,
        num_games=rating_games,
        device=device,
        num_sims=rating_sims,
        q_scale=q_scale,
        max_turns=max_turns,
        league=league,
    )


def load_or_compute_study_baseline(
    output_dir: str,
    *,
    init_from: str,
    objective_name: str,
    device: str,
    rating_games: int,
    rating_sims: int,
    q_scale: float,
    max_turns: int,
    league: League | None,
    force: bool = False,
) -> tuple[float, dict[str, float]]:
    path = _baseline_eval_path(output_dir, objective_name)
    if path.exists() and not force:
        payload = json.loads(path.read_text())
        if "objective_value" in payload:
            return float(payload["objective_value"]), payload.get("metrics", {})
        return float(payload["rating_2p"]), payload.get("metrics", {})

    if not init_from or not pathlib.Path(init_from).exists():
        raise FileNotFoundError(
            "Study baseline requires --init-from with an existing checkpoint."
        )

    objective_value, metrics = run_study_baseline_eval(
        init_from=init_from,
        objective_name=objective_name,
        device=device,
        rating_games=rating_games,
        rating_sims=rating_sims,
        q_scale=q_scale,
        max_turns=max_turns,
        league=league,
    )
    path.write_text(
        json.dumps(
            {
                "init_from": init_from,
                "objective": objective_name,
                "objective_value": objective_value,
                "rating_2p": objective_value,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return objective_value, metrics


def _training_loop_config(
    merged: dict,
    *,
    max_wall_minutes: float,
) -> tuple[LoopConfig, set[str]]:
    merged = {
        **merged,
        "compile_net": False,
        "save_buffer_in_checkpoints": False,
        "eval_games": 0,
        "league_selfplay_every": 0,
        "training_cycle_length": 4,
        "bot_selfplay_opus_prob": 0.5,
        "selfplay_turns_per_player": 60,
        "checkpoint_every": 999_999,
        "max_wall_minutes": max_wall_minutes,
        "max_iters": 999_999,
    }
    loop_fields = set(LoopConfig.__dataclass_fields__)
    cfg_fields = {k: v for k, v in merged.items() if k in loop_fields}
    return LoopConfig(**cfg_fields), set(cfg_fields.keys())


def trial_fn(
    trial: optuna.Trial,
    base_cfg: dict,
    output_dir: str,
    iters_per_trial: int,
    minutes_per_trial: float,
    sessions_per_trial: int,
    session_minutes: float,
    extrapolate_hours: float,
    study_baseline_rating: float,
    objective_name: str,
    narrow: bool,
    wide: bool,
    rating_games: int,
    rating_sims: int,
    keep_eval_checkpoints: bool,
    device: str,
    league: League | None,
) -> float:
    logger = logging.getLogger(__name__)
    try:
        sampled = define_search_space(trial, narrow=narrow, wide=wide)
        ensure_selfplay_mha_limit(
            int(sampled["selfplay_games"]),
            int(sampled["selfplay_sims"]),
        )
        merged = {**base_cfg, **sampled}
        q_scale = float(sampled.get("q_scale", 10.0))

        trial_run_id = f"tune_trial_{trial.number:03d}"
        trial_dir = pathlib.Path(output_dir) / trial_run_id
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        run = Run(trial_run_id, runs_root=output_dir)

        use_session_curve = sessions_per_trial > 0 and minutes_per_trial <= 0
        if not use_session_curve:
            wall = minutes_per_trial if minutes_per_trial > 0 else 60.0 * 24
            legacy_merged = {**merged}
            if minutes_per_trial <= 0:
                legacy_merged["max_iters"] = iters_per_trial
            cfg, explicit_fields = _training_loop_config(legacy_merged, max_wall_minutes=wall)
            run_loop(run, cfg, explicit_fields=explicit_fields)
            ckpt_path = run.ckpt_dir / "latest_resume.pt"
            net, _ = load_net_from_checkpoint(ckpt_path, map_location=device)
            net.eval()
            objective_value, wr = evaluate_objective(
                net,
                objective_name=objective_name,
                num_games=rating_games,
                device=device,
                num_sims=rating_sims,
                q_scale=q_scale,
                max_turns=cfg.eval_max_turns,
                league=league,
            )
            run.close()
            shutil.rmtree(trial_dir, ignore_errors=True)
            return objective_value

        session_hours = session_minutes / 60.0
        curve_times: list[float] = [0.0]
        curve_values: list[float] = [study_baseline_rating]
        session_values: list[float] = []

        for session_idx in range(sessions_per_trial):
            cfg, explicit_fields = _training_loop_config(
                merged,
                max_wall_minutes=session_minutes,
            )
            logger.info(
                "Trial %d session %d/%d: training %.0f min",
                trial.number,
                session_idx + 1,
                sessions_per_trial,
                session_minutes,
            )
            run_loop(run, cfg, explicit_fields=explicit_fields)

            ckpt_path = run.ckpt_dir / "latest_resume.pt"
            net, _ = load_net_from_checkpoint(ckpt_path, map_location=device)
            net.eval()
            objective_value, wr = evaluate_objective(
                net,
                objective_name=objective_name,
                num_games=rating_games,
                device=device,
                num_sims=rating_sims,
                q_scale=q_scale,
                max_turns=cfg.eval_max_turns,
                league=league,
            )
            t_hours = (session_idx + 1) * session_hours
            curve_times.append(t_hours)
            curve_values.append(objective_value)
            session_values.append(objective_value)
            logger.info(
                "Trial %d session %d/%d done: t=%.2fh %s=%.4f",
                trial.number,
                session_idx + 1,
                sessions_per_trial,
                t_hours,
                objective_name,
                objective_value,
            )
            run.event(
                "tune_session_eval",
                {
                    "trial": trial.number,
                    "session": session_idx + 1,
                    "t_hours": t_hours,
                    "objective_name": objective_name,
                    "objective_value": objective_value,
                    **{k: v for k, v in wr.items() if isinstance(v, (int, float))},
                },
            )

        curve: dict[str, float] = {}
        if objective_name == OBJECTIVE_RATING_2P:
            curve = extrapolate_rating(
                curve_times,
                curve_values,
                extra_hours=extrapolate_hours,
            )
            objective = curve["predicted_rating_2p"]
        else:
            curve = extrapolate_winrate(
                curve_times,
                curve_values,
                extra_hours=extrapolate_hours,
            )
            objective = curve["predicted_winrate"]

        trial.set_user_attr("objective_name", objective_name)
        trial.set_user_attr("study_baseline_objective", study_baseline_rating)
        trial.set_user_attr(f"study_baseline_{objective_name}", study_baseline_rating)
        trial.set_user_attr("curve_times_hours", curve_times)
        trial.set_user_attr("curve_objective_values", curve_values)
        trial.set_user_attr(f"curve_{objective_name}_values", curve_values)
        trial.set_user_attr("session_objective_values", session_values)
        trial.set_user_attr(f"session_{objective_name}_values", session_values)
        for key, val in curve.items():
            trial.set_user_attr(key, val)
        trial.set_user_attr("objective_last", curve_values[-1])
        trial.set_user_attr(f"{objective_name}_last", curve_values[-1])
        if objective_name == OBJECTIVE_RATING_2P:
            trial.set_user_attr("study_baseline_rating_2p", study_baseline_rating)
            trial.set_user_attr("curve_ratings_2p", curve_values)
            trial.set_user_attr("session_ratings_2p", session_values)
            trial.set_user_attr("rating_2p_last", curve_values[-1])
        else:
            trial.set_user_attr("predicted_opus_winrate", objective)

        if keep_eval_checkpoints:
            eval_ckpt_dir = pathlib.Path(output_dir) / "tune_eval_ckpts"
            eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                run.ckpt_dir / "latest_resume.pt",
                eval_ckpt_dir / f"trial_{trial.number:03d}.pt",
            )

        logger.info(
            "Trial %d finished: objective=%.4f (%s projected +%.0fh → t=%.1fh) "
            "curve=%s reward_mode=%s",
            trial.number,
            objective,
            objective_name,
            extrapolate_hours,
            curve["t_target_hours"],
            list(zip(curve_times, [round(r, 4) for r in curve_values])),
            sampled.get("reward_mode", "?"),
        )
        run.close()
        shutil.rmtree(trial_dir, ignore_errors=True)
        return objective
    except KeyboardInterrupt:
        raise
    except optuna.TrialPruned:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        # Treat CUDA OOM as a pruned trial so Optuna can continue cleanly.
        logger = logging.getLogger(__name__)
        logger.error("Trial %d failed with CUDA OOM: %s", trial.number, exc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise optuna.TrialPruned(f"CUDA OOM in trial {trial.number}")
    except Exception as exc:
        logging.getLogger(__name__).error("Trial %d failed: %s", trial.number, exc)
        return float("-inf")


def _prepare_study_for_optimize(study: optuna.Study, n_trials: int) -> int:
    """Configure sampler on resume and return remaining trial count."""
    completed = sum(
        1
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value != float("-inf")
    )
    if completed > 0:
        # Older trials used conditional selfplay_sims categories; reset TPE so
        # fixed [8,16,32,64] does not hit "dynamic value space" on resume.
        study.sampler = optuna.samplers.TPESampler(
            n_startup_trials=min(10, completed),
            warn_independent_sampling=False,
        )
    remaining = max(0, n_trials - len(study.trials))
    return remaining


def _fail_stale_running_trials(study: optuna.Study) -> int:
    """Mark RUNNING trials as FAIL after a crashed worker (zombie trials)."""
    n = 0
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ..train.device import resolve_device

    device = resolve_device(args.device)
    if args.narrow_ranges and args.wide_ranges:
        print("error: use only one of --narrow-ranges or --wide-ranges", file=sys.stderr)
        return 2

    base_cfg: dict = {
        "device": device,
        "num_players": args.num_players,
        "hidden": args.hidden,
        "arch": args.arch,
    }
    if args.init_from:
        base_cfg["init_from"] = args.init_from

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_session_curve = args.sessions_per_trial > 0 and args.minutes_per_trial <= 0
    if use_session_curve and not args.init_from:
        print(
            "error: session-curve tuning requires --init-from for the study baseline eval.",
            file=sys.stderr,
        )
        return 2

    league_root = output_dir / "league"
    league: League | None = None
    if (league_root / "league.json").exists():
        league = League(league_root)
        top = league.top_opponent_entry()
        if args.objective == OBJECTIVE_OPUS_WINRATE:
            print("Objective eval opponent: opus only")
        elif top is not None:
            print(
                f"League eval opponent: iter={top.get('iteration')} "
                f"rating_2p={top.get('rating_2p', top.get('rating'))}"
            )
        else:
            print("League manifest present but no available checkpoints for eval.")
    else:
        print(f"No league at {league_root}; rating eval uses reference bots only.")

    study_baseline_rating = 0.0
    if use_session_curve:
        print("Running study baseline eval (t=0)…")
        study_baseline_rating, baseline_metrics = load_or_compute_study_baseline(
            str(output_dir),
            init_from=args.init_from,
            objective_name=args.objective,
            device=device,
            rating_games=args.rating_games,
            rating_sims=args.rating_sims,
            q_scale=args.baseline_q_scale,
            max_turns=300,
            league=league,
            force=args.force_baseline,
        )
        print(
            f"Study baseline {args.objective}={study_baseline_rating:.4f} "
            f"(saved to {_baseline_eval_path(output_dir, args.objective)})"
        )
        if baseline_metrics:
            if args.objective == OBJECTIVE_OPUS_WINRATE:
                print(f"  vs opus={baseline_metrics.get('vs_opus_winrate', 0):.3f}")
            else:
                print(
                    f"  vs random={baseline_metrics.get('vs_random_winrate', 0):.3f} "
                    f"heuristic={baseline_metrics.get('vs_heuristic_winrate', 0):.3f} "
                    f"opus={baseline_metrics.get('vs_opus_winrate', 0):.3f}"
                )

    storage = args.storage or f"sqlite:///{output_dir}/optuna_{args.study_name}.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    stale = _fail_stale_running_trials(study)
    if stale:
        print(f"Marked {stale} stale RUNNING trial(s) as FAIL (prior worker exited).")

    remaining = _prepare_study_for_optimize(study, args.n_trials)
    if remaining == 0:
        print(f"Study already has {len(study.trials)} trials (target {args.n_trials}).")
    else:
        if use_session_curve:
            total_train_h = args.sessions_per_trial * (args.session_minutes / 60.0)
            objective_detail = (
                f"log fit, extrapolate +{args.extrapolate_hours:.0f}h"
                if args.objective == OBJECTIVE_RATING_2P
                else f"logit fit, extrapolate +{args.extrapolate_hours:.0f}h"
            )
            print(
                f"Running {remaining} trial(s): {args.sessions_per_trial}×"
                f"{args.session_minutes:.0f}min train + eval "
                f"({total_train_h:.1f}h), objective={args.objective}, "
                f"{objective_detail}"
            )
        else:
            print(
                f"Running {remaining} more trial(s) "
                f"({len(study.trials)} existing, target {args.n_trials})."
            )

    objective = functools.partial(
        trial_fn,
        base_cfg=base_cfg,
        output_dir=args.output_dir,
        iters_per_trial=args.iters_per_trial,
        minutes_per_trial=args.minutes_per_trial,
        sessions_per_trial=args.sessions_per_trial,
        session_minutes=args.session_minutes,
        extrapolate_hours=args.extrapolate_hours,
        study_baseline_rating=study_baseline_rating,
        objective_name=args.objective,
        narrow=args.narrow_ranges,
        wide=args.wide_ranges,
        rating_games=args.rating_games,
        rating_sims=args.rating_sims,
        keep_eval_checkpoints=args.keep_eval_checkpoints,
        device=device,
        league=league,
    )
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    completed = [t for t in study.trials if t.value is not None]
    ranked = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    if ranked:
        objective_label = (
            f"log-curve extrapolated 2p rating @ +{args.extrapolate_hours:.0f}h"
            if args.objective == OBJECTIVE_RATING_2P
            else f"logit-curve projected {args.objective} @ +{args.extrapolate_hours:.0f}h"
        )
        print(
            f"\nTop trials ({objective_label}, "
            f"self-play {FIXED_SELFPLAY_GAMES}×{FIXED_SELFPLAY_SIMS}):"
        )
        for t in ranked:
            pred = t.user_attrs.get(
                "predicted_rating_2p",
                t.user_attrs.get("predicted_opus_winrate", t.value),
            )
            last = t.user_attrs.get(
                "objective_last",
                t.user_attrs.get("rating_2p_last", float("nan")),
            )
            print(
                f"  #{t.number}: objective={pred:.4f} "
                f"(last_session={last:.4f}) params={t.params}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
