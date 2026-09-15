"""The iterative train-evaluate-improve loop."""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random as stdlib_random
import time
import signal
import threading
from typing import Callable, Optional

import torch

from ..env import actions as A
from ..env import engine as BE
from ..net import encoder as ENC
from ..net.model import AzulNet
from ..obs.run import Run
from .checkpointing import (
    checkpoint_net_state_dict,
    load_checkpoint,
    load_checkpoint_payload,
    load_model_state_dict_compatible,
    save_checkpoint,
    warm_start_net,
)
from .device import configure_device, resolve_device
from .instrumentation import (
    PerfCounters,
    cuda_memory_snapshot,
    maybe_time,
    nvidia_smi_snapshot,
    reset_cuda_peak_memory,
    resource_delta,
    resource_snapshot,
)
from .league import League
from .bot_selfplay import run_bot_selfplay
from .league_selfplay import run_league_selfplay
from .learner import make_optimizer, step_from_buffer
from .replay_buffer import ReplayBuffer
from .selfplay import run_selfplay
from .stability import net_parameters_finite, reset_buffer, restore_net_from_checkpoint
from .unified_eval import UnifiedEvalConfig, UnifiedEvalHandle
from .reproducibility import seed_all, provenance, require_disk_space


@dataclasses.dataclass
class LoopConfig:
    num_players: int = 2
    device: str = "auto"
    hidden: int = 256
    arch: str = "attn"
    selfplay_games: int = 1023
    selfplay_sims: int = 32
    selfplay_max_turns: int = 200
    selfplay_turns_per_player: int = 60
    replay_capacity: int = 1_000_000
    learner_batch: int = 256
    learner_steps_per_iter: int = 64
    entropy_bonus: float = 0.034
    checkpoint_every: int = 25
    lr: float = 2.75e-3
    weight_decay: float = 1.2e-5
    max_iters: int = 1000
    max_wall_minutes: float = 1440.0
    init_from: str = ""
    run_id: str = "default"
    runs_root: str = ""
    dirichlet_alpha: float = 0.27
    dirichlet_mix: float = 0.47
    q_scale: float = 25.0
    time_discount: float = 1.0
    reward_mode: str = "score_scaled"
    training_cycle_length: int = 4
    bot_selfplay_opus_prob: float = 0.5
    bot_selfplay_astra_prob: float = 0.0
    bot_selfplay_workers: int = 1
    league_selfplay_every: int = 3
    league_opponent_prob: float = 0.5
    league_opponent_sims: int = 4
    league_max_entries: int = 24
    league_keep_recent: int = 8
    eval_games: int = 512
    eval_sims: int = 32
    eval_astra_fraction: float = 0.125
    eval_max_turns: int = 200
    eval_turns_per_player: int = 60
    eval_league_opponents: int = 4
    use_amp: bool = False
    compile_net: bool = False
    keep_recent_checkpoints: int = 3
    save_buffer_in_checkpoints: bool = True
    profile_training: bool = False
    profile_sync_cuda: bool = False
    bot_policy: str = "batched"
    eval_workers: int = 1
    eval_device: str = "cpu"
    eval_q_scale: float = 28.0
    eval_temperature: float = 0.25
    search_backend: str = "one_ply"
    eval_search_backend: str = "one_ply"
    seed: int = 20260913
    torch_threads: int = 1
    league_root: str = ""
    provenance: dict | None = None
    fail_on_nonfinite: bool = True
    profile_every: int = 100

    def __post_init__(self) -> None:
        if not 0 <= self.eval_astra_fraction <= 1:
            raise ValueError("eval_astra_fraction must be between zero and one")
        if (not math.isfinite(self.bot_selfplay_opus_prob)
                or not math.isfinite(self.bot_selfplay_astra_prob)
                or not 0 <= self.bot_selfplay_opus_prob <= 1
                or not 0 <= self.bot_selfplay_astra_prob <= 1
                or self.bot_selfplay_opus_prob + self.bot_selfplay_astra_prob > 1):
            raise ValueError("bot self-play probabilities must be finite, in [0, 1], and sum to at most one")
        if type(self.bot_selfplay_workers) is not int or not 1 <= self.bot_selfplay_workers <= 8:
            raise ValueError("bot_selfplay_workers must be an integer from one to eight")


_GPU_DEFAULTS: dict[str, object] = {
    "selfplay_games": 1023,
    "selfplay_sims": 32,
    "learner_batch": 256,
    "replay_capacity": 1_000_000,
    "learner_steps_per_iter": 64,
    "use_amp": True,
    "compile_net": False,
}

# Max selfplay batch for attn (games×sims must be <= 65535).
_GPU_ATTN_SELFPLAY_GAMES = 2047


def apply_device_defaults(
    cfg: LoopConfig, device: str, explicit_fields: set[str] | None = None
) -> LoopConfig:
    if not device.startswith("cuda"):
        return dataclasses.replace(cfg)
    explicit_fields = explicit_fields or set()
    factory = LoopConfig()
    overrides: dict[str, object] = {}
    for field_name, gpu_value in _GPU_DEFAULTS.items():
        if field_name in explicit_fields:
            continue
        if getattr(cfg, field_name) == getattr(factory, field_name):
            overrides[field_name] = gpu_value
    cfg_out = dataclasses.replace(cfg, **overrides)
    if cfg_out.arch in {"attn", "source_attn"} and "selfplay_games" not in explicit_fields:
        if cfg_out.selfplay_games > _GPU_ATTN_SELFPLAY_GAMES:
            cfg_out = dataclasses.replace(cfg_out, selfplay_games=_GPU_ATTN_SELFPLAY_GAMES)
    return cfg_out


def _training_phase(cur_iter: int, cycle_length: int) -> str:
    """4-step cycle (iter 1..4): selfplay, league, selfplay, bot."""
    if cycle_length <= 0:
        return "selfplay"
    r = cur_iter % cycle_length
    if r == 2:
        return "league"
    if r == 0:
        return "bot"
    return "selfplay"


_SELFPLAY_KIND_LABEL: dict[str, str] = {
    "selfplay": "standard_selfplay",
    "league": "league_selfplay",
    "bot": "bot_selfplay",
}


def _resolve_selfplay_kind(
    scheduled_phase: str,
    league_has_entries: bool,
) -> tuple[str, str]:
    """Return (kind label, scheduled_phase) for logging; kind is what actually runs."""
    if scheduled_phase == "league" and not league_has_entries:
        return "standard_selfplay", scheduled_phase
    return _SELFPLAY_KIND_LABEL.get(scheduled_phase, scheduled_phase), scheduled_phase


def _latest_ckpt(ckpt_dir: pathlib.Path) -> Optional[pathlib.Path]:
    resume = ckpt_dir / "latest_resume.pt"
    if resume.exists():
        return resume
    ckpts = sorted(ckpt_dir.glob("iter_*.pt"))
    return ckpts[-1] if ckpts else None


def _get_league_opponent_paths(league: League, count: int, seed: int) -> list[str]:
    entries = [e for e in league.list_entries() if league._entry_available(e)]
    if not entries:
        return []
    rng = stdlib_random.Random(seed)
    sampled = rng.sample(entries, min(count, len(entries)))
    return [str(league._resolve_path(e["path"])) for e in sampled]


def _apply_eval_results(
    league: League,
    eval_results: dict,
    eval_agent_entity: str,
    league_entry_map: dict[str, int],
) -> dict[str, float]:
    context = eval_results.get("job_context", {})
    eval_agent_entity = context.get("entity", eval_agent_entity)
    league_entry_map = context.get("league_map", league_entry_map)
    def _to_entity(name: str) -> str:
        if name == "eval_agent":
            return eval_agent_entity
        if name in league_entry_map:
            return f"ckpt:{league_entry_map[name]}"
        if name == "opus":
            return "heuristic_opus"
        return name

    wins: dict[tuple[str, str, int], float] = {}
    ties: dict[tuple[str, str, int], float] = {}
    for result in eval_results.get("pairwise", []):
        winner_entity = _to_entity(result["winner"])
        loser_entity = _to_entity(result["loser"])
        weight = float(result["weight"])
        num_players = int(result.get("num_players", 2))
        if winner_entity == loser_entity:
            continue
        # Older workers encoded ties as reciprocal half-wins. Accumulate before
        # writing: the league stores integer counts and rounds each write.
        if result.get("is_tie", weight == 0.5):
            a, b = sorted((winner_entity, loser_entity))
            key = (a, b, num_players)
            ties[key] = ties.get(key, 0.0) + weight
        else:
            key = (winner_entity, loser_entity, num_players)
            wins[key] = wins.get(key, 0.0) + weight
    for (winner, loser, players), weight in wins.items():
        league.record_result(winner, loser, weight, 0.0, 0.0, num_players=players)
    for (a, b, players), weight in ties.items():
        league.record_result(a, b, 0.0, 0.0, weight, num_players=players)
    return league.recompute_ratings()


def _record_eval_completion(
    run: Run, league: League, iteration: int, result: dict,
    entity: str, league_map: dict[str, int],
    elapsed_min: float | None = None,
) -> None:
    """Persist diagnostics on every collection path, including shutdown."""
    if "error" in result:
        run.event("unified_eval_failed", {"iteration": iteration, **result})
        return
    ratings = _apply_eval_results(league, result, entity, league_map)
    entity = result.get("job_context", {}).get("entity", entity)
    rating = ratings.get(entity, 0.0)
    row = {"iter": iteration, "rating": rating, **result.get("metrics", {})}
    if elapsed_min is not None:
        row["elapsed_min"] = elapsed_min
    run.metric(row)
    run.event("unified_eval_done", {"iteration": iteration, "rating": rating,
                                  "astra_identity": result.get("astra_identity", {})})


def _run_loop(
    run: Run,
    config: LoopConfig,
    explicit_fields: set[str] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> dict:
    if config.num_players not in (2, 3, 4):
        raise ValueError("num_players must be 2, 3, or 4")
    if config.max_wall_minutes > 480:
        run.event("long_run_budget", {"minutes": config.max_wall_minutes})
    torch.set_num_threads(config.torch_threads)
    seed_all(config.seed)
    config = dataclasses.replace(config, provenance=provenance())
    run.write_config_if_missing(dataclasses.asdict(config))
    run.event("loop_start", {"config": dataclasses.asdict(config)})

    device = resolve_device(config.device)
    dev_info = configure_device(device)
    run.event("device_selected", {"requested": config.device, **dev_info})

    config = apply_device_defaults(config, device, explicit_fields=explicit_fields)
    run.event("effective_config", dataclasses.asdict(config))

    # Resume archives use lossless compression. Keep one raw replay equivalent
    # of launch headroom; each compressed write checks actual free disk space and
    # preserves the previous archive if a replacement cannot be completed.
    replay_bytes = config.replay_capacity * (ENC.D_GLOBAL * 4 + ENC.NUM_SOURCES * ENC.D_SOURCE * 4
                                            + A.NUM_ACTIONS * 5 + BE.MAX_PLAYERS * 4 + 8)
    require_disk_space(run.ckpt_dir, replay_bytes if config.save_buffer_in_checkpoints else 0)
    net = AzulNet(hidden=config.hidden, arch=config.arch).to(device)
    net.trained_player_counts = [config.num_players]
    if config.compile_net:
        net.enable_compile()

    optim = make_optimizer(net, lr=config.lr, weight_decay=config.weight_decay)
    buffer = ReplayBuffer(
        capacity=config.replay_capacity,
        d_global=ENC.D_GLOBAL,
        n_sources=ENC.NUM_SOURCES,
        d_source=ENC.D_SOURCE,
        num_actions=A.NUM_ACTIONS,
        max_players=BE.MAX_PLAYERS,
        device=device,
    )

    if config.use_amp and device.startswith("cuda"):
        run.event("amp_note", {"detail": "learner uses fp32; AMP not applied to weight updates"})

    start_iter = 0
    prior_wall_s = 0.0
    recover_path: Optional[str] = None
    ckpt = _latest_ckpt(run.ckpt_dir)
    if ckpt is not None:
        payload = load_checkpoint(ckpt, net, optim, buffer, map_location="cpu")
        start_iter = int(payload.get("iteration", 0))
        prior_wall_s = float(payload.get("progress", {}).get("training_wall_s", 0))
        saved_config = payload.get("config", {})
        immutable = ("arch", "hidden", "num_players", "seed", "reward_mode", "search_backend",
                     "replay_capacity", "lr", "weight_decay", "learner_steps_per_iter",
                     "selfplay_games", "selfplay_sims", "training_cycle_length", "q_scale",
                     "dirichlet_alpha", "dirichlet_mix", "entropy_bonus", "time_discount")
        changed = [key for key in immutable if key in saved_config and saved_config[key] != getattr(config, key)]
        if changed:
            raise ValueError(f"Resume configuration differs ({changed}); use a new run with --init-from")
        del payload
        recover_path = str(ckpt)
        run.event("loop_resumed", {"from": str(ckpt), "iter": start_iter})
    elif config.init_from:
        payload = load_checkpoint_payload(config.init_from, map_location="cpu")
        migrated = warm_start_net(net, payload)
        del payload
        recover_path = config.init_from
        run.event("loop_init_from", {"from": config.init_from, "migrated": migrated})

    def _recover_training(reason: str) -> None:
        nonlocal optim
        if config.fail_on_nonfinite:
            raise RuntimeError(f"Training stopped: {reason}")
        if not recover_path or not pathlib.Path(recover_path).exists():
            run.event("training_recover_failed", {"reason": reason}, level="ERROR")
            raise RuntimeError(f"cannot recover weights: {reason}")
        restore_net_from_checkpoint(net, recover_path, device)
        optim = make_optimizer(net, lr=config.lr, weight_decay=config.weight_decay)
        reset_buffer(buffer)
        run.event(
            "training_recovered",
            {"reason": reason, "from": recover_path, "buffer_cleared": True},
            level="WARNING",
        )

    league_root = pathlib.Path(config.league_root) if config.league_root else run.root.parent / "league"
    league = League(
        league_root,
        max_entries=config.league_max_entries,
        keep_recent=config.league_keep_recent,
    )

    eval_handle = UnifiedEvalHandle(
        UnifiedEvalConfig(
            total_games=config.eval_games,
            num_sims=config.eval_sims,
            astra_opponent_fraction=config.eval_astra_fraction,
            max_turns=config.eval_max_turns,
            turns_per_player=config.eval_turns_per_player,
            weight_2p=1.0,
            weight_3p=0.0,
            weight_4p=0.0,
            league_opponents=config.eval_league_opponents,
            q_scale=config.eval_q_scale,
            temperature=config.eval_temperature,
            inference_device=config.eval_device,
            search_backend=config.eval_search_backend,
            profile=config.profile_training,
            num_workers=max(1, config.eval_workers),
        ),
        hidden=config.hidden,
        arch=config.arch,
    )
    try:
        _last_eval_league_map: dict[str, int] = {}
        _last_eval_entity = ""

        t_start = time.monotonic()
        cur_iter = start_iter

        while True:
            # `prior_wall_s` comes from a durable resume checkpoint.  A resumed
            # bounded experiment must spend only its remaining wall-clock budget,
            # rather than silently starting a fresh full-length allocation.
            elapsed_min = (prior_wall_s + time.monotonic() - t_start) / 60.0
            iters_done = cur_iter - start_iter

            result = eval_handle.try_collect()
            if result is not None:
                iter_tag, eval_results = result
                _record_eval_completion(run, league, iter_tag, eval_results,
                                        _last_eval_entity or "eval_agent", _last_eval_league_map, elapsed_min)

            if should_stop():
                run.event("stop_requested", {"iter": cur_iter})
                break
            if iters_done >= config.max_iters:
                break
            if elapsed_min >= config.max_wall_minutes:
                break

            cur_iter += 1
            buffer.iteration = cur_iter
            profiling = config.profile_training or (config.profile_every > 0 and cur_iter % config.profile_every == 0)
            perf = PerfCounters(
                enabled=profiling,
                device=device,
                sync_cuda=config.profile_sync_cuda,
            )
            perf_arg = perf if profiling else None
            iter_resource_start = resource_snapshot()
            phase_started = time.monotonic()
            if profiling:
                reset_cuda_peak_memory(device)

            cycle_len = config.training_cycle_length
            if cycle_len <= 0 and config.league_selfplay_every > 0:
                cycle_len = config.league_selfplay_every * 2
            scheduled_phase = _training_phase(cur_iter, cycle_len)
            league_has_entries = len(league.list_entries()) > 0
            selfplay_kind, _ = _resolve_selfplay_kind(scheduled_phase, league_has_entries)
            cycle_step = (cur_iter % cycle_len) if cycle_len > 0 else 0

            run.write_heartbeat(
                {
                    "iter": cur_iter,
                    "phase": selfplay_kind,
                    "buffer_size": buffer.size,
                }
            )
            run.event(
                "iter_started",
                {
                    "iter": cur_iter,
                    "buffer_size": buffer.size,
                    "scheduled_phase": scheduled_phase,
                    "selfplay_kind": selfplay_kind,
                    "training_cycle_length": cycle_len,
                    "training_cycle_step": cycle_step,
                },
            )
            run.event(
                "selfplay_started",
                {
                    "iter": cur_iter,
                    "selfplay_kind": selfplay_kind,
                    "scheduled_phase": scheduled_phase,
                    "games": config.selfplay_games,
                    "sims": config.selfplay_sims,
                    "training_cycle_length": cycle_len,
                    "training_cycle_step": cycle_step,
                    "league_fallback": scheduled_phase == "league" and not league_has_entries,
                },
            )
            print(
                f"Iter {cur_iter}: starting {selfplay_kind} "
                f"(cycle {cycle_step}/{cycle_len}, buffer={buffer.size}, "
                f"games={config.selfplay_games}, sims={config.selfplay_sims})"
            )

            if not net_parameters_finite(net):
                _recover_training("nonfinite_weights_before_selfplay")

            sp_max_turns = (
                config.selfplay_turns_per_player * config.num_players
                if config.selfplay_turns_per_player > 0
                else config.selfplay_max_turns
            )

            phase = scheduled_phase

            def _selfplay_progress(info: dict) -> None:
                run.write_heartbeat(
                    {
                        "iter": cur_iter,
                        "phase": selfplay_kind,
                        "buffer_size": buffer.size,
                        **info,
                    }
                )
                run.event(
                    "selfplay_progress",
                    {"iter": cur_iter, "selfplay_kind": selfplay_kind, **info},
                )

            sp_common = dict(
                num_players=config.num_players,
                num_games=config.selfplay_games,
                device=device,
                max_turns=sp_max_turns,
                num_sims=config.selfplay_sims,
                seed=config.seed + cur_iter,
                search_backend=config.search_backend,
                time_discount=config.time_discount,
                reward_mode=config.reward_mode,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_mix=config.dirichlet_mix,
                q_scale=config.q_scale,
            )

            if phase == "league" and len(league.list_entries()) > 0:
                with perf.time("phase_selfplay_total"):
                    sp_metrics = run_league_selfplay(
                        net,
                        buffer,
                        league,
                        league_prob=config.league_opponent_prob,
                        opponent_sims=config.league_opponent_sims,
                        perf=perf_arg,
                        **sp_common,
                    )
                run.event(
                    "league_selfplay_done",
                    {"iter": cur_iter, "selfplay_kind": selfplay_kind, **sp_metrics},
                )
            elif phase == "bot":
                with perf.time("phase_selfplay_total"):
                    sp_metrics = run_bot_selfplay(
                        net,
                        buffer,
                        opus_prob=config.bot_selfplay_opus_prob,
                        astra_prob=config.bot_selfplay_astra_prob,
                        bot_policy=config.bot_policy,
                        bot_workers=config.bot_selfplay_workers,
                        perf=perf_arg,
                        **sp_common,
                    )
                run.event(
                    "bot_selfplay_done",
                    {"iter": cur_iter, "selfplay_kind": selfplay_kind, **sp_metrics},
                )
            else:
                if phase == "league":
                    run.event(
                        "league_selfplay_skipped",
                        {"iter": cur_iter, "reason": "empty_league"},
                        level="WARNING",
                    )
                with perf.time("phase_selfplay_total"):
                    sp_metrics = run_selfplay(
                        net,
                        buffer,
                        on_progress=_selfplay_progress,
                        perf=perf_arg,
                        **sp_common,
                    )
                run.event(
                    "selfplay_done",
                    {"iter": cur_iter, "selfplay_kind": selfplay_kind, **sp_metrics},
                )

            selfplay_wall_s = time.monotonic() - phase_started
            learner_started = time.monotonic()
            if buffer.size >= config.learner_batch:
                run.write_heartbeat({"iter": cur_iter, "phase": "learner"})
                net.train()
                accum = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
                steps = 0
                skipped_steps = 0
                with perf.time("phase_learner_total"):
                    if not net_parameters_finite(net):
                        _recover_training("nonfinite_weights_before_learner")
                    for _ in range(config.learner_steps_per_iter):
                        m = step_from_buffer(
                            net,
                            buffer,
                            optim,
                            batch_size=config.learner_batch,
                            num_players=config.num_players,
                            entropy_bonus=config.entropy_bonus,
                            perf=perf_arg,
                        )
                        if m.get("skipped", 0.0) > 0:
                            skipped_steps += 1
                            run.event("learner_step_failed", {"iter": cur_iter, "update": steps+1, **m}, level="ERROR")
                            if config.fail_on_nonfinite:
                                raise RuntimeError(f"Invalid learner update at iteration {cur_iter}: {m}")
                            continue
                        for k, v in m.items():
                            if k == "skipped":
                                continue
                            accum[k] = accum.get(k, 0.0) + v
                        steps += 1
                accum["learner_steps_skipped"] = skipped_steps
                accum["learner_steps_ok"] = steps
                if steps == 0:
                    accum["loss"] = float("nan")
                    accum["policy_loss"] = float("nan")
                    accum["value_loss"] = float("nan")
                    accum["entropy"] = float("nan")
                    run.event(
                        "learner_all_steps_skipped",
                        {"iter": cur_iter, "skipped": skipped_steps},
                        level="WARNING",
                    )
                else:
                    for k in ("loss", "policy_loss", "value_loss", "entropy"):
                        accum[k] /= steps
                for key in ("value_bias", "value_sign_accuracy", "grad_norm"):
                    if key in accum:
                        accum[key] /= max(steps, 1)
                accum.update(replay_sample_age_iters=buffer.last_sample_age,
                             replay_reuse=buffer.total_sampled / max(buffer.total_added, 1))
                run.event("learner_done", {"iter": cur_iter, **accum})
                if skipped_steps and config.fail_on_nonfinite:
                    raise RuntimeError(f"Non-finite learner updates at iteration {cur_iter}")
                if steps == 0 or not net_parameters_finite(net):
                    _recover_training(
                        "learner_all_steps_skipped"
                        if steps == 0
                        else "nonfinite_weights_after_learner"
                    )
            else:
                run.event("learner_skipped", {"iter": cur_iter, "buffer_size": buffer.size})

            run.metric({"iter": cur_iter, "phase": selfplay_kind,
                "selfplay_wall_s": round(selfplay_wall_s, 3),
                "learner_wall_s": round(time.monotonic() - learner_started, 3),
                "replay_reuse": buffer.total_sampled / max(buffer.total_added, 1),
                "replay_sample_age_iters": buffer.last_sample_age,
                "samples_added": sp_metrics.get("samples_added", 0),
                "unfinished_games": sp_metrics.get("games_total", 0) - sp_metrics.get("finished", 0),
                **cuda_memory_snapshot(device)})
            if cur_iter % config.checkpoint_every == 0:
                path = run.ckpt_dir / f"iter_{cur_iter:06d}.pt"
                resume_path = run.ckpt_dir / "latest_resume.pt"
                cfg_dict = dataclasses.asdict(config)
                ckpt_buffer = buffer if config.save_buffer_in_checkpoints else None
                with perf.time("checkpoint_save"):
                    save_checkpoint(path, net, optim, cur_iter, cfg_dict, buffer=None, progress={"training_wall_s": prior_wall_s + time.monotonic() - t_start})
                    save_checkpoint(resume_path, net, optim, cur_iter, cfg_dict, buffer=ckpt_buffer, progress={"training_wall_s": prior_wall_s + time.monotonic() - t_start})
                run.write_state(
                    {
                        "iter": cur_iter,
                        "last_checkpoint": str(resume_path),
                        "last_archive_checkpoint": str(path),
                    }
                )
                run.event("checkpoint_saved", {"iter": cur_iter, "path": str(path)})

                if config.keep_recent_checkpoints > 0:
                    ckpts = sorted(run.ckpt_dir.glob("iter_*.pt"))
                    for old in ckpts[: -config.keep_recent_checkpoints]:
                        old.unlink(missing_ok=True)

                with perf.time("league_add_checkpoint"):
                    entry = league.add_checkpoint(net, tag=f"i{cur_iter}", iteration=cur_iter)
                if eval_handle.is_active():
                    with perf.time("eval_wait_collect"):
                        prev = eval_handle.wait_and_collect()
                    if prev:
                        _record_eval_completion(run, league, prev[0], prev[1],
                                                _last_eval_entity, _last_eval_league_map)

                _last_eval_entity = f"ckpt:{entry['idx']}"
                if config.eval_games > 0:
                    with perf.time("eval_launch_prepare"):
                        league_paths = _get_league_opponent_paths(
                            league, config.eval_league_opponents, seed=cur_iter * 31
                        )
                        _last_eval_league_map = {}
                        for i, lpath in enumerate(league_paths):
                            for e in league.list_entries():
                                if str(league._resolve_path(e["path"])) == lpath:
                                    _last_eval_league_map[f"league_{i}"] = int(e["idx"])
                                    break
                        snapshot = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    with perf.time("eval_launch_process"):
                        eval_handle.launch(snapshot, league_paths, cur_iter, seed=config.seed + cur_iter * 7,
                                           context={"entity": _last_eval_entity, "league_map": dict(_last_eval_league_map)})
                    run.event("unified_eval_launched", {"iter": cur_iter})

            if profiling and iter_resource_start is not None:
                profile_fields = {
                    "iter": cur_iter,
                    "selfplay_kind": selfplay_kind,
                    "buffer_size": buffer.size,
                    **resource_delta(iter_resource_start, resource_snapshot()),
                    **perf.snapshot(),
                    **cuda_memory_snapshot(device),
                    **nvidia_smi_snapshot(),
                }
                profile_wall_s = float(profile_fields.get("profile_wall_s", 0.0))
                if profile_wall_s > 0:
                    profile_fields["profile_samples_added_per_s"] = round(
                        float(sp_metrics.get("samples_added", 0)) / profile_wall_s, 3
                    )
                    profile_fields["profile_games_finished_per_s"] = round(
                        float(sp_metrics.get("finished", 0)) / profile_wall_s, 3
                    )
                run.event("iter_profile", profile_fields)
                run.metric(profile_fields)

            print(
                f"Iter {cur_iter}: buf={buffer.size}, "
                f"samples={sp_metrics.get('samples_added', 0)}, "
                f"elapsed={elapsed_min:.1f}m"
            )

        result = eval_handle.wait_and_collect()
        if result:
            _record_eval_completion(run, league, result[0], result[1],
                                    _last_eval_entity or "eval_agent", _last_eval_league_map)
        eval_handle.cleanup()

        final_resume = run.ckpt_dir / "latest_resume.pt"
        final_buffer = buffer if config.save_buffer_in_checkpoints else None
        save_checkpoint(
            final_resume,
            net,
            optim,
            cur_iter,
            dataclasses.asdict(config),
            buffer=final_buffer,
            progress={"training_wall_s": prior_wall_s + time.monotonic() - t_start},
        )
        run.write_state({"iter": cur_iter, "last_checkpoint": str(final_resume)})
        run.write_heartbeat({"iter": cur_iter, "phase": "stopped" if should_stop() else "completed"})
        run.event("loop_end", {"iter": cur_iter})
        return {"iter": cur_iter, "stopped": should_stop(), "training_wall_s": prior_wall_s + time.monotonic() - t_start}

    finally:
        eval_handle.cleanup()


def run_loop(run: Run, config: LoopConfig, explicit_fields: set[str] | None = None) -> dict:
    """SIGINT/SIGTERM request a durable stop at the next iteration boundary."""
    stopping = False
    def request_stop(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
    try:
        return _run_loop(run, config, explicit_fields, lambda: stopping)
    except Exception as exc:
        run.event("loop_failed", {"error": str(exc)}, level="ERROR")
        run.write_heartbeat({"phase": "failed", "error": str(exc)})
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_loop_legacy(config: LoopConfig) -> None:
    """Backward-compatible entry using run_dir instead of Run."""
    run = Run(config.run_id, runs_root=config.runs_root or None)
    cfg = dataclasses.replace(config, run_id=run.run_id)
    run_loop(run, cfg)
