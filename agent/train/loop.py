"""The iterative train-evaluate-improve loop."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import random as stdlib_random
import time
from typing import Optional

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import encoder as ENC
from ..net.model import AzulNet
from ..obs.run import Run
from .checkpointing import (
    checkpoint_net_state_dict,
    load_checkpoint,
    load_checkpoint_payload,
    save_checkpoint,
)
from .device import configure_device, resolve_device
from .league import League
from .league_selfplay import run_league_selfplay
from .learner import make_optimizer, step_from_buffer
from .replay_buffer import ReplayBuffer
from .selfplay import run_selfplay
from .unified_eval import UnifiedEvalConfig, UnifiedEvalHandle


@dataclasses.dataclass
class LoopConfig:
    num_players: int = 2
    device: str = "auto"
    hidden: int = 256
    arch: str = "attn"
    selfplay_games: int = 512
    selfplay_sims: int = 8
    selfplay_max_turns: int = 200
    selfplay_turns_per_player: int = 50
    replay_capacity: int = 600_000
    learner_batch: int = 256
    learner_steps_per_iter: int = 192
    entropy_bonus: float = 0.015
    checkpoint_every: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_iters: int = 500
    max_wall_minutes: float = 60.0
    init_from: str = ""
    run_id: str = "default"
    runs_root: str = ""
    dirichlet_alpha: float = 0.15
    dirichlet_mix: float = 0.40
    q_scale: float = 22.0
    time_discount: float = 1.0
    reward_mode: str = "score_scaled"
    league_selfplay_every: int = 3
    league_opponent_prob: float = 0.5
    league_opponent_sims: int = 4
    league_max_entries: int = 24
    league_keep_recent: int = 8
    eval_games: int = 512
    eval_sims: int = 64
    eval_max_turns: int = 200
    eval_turns_per_player: int = 60
    eval_league_opponents: int = 4
    use_amp: bool = False
    compile_net: bool = False
    keep_recent_checkpoints: int = 3


_GPU_DEFAULTS: dict[str, object] = {
    "selfplay_games": 2048,
    "selfplay_sims": 32,
    "learner_batch": 4096,
    "replay_capacity": 820_000,
    "learner_steps_per_iter": 64,
    "use_amp": True,
    "compile_net": True,
}

# Attention models use more VRAM per game than flat MLP.
_GPU_ATTN_SELFPLAY_GAMES = 1024


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
    if cfg_out.arch == "attn" and "selfplay_games" not in explicit_fields:
        if cfg_out.selfplay_games > _GPU_ATTN_SELFPLAY_GAMES:
            cfg_out = dataclasses.replace(cfg_out, selfplay_games=_GPU_ATTN_SELFPLAY_GAMES)
    return cfg_out


def _league_trigger(cur_iter: int, num_players: int, every: int) -> bool:
    if every <= 0:
        return False
    if every == 1:
        return True
    h = hashlib.md5(f"{cur_iter}|{num_players}".encode()).digest()
    n = int.from_bytes(h[:8], "big")
    return (n % every) == 0


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
    for result in eval_results.get("pairwise", []):
        winner = result["winner"]
        loser = result["loser"]
        if winner == "eval_agent":
            winner = eval_agent_entity
        if loser == "eval_agent":
            loser = eval_agent_entity
        for i, idx in league_entry_map.items():
            if winner == i:
                winner = f"ckpt:{idx}"
            if loser == i:
                loser = f"ckpt:{idx}"
        league.record_result(
            winner,
            loser,
            wins_w=result["weight"],
            wins_l=0.0,
            num_players=int(result.get("num_players", 2)),
        )
    return league.recompute_ratings()


def run_loop(
    run: Run,
    config: LoopConfig,
    explicit_fields: set[str] | None = None,
) -> dict:
    run.write_config_if_missing(dataclasses.asdict(config))
    run.event("loop_start", {"config": dataclasses.asdict(config)})

    device = resolve_device(config.device)
    dev_info = configure_device(device)
    run.event("device_selected", {"requested": config.device, **dev_info})

    config = apply_device_defaults(config, device, explicit_fields=explicit_fields)
    run.event("effective_config", dataclasses.asdict(config))

    net = AzulNet(hidden=config.hidden, arch=config.arch).to(device)
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

    grad_scaler: Optional[torch.amp.GradScaler] = None
    if config.use_amp and device.startswith("cuda"):
        grad_scaler = torch.amp.GradScaler("cuda")
        run.event("amp_enabled", {"device": device})

    start_iter = 0
    ckpt = _latest_ckpt(run.ckpt_dir)
    if ckpt is not None:
        payload = load_checkpoint(ckpt, net, optim, buffer, map_location=device)
        start_iter = int(payload.get("iteration", 0))
        run.event("loop_resumed", {"from": str(ckpt), "iter": start_iter})
    elif config.init_from and pathlib.Path(config.init_from).exists():
        payload = load_checkpoint_payload(config.init_from, map_location=device)
        net.load_state_dict(checkpoint_net_state_dict(payload))
        run.event("loop_init_from", {"from": config.init_from})

    league_root = run.root.parent / "league"
    league = League(
        league_root,
        max_entries=config.league_max_entries,
        keep_recent=config.league_keep_recent,
    )

    eval_handle = UnifiedEvalHandle(
        UnifiedEvalConfig(
            total_games=config.eval_games,
            num_sims=config.eval_sims,
            max_turns=config.eval_max_turns,
            turns_per_player=config.eval_turns_per_player,
            weight_2p=1.0,
            weight_3p=0.0,
            weight_4p=0.0,
            league_opponents=config.eval_league_opponents,
        ),
        hidden=config.hidden,
        arch=config.arch,
    )
    _last_eval_league_map: dict[str, int] = {}
    _last_eval_entity = ""

    t_start = time.monotonic()
    cur_iter = start_iter

    while True:
        elapsed_min = (time.monotonic() - t_start) / 60.0
        iters_done = cur_iter - start_iter

        result = eval_handle.try_collect()
        if result is not None:
            iter_tag, eval_results = result
            if "error" in eval_results:
                run.event("unified_eval_failed", {"iteration": iter_tag, **eval_results})
            else:
                entity = _last_eval_entity or "eval_agent"
                ratings = _apply_eval_results(
                    league, eval_results, entity, _last_eval_league_map
                )
                rating = ratings.get(entity, 0.0)
                row = {"iter": iter_tag, "elapsed_min": elapsed_min, "rating": rating}
                row.update(eval_results.get("metrics", {}))
                run.metric(row)
                run.event("unified_eval_done", {"iteration": iter_tag, "rating": rating})

        if iters_done >= config.max_iters:
            break
        if elapsed_min >= config.max_wall_minutes:
            break

        cur_iter += 1
        run.write_heartbeat({"iter": cur_iter, "phase": "selfplay", "buffer_size": buffer.size})

        sp_max_turns = (
            config.selfplay_turns_per_player * config.num_players
            if config.selfplay_turns_per_player > 0
            else config.selfplay_max_turns
        )

        use_league = (
            config.league_selfplay_every > 0
            and _league_trigger(cur_iter, config.num_players, config.league_selfplay_every)
            and len(league.list_entries()) > 0
        )

        if use_league:
            sp_metrics = run_league_selfplay(
                net,
                buffer,
                league,
                num_players=config.num_players,
                num_games=config.selfplay_games,
                device=device,
                max_turns=sp_max_turns,
                num_sims=config.selfplay_sims,
                seed=cur_iter,
                league_prob=config.league_opponent_prob,
                time_discount=config.time_discount,
                opponent_sims=config.league_opponent_sims,
                reward_mode=config.reward_mode,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_mix=config.dirichlet_mix,
                q_scale=config.q_scale,
            )
            run.event("league_selfplay_done", {"iter": cur_iter, **sp_metrics})
        else:
            sp_metrics = run_selfplay(
                net,
                buffer=buffer,
                num_games=config.selfplay_games,
                num_players=config.num_players,
                num_sims=config.selfplay_sims,
                max_turns=sp_max_turns,
                device=device,
                seed=cur_iter,
                time_discount=config.time_discount,
                reward_mode=config.reward_mode,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_mix=config.dirichlet_mix,
                q_scale=config.q_scale,
            )
            run.event("selfplay_done", {"iter": cur_iter, **sp_metrics})

        if buffer.size >= config.learner_batch:
            run.write_heartbeat({"iter": cur_iter, "phase": "learner"})
            net.train()
            accum = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            steps = 0
            for _ in range(config.learner_steps_per_iter):
                m = step_from_buffer(
                    net,
                    buffer,
                    optim,
                    batch_size=config.learner_batch,
                    num_players=config.num_players,
                    entropy_bonus=config.entropy_bonus,
                    grad_scaler=grad_scaler,
                )
                for k, v in m.items():
                    accum[k] = accum.get(k, 0.0) + v
                steps += 1
            for k in accum:
                accum[k] /= max(steps, 1)
            run.event("learner_done", {"iter": cur_iter, **accum})
        else:
            run.event("learner_skipped", {"iter": cur_iter, "buffer_size": buffer.size})

        if cur_iter % config.checkpoint_every == 0:
            path = run.ckpt_dir / f"iter_{cur_iter:06d}.pt"
            resume_path = run.ckpt_dir / "latest_resume.pt"
            cfg_dict = dataclasses.asdict(config)
            save_checkpoint(path, net, optim, cur_iter, cfg_dict, buffer=None)
            save_checkpoint(resume_path, net, optim, cur_iter, cfg_dict, buffer=buffer)
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

            entry = league.add_checkpoint(net, tag=f"i{cur_iter}", iteration=cur_iter)
            _last_eval_entity = f"ckpt:{entry['idx']}"

            if eval_handle.is_active():
                prev = eval_handle.wait_and_collect()
                if prev and "error" not in prev[1]:
                    ratings = _apply_eval_results(
                        league, prev[1], _last_eval_entity, _last_eval_league_map
                    )
                    run.event(
                        "unified_eval_done",
                        {"iteration": prev[0], "rating": ratings.get(_last_eval_entity, 0.0)},
                    )

            if config.eval_games > 0:
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
                eval_handle.launch(snapshot, league_paths, cur_iter, seed=cur_iter * 7)
                run.event("unified_eval_launched", {"iter": cur_iter})

        print(
            f"Iter {cur_iter}: buf={buffer.size}, "
            f"samples={sp_metrics.get('samples_added', 0)}, "
            f"elapsed={elapsed_min:.1f}m"
        )

    result = eval_handle.wait_and_collect()
    if result and "error" not in result[1]:
        entity = _last_eval_entity or "eval_agent"
        ratings = _apply_eval_results(league, result[1], entity, _last_eval_league_map)
        run.metric({"iter": result[0], "rating": ratings.get(entity, 0.0)})
    eval_handle.cleanup()

    final_resume = run.ckpt_dir / "latest_resume.pt"
    save_checkpoint(
        final_resume,
        net,
        optim,
        cur_iter,
        dataclasses.asdict(config),
        buffer=buffer,
    )
    run.event("loop_end", {"iter": cur_iter})
    return {"iter": cur_iter}


def run_loop_legacy(config: LoopConfig) -> None:
    """Backward-compatible entry using run_dir instead of Run."""
    run = Run(config.run_id, runs_root=config.runs_root or None)
    cfg = dataclasses.replace(config, run_id=run.run_id)
    run_loop(run, cfg)
