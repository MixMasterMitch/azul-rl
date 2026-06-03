"""Self-play data generation using Gumbel MCTS."""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch

from ..env import batched_engine as BE
from ..net import encoder as ENC
from ..net import model as M
from ..search.gumbel_mcts import gumbel_root_act
from .instrumentation import PerfCounters, maybe_time, tensor_nbytes
from .replay_buffer import ReplayBuffer

REWARD_MODES = ("binary", "score_scaled")


def _rotate_for_cp(
    values: torch.Tensor,
    cp: torch.Tensor,
    num_players: int | None = None,
) -> torch.Tensor:
    """Rotate per-seat values so index 0 is the acting player at record time."""
    _, p = values.shape
    nP = p if num_players is None else num_players
    active = values[:, :nP]
    idx = (torch.arange(nP, device=values.device).unsqueeze(0) + cp.unsqueeze(-1)) % nP
    rotated_active = active.gather(1, idx.to(torch.long))
    rotated = values.clone()
    rotated[:, :nP] = rotated_active
    return rotated


def _select_finished_samples(
    global_feat: torch.Tensor,
    source_feat: torch.Tensor,
    legal_mask: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    game_idx: torch.Tensor,
    ended_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...] | None:
    """Drop positions from stall-capped or degenerate games (no legal actions / non-finite targets)."""
    finished = ended_mask[game_idx]
    legal = legal_mask.any(dim=-1)
    finite = torch.isfinite(policy_target).all(dim=-1) & torch.isfinite(value_target).all(
        dim=-1
    )
    keep = finished & legal & finite
    if not keep.any():
        return None
    return (
        global_feat[keep],
        source_feat[keep],
        legal_mask[keep],
        policy_target[keep],
        value_target[keep],
    )


def _final_rank_values_binary(
    engine: BE.BatchedEngine, num_players: int
) -> torch.Tensor:
    """Winner +1, losers -1 per absolute seat."""
    B = engine.batch_size
    dev = engine.device
    values = torch.full((B, BE.MAX_PLAYERS), -1.0, dtype=torch.float32, device=dev)
    winners = engine.get_winners().to(torch.long)
    ended = engine.ended
    if ended.any():
        rows = ended.nonzero(as_tuple=True)[0]
        w = winners[rows]
        single = w >= 0
        if single.any():
            r = rows[single]
            values[r, w[single].long()] = 1.0
        shared = w == BE.SHARED_VICTORY
        if shared.any():
            r = rows[shared]
            tied = _shared_victory_mask(engine, r, num_players)
            values[r, :num_players] = torch.where(
                tied,
                torch.ones_like(tied, dtype=torch.float32),
                -torch.ones_like(tied, dtype=torch.float32),
            )
    return values


def _shared_victory_mask(
    engine: BE.BatchedEngine,
    rows: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    """Return active seats sharing the official win on ended tied rows."""
    scores = engine.scores[rows, :num_players].to(torch.int32)
    rows_complete = engine.wall[rows, :num_players].all(dim=-1).sum(dim=-1).to(torch.int32)
    max_score = scores.max(dim=-1, keepdim=True).values
    score_best = scores == max_score
    rows_for_best = rows_complete.masked_fill(~score_best, -1)
    max_rows = rows_for_best.max(dim=-1, keepdim=True).values
    return score_best & (rows_complete == max_rows)


def _final_rank_values_score_scaled(
    engine: BE.BatchedEngine, num_players: int
) -> torch.Tensor:
    """Winner(s) +1; losers -1/(n-1) + (score/winner_score)^2."""
    B = engine.batch_size
    dev = engine.device
    values = torch.full((B, BE.MAX_PLAYERS), -1.0, dtype=torch.float32, device=dev)
    winners = engine.get_winners().to(torch.long)
    scores = engine.scores[:, :num_players].float()
    ended = engine.ended
    if not ended.any():
        return values

    rows = ended.nonzero(as_tuple=True)[0]
    w = winners[rows]
    single = w >= 0
    if single.any():
        r = rows[single]
        ws = w[single].long()
        winner_score = scores[r].gather(1, ws.unsqueeze(1)).squeeze(1).clamp_min(1.0)
        loss_base = -1.0 / max(num_players - 1, 1)
        ratio = (scores[r] / winner_score.unsqueeze(1)).pow(2)
        row_values = (loss_base + ratio).clamp(-1.0, 1.0)
        row_values.scatter_(1, ws.unsqueeze(1), 1.0)
        values[r, :num_players] = row_values
    shared = w == BE.SHARED_VICTORY
    if shared.any():
        r = rows[shared]
        tied = _shared_victory_mask(engine, r, num_players)
        winner_score = scores[r].masked_fill(~tied, -1).max(dim=-1).values.clamp_min(1.0)
        loss_base = -1.0 / max(num_players - 1, 1)
        ratio = (scores[r] / winner_score.unsqueeze(1)).pow(2)
        row_values = (loss_base + ratio).clamp(-1.0, 1.0)
        values[r, :num_players] = torch.where(tied, torch.ones_like(row_values), row_values)
    return values


def _final_rank_values(
    engine: BE.BatchedEngine, num_players: int, reward_mode: str
) -> torch.Tensor:
    if reward_mode == "binary":
        return _final_rank_values_binary(engine, num_players)
    if reward_mode == "score_scaled":
        return _final_rank_values_score_scaled(engine, num_players)
    raise ValueError(f"Unknown reward_mode={reward_mode!r}. Choose from {REWARD_MODES}.")


def _default_temperature_schedule(step: int) -> float:
    if step < 30:
        return 1.0
    if step < 60:
        return 0.5
    return 0.25


def run_selfplay(
    net: M.AzulNet,
    buffer: ReplayBuffer,
    num_games: int = 512,
    num_players: int = 2,
    num_sims: int = 8,
    max_turns: int = 200,
    temperature_schedule: Optional[callable] = None,
    dirichlet_alpha: float = 0.3,
    dirichlet_mix: float = 0.25,
    q_scale: float = 10.0,
    time_discount: float = 1.0,
    reward_mode: str = "score_scaled",
    device: torch.device | str = "cpu",
    seed: Optional[int] = None,
    on_progress: Optional[Callable[[dict], None]] = None,
    progress_every_turns: int = 10,
    progress_every_s: float = 30.0,
    perf: PerfCounters | None = None,
) -> dict:
    """Run self-play games and collect training samples.

    Writes finished-game samples into ``buffer`` and returns aggregate metrics.
    """
    if temperature_schedule is None:
        temperature_schedule = _default_temperature_schedule

    device_t = torch.device(device)
    engine = BE.BatchedEngine(num_games, num_players, device_t, seed)
    net = net.to(device_t)
    net.eval()

    storage_device = buffer.device

    rec_global: list[torch.Tensor] = []
    rec_source: list[torch.Tensor] = []
    rec_legal: list[torch.Tensor] = []
    rec_policy: list[torch.Tensor] = []
    rec_cp: list[torch.Tensor] = []
    rec_game_idx: list[torch.Tensor] = []
    rec_step: list[torch.Tensor] = []

    game_end_step = torch.full(
        (num_games,), fill_value=max_turns, dtype=torch.int32, device=storage_device
    )
    prev_ended = torch.zeros(num_games, dtype=torch.bool, device=storage_device)

    t_start = time.monotonic()
    last_progress_t = t_start
    turn = 0
    while turn < max_turns and not engine.ended.all():
        alive = ~engine.ended
        if alive.any():
            with maybe_time(perf, "selfplay_encode"):
                global_feat, source_feat = ENC.encode_state(engine)
            with maybe_time(perf, "selfplay_legal_mask"):
                legal_mask = engine.legal_action_mask()
            temp = temperature_schedule(turn)

            with maybe_time(perf, "selfplay_mcts"):
                actions, improved_policy = gumbel_root_act(
                    engine,
                    net,
                    num_sims=num_sims,
                    temperature=temp,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_mix=dirichlet_mix,
                    q_scale=q_scale,
                    precomputed=(global_feat, source_feat, legal_mask),
                    perf=perf,
                )

            with maybe_time(perf, "selfplay_record"):
                alive_idx = alive.nonzero(as_tuple=True)[0]
                if perf is not None:
                    perf.add_count("selfplay_turns", 1)
                    perf.add_count("selfplay_alive_positions", alive_idx.numel())
                g_rec = global_feat[alive].to(storage_device)
                s_rec = source_feat[alive].to(storage_device)
                l_rec = legal_mask[alive].to(storage_device)
                p_rec = improved_policy[alive].to(storage_device)
                cp_rec = engine.current_player[alive].to(storage_device)
                gi_rec = alive_idx.to(storage_device)
                step_rec = torch.full(
                    (alive_idx.numel(),),
                    turn,
                    dtype=torch.int32,
                    device=storage_device,
                )
                if perf is not None:
                    record_bytes = (
                        tensor_nbytes(g_rec)
                        + tensor_nbytes(s_rec)
                        + tensor_nbytes(l_rec)
                        + tensor_nbytes(p_rec)
                        + tensor_nbytes(cp_rec)
                        + tensor_nbytes(gi_rec)
                        + tensor_nbytes(step_rec)
                    )
                    perf.add_count("selfplay_record_mb", record_bytes / (1024**2))
                    if device_t != storage_device:
                        perf.add_count("selfplay_device_transfer_mb", record_bytes / (1024**2))
                rec_global.append(g_rec)
                rec_source.append(s_rec)
                rec_legal.append(l_rec)
                rec_policy.append(p_rec)
                rec_cp.append(cp_rec)
                rec_game_idx.append(gi_rec)
                rec_step.append(step_rec)
            with maybe_time(perf, "selfplay_env_step"):
                engine.step(actions)
        else:
            break

        cur_ended = engine.ended.to(storage_device)
        newly_ended = cur_ended & ~prev_ended
        if newly_ended.any():
            game_end_step[newly_ended] = turn + 1
        prev_ended = cur_ended
        turn += 1

        if on_progress is not None:
            now = time.monotonic()
            if turn % progress_every_turns == 0 or (now - last_progress_t) >= progress_every_s:
                last_progress_t = now
                alive_n = int((~engine.ended).sum().item())
                finished_n = int(engine.ended.sum().item())
                on_progress(
                    {
                        "turn": turn,
                        "alive_games": alive_n,
                        "finished_games": finished_n,
                        "elapsed_s": round(now - t_start, 1),
                    }
                )

    wall_s = time.monotonic() - t_start
    ended_mask = engine.ended.to(storage_device)

    if not rec_global:
        empty_metrics = {
            "steps": turn,
            "samples_added": 0,
            "finished": int(ended_mask.sum().item()),
            "games_total": num_games,
            "wall_s": round(wall_s, 3),
        }
        return empty_metrics

    with maybe_time(perf, "selfplay_concat"):
        all_g = torch.cat(rec_global, dim=0)
        all_s = torch.cat(rec_source, dim=0)
        all_l = torch.cat(rec_legal, dim=0)
        all_p = torch.cat(rec_policy, dim=0)
        all_cp = torch.cat(rec_cp, dim=0)
        all_gi = torch.cat(rec_game_idx, dim=0)
        all_step = torch.cat(rec_step, dim=0)

    with maybe_time(perf, "selfplay_value_targets"):
        final_values = _final_rank_values(engine, num_players, reward_mode).to(storage_device)
        unfinished = ~ended_mask
        if unfinished.any():
            final_values[unfinished, :num_players] = -1.0

        per_sample_values = final_values[all_gi]
        rotated = _rotate_for_cp(per_sample_values, all_cp.to(torch.long), num_players)

        finished_mask = ended_mask[all_gi]
        ttg = (game_end_step[all_gi].to(torch.float32) - all_step.to(torch.float32)).clamp_min(0)
        discount = torch.pow(
            torch.tensor(time_discount, dtype=torch.float32, device=storage_device), ttg
        )
        effective_discount = finished_mask.to(discount.dtype) * discount + (
            1.0 - finished_mask.to(discount.dtype)
        )
        value_targets = rotated * effective_discount.unsqueeze(-1)

    total_positions = int(all_g.shape[0])
    with maybe_time(perf, "selfplay_select_samples"):
        selected = _select_finished_samples(
            all_g, all_s, all_l, all_p, value_targets, all_gi, ended_mask
        )
    samples_added = 0
    if selected is not None:
        from .stability import sanitize_policy_targets, sanitize_value_targets

        with maybe_time(perf, "selfplay_buffer_add"):
            g, s, l, p, v = selected
            buffer.add(
                g,
                s,
                l,
                sanitize_policy_targets(p, l),
                sanitize_value_targets(v),
            )
            samples_added = int(g.shape[0])

    metrics = {
        "steps": turn,
        "samples_added": samples_added,
        "samples_total": total_positions,
        "samples_skipped_unfinished": total_positions - samples_added,
        "finished": int(ended_mask.sum().item()),
        "games_total": num_games,
        "wall_s": round(wall_s, 3),
        "games_per_s": round(int(ended_mask.sum().item()) / wall_s, 2) if wall_s > 0 else 0.0,
    }

    return metrics
