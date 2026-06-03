"""Self-play vs heuristic / heuristic-opus bots (main net positions only)."""

from __future__ import annotations

import random
import time
from typing import Literal, Optional

import torch

from ..env import batched_engine as BE
from ..eval import bots as B
from ..eval.heuristic_opus import HeuristicOpusBot
from .batched_bot_policy import batched_heuristic_actions
from ..net import encoder as ENC
from ..net import model as M
from ..search.gumbel_mcts import gumbel_root_act
from .instrumentation import PerfCounters, maybe_time, tensor_nbytes
from .replay_buffer import ReplayBuffer
from .selfplay import (
    _default_temperature_schedule,
    _final_rank_values,
    _rotate_for_cp,
    _select_finished_samples,
)

BotKind = Literal["heuristic", "heuristic_opus"]


def run_bot_selfplay(
    net: M.AzulNet,
    buffer: ReplayBuffer,
    num_players: int = 2,
    num_games: int = 512,
    device: str = "cpu",
    max_turns: int = 200,
    num_sims: int = 32,
    seed: int = 0,
    time_discount: float = 1.0,
    reward_mode: str = "score_scaled",
    dirichlet_alpha: float = 0.3,
    dirichlet_mix: float = 0.25,
    q_scale: float = 10.0,
    opus_prob: float = 0.5,
    temperature_schedule: Optional[callable] = None,
    bot_policy: str = "batched",
    perf: PerfCounters | None = None,
) -> dict:
    """Play vs rule bots; 50/50 heuristic vs heuristic_opus per game by default."""
    if temperature_schedule is None:
        temperature_schedule = _default_temperature_schedule

    device_t = torch.device(device)
    rng = random.Random(seed)
    net = net.to(device_t)
    net.eval()

    use_batched_bot = bot_policy == "batched"
    heuristic_bot = B.HeuristicBot(seed=seed + 1) if not use_batched_bot else None
    opus_bot = HeuristicOpusBot(seed=seed + 2)

    engine = BE.BatchedEngine(num_games, num_players, device_t, seed)
    storage_device = buffer.device

    main_seat = torch.zeros((num_games,), dtype=torch.int8, device=device_t)
    bot_kind: list[BotKind] = []
    n_heuristic = 0
    n_opus = 0
    for b in range(num_games):
        main_seat[b] = rng.randint(0, num_players - 1)
        if rng.random() < opus_prob:
            bot_kind.append("heuristic_opus")
            n_opus += 1
        else:
            bot_kind.append("heuristic")
            n_heuristic += 1

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
    turn = 0
    while turn < max_turns and not engine.ended.all():
        alive = ~engine.ended
        if not alive.any():
            break

        cp = engine.current_player
        is_main_turn = (cp == main_seat) & alive

        actions = torch.zeros((num_games,), dtype=torch.long, device=device_t)

        if is_main_turn.any():
            main_idx = is_main_turn.nonzero(as_tuple=True)[0]
            with maybe_time(perf, "bot_selfplay_main_select"):
                sub = engine.index_select(main_idx)
            with maybe_time(perf, "bot_selfplay_encode"):
                g, s = ENC.encode_state(sub)
            with maybe_time(perf, "bot_selfplay_legal_mask"):
                legal = sub.legal_action_mask()
            temp = temperature_schedule(turn)
            with torch.no_grad():
                with maybe_time(perf, "bot_selfplay_main_mcts"):
                    actions_sub, improved = gumbel_root_act(
                        sub,
                        net,
                        num_sims=num_sims,
                        temperature=temp,
                        dirichlet_alpha=dirichlet_alpha,
                        dirichlet_mix=dirichlet_mix,
                        q_scale=q_scale,
                        perf=perf,
                    )
            with maybe_time(perf, "bot_selfplay_record"):
                if perf is not None:
                    perf.add_count("bot_selfplay_main_turns", 1)
                    perf.add_count("bot_selfplay_main_positions", main_idx.numel())
                actions.index_copy_(0, main_idx, actions_sub)

                g_rec = g.to(storage_device)
                s_rec = s.to(storage_device)
                l_rec = legal.to(storage_device)
                p_rec = improved.to(storage_device)
                cp_rec = sub.current_player.to(storage_device)
                gi_rec = main_idx.to(storage_device)
                step_rec = torch.full(
                    (main_idx.numel(),), turn, dtype=torch.int32, device=storage_device
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
                    perf.add_count("bot_selfplay_record_mb", record_bytes / (1024**2))
                    if device_t != storage_device:
                        perf.add_count("bot_selfplay_device_transfer_mb", record_bytes / (1024**2))
                rec_global.append(g_rec)
                rec_source.append(s_rec)
                rec_legal.append(l_rec)
                rec_policy.append(p_rec)
                rec_cp.append(cp_rec)
                rec_game_idx.append(gi_rec)
                rec_step.append(step_rec)

        bot_turn = alive & ~is_main_turn
        if bot_turn.any():
            with maybe_time(perf, "bot_selfplay_bot_policy"):
                if use_batched_bot:
                    if perf is not None:
                        perf.add_count("bot_selfplay_bot_positions", int(bot_turn.sum().item()))
                    bot_idx = bot_turn.nonzero(as_tuple=True)[0]
                    opus_idx = torch.tensor(
                        [
                            b
                            for b in bot_idx.tolist()
                            if bot_kind[int(b)] == "heuristic_opus"
                        ],
                        dtype=torch.long,
                        device=device_t,
                    )
                    opus_turn = torch.zeros_like(bot_turn)
                    if opus_idx.numel() > 0:
                        opus_turn[opus_idx] = True
                    heuristic_turn = bot_turn & ~opus_turn
                    if heuristic_turn.any():
                        bot_actions = batched_heuristic_actions(engine)
                        actions = torch.where(heuristic_turn, bot_actions, actions)
                    for bi in opus_idx.tolist():
                        actions[int(bi)] = opus_bot.select_action(engine, int(bi))
                else:
                    bot_idx = bot_turn.nonzero(as_tuple=True)[0]
                    if perf is not None:
                        perf.add_count("bot_selfplay_bot_positions", bot_idx.numel())
                    for bi in bot_idx.tolist():
                        b = int(bi)
                        bot = opus_bot if bot_kind[b] == "heuristic_opus" else heuristic_bot
                        actions[b] = bot.select_action(engine, b)

        with maybe_time(perf, "bot_selfplay_env_step"):
            engine.step(actions)

        cur_ended = engine.ended.to(storage_device)
        newly_ended = cur_ended & ~prev_ended
        if newly_ended.any():
            game_end_step[newly_ended] = turn + 1
        prev_ended = cur_ended
        turn += 1

    wall_s = time.monotonic() - t_start
    ended_mask = engine.ended.to(storage_device)

    if not rec_global:
        return {
            "steps": turn,
            "samples_added": 0,
            "finished": int(ended_mask.sum().item()),
            "games_total": num_games,
            "wall_s": round(wall_s, 3),
            "bot_heuristic_games": n_heuristic,
            "bot_opus_games": n_opus,
        }

    with maybe_time(perf, "bot_selfplay_concat"):
        all_g = torch.cat(rec_global, dim=0)
        all_s = torch.cat(rec_source, dim=0)
        all_l = torch.cat(rec_legal, dim=0)
        all_p = torch.cat(rec_policy, dim=0)
        all_cp = torch.cat(rec_cp, dim=0)
        all_gi = torch.cat(rec_game_idx, dim=0)
        all_step = torch.cat(rec_step, dim=0)

    with maybe_time(perf, "bot_selfplay_value_targets"):
        final_values = _final_rank_values(engine, num_players, reward_mode).to(storage_device)
        unfinished = ~ended_mask
        if unfinished.any():
            final_values[unfinished, :num_players] = -1.0

        per_sample = final_values[all_gi]
        rotated = _rotate_for_cp(per_sample, all_cp.to(torch.long), num_players)
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
    with maybe_time(perf, "bot_selfplay_select_samples"):
        selected = _select_finished_samples(
            all_g, all_s, all_l, all_p, value_targets, all_gi, ended_mask
        )
    samples_added = 0
    if selected is not None:
        from .stability import sanitize_policy_targets, sanitize_value_targets

        with maybe_time(perf, "bot_selfplay_buffer_add"):
            g, s, l, p, v = selected
            buffer.add(
                g,
                s,
                l,
                sanitize_policy_targets(p, l),
                sanitize_value_targets(v),
            )
            samples_added = int(g.shape[0])

    return {
        "steps": turn,
        "samples_added": samples_added,
        "samples_total": total_positions,
        "samples_skipped_unfinished": total_positions - samples_added,
        "finished": int(ended_mask.sum().item()),
        "games_total": num_games,
        "wall_s": round(wall_s, 3),
        "bot_heuristic_games": n_heuristic,
        "bot_opus_games": n_opus,
    }
