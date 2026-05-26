"""Self-play with league opponents in some seats."""

from __future__ import annotations

import random
import time
from typing import Optional

import torch

from ..env import batched_engine as BE
from ..net import encoder as ENC
from ..net import model as M
from ..search.gumbel_mcts import gumbel_root_act
from .league import League
from .replay_buffer import ReplayBuffer
from .selfplay import (
    REWARD_MODES,
    _final_rank_values,
    _rotate_for_cp,
    _default_temperature_schedule,
)


def run_league_selfplay(
    net: M.AzulNet,
    buffer: ReplayBuffer,
    league: League,
    num_players: int = 2,
    num_games: int = 512,
    device: str = "cpu",
    max_turns: int = 200,
    num_sims: int = 8,
    seed: int = 0,
    league_prob: float = 0.5,
    time_discount: float = 1.0,
    opponent_sims: int = 4,
    reward_mode: str = "score_scaled",
    dirichlet_alpha: float = 0.3,
    dirichlet_mix: float = 0.25,
    q_scale: float = 10.0,
    temperature_schedule: Optional[callable] = None,
) -> dict:
    """Play games vs league checkpoints; buffer only main-agent positions."""
    if temperature_schedule is None:
        temperature_schedule = _default_temperature_schedule

    device_t = torch.device(device)
    rng = random.Random(seed)
    opp_path = league.sample_opponent_path(rng)
    if opp_path is None:
        from .selfplay import run_selfplay

        return run_selfplay(
            net,
            buffer=buffer,
            num_games=num_games,
            num_players=num_players,
            num_sims=num_sims,
            max_turns=max_turns,
            device=device,
            seed=seed,
            time_discount=time_discount,
            reward_mode=reward_mode,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_mix=dirichlet_mix,
            q_scale=q_scale,
        )

    opp_net = league.load_net(opp_path, device=str(device_t))
    net = net.to(device_t)
    net.eval()

    engine = BE.BatchedEngine(num_games, num_players, device_t, seed)
    storage_device = buffer.device

    main_seat = torch.zeros((num_games,), dtype=torch.int8, device=device_t)
    for b in range(num_games):
        if rng.random() < league_prob:
            main_seat[b] = rng.randint(0, num_players - 1)

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

        if is_main_turn.any():
            main_idx = is_main_turn.nonzero(as_tuple=True)[0]
            sub = engine.index_select(main_idx)
            g, s = ENC.encode_state(sub)
            legal = sub.legal_action_mask()
            temp = temperature_schedule(turn)
            with torch.no_grad():
                actions_sub, improved = gumbel_root_act(
                    sub,
                    net,
                    num_sims=num_sims,
                    temperature=temp,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_mix=dirichlet_mix,
                    q_scale=q_scale,
                )

            actions = torch.zeros((num_games,), dtype=torch.long, device=device_t)
            actions.index_copy_(0, main_idx, actions_sub)

            rec_global.append(g.to(storage_device))
            rec_source.append(s.to(storage_device))
            rec_legal.append(legal.to(storage_device))
            rec_policy.append(improved.to(storage_device))
            rec_cp.append(sub.current_player.to(storage_device))
            rec_game_idx.append(main_idx.to(storage_device))
            rec_step.append(
                torch.full((main_idx.numel(),), turn, dtype=torch.int32, device=storage_device)
            )
        else:
            actions = torch.zeros((num_games,), dtype=torch.long, device=device_t)

        opp_turn = alive & ~is_main_turn
        if opp_turn.any():
            opp_idx = opp_turn.nonzero(as_tuple=True)[0]
            sub = engine.index_select(opp_idx)
            with torch.no_grad():
                opp_actions, _ = gumbel_root_act(sub, opp_net, num_sims=opponent_sims)
            actions.index_copy_(0, opp_idx, opp_actions)

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
            "league_opponent": opp_path,
        }

    all_g = torch.cat(rec_global, dim=0)
    all_s = torch.cat(rec_source, dim=0)
    all_l = torch.cat(rec_legal, dim=0)
    all_p = torch.cat(rec_policy, dim=0)
    all_cp = torch.cat(rec_cp, dim=0)
    all_gi = torch.cat(rec_game_idx, dim=0)
    all_step = torch.cat(rec_step, dim=0)

    final_values = _final_rank_values(engine, num_players, reward_mode).to(storage_device)
    unfinished = ~ended_mask
    if unfinished.any():
        final_values[unfinished, :num_players] = -1.0

    per_sample = final_values[all_gi]
    rotated = _rotate_for_cp(per_sample, all_cp.to(torch.long))
    finished_mask = ended_mask[all_gi]
    ttg = (game_end_step[all_gi].to(torch.float32) - all_step.to(torch.float32)).clamp_min(0)
    discount = torch.pow(
        torch.tensor(time_discount, dtype=torch.float32, device=storage_device), ttg
    )
    effective_discount = finished_mask.to(discount.dtype) * discount + (
        1.0 - finished_mask.to(discount.dtype)
    )
    value_targets = rotated * effective_discount.unsqueeze(-1)

    buffer.add(all_g, all_s, all_l, all_p, value_targets)

    return {
        "steps": turn,
        "samples_added": int(all_g.shape[0]),
        "finished": int(ended_mask.sum().item()),
        "games_total": num_games,
        "wall_s": round(wall_s, 3),
        "league_opponent": opp_path,
    }
