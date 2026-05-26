"""Self-play data generation using Gumbel MCTS."""

from __future__ import annotations

import time
from typing import Optional

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import encoder as ENC
from ..net import model as M
from ..search.gumbel_mcts import gumbel_root_act
from .replay_buffer import ReplayBuffer

REWARD_MODES = ("binary", "score_scaled")


def _rotate_for_cp(values: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
    """Rotate per-seat values so index 0 is the acting player at record time."""
    b, p = values.shape
    idx = (torch.arange(p, device=values.device).unsqueeze(0) + cp.unsqueeze(-1)) % p
    return values.gather(1, idx.to(torch.long))


def _final_rank_values_binary(
    engine: BE.BatchedEngine, num_players: int
) -> torch.Tensor:
    """Winner +1, losers -1 per absolute seat."""
    values = torch.zeros(
        (engine.batch_size, BE.MAX_PLAYERS), dtype=torch.float32, device=engine.device
    )
    winners = engine.get_winners()
    for b in range(engine.batch_size):
        if not engine.ended[b]:
            values[b, :num_players] = -1.0
            continue
        w = int(winners[b].item())
        for p in range(num_players):
            values[b, p] = 1.0 if p == w else -1.0
    return values


def _final_rank_values_score_scaled(
    engine: BE.BatchedEngine, num_players: int
) -> torch.Tensor:
    """Winner +1; losers -1/(n-1) + (score/winner_score)^2."""
    values = torch.zeros(
        (engine.batch_size, BE.MAX_PLAYERS), dtype=torch.float32, device=engine.device
    )
    winners = engine.get_winners()
    scores = engine.scores.float()
    for b in range(engine.batch_size):
        if not engine.ended[b]:
            values[b, :num_players] = -1.0
            continue
        w = int(winners[b].item())
        winner_score = max(float(scores[b, w].item()), 1.0)
        n = float(num_players)
        loss_base = -1.0 / max(n - 1.0, 1.0)
        for p in range(num_players):
            if p == w:
                values[b, p] = 1.0
            else:
                ratio = (float(scores[b, p].item()) / winner_score) ** 2
                values[b, p] = loss_base + ratio
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
    buffer: Optional[ReplayBuffer] = None,
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
) -> dict | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run self-play games and collect training samples.

    When ``buffer`` is provided, writes samples and returns metrics dict.
    Otherwise returns legacy tensor tuple for backward compatibility.
    """
    if temperature_schedule is None:
        temperature_schedule = _default_temperature_schedule

    device_t = torch.device(device)
    engine = BE.BatchedEngine(num_games, num_players, device_t, seed)
    net = net.to(device_t)
    net.eval()

    storage_device = buffer.device if buffer is not None else torch.device("cpu")

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
        if alive.any():
            global_feat, source_feat = ENC.encode_state(engine)
            legal_mask = engine.legal_action_mask()
            temp = temperature_schedule(turn)

            actions, improved_policy = gumbel_root_act(
                engine,
                net,
                num_sims=num_sims,
                temperature=temp,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_mix=dirichlet_mix,
                q_scale=q_scale,
                precomputed=(global_feat, source_feat, legal_mask),
            )

            alive_idx = alive.nonzero(as_tuple=True)[0]
            rec_global.append(global_feat[alive].to(storage_device))
            rec_source.append(source_feat[alive].to(storage_device))
            rec_legal.append(legal_mask[alive].to(storage_device))
            rec_policy.append(improved_policy[alive].to(storage_device))
            rec_cp.append(engine.current_player[alive].to(storage_device))
            rec_game_idx.append(alive_idx.to(storage_device))
            rec_step.append(
                torch.full(
                    (alive_idx.numel(),),
                    turn,
                    dtype=torch.int32,
                    device=storage_device,
                )
            )
            engine.step(actions)
        else:
            break

        cur_ended = engine.ended.to(storage_device)
        newly_ended = cur_ended & ~prev_ended
        if newly_ended.any():
            game_end_step[newly_ended] = turn + 1
        prev_ended = cur_ended
        turn += 1

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
        if buffer is None:
            empty = torch.zeros((0, ENC.D_GLOBAL))
            return (
                empty,
                torch.zeros((0, ENC.NUM_SOURCES, ENC.D_SOURCE)),
                torch.zeros((0, A.NUM_ACTIONS), dtype=torch.bool),
                torch.zeros((0, A.NUM_ACTIONS)),
                torch.zeros((0, BE.MAX_PLAYERS)),
            )
        return empty_metrics

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

    per_sample_values = final_values[all_gi]
    rotated = _rotate_for_cp(per_sample_values, all_cp.to(torch.long))

    finished_mask = ended_mask[all_gi]
    ttg = (game_end_step[all_gi].to(torch.float32) - all_step.to(torch.float32)).clamp_min(0)
    discount = torch.pow(
        torch.tensor(time_discount, dtype=torch.float32, device=storage_device), ttg
    )
    effective_discount = finished_mask.to(discount.dtype) * discount + (
        1.0 - finished_mask.to(discount.dtype)
    )
    value_targets = rotated * effective_discount.unsqueeze(-1)

    if buffer is not None:
        buffer.add(all_g, all_s, all_l, all_p, value_targets)

    metrics = {
        "steps": turn,
        "samples_added": int(all_g.shape[0]),
        "finished": int(ended_mask.sum().item()),
        "games_total": num_games,
        "wall_s": round(wall_s, 3),
        "games_per_s": round(int(ended_mask.sum().item()) / wall_s, 2) if wall_s > 0 else 0.0,
    }

    if buffer is not None:
        return metrics

    return all_g, all_s, all_l, all_p, value_targets
