"""Tournament evaluation: play games between agents and collect results."""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch

from ..env import engine as BE
from ..net import encoder as ENC
from ..net import model as M
from ..search import gumbel_mcts as G
from .bots import Bot, HeuristicBot, RandomBot
from .heuristic_opus import HeuristicOpusBot

BotFactory = Callable[[int], Bot]

DEFAULT_OPPONENTS: tuple[tuple[str, BotFactory], ...] = (
    ("random", lambda seed: RandomBot(seed=seed)),
    ("heuristic", lambda seed: HeuristicBot(seed=seed)),
    ("opus", lambda seed: HeuristicOpusBot(seed=seed)),
)


@dataclasses.dataclass
class EvalConfig:
    num_games: int = 256
    num_players: int = 2
    num_sims: int = 32
    max_turns: int = 300
    temperature: float = 0.25
    q_scale: float = 10.0
    device: str = "cpu"


def _ml_select_action(
    engine: BE.BatchedEngine,
    net: M.AzulNet,
    num_players: int,
    cfg: EvalConfig,
) -> int:
    """Gumbel-root MCTS action for batch index 0."""
    global_feat, source_feat = ENC.encode_state(engine)
    legal_mask = engine.legal_action_mask()
    actions, _ = G.gumbel_root_act(
        engine,
        net,
        num_sims=cfg.num_sims,
        temperature=cfg.temperature,
        dirichlet_alpha=0.0,
        dirichlet_mix=0.0,
        q_scale=cfg.q_scale,
        precomputed=(global_feat, source_feat, legal_mask),
    )
    return int(actions[0].item())


def _play_vs_opponent(
    net: M.AzulNet,
    opponent_factory: BotFactory,
    cfg: EvalConfig,
) -> float:
    """Win rate for the net (seat alternates) vs one opponent type."""
    device = torch.device(cfg.device)
    wins = 0
    for game_idx in range(cfg.num_games):
        ml_seat = game_idx % cfg.num_players
        opponent = opponent_factory(game_idx)
        engine = BE.BatchedEngine(1, cfg.num_players, device, seed=game_idx * 1000)

        for _ in range(cfg.max_turns):
            if engine.ended[0]:
                break
            cp = int(engine.current_player[0].item())
            if cp == ml_seat:
                action = _ml_select_action(engine, net, cfg.num_players, cfg)
            else:
                action = opponent.select_action(engine, 0)
            engine.step(torch.tensor([action], dtype=torch.long, device=device))

        if engine.ended[0]:
            winner = int(engine.get_winners()[0].item())
            if winner >= 0 and winner == ml_seat:
                wins += 1

    return wins / max(cfg.num_games, 1)


def evaluate_checkpoint(
    net: M.AzulNet,
    num_games: int = 256,
    num_players: int = 2,
    num_sims: int = 32,
    device: torch.device | str = "cpu",
    q_scale: float = 10.0,
    max_turns: int = 300,
    opponents: tuple[tuple[str, BotFactory], ...] = DEFAULT_OPPONENTS,
) -> dict[str, float]:
    """Evaluate a checkpoint vs reference bots using Gumbel MCTS at the root.

    Returns per-opponent win rates keyed as ``vs_{name}_winrate``.
    """
    device = torch.device(device)
    net = net.to(device)
    net.eval()

    cfg = EvalConfig(
        num_games=num_games,
        num_players=num_players,
        num_sims=num_sims,
        max_turns=max_turns,
        q_scale=q_scale,
        device=str(device),
    )

    results: dict[str, float] = {}
    for opponent_name, opponent_factory in opponents:
        results[f"vs_{opponent_name}_winrate"] = _play_vs_opponent(
            net, opponent_factory, cfg
        )
    return results


def combined_winrate(metrics: dict[str, float]) -> float:
    """Unweighted mean of all ``vs_*_winrate`` entries in *metrics*."""
    keys = [k for k in metrics if k.startswith("vs_") and k.endswith("_winrate")]
    if not keys:
        return 0.0
    return sum(metrics[k] for k in keys) / len(keys)


def play_vs_net_opponent(
    net: M.AzulNet,
    opponent: M.AzulNet,
    cfg: EvalConfig,
) -> float:
    """Win rate for *net* vs another net (seats alternate each game)."""
    device = torch.device(cfg.device)
    wins = 0
    for game_idx in range(cfg.num_games):
        ml_seat = game_idx % cfg.num_players
        engine = BE.BatchedEngine(1, cfg.num_players, device, seed=game_idx * 1000 + 7)

        for _ in range(cfg.max_turns):
            if engine.ended[0]:
                break
            cp = int(engine.current_player[0].item())
            actor = net if cp == ml_seat else opponent
            action = _ml_select_action(engine, actor, cfg.num_players, cfg)
            engine.step(torch.tensor([action], dtype=torch.long, device=device))

        if engine.ended[0]:
            winner = int(engine.get_winners()[0].item())
            if winner >= 0 and winner == ml_seat:
                wins += 1

    return wins / max(cfg.num_games, 1)
