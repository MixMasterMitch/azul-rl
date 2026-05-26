"""Tournament evaluation: play games between agents and collect results."""

from __future__ import annotations

from typing import Callable

import torch

from ..env import actions as A
from ..env import batched_engine as BE
from ..net import encoder as ENC
from ..net import model as M
from .bots import Bot, RandomBot, HeuristicBot


def evaluate_checkpoint(
    net: M.AzulNet,
    num_games: int = 256,
    num_players: int = 2,
    num_sims: int = 16,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Evaluate a checkpoint against reference bots.

    Returns win rates against each opponent type.
    """
    device = torch.device(device)
    net = net.to(device)
    net.eval()

    results = {}

    for opponent_name, opponent_cls in [("random", RandomBot), ("heuristic", HeuristicBot)]:
        wins = 0
        total = 0

        for game_idx in range(num_games):
            # ML agent gets a random seat
            ml_seat = game_idx % num_players
            opponent = opponent_cls(seed=game_idx)

            engine = BE.BatchedEngine(1, num_players, device, seed=game_idx * 1000)

            for turn in range(300):
                if engine.ended[0]:
                    break

                cp = engine.current_player[0].item()
                if cp == ml_seat:
                    # ML agent's turn
                    global_feat, source_feat = ENC.encode_state(engine)
                    legal_mask = engine.legal_action_mask()
                    with torch.no_grad():
                        logits, _ = net(global_feat, source_feat, legal_mask, num_players)
                    # Greedy action selection (no MCTS during eval for speed)
                    action = logits[0].argmax().item()
                else:
                    action = opponent.select_action(engine, 0)

                actions = torch.tensor([action], dtype=torch.long, device=device)
                engine.step(actions)

            if engine.ended[0]:
                winner = engine.get_winners()[0].item()
                if winner == ml_seat:
                    wins += 1
            total += 1

        results[f"vs_{opponent_name}_winrate"] = wins / max(total, 1)

    return results
