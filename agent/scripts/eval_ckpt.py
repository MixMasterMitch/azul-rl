"""Evaluate a checkpoint against reference bots."""

from __future__ import annotations

import argparse

from ..eval.tournament import evaluate_checkpoint
from ..train.checkpointing import load_net_from_checkpoint
from ..train.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Azul checkpoint")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--num-games", type=int, default=256)
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-sims", type=int, default=32)
    parser.add_argument("--q-scale", type=float, default=10.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    net, _ = load_net_from_checkpoint(args.checkpoint, map_location=device)
    metrics = evaluate_checkpoint(
        net,
        num_games=args.num_games,
        num_players=args.num_players,
        num_sims=args.num_sims,
        device=device,
        q_scale=args.q_scale,
    )
    from ..eval.tournament import combined_winrate

    print(f"combined_winrate: {combined_winrate(metrics):.3f}")
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
