"""High-confidence opus-only eval for one or more checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

from ..eval.tournament import DEFAULT_OPPONENTS, evaluate_checkpoint
from ..train.checkpointing import load_net_from_checkpoint
from ..train.device import resolve_device

OPUS_ONLY_OPPONENTS = tuple(
    opponent for opponent in DEFAULT_OPPONENTS if opponent[0] == "opus"
)


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def eval_checkpoint(
    checkpoint: str,
    *,
    num_games: int,
    num_sims: int,
    q_scale: float,
    device: str,
    max_turns: int,
) -> dict:
    net, _ = load_net_from_checkpoint(checkpoint, map_location=device)
    net.eval()
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
    winrate = float(metrics["vs_opus_winrate"])
    wins = int(round(winrate * num_games))
    lo, hi = wilson_ci(wins, num_games)
    return {
        "checkpoint": checkpoint,
        "num_games": num_games,
        "num_sims": num_sims,
        "q_scale": q_scale,
        "wins": wins,
        "losses": num_games - wins,
        "opus_winrate": winrate,
        "opus_ci_low": lo,
        "opus_ci_high": hi,
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="High-confidence opus eval for checkpoints")
    p.add_argument(
        "checkpoints",
        nargs="+",
        help="Checkpoint paths to evaluate",
    )
    p.add_argument("--num-games", type=int, default=1024)
    p.add_argument("--num-sims", type=int, default=64)
    p.add_argument("--q-scale", type=float, default=28.0)
    p.add_argument("--max-turns", type=int, default=300)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--report",
        type=str,
        default="",
        help="JSON report path (default: agent/runs/phase3_opus_eval.json)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    report_path = (
        pathlib.Path(args.report)
        if args.report
        else pathlib.Path("agent/runs/phase3_opus_eval.json")
    )

    rows: list[dict] = []
    for ckpt in args.checkpoints:
        if not pathlib.Path(ckpt).exists():
            print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
            return 2
        print(
            f"Evaluating {ckpt} ({args.num_games} games, "
            f"sims={args.num_sims}, q_scale={args.q_scale})…"
        )
        row = eval_checkpoint(
            ckpt,
            num_games=args.num_games,
            num_sims=args.num_sims,
            q_scale=args.q_scale,
            device=device,
            max_turns=args.max_turns,
        )
        rows.append(row)
        print(
            f"  opus={row['opus_winrate']:.4f} "
            f"[{row['opus_ci_low']:.4f}, {row['opus_ci_high']:.4f}] "
            f"({row['wins']}/{row['num_games']})"
        )

    best = max(rows, key=lambda r: r["opus_winrate"])
    payload = {
        "protocol": {
            "num_games": args.num_games,
            "num_sims": args.num_sims,
            "q_scale": args.q_scale,
            "max_turns": args.max_turns,
            "opponent": "heuristic_opus",
        },
        "best_checkpoint": best["checkpoint"],
        "results": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nBest: {best['checkpoint']} opus={best['opus_winrate']:.4f}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
