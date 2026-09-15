"""Fixed-checkpoint sweep over eval q_scale and num_sims vs opus."""

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
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep eval q_scale and num_sims vs opus")
    p.add_argument("checkpoint", type=str)
    p.add_argument("--num-games", type=int, default=512)
    p.add_argument(
        "--q-scales",
        type=str,
        default="8,12,18,25,32",
        help="Comma-separated q_scale values",
    )
    p.add_argument(
        "--num-sims-list",
        type=str,
        default="16,32,64",
        help="Comma-separated num_sims values",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-turns", type=int, default=300)
    p.add_argument(
        "--report",
        type=str,
        default="",
        help="JSON report path (default: agent/runs/phase4_search_sweep.json)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not pathlib.Path(args.checkpoint).exists():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    q_scales = parse_float_list(args.q_scales)
    num_sims_list = parse_int_list(args.num_sims_list)
    report_path = (
        pathlib.Path(args.report)
        if args.report
        else pathlib.Path("agent/runs/phase4_search_sweep.json")
    )

    net, _ = load_net_from_checkpoint(args.checkpoint, map_location=device)
    net.eval()

    rows: list[dict] = []
    total = len(q_scales) * len(num_sims_list)
    idx = 0
    for q_scale in q_scales:
        for num_sims in num_sims_list:
            idx += 1
            print(
                f"[{idx}/{total}] q_scale={q_scale} num_sims={num_sims} "
                f"({args.num_games} games)…"
            )
            metrics = evaluate_checkpoint(
                net,
                num_games=args.num_games,
                num_players=2,
                num_sims=num_sims,
                device=device,
                q_scale=q_scale,
                max_turns=args.max_turns,
                opponents=OPUS_ONLY_OPPONENTS,
            )
            winrate = float(metrics["vs_opus_winrate"])
            wins = int(round(winrate * args.num_games))
            lo, hi = wilson_ci(wins, args.num_games)
            row = {
                "q_scale": q_scale,
                "num_sims": num_sims,
                "num_games": args.num_games,
                "wins": wins,
                "opus_winrate": winrate,
                "opus_ci_low": lo,
                "opus_ci_high": hi,
            }
            rows.append(row)
            print(f"  opus={winrate:.4f} [{lo:.4f}, {hi:.4f}]")

    best = max(rows, key=lambda r: r["opus_winrate"])
    payload = {
        "checkpoint": args.checkpoint,
        "protocol": {
            "num_games": args.num_games,
            "max_turns": args.max_turns,
            "opponent": "heuristic_opus",
        },
        "best": best,
        "results": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"\nBest: q_scale={best['q_scale']} num_sims={best['num_sims']} "
        f"opus={best['opus_winrate']:.4f}"
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
