"""Re-run tournament eval on saved tune checkpoints (MCTS + opus).

Checkpoints are written to ``{output_dir}/tune_eval_ckpts/trial_NNN.pt`` when
``tune.py`` runs with ``--keep-eval-checkpoints`` (default). Trials completed
before that change have no saved weights and cannot be re-evaluated.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import optuna

from ..eval.tournament import combined_winrate, evaluate_checkpoint
from ..train.checkpointing import load_net_from_checkpoint
from ..train.device import resolve_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-evaluate saved tune trial checkpoints")
    p.add_argument("--study-name", type=str, default="azul-tune")
    p.add_argument("--output-dir", type=str, default="agent/runs")
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-games", type=int, default=256)
    p.add_argument("--num-sims", type=int, default=32)
    p.add_argument("--num-players", type=int, default=2)
    p.add_argument("--trials", type=str, default="", help="Comma-separated trial ids (default: all with ckpt)")
    p.add_argument("--report", type=str, default="", help="Optional JSON output path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    output_dir = pathlib.Path(args.output_dir)
    ckpt_dir = output_dir / "tune_eval_ckpts"

    storage = args.storage or f"sqlite:///{output_dir}/optuna_{args.study_name}.db"
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    if args.trials:
        trial_ids = [int(x.strip()) for x in args.trials.split(",") if x.strip()]
    else:
        trial_ids = sorted(
            int(p.stem.split("_")[1])
            for p in ckpt_dir.glob("trial_*.pt")
        )

    rows: list[dict] = []
    for tid in trial_ids:
        ckpt = ckpt_dir / f"trial_{tid:03d}.pt"
        row: dict = {"trial": tid, "checkpoint": str(ckpt)}
        if not ckpt.exists():
            row["status"] = "missing_checkpoint"
            rows.append(row)
            continue

        trial = study.trials[tid]
        q_scale = float(trial.params.get("q_scale", 10.0))
        net, _ = load_net_from_checkpoint(ckpt, map_location=device)
        net.eval()
        metrics = evaluate_checkpoint(
            net,
            num_games=args.num_games,
            num_players=args.num_players,
            num_sims=args.num_sims,
            device=device,
            q_scale=q_scale,
        )
        combined = combined_winrate(metrics)
        row["status"] = "ok"
        row["combined_winrate"] = combined
        row["optuna_value"] = trial.value
        row.update(metrics)
        row["params"] = trial.params
        rows.append(row)
        print(
            f"trial {tid}: combined={combined:.3f} "
            f"(random={metrics.get('vs_random_winrate', 0):.3f} "
            f"heuristic={metrics.get('vs_heuristic_winrate', 0):.3f} "
            f"opus={metrics.get('vs_opus_winrate', 0):.3f})"
        )

    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    missing = sum(1 for r in rows if r.get("status") == "missing_checkpoint")
    if missing:
        print(f"\n{missing} trial(s) had no saved checkpoint under {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
