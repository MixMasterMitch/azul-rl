"""Summarize iter_profile rows from a training run metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_profile_rows(metrics_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not metrics_path.exists():
        return rows
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "selfplay_kind" in row:
                rows.append(row)
    return rows


def _fmt(row: dict, key: str, default: str = "-") -> str:
    val = row.get(key)
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize training profile metrics")
    parser.add_argument("run_dir", type=str, help="Run directory containing metrics.jsonl")
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        help="Optional baseline run directory for delta comparison",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = _load_profile_rows(run_dir / "metrics.jsonl")
    if not rows:
        print(f"No profile rows in {run_dir / 'metrics.jsonl'}")
        return

    baseline_rows = _load_profile_rows(Path(args.baseline) / "metrics.jsonl") if args.baseline else []

    cols = [
        "iter",
        "selfplay_kind",
        "profile_wall_s",
        "profile_phase_selfplay_total_s",
        "profile_phase_learner_total_s",
        "profile_selfplay_env_step_s",
        "profile_selfplay_mcts_s",
        "profile_league_selfplay_env_step_s",
        "profile_bot_selfplay_bot_policy_s",
        "profile_samples_added_per_s",
        "profile_games_finished_per_s",
        "profile_process_cpu_pct_one_core",
        "profile_process_cpu_pct_all_cores",
        "profile_gpu_util_pct",
        "profile_cuda_peak_allocated_mb",
    ]

    print(f"Profile summary: {run_dir}")
    print("\t".join(cols))
    for row in rows:
        print("\t".join(_fmt(row, c) for c in cols))

    if baseline_rows:
        print(f"\nDelta vs baseline {args.baseline} (by selfplay_kind):")
        base_by_kind = {r["selfplay_kind"]: r for r in baseline_rows}
        for row in rows:
            kind = row["selfplay_kind"]
            base = base_by_kind.get(kind)
            if base is None:
                continue
            wall = float(row.get("profile_wall_s", 0)) - float(base.get("profile_wall_s", 0))
            sp = float(row.get("profile_phase_selfplay_total_s", 0)) - float(
                base.get("profile_phase_selfplay_total_s", 0)
            )
            print(f"  iter {row.get('iter')} {kind}: wall_s {wall:+.2f}, selfplay_s {sp:+.2f}")


if __name__ == "__main__":
    main()
