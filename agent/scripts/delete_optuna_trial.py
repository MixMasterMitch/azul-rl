"""Delete one trial from an Optuna SQLite study (e.g. stale RUNNING after interrupt)."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

TRIAL_CHILD_TABLES = (
    "trial_params",
    "trial_values",
    "trial_user_attributes",
    "trial_system_attributes",
    "trial_intermediate_values",
    "trial_heartbeats",
)


def delete_trial(db_path: Path, study_name: str, trial_number: int) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"study not found: {study_name}")
    study_id = row[0]

    cur.execute(
        "SELECT trial_id, state FROM trials WHERE study_id = ? AND number = ?",
        (study_id, trial_number),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"trial number {trial_number} not found in {study_name}")
    trial_id, state = row

    for table in TRIAL_CHILD_TABLES:
        cur.execute(f"DELETE FROM {table} WHERE trial_id = ?", (trial_id,))
    cur.execute("DELETE FROM trials WHERE trial_id = ?", (trial_id,))
    conn.commit()
    conn.close()
    print(f"Deleted trial #{trial_number} (trial_id={trial_id}, was {state}) from {study_name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Delete a trial from an Optuna SQLite study")
    p.add_argument("--study-name", required=True)
    p.add_argument("--trial-number", type=int, required=True)
    p.add_argument(
        "--db",
        type=str,
        default="",
        help="Path to optuna db (default: agent/runs/optuna_{study_name}.db)",
    )
    args = p.parse_args(argv)
    db_path = Path(args.db or f"agent/runs/optuna_{args.study_name}.db")
    if not db_path.exists():
        print(f"error: db not found: {db_path}", file=sys.stderr)
        return 2
    delete_trial(db_path, args.study_name, args.trial_number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
