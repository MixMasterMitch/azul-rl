"""Print league ratings table."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description="Azul league ratings")
    parser.add_argument("--league-dir", type=str, default="agent/runs/league")
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.league_dir) / "league.json"
    if not manifest_path.exists():
        print(f"No league manifest at {manifest_path}")
        return

    manifest = json.loads(manifest_path.read_text())
    ratings = manifest.get("ratings", {})
    entries = manifest.get("entries", [])

    print(f"{'Entity':<20} {'Rating':>10} {'Iter':>8}")
    print("-" * 42)
    for name, rating in sorted(ratings.items(), key=lambda x: -x[1]):
        print(f"{name:<20} {rating:>10.1f}")

    if entries:
        print(f"\n{len(entries)} checkpoints in league")


if __name__ == "__main__":
    main()
