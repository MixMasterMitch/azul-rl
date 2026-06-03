"""Print league ratings table from league.json."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description="Azul league ratings")
    parser.add_argument("--league-dir", type=str, default="agent/runs/league")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.league_dir) / "league.json"
    if not manifest_path.exists():
        print(f"No league manifest at {manifest_path}")
        return

    manifest = json.loads(manifest_path.read_text())
    rating_system = manifest.get("rating_system", "unknown")
    entries = [e for e in manifest.get("entries", []) if e.get("active", True)]
    floating = manifest.get("floating_entities", {})

    print(f"Rating system: {rating_system}")
    print(f"Anchors: {manifest.get('anchors', {})}")

    ranked = sorted(
        entries,
        key=lambda e: float(e.get("rating", 0)),
        reverse=True,
    )[: args.top]

    print(f"\n{'#':<3} {'Idx':<6} {'Tag':<12} {'Rating':>8} {'2p':>8} {'3p':>8} {'4p':>8} {'Games':>7}")
    print("-" * 68)
    for i, e in enumerate(ranked, 1):
        print(
            f"{i:<3} {e.get('idx', '?'):<6} {str(e.get('tag', '')):<12} "
            f"{e.get('rating', 0):>8.0f} "
            f"{e.get('rating_2p', 0):>8.0f} "
            f"{e.get('rating_3p', 0):>8.0f} "
            f"{e.get('rating_4p', 0):>8.0f} "
            f"{e.get('games', 0):>7}"
        )

    if floating:
        print("\nFloating entities:")
        for name, fe in sorted(
            floating.items(), key=lambda x: float(x[1].get("rating", 0)), reverse=True
        ):
            print(
                f"  {name:<20} rating={fe.get('rating', '?'):>6} "
                f"games={fe.get('games', 0)}"
            )

    inactive = sum(1 for e in manifest.get("entries", []) if not e.get("active", True))
    print(
        f"\n{len(entries)} active checkpoints"
        + (f", {inactive} inactive (history only)" if inactive else "")
    )


if __name__ == "__main__":
    main()
