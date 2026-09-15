"""Print league ratings table from league.json."""

from __future__ import annotations

import argparse
import json
import pathlib

from agent.train import ranking as R
from agent.train import rating_display as D


def display_entry(entry: dict, reference_anchors: dict, weights: dict[int, float]) -> dict:
    """Read old stored calibration without rewriting the league or fitting again."""
    converted = dict(entry)
    total = weight_sum = 0.0
    for pc in R.PLAYER_COUNTS:
        key = f'rating_{pc}p'
        if entry.get(key) is None:
            continue
        value = D.from_legacy_calibrated(entry[key], pc, reference_anchors)
        converted[key] = value
        weight = weights.get(pc, 0) or 1.0
        total += value * weight
        weight_sum += weight
    if weight_sum:
        converted['rating'] = total / weight_sum
    else:
        converted['rating'] = None  # No per-format evidence available for conversion.
    return converted


def format_rating(value: float | None) -> str:
    return '—' if value is None else f'{value:.0f}'


def main() -> None:
    parser = argparse.ArgumentParser(description="Azul league ratings")
    parser.add_argument("--league-dir", type=str, default="agent/runs/league")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument('--legacy-ratings', action='store_true', help='Show the original stored league scale')
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.league_dir) / "league.json"
    if not manifest_path.exists():
        print(f"No league manifest at {manifest_path}")
        return

    manifest = json.loads(manifest_path.read_text())
    rating_system = manifest.get("rating_system", "unknown")
    entries = [e for e in manifest.get("entries", []) if e.get("active", True)]
    floating = manifest.get("floating_entities", {})
    if not args.legacy_ratings:
        references = R.reference_anchors_from_manifest(manifest)
        weights = R._count_games_per_entity_per_pc(manifest.get('results', []))
        entries = [display_entry(e, references, weights.get(f"ckpt:{e['idx']}", {})) for e in entries]
        floating = {name: display_entry(e, references, weights.get(name, {})) for name, e in floating.items()}

    print(f"Rating system: {rating_system}")
    print(f"Display scale: {'legacy' if args.legacy_ratings else D.VERSION}")
    print(f"Statistical anchors: {manifest.get('anchors', {})}")

    ranked = sorted(
        entries,
        key=lambda e: float(e.get("rating") or 0),
        reverse=True,
    )[: args.top]

    print(f"\n{'#':<3} {'Idx':<6} {'Tag':<12} {'Rating':>8} {'2p':>8} {'3p':>8} {'4p':>8} {'Games':>7}")
    print("-" * 68)
    for i, e in enumerate(ranked, 1):
        print(
            f"{i:<3} {e.get('idx', '?'):<6} {str(e.get('tag', '')):<12} "
            f"{format_rating(e.get('rating')):>8} "
            f"{format_rating(e.get('rating_2p')):>8} "
            f"{format_rating(e.get('rating_3p')):>8} "
            f"{format_rating(e.get('rating_4p')):>8} "
            f"{e.get('games', 0):>7}"
        )

    if floating:
        print("\nFloating entities:")
        for name, fe in sorted(
            floating.items(), key=lambda x: float(x[1].get("rating") or 0), reverse=True
        ):
            print(
                f"  {name:<20} rating={format_rating(fe.get('rating')):>6} "
                f"games={fe.get('games', 0)}"
            )

    inactive = sum(1 for e in manifest.get("entries", []) if not e.get("active", True))
    print(
        f"\n{len(entries)} active checkpoints"
        + (f", {inactive} inactive (history only)" if inactive else "")
    )


if __name__ == "__main__":
    main()
