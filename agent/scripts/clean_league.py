"""Remove inactive ML agents from the league and recompute ratings.

Usage:
    python -m agent.scripts.clean_league [--league-root agent/runs/league] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

from agent.train.league import League


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove inactive ML agents from the league and recompute ratings."
    )
    parser.add_argument(
        "--league-root",
        type=str,
        default="agent/runs/league",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    league_root = pathlib.Path(args.league_root)
    manifest_path = league_root / "league.json"

    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    results = manifest.get("results", [])

    active_entries = [e for e in entries if e.get("active", True)]
    inactive_entries = [e for e in entries if not e.get("active", True)]

    if not inactive_entries:
        print("No inactive entries to remove. League is already clean.")
        return

    inactive_ids = {f"ckpt:{e['idx']}" for e in inactive_entries}
    keep_ids = {f"ckpt:{e['idx']}" for e in active_entries}
    keep_ids.update(manifest.get("anchors", {}).keys())
    keep_ids.update(manifest.get("floating_entities", {}).keys())
    for row in results:
        for side in ("a", "b"):
            entity = row.get(side, "")
            if not entity.startswith("ckpt:"):
                keep_ids.add(entity)

    clean_results = [
        row for row in results if row.get("a") in keep_ids and row.get("b") in keep_ids
    ]

    print(f"Inactive entries: {len(inactive_entries)}")
    print(f"Results rows: {len(results)} -> {len(clean_results)}")

    if args.dry_run:
        print("Dry run — no changes written.")
        return

    backup = manifest_path.with_suffix(".json.bak")
    shutil.copy2(manifest_path, backup)
    print(f"Backup saved to {backup}")

    manifest["entries"] = active_entries
    manifest["results"] = clean_results

    for entry in inactive_entries:
        ckpt_path = league_root / pathlib.Path(entry["path"]).name
        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"Deleted {ckpt_path}")

    manifest_path.write_text(json.dumps(manifest, indent=2))

    league = League(league_root)
    ratings = league.recompute_ratings()
    print(f"Recomputed {len(ratings)} entity ratings.")


if __name__ == "__main__":
    main()
