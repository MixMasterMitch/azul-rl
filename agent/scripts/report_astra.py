"""Build a compact, auditable index of completed Astra experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.eval.astra_tournament import atomic_json, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("agent/runs/astra"))
    parser.add_argument("--output", type=Path, default=Path("docs/astra/experiments.json"))
    args = parser.parse_args()
    experiments = []
    for path in sorted(args.root.glob("*/report.json")):
        report = json.loads(path.read_text())
        spec = report["spec"]
        manifest = json.loads((path.parent / "manifest.json").read_text())
        experiments.append({
            "directory": str(path.parent), "name": spec["name"], "split": spec["split"],
            "seed": spec["seed"], "players": spec["players"], "blocks": spec["blocks"],
            "fields": spec["fields"], "checkpoint_policy": spec.get("checkpoint_policy", "greedy"),
            "control": spec.get("control"), "config": spec.get("config"),
            "configs_per_pc": spec.get("configs_per_pc"), "complete": report["complete"],
            "games": report["total_games"], "started_at": manifest["started_at"],
            "wall_s_last_invocation": report["wall_s_this_run"],
            "raw_records_sha256": digest(path.parent / "games.jsonl"),
            "native_sha256": {k: v for k, v in spec["sources"].items() if k.endswith(".so")},
            "decision_sources": {k: v for k, v in spec["sources"].items()
                                 if k.startswith("native/astra/src/") or k == "agent/eval/heuristic_astra.py"},
            "checkpoints": manifest["checkpoints"],
            "summary": {k: {a: b for a, b in v.items() if a != "block_scores"}
                        for k, v in report["summary"].items()},
        })
    experiments.sort(key=lambda e: e["started_at"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {"experiments": experiments,
                             "games": sum(e["games"] for e in experiments),
                             "note": "Only completed report files are indexed. Raw records, manifests, frozen runtimes and seed-level statistics remain in each experiment directory. Different protocols are not pooled into a strength estimate."})
    print(f"Indexed {len(experiments)} experiments / {sum(e['games'] for e in experiments)} games")


if __name__ == "__main__":
    main()
