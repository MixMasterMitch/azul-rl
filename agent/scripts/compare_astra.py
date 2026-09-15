"""Compare two matched promotion experiments, resampling complete seed blocks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agent.eval.astra_tournament import STAT_EPS, atomic_json, interval, schedule_identity


def compare(candidate: dict[str, Any], incumbent: dict[str, Any]) -> dict[str, Any]:
    a, b = candidate["spec"], incumbent["spec"]
    if candidate.get("schedule_sha256") != incumbent.get("schedule_sha256"):
        raise ValueError("Actual engine/bot seeds or seat schedules differ")
    for key in ("players", "blocks", "fields", "seed", "split", "max_turns", "control"):
        if a[key] != b[key]:
            raise ValueError(f"Experiments differ in {key}")
    engine_files = {k for k in a["sources"] | b["sources"] if k.startswith("agent/env/")
                    or k in ("agent/eval/bots.py", "agent/eval/heuristic_opus.py")}
    if any(a["sources"].get(k) != b["sources"].get(k) for k in engine_files):
        raise ValueError("Engine/reference opponent sources differ")
    if "checkpoint" in a["fields"]:
        if a.get("checkpoint_policy") != b.get("checkpoint_policy"):
            raise ValueError("Checkpoint policies differ")
        for n in a["players"]:
            x = candidate.get("checkpoints", {}).get(str(n), {})
            y = incumbent.get("checkpoints", {}).get(str(n), {})
            if not x.get("sha256") or x.get("sha256") != y.get("sha256"):
                raise ValueError("Frozen checkpoint identities differ or are absent")
        model_files = {k for k in a["sources"] | b["sources"]
                       if k.startswith(("agent/net/", "agent/search/"))
                       or k == "agent/train/checkpointing.py"}
        if any(a["sources"].get(k) != b["sources"].get(k) for k in model_files):
            raise ValueError("Neural inference sources differ")
    if a["split"] != "promotion" or a["blocks"] < 128:
        raise ValueError("Promotion requires at least 128 fresh promotion blocks")
    if "self" in a["fields"]:
        raise ValueError("Self-play is a diagnostic, not a fixed promotion field")
    if "incumbent" in a["fields"] and a.get("incumbent_config") != b.get("incumbent_config"):
        raise ValueError("Reference incumbent configurations differ")
    results = {}
    for n in a["players"]:
        fields = a["fields"]
        if not {"opus", "mixed"} <= set(fields):
            raise ValueError("Promotion requires both Opus and mixed fields")
        deltas = []
        field_results = {}
        failed = False
        for field in fields:
            x = candidate["summary"][f"{n}p/{field}/astra"]
            y = incumbent["summary"][f"{n}p/{field}/astra"]
            if x["unfinished"] or y["unfinished"] or x["failures"] or y["failures"]:
                failed = True
                break
            xs = {int(k): v for k, v in x["block_scores"].items()}
            ys = {int(k): v for k, v in y["block_scores"].items()}
            if set(xs) != set(range(a["blocks"])) or set(ys) != set(xs):
                failed = True
                break
            delta = [xs[k] - ys[k] for k in sorted(xs)]
            deltas.append(delta)
            field_results[field] = {"delta": float(np.mean(delta)), "ci95": interval(delta)}
        if failed:
            results[str(n)] = {"promote": False, "reason": "Incomplete, failed, or unmatched games"}
            continue
        pooled = np.mean(deltas, axis=0).tolist()
        ci = interval(pooled)
        significant_opus_regression = field_results["opus"]["ci95"][1] < -STAT_EPS
        results[str(n)] = {"promote": ci[0] > STAT_EPS and not significant_opus_regression,
                           "competitive_field_delta": float(np.mean(pooled)), "ci95": ci,
                           "fields": field_results,
                           "reason": "passed" if ci[0] > STAT_EPS and not significant_opus_regression else "Improvement not established or Opus regression"}
    return {"candidate": a["name"], "incumbent": b["name"], "player_counts": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = []
    for path in (args.candidate, args.incumbent):
        report = json.loads(path.read_text())
        records = [json.loads(line) for line in path.with_name("games.jsonl").read_text().splitlines()]
        report["schedule_sha256"] = schedule_identity(records)
        reports.append(report)
    result = compare(*reports)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
