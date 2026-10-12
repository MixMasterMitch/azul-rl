"""Bridge the selected shared model onto the frozen Astra ranking scale.

The reference campaign supplies random/heuristic/Opus/Astra ratings. Held-out
shared-core confirmation games supply selected-model comparisons at 1024 simulations.
These are cross-campaign Bradley-Terry diagnostics, not a new round robin.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent.train import ranking as R
from agent.train import rating_display as D

ROOT = Path(__file__).resolve().parents[2]
RANKING = ROOT / "agent/runs/astra-ranking/full/report.json"
RANKING_MANIFEST = RANKING.parent / "manifest.json"
EVALUATIONS = ROOT / "agent/runs/shared_core_24h_20261008/evaluations"
RESULT = ROOT / "agent/runs/shared_core_24h_20261008/result.json"
OUTPUT = ROOT / "docs/deployment-ratings.json"
ENTITIES = ("random", "heuristic", "opus", "astra")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def placement(record: dict, a: int, b: int) -> int:
    """Compare recorded final scores, respecting official winner tiebreaks."""
    scores = record["scores"]
    if scores[a] != scores[b]:
        return (scores[a] > scores[b]) - (scores[a] < scores[b])
    winners = record["winning_seats"]
    return (a in winners) - (b in winners)


def produce() -> dict:
    ranking = json.loads(RANKING.read_text())
    ranking_manifest = json.loads(RANKING_MANIFEST.read_text())
    selection = json.loads(RESULT.read_text())
    if sha256(Path(selection["selected"])) != selection["selected_sha256"]:
        raise ValueError("Selected shared checkpoint hash changed")
    if not ranking["complete"] or ranking["failures"] or ranking["unfinished"]:
        raise ValueError("The frozen reference ranking is incomplete")
    ratings: dict[str, dict[str, float]] = {entity: {} for entity in (*ENTITIES, "top_model")}
    evidence: dict[str, dict] = {}
    for pc in R.PLAYER_COUNTS:
        baseline = ranking["ratings"]["per_player_count"][str(pc)]["ratings"]
        anchors = {entity: float(baseline[entity]) for entity in ENTITIES}
        if anchors["random"] != 1000.0:
            raise ValueError("Random is not anchored at 1000")
        expected_hash = selection["selected_sha256"]
        files = sorted(EVALUATIONS.glob(f"confirmation_selected_{pc}p_*.json"))
        files = [path for path in files if "direct" not in path.name]
        comparisons: list[dict] = []
        games = 0
        rated_games = 0
        for path in files:
            panel = json.loads(path.read_text())
            if panel["candidate_sha256"] != expected_hash or panel["num_players"] != pc:
                raise ValueError(f"Wrong selected-model identity: {path}")
            if panel["search"]["num_simulations"] != 1024 or not panel["finished"]:
                raise ValueError(f"Incomplete or wrong search profile: {path}")
            if panel["identities"]["astra"]["config"] != ranking_manifest["configs"][str(pc)]:
                raise ValueError(f"Astra reference configuration changed: {path}")
            if len(panel["records"]) != panel["games_total"]:
                raise ValueError(f"Missing game records: {path}")
            for record in panel["records"]:
                games += 1
                seat = record["candidate_seat"]
                if record["seats"][seat] != "candidate":
                    raise ValueError(f"Candidate seat mismatch: {path}")
                if any(name in ("astra", "opus") for name in record["seats"]):
                    rated_games += 1
                for other, name in enumerate(record["seats"]):
                    if name not in ("astra", "opus"):
                        continue
                    outcome = placement(record, seat, other)
                    R.add_match_result(comparisons, "top_model", name,
                                       int(outcome > 0), int(outcome < 0),
                                       ties=int(outcome == 0), num_players=pc)
        if not comparisons:
            raise ValueError(f"No rated selected-model comparisons at {pc}p")
        fit = R.fit_ratings_for_pc(comparisons, pc, anchors=anchors,
                                   use_reference_anchors=False, prior_sigma=10000.0)
        scale = D.scales_for({pc: {"random": 1000.0, "heuristic": anchors["heuristic"]}})
        for entity in ratings:
            ratings[entity][str(pc)] = round(D.to_display(fit[entity], pc, scale))
        evidence[str(pc)] = {"checkpoint_sha256": expected_hash, "games": games,
                             "rated_games": rated_games,
                             "pairwise_records": comparisons,
                             "files": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                                       for path in files],
                             "raw_ratings": {name: fit[name] for name in ratings}}
    return {"schema_version": 1, "rating_display": D.VERSION,
            "method": "Bradley-Terry pairwise final score comparisons with official winner tiebreaks, random=1000, scale=1000; frozen reference opponents anchored; top model fitted from held-out 1024-simulation shared-model panels; display maps heuristic to 2500 per count",
            "limitations": "Cross-campaign diagnostic: shared-model games use different draws and table mixes from the reference round robin. Multiplayer pairwise outcomes are correlated. The broad 2p retention panel did not clear its prespecified gate. Values are comparative estimates, not confidence-bounded strength claims.",
            "reference": {"path": str(RANKING.relative_to(ROOT)), "sha256": sha256(RANKING),
                          "manifest_sha256": sha256(RANKING_MANIFEST),
                          "games": ranking["total_games"]},
            "ratings": ratings, "evidence": evidence}


def main() -> None:
    result = produce()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["ratings"], indent=2))


if __name__ == "__main__":
    main()
