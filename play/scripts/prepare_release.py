"""Stage immutable, weights-only artifacts and validate a serving release."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from agent.env.engine import BatchedEngine
from agent.net.encoder import encode_state
from agent.search.config import SearchConfig
from agent.train.checkpointing import load_net_from_checkpoint
from agent.train.ranking import reference_anchors_from_manifest
from play import fixed_ratings as F

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate(directory: Path) -> dict:
    manifest = json.loads((directory / "registry.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("default_model_id") not in manifest["models"]:
        raise ValueError("A release must identify its default trained opponent")
    torch.set_num_threads(1)
    for model_id, entry in manifest["models"].items():
        path = (directory / entry["checkpoint"]).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != entry["sha256"]:
            raise ValueError(f"Missing or corrupt artifact: {model_id}")
        net, payload = load_net_from_checkpoint(path, "cpu")
        if any(k in payload for k in ("optimizer_state_dict", "buffer", "rng_state")):
            raise ValueError(f"Training-only state in serving artifact: {model_id}")
        if any(payload.get(k) != entry[k] for k in ("hidden", "arch", "model_version", "encoder_version")):
            raise ValueError(f"Model metadata mismatch: {model_id}")
        SearchConfig(**entry["search"])
        net.eval()
        for pc in entry["trained_player_counts"]:
            engine = BatchedEngine(1, pc, seed=42)
            global_feat, sources = encode_state(engine)
            mask = engine.legal_action_mask()
            with torch.inference_mode():
                policy, value = net(global_feat, sources, mask, pc)
            if not torch.isfinite(policy).all() or not torch.isfinite(value).all() or not bool(mask[0, policy[0].argmax()]):
                raise ValueError(f"Inference validation failed: {model_id}")
    return manifest


def prepare(config_path: Path, directory: Path) -> dict:
    config = json.loads(config_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    previous = directory / "registry.json"
    manifest = json.loads(previous.read_text()) if previous.exists() else {"schema_version": 1, "models": {}}
    league = json.loads((ROOT / config["league"]).read_text()) if "league" in config else {"entries": []}
    ratings = None
    if config.get("rating_scheme") == F.VERSION:
        refs = F.references()
        manifest["rating_scheme"] = F.VERSION
        manifest.pop("ratings_source_sha256", None)
        if config.get("fixed_rating_points") != {"random": 1000, "heuristic": 2000,
                                                 "opus": 2250, "astra": 2500, "agent": 3000}:
            raise ValueError("Fixed rating points must match the play rating scheme")
    elif "ratings" in config:
        ratings_path = ROOT / config["ratings"]["source"]
        if digest(ratings_path) != config["ratings"]["source_sha256"]:
            raise ValueError("Rating evidence changed; review the release selection")
        ratings = json.loads(ratings_path.read_text())
        refs = {int(pc): {name: values["raw_ratings"][name]
                          for name in ("random", "heuristic", "opus", "astra")}
                for pc, values in ratings["evidence"].items()}
        for values in refs.values():
            values["heuristic_opus"] = values.pop("opus")
    else:
        refs = reference_anchors_from_manifest(league)
    manifest.update(default_model_id=config["default_model_id"], reference_anchors_per_pc=refs)
    if ratings:
        manifest["ratings_source_sha256"] = config["ratings"]["source_sha256"]
    manifest["builtin_ratings"] = {}
    # Reference anchors are raw per-PC values; floating_entities contains display
    # calibration and must not be calibrated a second time.
    for bot, entity in (("random", "random"), ("heuristic", "heuristic"), ("opus", "heuristic_opus"), ("astra", "astra")):
        manifest["builtin_ratings"][bot] = {str(pc): values[entity] for pc, values in refs.items() if entity in values}
    for model_id, spec in config["models"].items():
        source = ROOT / spec["source"]
        if digest(source) != spec["source_sha256"]:
            raise ValueError(f"Source checksum changed for {model_id}; explicitly update the release selection")
        index = spec.get("league_index")
        league_entry = next((e for e in league["entries"] if e["idx"] == index), {})
        if config.get("rating_scheme") == F.VERSION and spec.get("rating_entity") == "top_model":
            raw_ratings = {str(pc): F.POINTS[F.AGENT_ID] for pc in spec["trained_player_counts"]}
        elif ratings and spec.get("rating_entity"):
            raw_ratings = {str(pc): ratings["evidence"][str(pc)]["raw_ratings"][spec["rating_entity"]]
                           for pc in spec["trained_player_counts"]}
        else:
            raw_ratings = {str(pc): league_entry[f"rating_{pc}p"] for pc in spec["trained_player_counts"]
                           if f"rating_{pc}p" in league_entry}
        existing = manifest["models"].get(model_id)
        if existing:
            if existing["source_sha256"] != spec["source_sha256"] or existing["search"] != spec["search"]:
                raise ValueError(f"Model IDs are immutable; assign a new ID for changes to {model_id}")
            existing["raw_ratings"] = raw_ratings
            continue
        net, source_payload = load_net_from_checkpoint(source, "cpu")
        metadata = {"hidden": net.hidden, "arch": net.arch, "model_version": net.model_version,
                    "encoder_version": 3, "trained_player_counts": spec["trained_player_counts"]}
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary) / "model.pt"
            torch.save({**metadata, "model_state_dict": net.state_dict(),
                        "iteration": source_payload.get("iteration", 0)}, path)
            checksum = digest(path)
            filename = checksum + ".pt"
            os.replace(path, directory / filename)
        entry = {**metadata, "name": spec["name"], "checkpoint": filename, "sha256": checksum,
                 "source_sha256": spec["source_sha256"], "search": spec["search"], "status": "baseline",
                 "available_for_new_games": True,
                 "raw_ratings": raw_ratings}
        manifest["models"][model_id] = entry
    # All previously shipped IDs remain resumable. Only configured IDs are offered
    # for new games; their binaries are deliberately never pruned here.
    for model_id, entry in manifest["models"].items():
        entry["available_for_new_games"] = model_id in config["models"]
    manifest.pop("release_id", None)
    manifest["release_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    temporary = directory / "registry.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, previous)
    return validate(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "play/release.json")
    parser.add_argument("--output", type=Path, default=ROOT / "play/artifacts")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = validate(args.output) if args.validate_only else prepare(args.config, args.output)
    print(json.dumps({"release_id": manifest["release_id"], "models": list(manifest["models"])}))


if __name__ == "__main__":
    main()
