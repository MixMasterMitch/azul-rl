"""Release-aware model catalog using the existing competitive model registry."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

from agent.train.model_registry import ModelRegistry
from agent.train import ranking as R
from agent.train import rating_display as D

BUILTINS = {"random": "Random Bot", "heuristic": "Heuristic Bot", "opus": "Opus Bot", "astra": "Astra Heuristic"}


def serving_registry() -> ModelRegistry:
    staged = Path(__file__).parent / "artifacts" / "registry.json"
    path = os.environ.get("AZUL_MODEL_REGISTRY")
    return ModelRegistry(path or (staged if staged.exists() else None))


class ModelCatalog:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.models = deepcopy(registry.entries())
        manifest = json.loads(registry.path.read_text()) if registry.path.exists() else {}
        self.reference_anchors = R.reference_anchors_from_manifest(manifest)
        self.default_model_id = manifest.get("default_model_id")
        self.release_id = manifest.get("release_id", "development")
        self.builtin_ratings = manifest.get("builtin_ratings", {})
        self.builtins = dict(BUILTINS)
        if importlib.util.find_spec("azul_astra") is None:
            self.builtins.pop("astra")
        self.scales = D.scales_for(self.reference_anchors)

    def resolve(self, model_id: str, num_players: int) -> dict:
        if model_id == "astra" and model_id not in self.builtins:
            raise ValueError("Astra is unavailable: its native extension is not installed on this server")
        if model_id in self.builtins:
            entity = "heuristic_opus" if model_id == "opus" else model_id
            raw = self.builtin_ratings.get(model_id, {}) or {
                str(pc): anchors[entity] for pc, anchors in self.reference_anchors.items() if entity in anchors}
            return {"id": model_id, "name": self.builtins[model_id], "supported_players": [2, 3, 4], "raw_ratings": raw}
        entry = self.models.get(model_id)
        if entry is None:
            raise ValueError(f"Unknown opponent: {model_id}")
        if num_players not in entry["trained_player_counts"]:
            raise ValueError("This agent is not trained for the selected player count")
        root = self.registry.path.parent.resolve()
        checkpoint = (root / entry["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(root):
            raise ValueError("Model artifact must be inside the release directory")
        return {**deepcopy(entry), "id": model_id, "supported_players": entry["trained_player_counts"],
                "checkpoint": str(checkpoint)}

    def list_opponents(self, num_players: int | None = None) -> list[dict]:
        result = []
        for model_id in [*self.builtins, *self.models]:
            if model_id in self.models and not self.models[model_id].get("available_for_new_games", True):
                continue
            pcs = self.models[model_id]["trained_player_counts"] if model_id in self.models else [2, 3, 4]
            if num_players is not None and num_players not in pcs:
                continue
            info = self.resolve(model_id, pcs[0])
            public = {"id": model_id, "name": info["name"], "supported_players": pcs,
                      "kind": "net" if model_id in self.models else "heuristic",
                      "default": model_id == self.default_model_id}
            rated = []
            for pc, raw in info.get("raw_ratings", {}).items():
                value = D.to_display(float(raw), int(pc), self.scales)
                public[f"rating_{pc}p"] = round(value)
                rated.append(value)
            if rated:
                public["rating"] = round(sum(rated) / len(rated))
            result.append(public)
        return result
