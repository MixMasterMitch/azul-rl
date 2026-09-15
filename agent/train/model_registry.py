"""Immutable model artifacts and an atomic registry for playable agents."""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path

from ..eval.arena import checkpoint_hash, write_report
from ..search.config import SearchConfig
from .checkpointing import load_net_from_checkpoint, save_checkpoint


def default_registry_path() -> Path:
    return Path(os.environ.get('AZUL_MODEL_REGISTRY', str(Path(__file__).resolve().parents[1] / 'runs/competitive/models/registry.json')))


class ModelRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_registry_path()

    def entries(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text())
        if payload.get('schema_version') != 1:
            raise ValueError('Unsupported model registry schema')
        return payload['models']

    def resolve(self, model_id: str, num_players: int) -> dict:
        entry = self.entries().get(model_id)
        if entry is None:
            raise ValueError(f'Unknown model: {model_id}')
        if num_players not in entry['trained_player_counts']:
            raise ValueError('This agent is not trained for the selected player count')
        result = dict(entry)
        result['checkpoint'] = str((self.path.parent / entry['checkpoint']).resolve())
        return result

    def register(self, model_id: str, checkpoint: str, search: SearchConfig,
                 *, name: str = 'Trained Agent', status: str = 'baseline',
                 evidence: dict | None = None) -> dict:
        if status not in {'baseline', 'promoted'}:
            raise ValueError('Invalid registry status')
        if status == 'promoted' and not (evidence and evidence.get('replicated') and evidence.get('promote')):
            raise ValueError('Promotion requires confirmation and training-seed replication')
        net, payload = load_net_from_checkpoint(checkpoint, 'cpu')
        trained = payload.get('trained_player_counts') or [payload.get('config', {}).get('num_players', 2)]
        if trained != [2]:
            raise ValueError('Competitive registry currently serves verified two-player agents only')
        source_hash = checkpoint_hash(checkpoint)
        artifact = self.path.parent / f'{source_hash[:20]}.pt'
        if not artifact.exists():
            save_checkpoint(artifact, net, iteration=payload.get('iteration', 0),
                            config={'num_players': 2, 'source_sha256': source_hash})
        entry = {'name': name, 'checkpoint': artifact.name, 'sha256': checkpoint_hash(artifact),
                 'source_sha256': source_hash, 'trained_player_counts': trained,
                 'search': asdict(search), 'status': status, 'evidence': evidence or {}}
        entries = self.entries()
        entries[model_id] = entry
        manifest = json.loads(self.path.read_text()) if self.path.exists() else {}
        manifest.update(schema_version=1, models=entries, default_model_id=model_id)
        write_report(self.path, manifest)
        return entry
