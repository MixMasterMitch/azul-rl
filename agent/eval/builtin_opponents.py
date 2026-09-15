"""Builtin evaluation policies and identities for reproducible reports."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

from .bots import RandomBot, HeuristicBot
from .heuristic_opus import HeuristicOpusBot
from .heuristic_astra import HeuristicAstraBot, native_module, production_config

BUILTIN_BOTS = {'random': RandomBot, 'heuristic': HeuristicBot,
                'opus': HeuristicOpusBot, 'astra': HeuristicAstraBot}


def builtin_identity(name: str, num_players: int = 2) -> dict[str, Any] | None:
    """Include Astra's actual native binary and configuration in cache keys."""
    if name != 'astra':
        return None
    module = native_module()
    binaries = sorted(Path(module.__file__).parent.glob('*.so'))
    return {'config': asdict(production_config(num_players)), 'native_api': module.CONFIG_VERSION,
            'native_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries}}
