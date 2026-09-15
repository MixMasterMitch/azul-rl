"""Frozen, balanced round-robin evaluation of playable Azul agents.

The campaign does not tune agents or modify training leagues. Every seed block
contains all seat rotations; uncertainty resamples blocks, including all tables
which shared their draw seed. Unsupported neural multiplayer heads are excluded.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import fcntl
from itertools import combinations, permutations
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

from . import astra_tournament as tournament
from .heuristic_astra import production_config
from ..search.config import SearchConfig

BUILTINS = ('astra', 'opus', 'heuristic', 'random')
SEED_BASE = 900_000_000


def schedule(participants: dict[str, dict[str, Any]], blocks_2p: int = 256,
             blocks_mp: int = 128, mixed_blocks: int = 64) -> list[dict[str, Any]]:
    """All unordered 2p pairs, both minority directions in MP, and mixed tables."""
    if not participants or any(not 1 <= x <= 10000 for x in (blocks_2p, blocks_mp, mixed_blocks)):
        raise ValueError('Participants and positive block counts up to 10000 are required')
    tasks = []
    for n in (2, 3, 4):
        names = sorted(name for name, spec in participants.items() if n in spec['players'])
        fields = [(f'{a}_vs_{b}', [a] + [b] * (n - 1), blocks_2p if n == 2 else blocks_mp, 0)
                  for a, b in (combinations(names, 2) if n == 2 else permutations(names, 2))]
        if n > 2:
            fields += [('mixed_' + '_'.join(table), list(table), mixed_blocks, 100_000)
                       for table in combinations(names, n)]
        for field_id, (field, table, blocks, offset) in enumerate(fields):
            for block in range(blocks):
                engine_seed = SEED_BASE + n * 1_000_000 + offset + block
                for seat in range(n):
                    rotated = table[-seat:] + table[:-seat] if seat else list(table)
                    tasks.append(dict(key=f'{n}:{field}:{block}:{seat}', n=n, field=field,
                                      block=block + offset, seat=seat, variant=table[0], names=rotated,
                                      engine_seed=engine_seed, bot_seed=engine_seed * 37 + field_id * 101))
    # Interleave fields and counts to expose setup problems early. RNGs are local.
    return sorted(tasks, key=lambda task: (task['block'], task['seat'], task['n'], task['field']))


class RegisteredBot:
    """A copied checkpoint using its recorded serving search configuration."""
    def __init__(self, spec: dict[str, Any], seed: int) -> None:
        from ..train.checkpointing import load_net_from_checkpoint
        self.net, _ = load_net_from_checkpoint(spec['checkpoint'], map_location='cpu')
        self.net.eval()
        self.search = SearchConfig(**spec['search'])
        self.seed = seed
        self.move = 0

    @torch.inference_mode()
    def select_action(self, engine: Any, game_idx: int) -> int:
        from ..search.gumbel_mcts import gumbel_root_act
        sub = engine.index_select(torch.tensor([game_idx]))
        settings = replace(self.search, seed=self.seed + self.move)
        self.move += 1
        return int(gumbel_root_act(sub, self.net, search_config=settings)[0][0])


def play_game(task: dict[str, Any]) -> dict[str, Any]:
    def factory(name: str, cfg: dict[str, Any], seed: int, checkpoint: str | None,
                checkpoint_policy: str) -> Any:
        if name in BUILTINS:
            return tournament.make_bot(name, cfg, seed, checkpoint, checkpoint_policy)
        return RegisteredBot(task['participants'][name], seed)
    return tournament.play_game(task, bot_factory=factory)


def discover_participants() -> dict[str, dict[str, Any]]:
    root = tournament.ROOT
    participants = {name: {'kind': 'builtin', 'players': [2, 3, 4]} for name in BUILTINS}
    # Published league champion: use the exact published weights/profile when available.
    registries = [('league', root / 'play/artifacts/registry.json'),
                  ('trained', root / 'agent/runs/competitive/models/registry.json')]
    for prefix, path in registries:
        if not path.is_file():
            continue
        registry = json.loads(path.read_text())
        model_id = registry.get('default_model_id')
        spec = registry.get('models', {}).get(model_id)
        if spec is None:
            continue
        checkpoint = (path.parent / spec['checkpoint']).resolve()
        if not checkpoint.is_file() or tournament.digest(checkpoint) != spec['sha256']:
            raise ValueError(f'Registry checkpoint hash mismatch: {model_id}')
        settings = dict(spec['search'])
        settings['move_deadline_s'] = 2.0
        SearchConfig(**settings)
        participants[f'{prefix}_2p'] = dict(
            kind='checkpoint', players=spec['trained_player_counts'], checkpoint=str(checkpoint),
            sha256=spec['sha256'], search=settings, original_search=spec['search'],
            model_id=model_id, name=spec['name'], registry_source=str(path),
        )
    return participants


def prepare(output: Path, *, blocks_2p: int = 256, blocks_mp: int = 128,
            mixed_blocks: int = 64) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Campaign already exists; use --resume')
    participants = discover_participants()
    (output / 'checkpoints').mkdir(exist_ok=True)
    for spec in participants.values():
        if spec['kind'] == 'checkpoint':
            source = Path(spec['checkpoint'])
            target = output / 'checkpoints' / f"{spec['sha256']}.pt"
            shutil.copy2(source, target)
            if tournament.digest(target) != spec['sha256']:
                raise ValueError('Checkpoint changed while being copied')
            spec['source_checkpoint'] = str(source)
            spec['checkpoint'] = str(target.relative_to(output))
    tasks = schedule(participants, blocks_2p, blocks_mp, mixed_blocks)
    tournament.freeze_sources(output)
    manifest = dict(schema_version=1, purpose='ranking-only; no policy tuning',
                    created_at=time.time(), participants=participants,
                    blocks_2p=blocks_2p, blocks_mp=blocks_mp, mixed_blocks=mixed_blocks,
                    expected_games=len(tasks), schedule_sha256=tournament.schedule_identity(tasks),
                    configs={str(n): asdict(production_config(n)) for n in (2, 3, 4)},
                    max_turns=400, environment={'python': platform.python_version(),
                    'torch': str(torch.__version__), 'numpy': np.__version__},
                    frozen_sha256=tournament.frozen_identity(output))
    tournament.atomic_json(output / 'manifest.json', manifest)
    return manifest


def load_records(path: Path, scheduled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n') + 1]
        path.write_bytes(raw)
    records = [json.loads(line) for line in raw.splitlines()]
    tasks = {task['key']: task for task in scheduled}
    seen = set()
    for record in records:
        key = record['key']
        if key in seen or key not in tasks:
            raise ValueError('Duplicate or unexpected game record')
        if any(record.get(k) != v for k, v in tasks[key].items()):
            raise ValueError('Recorded seeds, participants or seats differ from schedule')
        seen.add(key)
    return records


def run_archived(output: Path, workers: int = 4) -> dict[str, Any]:
    """Must execute from the archived runtime; resume only missing records."""
    if not 1 <= workers <= 4:
        raise ValueError('Use one to four workers')
    torch.set_num_threads(1)
    output = output.resolve()
    if not Path(__file__).resolve().is_relative_to(output / 'frozen'):
        raise ValueError('Run this campaign through the CLI to load its archived runtime')
    with (output / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((output / 'manifest.json').read_text())
        if tournament.frozen_identity(output) != manifest['frozen_sha256']:
            raise ValueError('Frozen source, native extension or checkpoint changed')
        environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__}
        if environment != manifest['environment']:
            raise ValueError('Restore the recorded Python/Torch/NumPy environment before resuming')
        tasks = schedule(manifest['participants'], manifest['blocks_2p'], manifest['blocks_mp'], manifest['mixed_blocks'])
        if tournament.schedule_identity(tasks) != manifest['schedule_sha256']:
            raise ValueError('Schedule identity changed')
        records_path = output / 'games.jsonl'
        records = load_records(records_path, tasks)
        seen = {record['key'] for record in records}
        participants = json.loads(json.dumps(manifest['participants']))
        for spec in participants.values():
            if spec['kind'] == 'checkpoint':
                spec['checkpoint'] = str(output / spec['checkpoint'])
        pending = []
        for task in tasks:
            if task['key'] not in seen:
                task = dict(task, config=manifest['configs'][str(task['n'])], max_turns=manifest['max_turns'],
                            participants=participants, checkpoint=None, trace=False)
                pending.append(task)
        start = time.monotonic()
        if pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=tournament._worker_init) as pool, records_path.open('a') as stream:
                futures = {pool.submit(play_game, task): task['key'] for task in pending}
                for future in as_completed(futures):
                    record = future.result()
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    records.append(record)
                    if len(records) % 32 == 0 or len(records) == len(tasks):
                        progress = dict(completed=len(records), total=len(tasks),
                                        failures=sum(bool(g['failure']) for g in records),
                                        unfinished=sum(not g['finished'] for g in records),
                                        elapsed_s=round(time.monotonic() - start, 1))
                        tournament.atomic_json(output / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
        report = summarize(records)
        report.update(complete=len(records) == len(tasks), expected_games=len(tasks),
                      manifest_sha256=tournament.digest(output / 'manifest.json'))
        tournament.atomic_json(output / 'report.json', report)
        return report


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Direct minority-table win shares, retaining every unfinished game."""
    from agent.scripts.rate_astra import ratings
    groups: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for game in records:
        names = game['names']
        n = game['n']
        for seat, name in enumerate(names):
            others = names[:seat] + names[seat + 1:]
            if len(set(others)) == 1 and name != others[0]:
                groups.setdefault(f'{n}p/{name}/vs_{others[0]}', []).append((game, seat))
    table = {}
    for key, games in sorted(groups.items()):
        blocks: dict[int, list[float]] = {}
        shares = []
        for game, seat in games:
            share = (float(seat in game['winners']) / len(game['winners'])
                     if game['finished'] and not game['failure'] and game['winners'] else None)
            if share is not None:
                blocks.setdefault(game['block'], []).append(share)
                shares.append(share)
        n = games[0][0]['n']
        complete_blocks = [sum(v) / n for _, v in sorted(blocks.items()) if len(v) == n]
        missing = len(games) - len(shares)
        table[key] = dict(games=len(games), unfinished_or_failed=missing,
                          win_share=sum(shares) / len(games),
                          ci95=tournament.interval(complete_blocks) if not missing and complete_blocks else [0., 1.],
                          complete_blocks=len(complete_blocks))
    return dict(total_games=len(records), unfinished=sum(not g['finished'] for g in records),
                failures=sum(bool(g['failure']) for g in records), matchups=table,
                ratings=ratings(records))
