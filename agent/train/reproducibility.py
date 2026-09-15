"""Reproducible runs, provenance, and storage preflight checks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            # manual_seed_all can be queued before CUDA's lazy initialization.
            # Materialize those generators so a later first CUDA draw resumes.
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from resumable RNG state')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    def git(*args: str) -> bytes:
        return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL)
    result: dict = {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                    'attention_pooling': 'explicit_softmax',
                    'cuda': torch.version.cuda, 'engine_backend': 'rust', 'dependencies': {}}
    for name in ('torch', 'numpy', 'optuna', 'flask', 'pytest', 'azul-astra'):
        try:
            result['dependencies'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        result['git_revision'] = git('rev-parse', 'HEAD').decode().strip()
        diff = git('diff', '--binary', 'HEAD')
        digest = hashlib.sha256(diff)
        untracked = git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0')
        for name in sorted(n for n in untracked if n):
            path = root / os.fsdecode(name)
            if path.is_file():
                digest.update(name)
                digest.update(path.read_bytes())
        result['dirty'] = bool(diff or any(untracked))
        result['dirty_state_sha256'] = digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        result['git_revision'] = 'unknown'
    return result


def tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def require_disk_space(path: Path, required_bytes: int, reserve_bytes: int = 256 * 1024**2) -> None:
    parent = path if path.is_dir() else path.parent
    while not parent.exists():
        parent = parent.parent
    available = shutil.disk_usage(parent).free
    needed = int(required_bytes * 1.1) + reserve_bytes
    if available < needed:
        raise OSError(f'Insufficient disk space for {path}: need {needed / 2**30:.2f} GiB free, have {available / 2**30:.2f} GiB')
