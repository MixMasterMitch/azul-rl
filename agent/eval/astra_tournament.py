"""Independent, resumable all-player-count Astra experiments.

Raw records are append-only; seeds and seat blocks, not individual seat games,
are the units of statistical resampling. Nothing is written to the training league.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
from functools import wraps
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
import zipfile

import numpy as np
import torch

from ..env import actions as A
# Keep tournament deals on the explicit PyTorch engine. Changing the application's
# default backend must not change this campaign's seeded reference games.
from ..env.batched_engine import BatchedEngine
from .bots import HeuristicBot, RandomBot
from .heuristic_astra import AstraConfig, HeuristicAstraBot, native_module, production_config, snapshot
from .heuristic_opus import HeuristicOpusBot

SPLITS = {"development": 1_000_000, "promotion": 100_000_000, "final": 200_000_000}
STAT_EPS = 1e-12  # Ignore floating-point residue at an exact zero/baseline.
ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIRS = ("agent/env", "agent/net", "agent/search", "agent/train")
REFERENCE_FILES = ("agent/eval/bots.py", "agent/eval/heuristic_opus.py",
                   "agent/eval/inference.py", "agent/eval/latency.py", "agent/eval/arena.py")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def schedule_identity(records: list[dict[str, Any]]) -> str:
    keys = ("key", "n", "field", "block", "seat", "variant", "names", "engine_seed", "bot_seed")
    rows = [{k: g[k] for k in keys} for g in sorted(records, key=lambda row: row["key"])]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def git_revision() -> str | None:
    """Source hashes remain authoritative when an archive has no Git metadata."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def source_identity(reference_experiment: Path | None = None) -> dict[str, str]:
    paths = sorted(p for folder in ("agent/env", "agent/eval", "agent/net", "agent/search", "agent/train", "native/astra/src")
                   for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".rs"))
    paths += [ROOT / name for name in ("native/astra/Cargo.lock", "native/astra/Cargo.toml",
                                      "native/astra/rust-toolchain.toml", "native/astra/pyproject.toml",
                                      "pyproject.toml", "agent/scripts/eval_astra.py")]
    module = native_module()
    native_files = list(Path(module.__file__).parent.glob("*.so"))
    result = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p)
              for p in paths + native_files}
    if reference_experiment is not None:
        frozen = reference_experiment / "frozen"
        if not (frozen / "agent/env/batched_engine.py").is_file():
            raise ValueError("Reference experiment has no frozen game engine")
        result = {k: v for k, v in result.items()
                  if not any(k.startswith(p + "/") for p in REFERENCE_DIRS) and k not in REFERENCE_FILES}
        for folder in REFERENCE_DIRS:
            for p in (frozen / folder).rglob("*.py"):
                result[str(p.relative_to(frozen))] = digest(p)
        for name in REFERENCE_FILES:
            p = frozen / name
            if p.is_file():
                result[name] = digest(p)
    return result


def freeze_sources(output: Path, reference_experiment: Path | None = None) -> None:
    with zipfile.ZipFile(output / "sources.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("pyproject.toml", "README.md", "MANIFEST.in", ".dockerignore", ".gitignore"):
            z.write(ROOT / name, name)
        for name in ("native/astra/README.md", "native/astra/SIMULATOR.md"):
            if (ROOT / name).is_file():
                z.write(ROOT / name, name)
        for folder in ("agent", "native/astra", "play", "infra"):
            for p in sorted((ROOT / folder).rglob("*")):
                if p.is_file() and not {"target", "runs", "play_data", "artifacts", "__pycache__"}.intersection(p.relative_to(ROOT).parts) and (
                    p.suffix in (".py", ".rs", ".toml", ".lock", ".json") or p.name.endswith("Dockerfile")
                ):
                    z.write(p, p.relative_to(ROOT))
    # Spawned workers import this private snapshot, so concurrent workspace edits
    # cannot change rules or policies partway through a tournament.
    frozen = output / "frozen"
    frozen.mkdir(exist_ok=True)
    with zipfile.ZipFile(output / "sources.zip") as z:
        z.extractall(frozen)
    module_dir = Path(native_module().__file__).parent
    target = frozen / "azul_astra"
    target.mkdir(exist_ok=True)
    for p in module_dir.iterdir():
        if p.suffix in (".py", ".so"):
            shutil.copy2(p, target / p.name)
    if reference_experiment is not None:
        reference = reference_experiment / "frozen"
        for folder in REFERENCE_DIRS:
            shutil.rmtree(frozen / folder)
            shutil.copytree(reference / folder, frozen / folder,
                            ignore=shutil.ignore_patterns("__pycache__"))
        for name in REFERENCE_FILES:
            (frozen / name).unlink(missing_ok=True)
            if (reference / name).exists():
                shutil.copy2(reference / name, frozen / name)
        # The source archive must describe the files actually used by workers.
        with zipfile.ZipFile(output / "sources.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(frozen.rglob("*")):
                if p.is_file() and p.suffix not in (".so", ".pyc") and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(frozen))


def frozen_identity(output: Path) -> dict[str, str]:
    """Verify the actual worker code, native extension, and checkpoint bytes."""
    paths = [output / "sources.zip"]
    paths += [p for folder in ("frozen", "checkpoints") for p in (output / folder).rglob("*")
              if p.is_file() and p.suffix in (".py", ".rs", ".json", ".toml", ".lock", ".so", ".pt")]
    return {str(p.relative_to(output)): digest(p) for p in sorted(paths)}


def freeze_checkpoints(output: Path, reference_experiment: Path | None = None) -> dict[str, dict[str, str]]:
    if reference_experiment is not None:
        previous = json.loads((reference_experiment / "manifest.json").read_text())["checkpoints"]
        selected = {}
        for n, entry in previous.items():
            path = reference_experiment / "checkpoints" / Path(entry["path"]).name
            if digest(path) != entry["sha256"]:
                raise ValueError("Reference checkpoint hash mismatch")
            target = output / "checkpoints" / path.name
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
            elif digest(target) != entry["sha256"]:
                raise ValueError("Different frozen checkpoints share the same filename")
            selected[n] = {**entry, "path": str(target.resolve()), "source": str(path.resolve())}
        return selected
    manifest = ROOT / "agent/runs/league/league.json"
    if not manifest.exists():
        return {}
    entries = json.loads(manifest.read_text()).get("entries", [])
    available = []
    for e in entries:
        raw = Path(e.get("path", ""))
        p = raw if raw.is_absolute() else manifest.parent / raw.name
        if p.is_file():
            available.append((e, p))
    selected: dict[str, dict[str, str]] = {}
    for n in (2, 3, 4):
        if not available:
            break
        rated = [(e, p) for e, p in available if e.get(f"rating_{n}p") is not None]
        field = f"rating_{n}p" if rated else "rating"
        entry, path = max(rated or available, key=lambda pair: pair[0].get(field) or 0)
        target = output / "checkpoints" / path.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        selected[str(n)] = {"path": str(target.resolve()), "sha256": digest(target),
                            "selection": field, "source": str(path)}
    return selected


class CheckpointBot:
    def __init__(self, path: str, seed: int = 0, policy: str = "greedy") -> None:
        from ..train.checkpointing import load_net_from_checkpoint
        self.net, _ = load_net_from_checkpoint(path, map_location="cpu")
        self.net.eval()
        self.seed = seed
        self.policy = policy
        self.move = 0

    @torch.inference_mode()
    def select_action(self, engine: BatchedEngine, game_idx: int) -> int:
        from ..net.encoder import encode_state
        sub = engine.index_select(torch.tensor([game_idx]))
        if self.policy == "search32":
            from ..search.config import SearchConfig
            from ..search.gumbel_mcts import gumbel_root_act
            settings = SearchConfig(num_simulations=32, root_noise_scale=0.0, temperature=0.0,
                                    seed=self.seed + self.move, move_deadline_s=2.0)
            self.move += 1
            return int(gumbel_root_act(sub, self.net, search_config=settings)[0][0])
        g, s = encode_state(sub)
        return int(self.net(g, s, sub.legal_action_mask(), sub.num_players)[0].argmax())


def make_bot(name: str, cfg: dict[str, Any], seed: int, checkpoint: str | None,
             checkpoint_policy: str = "greedy") -> Any:
    if name == "astra":
        return HeuristicAstraBot(seed=seed, config=AstraConfig.from_dict(cfg))
    if name == "checkpoint":
        if checkpoint is None:
            raise ValueError("No frozen checkpoint available")
        return CheckpointBot(checkpoint, seed, checkpoint_policy)
    return {"random": RandomBot, "heuristic": HeuristicBot, "opus": HeuristicOpusBot}[name](seed=seed)


def winners_and_share(engine: BatchedEngine) -> tuple[list[int], list[float]]:
    n = engine.num_players
    if not bool(engine.ended[0]):
        return [], [0.0] * n
    scores = engine.scores[0, :n].tolist()
    complete = engine.wall[0, :n].all(-1).sum(-1).tolist()
    best = max(zip(scores, complete))
    winners = [p for p in range(n) if (scores[p], complete[p]) == best]
    return winners, [1.0 / len(winners) if p in winners else 0.0 for p in range(n)]


def schedule(players: list[int], blocks: int, split: str, seed: int, fields: list[str],
             control: str | None = None) -> list[dict[str, Any]]:
    if split not in SPLITS or not 1 <= blocks <= 10000 or not 0 <= seed < 1000:
        raise ValueError("Invalid split, blocks, or seed (seed must be 0..999)")
    tasks = []
    for n in players:
        for field_idx, field in enumerate(fields):
            for block in range(blocks):
                engine_seed = SPLITS[split] + n * 1_000_000 + seed * 10000 + block
                for seat in range(n):
                    for variant in (["astra", control] if control else ["astra"]):
                        pool = ["opus", "heuristic", "random"] if field == "mixed" else ["astra" if field == "self" else field]
                        # Each mixed block rotates opponent types; rotate seats as a
                        # whole table to balance candidate's seat and turn order.
                        others = [pool[(block + i) % len(pool)] for i in range(n - 1)]
                        names = [""] * n
                        names[seat] = variant
                        for i, name in enumerate(others):
                            names[(seat + i + 1) % n] = name
                        key = f"{n}:{field}:{block}:{seat}:{variant}"
                        tasks.append({"key": key, "n": n, "field": field, "block": block,
                                      "seat": seat, "variant": variant, "names": names,
                                      "engine_seed": engine_seed,
                                      "bot_seed": engine_seed * 37 + field_idx * 101})
    return tasks


def _worker_init() -> None:
    torch.set_num_threads(1)


def play_game(task: dict[str, Any], *, bot_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    n = task["n"]
    torch.manual_seed(task["bot_seed"])
    rng = random.Random(task["bot_seed"])
    e = BatchedEngine(1, n, seed=task["engine_seed"])
    factory = bot_factory or make_bot
    bots = [factory("astra" if name == "incumbent" else name,
                     task["incumbent_config"] if name == "incumbent" else task["config"],
                     rng.randrange(1 << 32), task.get("checkpoint"), task.get("checkpoint_policy", "greedy"))
            for name in task["names"]]
    floor_losses = [0] * n
    timings: list[float] = []
    rounds = 0
    trajectory = []
    diagnostics: list[dict[str, Any]] = []
    failure = None
    turns = 0
    start = time.monotonic()
    for turn in range(task["max_turns"]):
        if e.ended[0]:
            break
        cp = int(e.current_player[0])
        bot = bots[cp]
        t = time.monotonic()
        try:
            if isinstance(bot, HeuristicAstraBot):
                detail = bot.analyze(e, 0)
                action = detail["action"]
            else:
                detail = None
                action = bot.select_action(e, 0)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break
        elapsed = time.monotonic() - t
        if cp == task["seat"]:
            timings.append(elapsed)
            if detail:
                diagnostics.append({k: detail[k] for k in ("nodes", "depth", "cutoff_reason", "solved")})
        if not 0 <= action < A.NUM_ACTIONS or not e.legal_action_mask()[0, action]:
            failure = f"illegal action {action} by {task['names'][cp]}"
            break
        if task.get("trace"):
            trajectory.append({"state": snapshot(e, 0), "action": action, "seat": cp,
                               "analysis": detail})
        e.step(torch.tensor([action]), finalize_round=False)
        turns += 1
        if not e.factory_tiles.any() and not e.center_tiles.any():
            rounds += 1
            for p in range(n):
                floor_losses[p] -= sum(A.FLOOR_PENALTIES[:int(e.floor_count[0, p])])
            e.finalize_round()
    winners, shares = winners_and_share(e)
    result = {k: task[k] for k in ("key", "n", "field", "block", "seat", "variant", "names", "engine_seed", "bot_seed")}
    result.update({"finished": bool(e.ended[0]), "failure": failure, "winners": winners,
                   "win_share": shares[task["seat"]] if e.ended[0] else None,
                   "scores": e.scores[0, :n].tolist(),
                   "walls": e.wall[0, :n].tolist(),
                   "rows": e.wall[0, :n].all(-1).sum(-1).tolist(),
                   "unfinished_lines": (e.pattern_count[0, :n] > 0).sum(-1).tolist(),
                   "floor_losses": floor_losses, "rounds": rounds, "turns": turns,
                   "move_latencies": timings, "search": diagnostics,
                   "wall_s": time.monotonic() - start})
    if task.get("trace"):
        result["trajectory"] = trajectory
    return result


def interval(values: list[float], seed: int = 0) -> list[float]:
    if len(values) < 2:
        return [0.0, 1.0]
    data = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = np.concatenate([data[rng.integers(0, len(data), (500, len(data)))].mean(1) for _ in range(20)])
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for g in records:
        groups.setdefault(f"{g['n']}p/{g['field']}/{g['variant']}", []).append(g)
    result: dict[str, Any] = {}
    for key, games in groups.items():
        n = games[0]["n"]
        blocks: dict[int, list[dict[str, Any]]] = {}
        for g in games:
            blocks.setdefault(g["block"], []).append(g)
        complete = {b: sum(g["win_share"] for g in gs) / n for b, gs in blocks.items()
                    if len(gs) == n and all(g["finished"] for g in gs)}
        missing = sum(not g["finished"] for g in games)
        failures = sum(g["failure"] is not None for g in games)
        total_share = sum(g["win_share"] or 0 for g in games)
        latencies = [v for g in games for v in g["move_latencies"]]
        searches = [s for g in games for s in g.get("search", [])]
        cutoffs = {reason: sum(s["cutoff_reason"] == reason for s in searches)
                   for reason in sorted({s["cutoff_reason"] for s in searches})}
        # Worker completion/append order must not change the seeded bootstrap.
        ci = interval([complete[b] for b in sorted(complete)]) if not missing else [0.0, 1.0]
        result[key] = {"games": len(games), "completed_blocks": len(complete), "unfinished": missing,
                       "failures": failures,
                       "stalls": sum(not g["finished"] and g["failure"] is None for g in games),
                       "illegal_actions": sum("illegal action" in (g["failure"] or "").lower() for g in games),
                       "win_share": total_share / len(games),
                       "unfinished_win_share_bounds": [total_share / len(games), (total_share + missing) / len(games)],
                       "ci95": ci, "block_scores": complete,
                       "mean_score": float(np.mean([g["scores"][g["seat"]] for g in games])),
                       "mean_margin": float(np.mean([g["scores"][g["seat"]] - max(s for p, s in enumerate(g["scores"]) if p != g["seat"]) for g in games])),
                       "mean_floor_losses": float(np.mean([g["floor_losses"][g["seat"]] for g in games])),
                       "mean_rounds": float(np.mean([g["rounds"] for g in games])),
                       "mean_turns": float(np.mean([g.get("turns", 0) for g in games])),
                       "mean_unfinished_lines": float(np.mean([g["unfinished_lines"][g["seat"]] for g in games])),
                       "latency_mean_s": float(np.mean(latencies)) if latencies else 0,
                       "latency_p95_s": float(np.quantile(latencies, .95)) if latencies else 0,
                       "latency_max_s": max(latencies, default=0),
                       "search_calls": len(searches), "search_cutoffs": cutoffs,
                       "solved_rounds": sum(s["solved"] for s in searches),
                       "mean_search_nodes": float(np.mean([s["nodes"] for s in searches])) if searches else 0,
                       "mean_search_depth": float(np.mean([s["depth"] for s in searches])) if searches else 0,
                       "opus_superiority": key.endswith("/opus/astra") and not missing and not failures and len(complete) == len(blocks) and ci[0] > 1 / n + STAT_EPS}
    for key, group in list(result.items()):
        if not key.endswith("/astra"):
            continue
        control = result.get(key.removesuffix("astra") + "opus")
        if (control and not group["unfinished"] and not control["unfinished"]
                and not group["failures"] and not control["failures"]):
            a, b = group["block_scores"], control["block_scores"]
            shared = sorted(a.keys() & b.keys())
            delta = [a[k] - b[k] for k in shared]
            group["paired_delta_vs_opus"] = float(np.mean(delta)) if delta else None
            group["paired_delta_ci95"] = interval(delta) if len(delta) >= 2 else [-1.0, 1.0]
    return result


def _exclusive_output(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def locked(output: Path, **settings: Any) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Experiment directory is already in use by another runner") from exc
            try:
                return function(output, **settings)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return locked


@_exclusive_output
def run_tournament(output: Path, *, config: AstraConfig | dict[int, AstraConfig] | None,
                   players: list[int], blocks: int,
                   fields: list[str], split: str, seed: int, workers: int,
                   control: str | None, max_turns: int = 400, trace: bool = False,
                   name: str = "astra", incumbent_config: AstraConfig | None = None,
                   checkpoint_policy: str = "greedy", reference_experiment: Path | None = None) -> dict[str, Any]:
    if not players or any(n not in (2, 3, 4) for n in players) or len(set(players)) != len(players):
        raise ValueError("players must be unique values from 2, 3, 4")
    if not fields or any(f not in ("random", "heuristic", "opus", "mixed", "checkpoint", "self", "incumbent") for f in fields):
        raise ValueError("Invalid opponent fields")
    if control not in (None, "opus"):
        raise ValueError("Control must be Opus or None")
    if len(set(fields)) != len(fields) or not 1 <= workers <= 4 or max_turns < 1:
        raise ValueError("Invalid fields, workers, or turn cap")
    if "incumbent" in fields and incumbent_config is None:
        raise ValueError("Incumbent field requires an incumbent configuration")
    if checkpoint_policy not in ("greedy", "search32"):
        raise ValueError("Invalid checkpoint policy")
    if reference_experiment is not None:
        reference_manifest = json.loads((reference_experiment / "manifest.json").read_text())
        if reference_manifest.get("frozen_sha256") != frozen_identity(reference_experiment):
            raise ValueError("Reference experiment artifacts changed or have no integrity manifest")
    configs = {n: config if isinstance(config, AstraConfig) else config[n] if config is not None
               else production_config(n) for n in players}
    if any(not isinstance(c, AstraConfig) for c in configs.values()):
        raise ValueError("Each player count requires an AstraConfig")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    spec = {"schema_version": 1, "name": name,
            "config": asdict(config) if isinstance(config, AstraConfig) else None,
            "configs_per_pc": {str(n): asdict(c) for n, c in configs.items()}, "players": players,
            "blocks": blocks, "fields": fields, "split": split, "seed": seed,
            "control": control, "max_turns": max_turns, "trace": trace,
            "checkpoint_policy": checkpoint_policy, "sources": source_identity(reference_experiment),
            "incumbent_config": asdict(incumbent_config) if incumbent_config else None}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != spec:
            raise ValueError("Resume requires identical source/native hashes and experiment specification")
        if manifest.get("frozen_sha256") != frozen_identity(output):
            raise ValueError("Frozen runtime or checkpoint files changed; refusing resume")
    else:
        manifest = {"spec": spec, "checkpoints": freeze_checkpoints(output, reference_experiment),
                    "reference_experiment": str(reference_experiment.resolve()) if reference_experiment else None,
                    "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                    "torch": torch.__version__, "numpy": np.__version__,
                                    "git_head": git_revision()},
                    "started_at": time.time()}
        freeze_sources(output, reference_experiment)
        manifest["frozen_sha256"] = frozen_identity(output)
        atomic_json(manifest_path, manifest)
    if "checkpoint" in fields and any(str(n) not in manifest["checkpoints"] for n in players):
        raise ValueError("Checkpoint field requires an available frozen checkpoint for every player count")
    records_path = output / "games.jsonl"
    records = []
    if records_path.exists():
        # Recover only an interrupted final append, never silently ignore a bad record.
        raw = records_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raw = raw[:raw.rfind(b"\n") + 1]
            records_path.write_bytes(raw)
        records = [json.loads(line) for line in raw.splitlines()]
    seen = {g["key"] for g in records}
    if len(seen) != len(records):
        raise ValueError("Duplicate game records")
    tasks = schedule(players, blocks, split, seed, fields, control)
    scheduled = {t["key"]: t for t in tasks}
    if not seen <= scheduled.keys():
        raise ValueError("Unexpected records for this schedule")
    for record in records:
        expected = scheduled[record["key"]]
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Recorded game metadata differs from its scheduled seeds or seats")
    pending = []
    for task in tasks:
        if task["key"] in seen:
            continue
        cp = manifest["checkpoints"].get(str(task["n"]))
        task.update(config=asdict(configs[task["n"]]), max_turns=max_turns, trace=trace,
                    checkpoint=str((output / "checkpoints" / Path(cp["path"]).name).resolve()) if cp else None,
                    checkpoint_policy=checkpoint_policy,
                    incumbent_config=asdict(incumbent_config) if incumbent_config else None)
        pending.append(task)
    started = time.monotonic()
    if pending:
        sys.path.insert(0, str((output / "frozen").resolve()))
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init) as pool, records_path.open("a") as f:
                for record in pool.map(play_game, pending, chunksize=1):
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                    f.flush()
                    records.append(record)
                    if len(records) % 32 == 0 or len(records) == len(tasks):
                        print(json.dumps({"completed": len(records), "total": len(tasks),
                                          "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        finally:
            sys.path.pop(0)
    report = {"spec": spec, "checkpoints": manifest["checkpoints"],
              "schedule_sha256": schedule_identity(records),
              "summary": summarize(records), "total_games": len(records),
              "wall_s_this_run": time.monotonic() - started, "complete": len(records) == len(tasks)}
    atomic_json(output / "report.json", report)
    return report
