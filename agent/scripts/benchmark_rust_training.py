"""Compare CUDA self-play with persistent Rust state + CUDA inference.

Measures the production Rust adapter against the explicit PyTorch reference.
Install the release extension with python -m pip install ./native/astra.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
import torch
import yaml

from agent.env import batched_engine as BE
from agent.env.rust_engine import RustEngine
from agent.env.engine import GameEngine
from agent.env.outcomes import final_values
from agent.net import encoder as ENC
from agent.search import gumbel_mcts as search
from agent.search.config import SearchConfig
from agent.train import selfplay
from agent.train.checkpointing import load_net_from_checkpoint
from agent.train.device import configure_device
from agent.train.instrumentation import PerfCounters
from agent.train.learner import make_optimizer, step_from_buffer
from agent.train.replay_buffer import ReplayBuffer

ORIGINAL_ENCODE = ENC.encode_state


def on_device(source: BE.BatchedEngine, device: str, seed: int) -> BE.BatchedEngine:
    result = BE.BatchedEngine.__new__(BE.BatchedEngine)
    result.batch_size, result.num_players = source.batch_size, source.num_players
    result.num_factories, result.device = source.num_factories, torch.device(device)
    result._rng = torch.Generator(device=device).manual_seed(seed)
    result._game_rngs = None
    for name in BE._STATE_TENSOR_ATTRS:
        setattr(result, name, getattr(source, name).to(device).clone())
    return result


@contextmanager
def backend_context(backend: str, initial: BE.BatchedEngine, seed: int) -> Iterator[None]:
    if backend == "torch_cuda":
        template = on_device(initial, "cuda", seed)
    else:
        template = GameEngine.from_batched(initial, seed=seed)
        template.device = torch.device("cuda")
    with patch.object(selfplay, "BE", SimpleNamespace(BatchedEngine=lambda *args, **kwargs: template.clone())):
        yield


def gpu_parity() -> dict[str, int]:
    """Exact CUDA state/refill parity with common float32 draw tapes, complete games."""
    moves = np.random.default_rng(191)
    turns = finished = 0
    for n in (2, 3, 4):
        initial = BE.BatchedEngine(8, n, "cpu", seed=71 + n)
        gpu = on_device(initial, "cuda", 3)
        native = RustEngine.from_batched(initial)
        for _ in range(500):
            if gpu.ended.all():
                break
            legal = native.legal_action_mask()
            actions = [int(moves.choice(mask.nonzero().flatten().numpy())) if mask.any() else 0 for mask in legal]
            draws = moves.random((8, gpu.num_factories * 4), dtype=np.float32)
            device_draws = torch.from_numpy(draws).to("cuda")
            cursor = 0

            def draw(rows: torch.Tensor) -> torch.Tensor:
                nonlocal cursor
                result = device_draws[rows, cursor]
                cursor += 1
                return result

            gpu._draw_uniform = draw
            gpu.step(torch.tensor(actions, device="cuda"))
            native.step(actions, draw_uniforms=draws.tolist())
            cpu = native.to_batched()
            for name in BE._STATE_TENSOR_ATTRS:
                assert torch.equal(getattr(cpu, name), getattr(gpu, name).cpu()), (n, turns, name)
            g, s = native.encode_state()
            gg, ss = ORIGINAL_ENCODE(gpu)
            torch.testing.assert_close(g, gg.cpu(), rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(s, ss.cpu(), rtol=0, atol=0)
            assert torch.equal(native.legal_action_mask(), gpu.legal_action_mask().cpu())
            assert torch.equal(native.get_winners(), gpu.get_winners().cpu())
            for mode in ("binary", "score_scaled"):
                torch.testing.assert_close(native.final_values(mode), final_values(gpu, n, mode).cpu(),
                                           rtol=1e-6, atol=1e-7)
            turns += 1
        assert gpu.ended.all()
        finished += int(gpu.ended.sum())
    return {"complete_games": finished, "batch_turns": turns}


def warm_and_check(net: Any, initial: BE.BatchedEngine, sims: int) -> dict[str, float]:
    gpu = on_device(initial, "cuda", 71)
    native = GameEngine.from_batched(initial)
    native.device = torch.device("cuda")
    cfg = SearchConfig(num_simulations=sims, temperature=1.0, seed=331)
    # Initial roots cannot finish a round in one move, so no RNG differences apply.
    with torch.no_grad():
        for _ in range(3):
            expected = search.gumbel_root_act(gpu, net, search_config=cfg)
            actual = search.gumbel_root_act(native, net, search_config=cfg)
        assert torch.equal(expected[0], actual[0]), "Candidate/action mismatch before stochastic refill"
        torch.testing.assert_close(expected[1], actual[1], rtol=2e-4, atol=2e-5)
    return {"max_policy_target_error": float((expected[1] - actual[1]).abs().max())}


def trial(backend: str, net: Any, weights: dict, initial: BE.BatchedEngine,
          cfg: dict, seed: int, profile: bool, learner_steps: int) -> dict[str, Any]:
    net.load_state_dict(weights)
    net.eval()
    buffer = ReplayBuffer(initial.batch_size * cfg["max_turns"], ENC.D_GLOBAL, 10, 5, 300, 4, "cuda")
    perf = PerfCounters(True, "cuda", sync_cuda=True) if profile else None
    optimizer = make_optimizer(net, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    with backend_context(backend, initial, seed):
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = selfplay.run_selfplay(
            net, buffer, num_games=initial.batch_size, num_players=initial.num_players,
            num_sims=cfg["sims"], max_turns=cfg["max_turns"], seed=seed, device="cuda",
            dirichlet_alpha=cfg["dirichlet_alpha"], dirichlet_mix=cfg["dirichlet_mix"],
            q_scale=cfg["q_scale"], time_discount=cfg["time_discount"], reward_mode=cfg["reward_mode"],
            perf=perf,
        )
        torch.cuda.synchronize()
        selfplay_s = time.perf_counter() - start
    assert buffer.size > 0 and result["finished"] > 0
    assert torch.isfinite(buffer.policy_target[:buffer.size]).all()
    torch.testing.assert_close(buffer.policy_target[:buffer.size].sum(1),
                               torch.ones(buffer.size, device="cuda"), rtol=1e-5, atol=1e-5)
    assert torch.isfinite(buffer.value_target[:buffer.size]).all()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(learner_steps):
        metrics = step_from_buffer(net, buffer, optimizer, 256, initial.num_players,
                                   entropy_bonus=cfg["entropy_bonus"])
        assert not metrics.get("skipped", 0), metrics
        assert np.isfinite(metrics["loss"]), metrics
    torch.cuda.synchronize()
    learner_s = time.perf_counter() - start
    result.update(backend=backend, seed=seed, instrumented=profile,
                  selfplay_total_s=selfplay_s, learner_total_s=learner_s,
                  cycle_s=selfplay_s + learner_s,
                  finished_samples_per_s=result["samples_added"] / selfplay_s,
                  finished_games_per_s=result["finished"] / selfplay_s,
                  peak_gpu_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2)
    if perf:
        result["profile"] = perf.snapshot()
    return result


def engine_only(initial: BE.BatchedEngine) -> dict[str, float]:
    """Same CPU-generated midgame state, a legal move, full round handling and encoding."""
    cpu = initial.clone()
    rng = torch.Generator().manual_seed(719)
    roots = []
    for turn in range(80):
        if turn in (0, 15, 30, 45, 60, 75):
            roots.append(cpu.clone())
        mask = cpu.legal_action_mask()
        actions = torch.rand(mask.shape, generator=rng).masked_fill(~mask, -1).argmax(1)
        cpu.step(actions)
    totals = {"torch_cuda": [], "rust_cpu": []}
    for root in roots:
        native = RustEngine.from_batched(root)
        gpu = on_device(root, "cuda", 1)
        action_cpu = root.legal_action_mask().long().argmax(1)
        action_gpu = action_cpu.to("cuda")
        for repeat in range(6):
            for backend in (list(totals) if repeat % 2 == 0 else list(reversed(totals))):
                child = gpu.clone() if backend == "torch_cuda" else native.clone()
                torch.cuda.synchronize()
                start = time.perf_counter()
                if backend == "torch_cuda":
                    child.step(action_gpu)
                    ORIGINAL_ENCODE(child)
                    child.legal_action_mask()
                else:
                    child.step(action_cpu)
                    child.encode_state_with_legal()
                torch.cuda.synchronize()
                if repeat:
                    totals[backend].append(time.perf_counter() - start)
    result = {f"{k}_mean_ms": statistics.mean(v) * 1000 for k, v in totals.items()}
    result["speedup"] = result["torch_cuda_mean_ms"] / result["rust_cpu_mean_ms"]
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("agent/runs/competitive/baselines/v4_latest.pt"))
    p.add_argument("--config", type=Path, default=Path("agent/runs/competitive/experiments/policy_attn_seed20260913/config.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--games", type=int, default=1023)
    p.add_argument("--sims", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--learner-steps", type=int, default=72)
    args = p.parse_args()
    torch.set_num_threads(1)
    configure_device("cuda")
    config = yaml.safe_load(args.config.read_text())
    config.update(sims=args.sims, max_turns=args.max_turns)
    net, payload = load_net_from_checkpoint(args.checkpoint, "cpu")
    del payload
    weights = copy.deepcopy(net.state_dict())
    net.to("cuda").eval()
    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "platform": platform.platform(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "arch": net.arch, "hidden": net.hidden, "games": args.games, "sims": args.sims,
        "num_players": config["num_players"], "max_turns": args.max_turns,
        "learner_steps": args.learner_steps, "repeats": args.repeats,
        "notes": ["Actual run_selfplay, GPU replay and FP32 learner; production Rust adapter.",
                  "Paired initial states; subsequent stochastic refills use each engine's own RNG.",
                  "Transfers and metadata updates included; no CPU/GPU overlap or pinned-memory optimization.",
                  "Replay capacity is games*max_turns, not the production 1M; allocation is outside timing.",
                  "Main runs synchronize only at phase boundaries; stage profiles are separate instrumented runs.",
                  "Cycle covers neural self-play plus learner, not bot rounds, evaluation or checkpoint I/O."],
        "trials": [],
    }
    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("Validating full games against CUDA...", flush=True)
    report["gpu_parity"] = gpu_parity()
    print("GPU parity:", report["gpu_parity"], flush=True)
    initial = BE.BatchedEngine(args.games, config["num_players"], "cpu", seed=191)
    report["search_parity"] = warm_and_check(net, initial, args.sims)
    report["engine_only"] = engine_only(initial)
    print("Engine only:", report["engine_only"], flush=True)
    save()
    for repetition in range(args.repeats + 1):
        profile = repetition == args.repeats
        seed = 1719 + repetition
        initial = BE.BatchedEngine(args.games, config["num_players"], "cpu", seed=seed)
        order = ["torch_cuda", "rust_cuda"] if repetition % 2 == 0 else ["rust_cuda", "torch_cuda"]
        for backend in order:
            print(f"Starting {backend} repetition={repetition} profile={profile}", flush=True)
            result = trial(backend, net, weights, initial, config, seed, profile, args.learner_steps)
            report["trials"].append(result)
            save()
            print(json.dumps({k: v for k, v in result.items() if k != "profile"}), flush=True)
    report["summary"] = {}
    for backend in ("torch_cuda", "rust_cuda"):
        rows = [r for r in report["trials"] if r["backend"] == backend and not r["instrumented"]]
        report["summary"][backend] = {
            k: statistics.median(r[k] for r in rows)
            for k in ("selfplay_total_s", "learner_total_s", "cycle_s", "finished_samples_per_s",
                      "finished_games_per_s", "peak_gpu_allocated_mb", "samples_added", "finished")
        }
    a, b = report["summary"]["torch_cuda"], report["summary"]["rust_cuda"]
    report["speedup"] = {"selfplay_wall": a["selfplay_total_s"] / b["selfplay_total_s"],
                         "cycle_wall": a["cycle_s"] / b["cycle_s"],
                         "samples_per_s": b["finished_samples_per_s"] / a["finished_samples_per_s"],
                         "games_per_s": b["finished_games_per_s"] / a["finished_games_per_s"]}
    save()
    print(json.dumps({"summary": report["summary"], "speedup": report["speedup"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
