"""Reproducible transition, boundary, and complete-move Astra benchmarks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable

import torch

# Match the authoritative reference used in Astra's differential experiments.
from agent.env.batched_engine import BatchedEngine
from agent.eval import astra_reference
from agent.eval.heuristic_astra import AstraConfig, HeuristicAstraBot, native_module, production_config, snapshot
from agent.eval.heuristic_opus import HeuristicOpusBot


def measure(call: Callable[[], Any], repeats: int) -> float:
    for _ in range(10):
        call()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--include-play-engine", action="store_true",
                        help="Also measure the current GameEngine adapter on identical public positions")
    args = parser.parse_args()
    torch.set_num_threads(1)
    native = native_module()
    cfg = AstraConfig.from_dict(json.loads(args.config.read_text())) if args.config else None
    rows = []
    resolutions = []
    for n in (2, 3, 4):
        e = BatchedEngine(1, n, seed=907 + n)
        bot = HeuristicOpusBot(seed=17)
        samples: dict[str, BatchedEngine] = {}
        boundaries: dict[str, list[int]] = {}
        for _ in range(200):
            if e.ended[0]:
                break
            wall_count = int(e.wall.sum())
            phase = "late_round" if not e.factory_tiles.any() else "opening" if wall_count == 0 else "middle"
            if phase not in samples:
                samples[phase] = e.clone()
            e.step(torch.tensor([bot.select_action(e, 0)]), finalize_round=False)
            if not e.factory_tiles.any() and not e.center_tiles.any():
                progress = int(e.wall.sum(-1).max())
                phase = "opening" if wall_count == 0 else "late_game" if progress >= 4 else "middle"
                boundaries.setdefault(phase, snapshot(e, 0))
                e.finalize_round()
        for phase, engine in samples.items():
            data = snapshot(engine, 0)
            action = native.legal_actions(data)[0]
            expected = astra_reference.transition(data, action)
            assert native.transition(data, action) == expected
            python_s = measure(lambda: astra_reference.transition(data, action), 3000)
            rust_s = measure(lambda: native.transition(data, action), 3000)
            choose = HeuristicAstraBot(seed=17, config=cfg)
            timings = []
            native_timings = []
            for _ in range(5):
                t = time.perf_counter()
                analysis = choose.analyze(engine, 0)
                timings.append(time.perf_counter() - t)
                native_timings.append(analysis["elapsed_s"])
            row = {"players": n, "phase": phase, "snapshot": data, "action": action,
                         "python_transition_us": python_s * 1e6,
                         "rust_transition_with_binding_us": rust_s * 1e6,
                         "transition_speedup": python_s / rust_s,
                         "snapshot_us": measure(lambda: snapshot(engine, 0), 500) * 1e6,
                         "binding_integer_roundtrip_us": measure(lambda: native._binding_roundtrip(data), 3000) * 1e6,
                         "parse_and_legal_with_binding_us": measure(lambda: native.legal_actions(data), 3000) * 1e6,
                         "move_mean_ms": statistics.mean(timings) * 1000,
                         "move_samples_ms": [t * 1000 for t in timings],
                         "native_search_ms": statistics.mean(native_timings) * 1000,
                         "nodes": analysis["nodes"], "depth": analysis["depth"],
                         "evaluations": analysis["evaluations"], "tt_hits": analysis["tt_hits"],
                         "evaluation_cache_hits": analysis["evaluation_cache_hits"],
                         "cutoff_reason": analysis["cutoff_reason"], "solved": analysis["solved"],
                         "selected_action": analysis["action"], "principal_variation": analysis["principal_variation"],
                         "values": analysis["values"]}
            if args.include_play_engine:
                from agent.env.engine import GameEngine

                play_engine = GameEngine.from_batched(engine)
                assert snapshot(play_engine, 0) == data
                play_timings = []
                play_native_timings = []
                for _ in range(5):
                    t = time.perf_counter()
                    play_analysis = choose.analyze(play_engine, 0)
                    play_timings.append(time.perf_counter() - t)
                    play_native_timings.append(play_analysis["elapsed_s"])
                if analysis["cutoff_reason"] != "time" and play_analysis["cutoff_reason"] != "time":
                    assert play_analysis["action"] == analysis["action"]
                row.update(play_snapshot_us=measure(lambda: snapshot(play_engine, 0), 500) * 1e6,
                           play_move_mean_ms=statistics.mean(play_timings) * 1000,
                           play_move_samples_ms=[t * 1000 for t in play_timings],
                           play_native_search_ms=statistics.mean(play_native_timings) * 1000,
                           play_selected_action=play_analysis["action"])
            rows.append(row)
        for phase, data in boundaries.items():
            assert native.resolve_round(data) == astra_reference.resolve_round(data)
            python_s = measure(lambda: astra_reference.resolve_round(data), 3000)
            rust_s = measure(lambda: native.resolve_round(data), 3000)
            resolutions.append({"players": n, "phase": phase, "snapshot": data,
                                "python_resolution_us": python_s * 1e6,
                                "rust_resolution_with_binding_us": rust_s * 1e6,
                                "resolution_speedup": python_s / rust_s})
    report = {"platform": platform.platform(), "processor": platform.processor(),
              "python": platform.python_version(), "torch": torch.__version__, "native_state_bytes": native.STATE_BYTES,
              "configs_per_pc": {str(n): asdict(cfg or production_config(n)) for n in (2, 3, 4)},
              "positions": rows, "round_resolutions": resolutions,
              "notes": "Transitions compare identical public-state contents and include legality generation. Round resolution uses actual pre-refill game boundaries and checks equal results. Native operation times include binding and validation. Binding roundtrip measures integer-array transport/allocation only, not the full search result dictionary. No Rust-versus-Python search speedup is inferred."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
