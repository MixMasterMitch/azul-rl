"""Find max replay buffer capacity before CUDA OOM for a given self-play batch size."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from datetime import datetime, timezone

import torch

from agent.env import actions as A
from agent.env import engine as BE
from agent.net import encoder as ENC
from agent.net.model import AzulNet
from agent.search import gumbel_mcts as G
from agent.train.learner import make_optimizer, step_from_buffer
from agent.train.replay_buffer import ReplayBuffer


def _bytes_per_sample() -> int:
    return (
        ENC.D_GLOBAL * 4
        + ENC.NUM_SOURCES * ENC.D_SOURCE * 4
        + A.NUM_ACTIONS * 1
        + A.NUM_ACTIONS * 4
        + BE.MAX_PLAYERS * 4
    )


def _gpu_mem_snapshot(device: torch.device) -> str:
    if not torch.cuda.is_available():
        return "cuda=n/a"
    torch.cuda.synchronize(device)
    alloc = torch.cuda.memory_allocated(device) / (1024**2)
    reserved = torch.cuda.memory_reserved(device) / (1024**2)
    peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    return f"alloc={alloc:.0f}MB reserved={reserved:.0f}MB peak={peak:.0f}MB"


def try_setup(
    log: logging.Logger,
    *,
    num_games: int,
    capacity: int,
    num_sims: int,
    learner_batch: int,
    device: str,
    skip_mcts_child_expand: bool = False,
) -> tuple[bool, str]:
    """Return (ok, detail). Simulates peak training memory for one iteration."""
    if not torch.cuda.is_available():
        return False, "cuda unavailable"

    torch.cuda.empty_cache()
    gc.collect()
    dev = torch.device(device)
    torch.cuda.reset_peak_memory_stats(dev)

    engine = net = optim = buffer = scaler = None
    t0 = time.monotonic()

    try:
        log.info("  [1/5] BatchedEngine(batch=%d)...", num_games)
        engine = BE.BatchedEngine(num_games, 2, dev, seed=0)
        log.info("        engine OK — %s", _gpu_mem_snapshot(dev))

        log.info("  [2/5] AzulNet + optimizer + ReplayBuffer(capacity=%s)...", f"{capacity:,}")
        net = AzulNet(hidden=256, arch="attn").to(dev)
        net.train()
        optim = make_optimizer(net)
        scaler = torch.amp.GradScaler("cuda")
        buffer = ReplayBuffer(
            capacity=capacity,
            d_global=ENC.D_GLOBAL,
            n_sources=ENC.NUM_SOURCES,
            d_source=ENC.D_SOURCE,
            num_actions=A.NUM_ACTIONS,
            max_players=BE.MAX_PLAYERS,
            device=dev,
        )
        log.info("        buffer OK — %s", _gpu_mem_snapshot(dev))

        global_feat, source_feat = ENC.encode_state(engine)
        legal_mask = engine.legal_action_mask()
        if skip_mcts_child_expand:
            log.info("  [3/5] root forward only (skip MCTS child expand)...")
            with torch.no_grad():
                logits, _ = net(global_feat, source_feat, legal_mask, 2)
            actions = logits.argmax(dim=-1)
            policy = torch.softmax(logits, dim=-1)
            engine.step(actions)
        else:
            log.info("  [3/5] MCTS self-play step (sims=%d)...", num_sims)
            actions, policy = G.gumbel_root_act(
                engine,
                net,
                num_sims=num_sims,
                dirichlet_alpha=0.0,
                dirichlet_mix=0.0,
                q_scale=10.0,
                precomputed=(global_feat, source_feat, legal_mask),
            )
            engine.step(actions)
        log.info("        step OK — %s", _gpu_mem_snapshot(dev))

        n_add = min(capacity, max(learner_batch, num_games))
        log.info("  [4/5] buffer.add(%d samples)...", n_add)
        buffer.add(
            global_feat[:1].expand(n_add, -1),
            source_feat[:1].expand(n_add, -1, -1),
            legal_mask[:1].expand(n_add, -1),
            policy[:1].expand(n_add, -1),
            torch.zeros((n_add, BE.MAX_PLAYERS), device=dev),
        )
        log.info("        add OK — %s", _gpu_mem_snapshot(dev))

        if buffer.size >= learner_batch:
            log.info("  [5/5] learner step (batch=%d)...", learner_batch)
            step_from_buffer(
                net,
                buffer,
                optim,
                batch_size=learner_batch,
                num_players=2,
                grad_scaler=scaler,
            )
            log.info("        learner OK — %s", _gpu_mem_snapshot(dev))
        else:
            log.info("  [5/5] learner skipped (buffer too small for batch)")

        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0
        peak_mb = torch.cuda.max_memory_allocated(dev) / (1024**2)
        return True, f"peak_alloc={peak_mb:.0f}MB elapsed={elapsed:.1f}s"
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        elapsed = time.monotonic() - t0
        return False, f"FAIL after {elapsed:.1f}s: {str(exc)[:100]}"
    finally:
        for obj in (engine, net, optim, buffer, scaler):
            del obj
        torch.cuda.empty_cache()
        gc.collect()


# (selfplay_games, selfplay_sims) — product must be <= 65535 for attn MCTS child batch
DEFAULT_SUITE: list[tuple[int, int]] = [
    (1024, 63),
    (2048, 31),
    (4096, 15),
]

MHA_CHILD_BATCH_LIMIT = 65535


def search_max_capacity(
    log: logging.Logger,
    *,
    num_games: int,
    num_sims: int,
    learner_batch: int,
    device: str,
    hi: int,
    max_probes: int = 10,
    skip_mcts_child_expand: bool = False,
) -> tuple[int, int]:
    """Find largest capacity <= hi. Try hi first; at most max_probes attempts total."""
    ok_at = 0
    fail_at = hi + 1
    probes_left = max_probes

    def _probe(capacity: int, label: str) -> bool:
        nonlocal probes_left, ok_at, fail_at
        probes_left -= 1
        log.info(
            "--- %s (probe %d/%d): capacity=%s ---",
            label,
            max_probes - probes_left,
            max_probes,
            f"{capacity:,}",
        )
        ok, detail = try_setup(
            log,
            num_games=num_games,
            capacity=capacity,
            num_sims=num_sims,
            learner_batch=learner_batch,
            device=device,
            skip_mcts_child_expand=skip_mcts_child_expand,
        )
        status = "OK" if ok else "FAIL"
        log.info(">>> result: %s — %s", status, detail)
        if ok:
            ok_at = max(ok_at, capacity)
        else:
            fail_at = min(fail_at, capacity)
        return ok

    log.info(
        "Search: try %s first, then up to %d total probes (binary search if needed)",
        f"{hi:,}",
        max_probes,
    )

    if _probe(hi, "quick check at hi"):
        log.info("Capacity %s OK — no further probes needed", f"{hi:,}")
        return hi, hi + 1

    lo = 0
    hi_search = hi - 1
    while lo <= hi_search and probes_left > 0:
        mid = (lo + hi_search) // 2
        log.info("Binary bracket: lo=%s hi=%s", f"{lo:,}", f"{hi_search:,}")
        if _probe(mid, "binary step"):
            lo = mid + 1
        else:
            hi_search = mid - 1

    if probes_left == 0 and lo <= hi_search:
        log.warning(
            "Probe budget exhausted (%d); best so far=%s (bracket %s..%s)",
            max_probes,
            f"{ok_at:,}",
            f"{lo:,}",
            f"{hi_search:,}",
        )

    return ok_at, fail_at


def run_probe(
    log: logging.Logger,
    *,
    num_games: int,
    num_sims: int,
    learner_batch: int,
    device: str,
    hi: int,
    max_probes: int = 10,
    skip_mcts_child_expand: bool = False,
) -> dict[str, int | float | str]:
    bps = _bytes_per_sample()
    log.info("=" * 60)
    log.info(
        "PROBE START games=%d sims=%d learner_batch=%d hi=%s",
        num_games,
        num_sims,
        learner_batch,
        f"{hi:,}",
    )
    log.info("Per-sample buffer storage: %.2f KiB (%d bytes)", bps / 1024, bps)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s (%.1f GiB total)", torch.cuda.get_device_name(0), props.total_memory / 2**30)
    log.info("=" * 60)

    t_probe = time.monotonic()
    child_batch = num_games * num_sims
    if child_batch > MHA_CHILD_BATCH_LIMIT and not skip_mcts_child_expand:
        log.warning(
            "batch*sims=%d exceeds PyTorch MHA limit (%d); expect RuntimeError",
            child_batch,
            MHA_CHILD_BATCH_LIMIT,
        )

    max_ok, first_fail = search_max_capacity(
        log,
        num_games=num_games,
        num_sims=num_sims,
        learner_batch=learner_batch,
        device=device,
        hi=hi,
        max_probes=max_probes,
        skip_mcts_child_expand=skip_mcts_child_expand,
    )
    elapsed = time.monotonic() - t_probe
    est_gb = max_ok * bps / (1024**3)

    log.info("=" * 60)
    log.info("PROBE DONE games=%d wall_s=%.1f (%.1f min)", num_games, elapsed, elapsed / 60)
    mode = "buffer-only" if skip_mcts_child_expand else "full (MCTS+learner)"
    log.info("Mode: %s", mode)
    log.info("Max OK capacity: %s samples (~%.2f GiB buffer only)", f"{max_ok:,}", est_gb)
    if first_fail <= hi:
        log.info("First failure at: %s", f"{first_fail:,}")
    log.info("=" * 60)

    return {
        "num_games": num_games,
        "num_sims": num_sims,
        "max_ok": max_ok,
        "first_fail": first_fail,
        "buffer_gib": est_gb,
        "wall_s": elapsed,
        "at_hi": max_ok >= hi,
    }


def _configure_logging(log_path: str | None) -> logging.Logger:
    log = logging.getLogger("probe_replay_oom")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_path:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
        log.info("Logging to %s", log_path)
    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run default suite: 1024x63, 2048x31, 4096x15",
    )
    parser.add_argument("--num-games", type=int, default=0, help="Single preset (with --num-sims)")
    parser.add_argument("--num-sims", type=int, default=32)
    parser.add_argument("--learner-batch", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hi", type=int, default=1_000_000, help="Target capacity (try first)")
    parser.add_argument("--max-probes", type=int, default=10, help="Max try_setup calls per preset")
    parser.add_argument("--log-file", type=str, default="/tmp/replay_oom_probe.log")
    parser.add_argument(
        "--buffer-only",
        action="store_true",
        help="Skip MCTS child expand (buffer sizing only)",
    )
    parser.add_argument("--stdout-only", action="store_true", help="Do not write log file")
    args = parser.parse_args()

    log_path = None if args.stdout_only else args.log_file
    log = _configure_logging(log_path)
    log.info("Run started at %s", datetime.now(timezone.utc).isoformat())

    if args.suite or args.num_games == 0:
        run_list: list[tuple[int, int]] = DEFAULT_SUITE
    else:
        run_list = [(args.num_games, args.num_sims)]

    summaries: list[dict] = []
    for num_games, num_sims in run_list:
        summaries.append(
            run_probe(
                log,
                num_games=num_games,
                num_sims=num_sims,
                learner_batch=args.learner_batch,
                device=args.device,
                hi=args.hi,
                max_probes=args.max_probes,
                skip_mcts_child_expand=args.buffer_only,
            )
        )

    log.info("SUMMARY")
    for s in summaries:
        hi_note = " (1M OK)" if s["at_hi"] else ""
        log.info(
            "  %dx%d: max=%s (~%.2f GiB)%s first_fail=%s time=%.1fm",
            s["num_games"],
            s["num_sims"],
            f"{s['max_ok']:,}",
            s["buffer_gib"],
            hi_note,
            f"{s['first_fail']:,}" if not s["at_hi"] else "n/a",
            s["wall_s"] / 60,
        )


if __name__ == "__main__":
    main()
