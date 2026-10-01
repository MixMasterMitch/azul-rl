"""Quiet local supervision of one bounded competitive training campaign.

No desktop notifications or outbound messages. Recovery and health observations
are recorded in the campaign directory; the experiment/promotion gates still apply.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any


REPO = Path(__file__).resolve().parents[2]
GIB = 1024**3


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def tail_events(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 128 * 1024))
            lines = stream.read().splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except (ValueError, UnicodeError):
            pass  # The first/last record can be partially written.
    return events


def health_snapshot(root: Path, launched_at: float, now: float) -> dict[str, Any]:
    status = read_json(root / "status.json")
    if status.get("updated_at", 0) < launched_at:
        status = {}
    snapshot: dict[str, Any] = {
        "time": now,
        "stage": status.get("stage", "starting"),
        "disk_free_gib": shutil.disk_usage(root).free / GIB,
    }
    if status.get("stage") == "evaluation":
        for key in (
            "completed_games_total",
            "unfinished",
            "search_profile",
            "label",
            "turn",
        ):
            if key in status:
                snapshot[key] = status[key]
    latest_activity = max(launched_at, status.get("updated_at", 0))
    heartbeat_path = status.get("heartbeat")
    if heartbeat_path:
        path = Path(heartbeat_path).resolve()
        # Never follow a stale/foreign campaign's heartbeat into other run files.
        if path.is_relative_to((root / "experiments").resolve()):
            heartbeat = read_json(path)
            try:
                heartbeat_time = datetime.fromisoformat(heartbeat["t"]).timestamp()
            except (KeyError, ValueError):
                heartbeat_time = 0
            latest_activity = max(latest_activity, heartbeat_time)
            snapshot.update(
                run_id=path.parent.name,
                iteration=heartbeat.get("iter"),
                phase=heartbeat.get("phase"),
                heartbeat_age_s=max(0, now - heartbeat_time),
            )
            if heartbeat_time >= launched_at:
                rows = [
                    event
                    for event in tail_events(path.parent / "events.log")
                    if event.get("event") == "learner_done"
                ]
                if rows:
                    row = rows[-1]["fields"]
                    # Nonfinite JSON numbers cannot be written to the health file.
                    for key in (
                        "loss",
                        "grad_norm",
                        "learner_steps_skipped",
                        "learner_steps_ok",
                    ):
                        value = row.get(key)
                        snapshot[key] = (
                            value
                            if isinstance(value, (int, float)) and math.isfinite(value)
                            else None
                        )
                checkpoint = path.parent / "checkpoints/latest_resume.pt"
                if checkpoint.exists():
                    stat = checkpoint.stat()
                    snapshot.update(
                        checkpoint_age_s=max(0, now - stat.st_mtime),
                        checkpoint_mib=stat.st_size / 1024**2,
                    )
    snapshot["activity_age_s"] = max(0, now - latest_activity)
    return snapshot


def recovery_reason(health: dict[str, Any], stale_seconds: float = 600) -> str | None:
    if health["disk_free_gib"] < health.get("disk_floor_gib", 1):
        return "low_disk"
    if health["activity_age_s"] > stale_seconds:
        return "stalled"
    return None


def healthy_progress(health: dict[str, Any], previous_iteration: int | None) -> bool:
    """A check qualifies after a full four-phase cycle with finite learner updates."""
    if health.get("stage") == "evaluation":
        completed = health.get("completed_games_total")
        return (
            isinstance(completed, int)
            and previous_iteration is not None
            and completed > previous_iteration
            and health.get("unfinished") == 0
            and recovery_reason(health) is None
        )
    iteration = health.get("iteration")
    return (
        health.get("stage") == "training"
        and isinstance(iteration, int)
        and previous_iteration is not None
        and iteration >= previous_iteration + 4
        and health.get("learner_steps_ok", 0) > 0
        and health.get("learner_steps_skipped") == 0
        and health.get("loss") is not None
        and health.get("grad_norm") is not None
        and health["grad_norm"] < 1000
        and recovery_reason(health) is None
    )


def reclaim_build_cache(root: Path) -> dict[str, Any]:
    """Only old dangling images and abandoned atomic temp files from this campaign."""
    before = shutil.disk_usage(root).free
    removed = []
    # Caller has reaped the training process before deleting checkpoint temporaries.
    for path in (root / "experiments").glob("*/checkpoints/*.tmp"):
        path.unlink()
        removed.append(str(path))
    result: dict[str, Any] = {"removed_temporaries": removed}
    if shutil.which("docker"):
        try:
            proc = subprocess.run(
                ["docker", "image", "prune", "--force", "--filter", "until=24h"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            result.update(
                docker_returncode=proc.returncode,
                docker_output=proc.stdout[-4000:],
                docker_error=proc.stderr[-1000:],
            )
        except subprocess.TimeoutExpired:
            result["docker_error"] = "cache cleanup timed out"
    result.update(
        free_before_gib=before / GIB, free_after_gib=shutil.disk_usage(root).free / GIB
    )
    return result


def stop_child(child: subprocess.Popen, *, grace_seconds: float = 120) -> None:
    if child.poll() is not None:
        return
    # Only this supervisor's own child process group is signalled.
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def supervise(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    # A distillation iteration includes teacher search and an atomic full-bank save.
    grace_seconds = (
        300
        if args.stage
        in {"distillation-campaign", "distillation-resume", "aux-score", "surprise"}
        else 120
    )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This campaign already has a supervisor")
        state_path = root / "supervisor.json"
        command = [
            sys.executable,
            "-u",
            "-m",
            "agent.scripts.competitive",
            args.stage,
            "--root",
            str(root),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        if args.minutes_per_arm is not None:
            command.extend(["--minutes-per-arm", str(args.minutes_per_arm)])
        if getattr(args, "learner_updates", None) is not None:
            command.extend(["--learner-updates", *map(str, args.learner_updates)])
        if getattr(args, "bot_workers", None) is not None:
            command.extend(["--bot-workers", str(args.bot_workers)])
        if getattr(args, "bounded_search", False):
            command.append("--bounded-search")
        if args.hours > 8 or args.stage in {
            "noise-ablation",
            "lr-campaign",
            "enhancements",
            "reanalysis",
            "league-campaign",
            "finetune-campaign",
            "policy-weight",
            "weight-refine",
            "capacity-campaign",
            "distillation-campaign",
            "distillation-resume",
            "aux-score",
            "surprise",
        }:
            command.extend(["--campaign-hours", str(args.hours)])
        if getattr(args, "initializer", None):
            command.extend(["--initializer", str(args.initializer)])
        state = read_json(state_path)
        if not state:
            state = {
                "started_at": time.time(),
                "deadline": time.time() + args.hours * 3600,
                "command": command,
                "attempts": 0,
                "failures_without_progress": 0,
            }
        elif state["command"] != command:
            raise ValueError("Supervisor command changed; use a separate campaign root")
        if state.get("state") in {
            "complete",
            "budget_exhausted",
            "recovery_exhausted",
            "native_failure",
        }:
            return 1 if state["state"] == "native_failure" else 0
        # A preflight budget includes setup in the user's wall-clock allowance.
        preflight = read_json(root / "preflight_budget.json")
        if preflight:
            if (
                not all(
                    isinstance(preflight.get(key), (int, float))
                    and math.isfinite(preflight[key])
                    for key in ("started_at", "deadline")
                )
                or preflight["deadline"] <= preflight["started_at"]
            ):
                raise ValueError("Invalid preflight budget")
            state["started_at"] = min(state["started_at"], preflight["started_at"])
            state["deadline"] = min(state["deadline"], preflight["deadline"])
        state["supervisor_pid"] = os.getpid()
        stopping = threading.Event()

        def request_stop(signum: int, frame: object) -> None:
            stopping.set()

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, request_stop)

        def event(name: str, **fields: Any) -> None:
            with (root / "supervisor_events.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {"time": time.time(), "event": name, **fields}, allow_nan=False
                    )
                    + "\n"
                )

        event("supervisor_started", deadline=state["deadline"], command=command)
        child: subprocess.Popen | None = None
        try:
            while (
                not stopping.is_set()
                and time.time() < state["deadline"] - grace_seconds
            ):
                # Respect a manually launched campaign instead of racing its writes.
                with (root / ".campaign.lock").open("a") as campaign_lock:
                    try:
                        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        event("waiting_for_existing_campaign")
                        stopping.wait(args.interval)
                        continue
                disk_floor = (
                    0.6 * GIB
                    if args.stage
                    in {
                        "weight-refine",
                        "capacity-campaign",
                        "distillation-campaign",
                        "distillation-resume",
                        "aux-score",
                        "surprise",
                    }
                    else 1 * GIB
                    if args.stage == "search"
                    else 1.5 * GIB
                    if args.stage
                    in {"league-campaign", "finetune-campaign", "policy-weight"}
                    else 3.25 * GIB
                )
                if shutil.disk_usage(root).free < disk_floor + 0.25 * GIB:
                    event("disk_cleanup", **reclaim_build_cache(root))
                    if shutil.disk_usage(root).free < disk_floor:
                        state.update(state="waiting_for_disk", updated_at=time.time())
                        write_json(state_path, state)
                        stopping.wait(300)
                        continue
                launched_at = time.time()
                state["attempts"] += 1
                with (root / f"{args.stage}.log").open("a") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=REPO,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                state.update(
                    state="running", child_pid=child.pid, updated_at=launched_at
                )
                write_json(state_path, state)
                (root / f"{args.stage}.pid").write_text(str(child.pid) + "\n")
                write_json(
                    root / f"{args.stage}_launch.json",
                    {
                        "pid": child.pid,
                        "started_at": datetime.fromtimestamp(
                            launched_at, timezone.utc
                        ).isoformat(),
                        "command": command,
                        "supervisor_pid": os.getpid(),
                        "attempt": state["attempts"],
                        "deadline": state["deadline"],
                    },
                )
                event("child_started", pid=child.pid, attempt=state["attempts"])
                next_check = 0.0
                interval = args.interval
                healthy_checks = 0
                previous_iteration = 0
                previous_run = None
                while child.poll() is None:
                    reason = None
                    # A new arm needs fresh stability checks even when the prior
                    # arm had already graduated to a 30-minute interval.
                    current_status = read_json(root / "status.json")
                    current_run = (
                        current_status.get("run_id")
                        if current_status.get("stage") in {"training", "evaluation"}
                        else None
                    )
                    if current_run is not None and current_run != previous_run:
                        if previous_run is not None:
                            event(
                                "monitor_arm_changed",
                                run_id=current_run,
                                interval_seconds=args.interval,
                            )
                        previous_run = current_run
                        healthy_checks, previous_iteration = 0, 0
                        interval, next_check = args.interval, 0.0
                    if time.time() >= next_check:
                        health = health_snapshot(root, launched_at, time.time())
                        # This campaign compacts retained states and retires its
                        # completed scratch. Atomic writers still enforce 256 MiB.
                        if args.stage in {
                            "weight-refine",
                            "capacity-campaign",
                            "distillation-campaign",
                            "distillation-resume",
                            "aux-score",
                            "surprise",
                        }:
                            health["disk_floor_gib"] = 0.3
                        if healthy_progress(health, previous_iteration):
                            healthy_checks += 1
                        else:
                            healthy_checks = 0
                        previous_iteration = health.get(
                            "iteration",
                            health.get("completed_games_total", previous_iteration),
                        )
                        new_interval = (
                            args.steady_interval
                            if healthy_checks >= args.stable_checks
                            else args.interval
                        )
                        if new_interval != interval:
                            event(
                                "monitor_interval_changed",
                                interval_seconds=new_interval,
                                healthy_checks=healthy_checks,
                            )
                        interval = new_interval
                        next_check = time.time() + interval
                        health.update(
                            healthy_checks=healthy_checks,
                            interval_seconds=interval,
                            next_check_at=next_check,
                        )
                        write_json(root / "health.json", health)
                        event("health_check", **health)
                        reason = recovery_reason(health, args.stale_seconds)
                    if (
                        reason
                        or stopping.is_set()
                        or time.time() >= state["deadline"] - grace_seconds
                    ):
                        event(
                            "child_stop_requested",
                            reason=reason
                            or ("manual_stop" if stopping.is_set() else "budget"),
                        )
                        stop_child(child, grace_seconds=grace_seconds)
                        break
                    # Process exit wakes this wait immediately; full health checks
                    # follow the requested 5-minute / 30-minute schedule.
                    try:
                        child.wait(
                            timeout=max(
                                0.01,
                                min(
                                    5,
                                    next_check - time.time(),
                                    state["deadline"] - grace_seconds - time.time(),
                                ),
                            )
                        )
                    except subprocess.TimeoutExpired:
                        pass
                status = read_json(root / "status.json")
                returncode = child.wait()
                child = None
                event("child_exited", returncode=returncode, campaign_status=status)
                if stopping.is_set():
                    state["state"] = "stopped"
                    break
                if args.stage in {
                    "finetune-campaign",
                    "policy-weight",
                    "weight-refine",
                    "capacity-campaign",
                    "distillation-campaign",
                    "distillation-resume",
                    "aux-score",
                    "surprise",
                } and returncode in {-signal.SIGABRT, -signal.SIGSEGV, -signal.SIGBUS}:
                    state["state"] = "native_failure"
                    event(
                        "native_failure",
                        returncode=returncode,
                        reason="Native-memory failures require investigation; do not retry this campaign automatically.",
                    )
                    break
                if (
                    returncode == 0
                    and status.get("stage")
                    == args.stage.replace("-", "_") + "_complete"
                ):
                    state["state"] = "complete"
                    break
                # Advancing the durable checkpoint resets the retry counter.
                durable = (
                    (root / "evaluations").glob("*/batch_*.json")
                    if args.stage == "search"
                    else (root / "experiments").glob("*/checkpoints/latest_resume.pt")
                )
                if args.stage in {
                    "lr-campaign",
                    "enhancements",
                    "reanalysis",
                    "league-campaign",
                    "finetune-campaign",
                    "policy-weight",
                    "weight-refine",
                    "capacity-campaign",
                    "distillation-campaign",
                    "distillation-resume",
                    "aux-score",
                    "surprise",
                }:
                    durable = [
                        *(root / "experiments").glob("*/checkpoints/latest_resume.pt"),
                        *(root / "experiments").glob("*/milestones/*.json"),
                        *(root / "evaluations").glob("*.json"),
                    ]
                progress = sorted(
                    (str(path), path.stat().st_mtime_ns) for path in durable
                )
                progress = [list(item) for item in progress]
                if progress == state.get("checkpoint_progress"):
                    state["failures_without_progress"] += 1
                else:
                    state["failures_without_progress"] = 1
                    state["checkpoint_progress"] = progress
                if state["failures_without_progress"] >= args.max_retries:
                    state["state"] = "recovery_exhausted"
                    event("recovery_exhausted", error=status.get("error"))
                    break
                state.update(state="recovering", updated_at=time.time())
                write_json(state_path, state)
                # Same experiment, settings, optimizer, replay and RNG are restored.
                # Invalid numerical updates fail before replacing the durable file.
                stopping.wait(
                    min(30 * 2 ** (state["failures_without_progress"] - 1), 300)
                )
            else:
                state["state"] = "stopped" if stopping.is_set() else "budget_exhausted"
        finally:
            if child is not None:
                stop_child(child, grace_seconds=grace_seconds)
            state.update(updated_at=time.time(), child_pid=None)
            write_json(state_path, state)
            event("supervisor_finished", state=state["state"])
    return 1 if state.get("state") == "native_failure" else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "agent/runs/competitive")
    parser.add_argument(
        "--stage",
        choices=[
            "policy",
            "learner",
            "search",
            "tree-training",
            "tree-distill",
            "noise-ablation",
            "lr-campaign",
            "enhancements",
            "reanalysis",
            "league-campaign",
            "finetune-campaign",
            "policy-weight",
            "weight-refine",
            "capacity-campaign",
            "distillation-campaign",
            "distillation-resume",
            "aux-score",
            "surprise",
        ],
        default="policy",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--minutes-per-arm", type=float)
    parser.add_argument("--learner-updates", type=int, nargs="+")
    parser.add_argument("--bot-workers", type=int, choices=range(1, 9))
    parser.add_argument("--initializer")
    parser.add_argument("--bounded-search", action="store_true")
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--steady-interval", type=float, default=1800)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--stale-seconds", type=float, default=600)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()
    if (
        not 0 < args.hours <= 24
        or args.interval <= 0
        or args.steady_interval <= 0
        or args.stable_checks < 1
        or args.stale_seconds <= 0
        or args.max_retries < 1
    ):
        parser.error(
            "Use a positive budget of at most 24 hours and positive monitoring/retry settings"
        )
    raise SystemExit(supervise(args))


if __name__ == "__main__":
    main()
