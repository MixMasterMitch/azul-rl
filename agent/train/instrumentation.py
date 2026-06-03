"""Lightweight training instrumentation helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import resource
import subprocess
import time
from typing import Iterator

import torch


@dataclass
class _Timing:
    total_s: float = 0.0
    count: int = 0
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.total_s += elapsed_s
        self.count += 1
        self.max_s = max(self.max_s, elapsed_s)


class PerfCounters:
    """Aggregate stage timings and counters for one training iteration."""

    def __init__(
        self,
        enabled: bool,
        device: torch.device | str = "cpu",
        sync_cuda: bool = False,
    ):
        self.enabled = enabled
        self.device = torch.device(device)
        self.sync_cuda = sync_cuda and self.device.type == "cuda" and torch.cuda.is_available()
        self._timings: dict[str, _Timing] = {}
        self._counters: dict[str, float] = {}

    def _sync(self) -> None:
        if self.sync_cuda:
            torch.cuda.synchronize(self.device)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed_s = time.perf_counter() - started
            self._timings.setdefault(name, _Timing()).add(elapsed_s)

    def add_count(self, name: str, value: float = 1.0) -> None:
        if self.enabled:
            self._counters[name] = self._counters.get(name, 0.0) + float(value)

    def snapshot(self, prefix: str = "profile") -> dict[str, float | int | bool]:
        if not self.enabled:
            return {}

        out: dict[str, float | int | bool] = {
            f"{prefix}_sync_cuda": self.sync_cuda,
        }
        for name, timing in sorted(self._timings.items()):
            key = f"{prefix}_{name}"
            out[f"{key}_s"] = round(timing.total_s, 6)
            out[f"{key}_n"] = timing.count
            out[f"{key}_avg_ms"] = round((timing.total_s / max(timing.count, 1)) * 1000.0, 3)
            out[f"{key}_max_ms"] = round(timing.max_s * 1000.0, 3)
        for name, value in sorted(self._counters.items()):
            out[f"{prefix}_{name}"] = int(value) if value.is_integer() else round(value, 6)
        return out


@dataclass(frozen=True)
class ResourceSnapshot:
    wall_s: float
    process_cpu_s: float
    child_cpu_s: float
    read_bytes: int
    write_bytes: int


def _self_io_bytes() -> tuple[int, int]:
    try:
        read_bytes = 0
        write_bytes = 0
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    read_bytes = int(line.split()[1])
                elif line.startswith("write_bytes:"):
                    write_bytes = int(line.split()[1])
        return read_bytes, write_bytes
    except OSError:
        return 0, 0


def resource_snapshot() -> ResourceSnapshot:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    read_bytes, write_bytes = _self_io_bytes()
    return ResourceSnapshot(
        wall_s=time.perf_counter(),
        process_cpu_s=usage.ru_utime + usage.ru_stime,
        child_cpu_s=child_usage.ru_utime + child_usage.ru_stime,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
    )


def resource_delta(
    start: ResourceSnapshot,
    end: ResourceSnapshot,
    prefix: str = "profile",
) -> dict[str, float | int]:
    wall_s = max(end.wall_s - start.wall_s, 1e-9)
    proc_cpu_s = max(end.process_cpu_s - start.process_cpu_s, 0.0)
    child_cpu_s = max(end.child_cpu_s - start.child_cpu_s, 0.0)
    cpu_count = os.cpu_count() or 1
    out: dict[str, float | int] = {
        f"{prefix}_wall_s": round(wall_s, 6),
        f"{prefix}_process_cpu_s": round(proc_cpu_s, 6),
        f"{prefix}_process_cpu_pct_one_core": round((proc_cpu_s / wall_s) * 100.0, 2),
        f"{prefix}_process_cpu_pct_all_cores": round((proc_cpu_s / wall_s) * 100.0 / cpu_count, 2),
        f"{prefix}_child_cpu_s": round(child_cpu_s, 6),
        f"{prefix}_read_mb": round(max(end.read_bytes - start.read_bytes, 0) / (1024**2), 3),
        f"{prefix}_write_mb": round(max(end.write_bytes - start.write_bytes, 0) / (1024**2), 3),
        f"{prefix}_cpu_count": cpu_count,
    }
    try:
        load1, load5, load15 = os.getloadavg()
        out[f"{prefix}_load1"] = round(load1, 3)
        out[f"{prefix}_load5"] = round(load5, 3)
        out[f"{prefix}_load15"] = round(load15, 3)
    except OSError:
        pass
    return out


def nvidia_smi_snapshot(prefix: str = "profile_gpu") -> dict[str, float]:
    query = "utilization.gpu,utilization.memory,power.draw,temperature.gpu"
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    if not first:
        return {}
    try:
        gpu_util, mem_util, power_w, temp_c = [float(x.strip()) for x in first.split(",")[:4]]
    except ValueError:
        return {}
    return {
        f"{prefix}_util_pct": round(gpu_util, 2),
        f"{prefix}_mem_util_pct": round(mem_util, 2),
        f"{prefix}_power_w": round(power_w, 2),
        f"{prefix}_temp_c": round(temp_c, 2),
    }


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@contextmanager
def maybe_time(perf: PerfCounters | None, name: str) -> Iterator[None]:
    if perf is None:
        yield
        return
    with perf.time(name):
        yield


def reset_cuda_peak_memory(device: torch.device | str) -> None:
    device_t = torch.device(device)
    if device_t.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_t)


def cuda_memory_snapshot(
    device: torch.device | str,
    prefix: str = "profile_cuda",
) -> dict[str, float]:
    device_t = torch.device(device)
    if device_t.type != "cuda" or not torch.cuda.is_available():
        return {}

    torch.cuda.synchronize(device_t)
    return {
        f"{prefix}_allocated_mb": round(torch.cuda.memory_allocated(device_t) / (1024**2), 2),
        f"{prefix}_reserved_mb": round(torch.cuda.memory_reserved(device_t) / (1024**2), 2),
        f"{prefix}_peak_allocated_mb": round(
            torch.cuda.max_memory_allocated(device_t) / (1024**2), 2
        ),
        f"{prefix}_peak_reserved_mb": round(
            torch.cuda.max_memory_reserved(device_t) / (1024**2), 2
        ),
    }
