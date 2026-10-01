"""Process-parallel tree traversal with one centralized inference batcher.

Each worker owns independent CPU/Rust trees.  Neural leaves are returned to the
parent, combined across workers, and evaluated by the one authoritative model
on its inference device.  Workers never initialize CUDA or retain model state.
"""

from __future__ import annotations

import atexit
from dataclasses import replace
import multiprocessing as mp
from queue import Empty
import signal
import threading
import time
import traceback
from typing import Any

import numpy as np
import torch

from ..env.engine import GameEngine
from ..eval.inference import InferenceModel
from ..net.model import AzulNet
from ..train.instrumentation import PerfCounters
from .config import SearchConfig


class _RemoteModel:
    def __init__(self, worker: int, requests: Any, responses: Any) -> None:
        self.worker = worker
        self.requests = requests
        self.responses = responses
        self.task = -1
        self.sequence = 0

    def bind(self, task: int) -> None:
        self.task = task
        self.sequence = 0

    def __call__(
        self,
        global_feat: torch.Tensor,
        source_feat: torch.Tensor,
        legal: torch.Tensor,
        num_players: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        request_id = (self.task, self.sequence)
        self.sequence += 1
        self.requests.put(
            (
                self.worker,
                request_id,
                global_feat.numpy(),
                source_feat.numpy(),
                legal.numpy(),
                num_players,
            )
        )
        response_id, policy, value = self.responses.get()
        if response_id != request_id:
            raise RuntimeError(
                f"parallel inference response mismatch: {response_id} != {request_id}"
            )
        return torch.from_numpy(policy), torch.from_numpy(value)


def _worker_main(
    worker: int, tasks: Any, requests: Any, responses: Any, results: Any
) -> None:
    # The training parent owns iteration-boundary shutdown and sends sentinels
    # after its current search returns. Supervisors signal the whole process
    # group, so workers must not exit underneath an in-flight parent search.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    torch.set_num_threads(1)
    model = _RemoteModel(worker, requests, responses)
    while True:
        task = tasks.get()
        if task is None:
            return
        task_id, state, config = task
        try:
            model.bind(task_id)
            engine = GameEngine.from_state_dict(state, device="cpu")
            if config.tree_core == "rust":
                from .native_tree import native_tree_act

                actions, policies = native_tree_act(engine, model, config)
            else:
                from .tree import gumbel_tree_act

                actions, policies = gumbel_tree_act(engine, model, config)
            results.put((worker, task_id, actions.numpy(), policies.numpy(), None))
        except BaseException:
            results.put((worker, task_id, None, None, traceback.format_exc()))


class ParallelTreePool:
    """Persistent workers and synchronous centralized inference coordination."""

    def __init__(
        self, workers: int, inference_batch_size: int, inference_wait_ms: float
    ) -> None:
        self.workers = workers
        self.inference_batch_size = inference_batch_size
        self.inference_wait_s = inference_wait_ms / 1000.0
        self.context = mp.get_context("spawn")
        self.requests = self.context.Queue()
        self.results = self.context.Queue()
        self.tasks = [self.context.Queue() for _ in range(workers)]
        self.responses = [self.context.Queue() for _ in range(workers)]
        self.processes = [
            self.context.Process(
                target=_worker_main,
                args=(i, self.tasks[i], self.requests, self.responses[i], self.results),
                name=f"azul-tree-{i}",
                daemon=True,
            )
            for i in range(workers)
        ]
        for process in self.processes:
            process.start()
        self._task_id = 0
        self._call_lock = threading.Lock()

    def _assert_alive(self) -> None:
        dead = [(p.name, p.exitcode) for p in self.processes if not p.is_alive()]
        if dead:
            raise RuntimeError(f"parallel tree worker exited: {dead}")

    def _infer(self, first: tuple, model: Any) -> tuple[int, int]:
        requests = [first]
        rows = len(first[2])
        deadline = time.monotonic() + self.inference_wait_s
        while rows < self.inference_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self.requests.get(timeout=remaining)
            except Empty:
                break
            requests.append(item)
            rows += len(item[2])
        requests.sort(key=lambda item: item[0])
        global_feat = torch.from_numpy(np.concatenate([item[2] for item in requests]))
        source_feat = torch.from_numpy(np.concatenate([item[3] for item in requests]))
        legal = torch.from_numpy(np.concatenate([item[4] for item in requests]))
        player_counts = {int(item[5]) for item in requests}
        if len(player_counts) != 1:
            raise RuntimeError("one inference batch cannot mix player counts")
        policy, value = model(global_feat, source_feat, legal, player_counts.pop())
        policy_np = policy.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()
        offset = 0
        for worker, request_id, g, *_ in requests:
            stop = offset + len(g)
            self.responses[worker].put(
                (request_id, policy_np[offset:stop], value_np[offset:stop])
            )
            offset = stop
        return len(requests), rows

    def search(
        self, engine: GameEngine, net: Any, config: SearchConfig
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        with self._call_lock:
            self._assert_alive()
            self._task_id += 1
            task_id = self._task_id
            worker_count = min(self.workers, engine.batch_size)
            base_seed = (
                config.seed
                if config.seed is not None
                else int(torch.randint(2**31, (), device="cpu").item())
            )
            bounds = np.linspace(0, engine.batch_size, worker_count + 1, dtype=int)
            for worker, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
                indices = torch.arange(int(start), int(stop), device=engine.device)
                state = engine.index_select(indices).state_dict()
                worker_config = replace(
                    config,
                    cpu_workers=1,
                    seed=(base_seed + int(start) * 1_000_003) % (1 << 63),
                )
                self.tasks[worker].put((task_id, state, worker_config))

            outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            inference_batches = inference_requests = inference_rows = 0
            while len(outputs) < worker_count:
                self._assert_alive()
                try:
                    worker, returned_task, actions, policies, error = (
                        self.results.get_nowait()
                    )
                except Empty:
                    try:
                        request = self.requests.get(timeout=0.05)
                    except Empty:
                        continue
                    request_count, row_count = self._infer(request, net)
                    inference_batches += 1
                    inference_requests += request_count
                    inference_rows += row_count
                    continue
                if returned_task != task_id:
                    raise RuntimeError(
                        f"stale parallel tree result for task {returned_task}"
                    )
                if error is not None:
                    raise RuntimeError(
                        f"parallel tree worker {worker} failed:\n{error}"
                    )
                outputs[worker] = (actions, policies)

            ordered = [outputs[i] for i in range(worker_count)]
            actions = torch.from_numpy(
                np.concatenate([item[0] for item in ordered])
            ).to(engine.device)
            policies = torch.from_numpy(
                np.concatenate([item[1] for item in ordered])
            ).to(engine.device)
            return (
                actions,
                policies,
                {
                    "workers": worker_count,
                    "inference_batches": inference_batches,
                    "inference_requests": inference_requests,
                    "inference_rows": inference_rows,
                },
            )

    def close(self) -> None:
        for task in self.tasks:
            task.put(None)
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for queue in [self.requests, self.results, *self.tasks, *self.responses]:
            queue.close()
            queue.join_thread()


_POOLS: dict[tuple[int, int, float], ParallelTreePool] = {}
_POOLS_LOCK = threading.Lock()


def _pool(config: SearchConfig) -> ParallelTreePool:
    key = (config.cpu_workers, config.inference_batch_size, config.inference_wait_ms)
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = ParallelTreePool(*key)
            _POOLS[key] = pool
        return pool


def parallel_tree_act(
    engine: GameEngine, net: Any, config: SearchConfig, perf: PerfCounters | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    model = net
    if isinstance(net, AzulNet):
        model = InferenceModel(
            net,
            str(next(net.parameters()).device),
            batch_size=config.inference_batch_size,
        )
    actions, policies, metrics = _pool(config).search(engine, model, config)
    if perf is not None:
        for name, value in metrics.items():
            perf.add_count(f"parallel_tree_{name}", value)
    return actions, policies


def shutdown_parallel_tree_pools() -> None:
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.close()


atexit.register(shutdown_parallel_tree_pools)
