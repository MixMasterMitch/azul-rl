"""Python boundary for the native Rust Gumbel tree.

Rust owns games, nodes, edges, traversal, chance outcomes, and backups.  The
callback is deliberately narrow: it sends contiguous encoded leaves to the
existing inference model and returns contiguous policy/value outputs.
"""

from __future__ import annotations

from typing import Any
from collections import OrderedDict
from threading import RLock

import numpy as np
import torch

from ..env import actions as A
from ..env.engine import GameEngine
from ..env.rust_engine import _native_module
from ..net import encoder as ENC
from ..net.model import AzulNet
from ..train.instrumentation import PerfCounters
from .config import SearchConfig


class NativeInferenceAdapter:
    """Translate native byte buffers without copying encoded FP32 features."""

    def __init__(self, model: Any, cache_size: int = 0) -> None:
        self.model = model
        self.cache_size = cache_size
        self.cache: OrderedDict[bytes, tuple[bytes, bytes]] = OrderedDict()
        self.cache_hits = 0
        self.inferred_rows = 0
        self.lock = RLock()

    def __call__(
        self,
        global_bytes: bytearray,
        source_bytes: bytearray,
        legal_bytes: bytearray,
        rows: int,
        num_players: int,
    ) -> tuple[bytes, bytes]:
        global_feat = torch.frombuffer(global_bytes, dtype=torch.float32).reshape(
            rows, ENC.D_GLOBAL
        )
        source_feat = torch.frombuffer(source_bytes, dtype=torch.float32).reshape(
            rows, ENC.NUM_SOURCES, ENC.D_SOURCE
        )
        legal = (
            torch.frombuffer(legal_bytes, dtype=torch.uint8)
            .reshape(rows, A.NUM_ACTIONS)
            .bool()
        )
        if self.cache_size:
            with self.lock:
                return self._cached(
                    global_bytes,
                    source_bytes,
                    legal_bytes,
                    global_feat,
                    source_feat,
                    legal,
                    num_players,
                )
        self.inferred_rows += rows
        policy, value = self.model(global_feat, source_feat, legal, num_players)
        policy_bytes = (
            policy.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .tobytes()
        )
        value_bytes = (
            value.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .tobytes()
        )
        return policy_bytes, value_bytes

    def _cached(
        self,
        gb: bytearray,
        sb: bytearray,
        lb: bytearray,
        g: torch.Tensor,
        s: torch.Tensor,
        legal: torch.Tensor,
        num_players: int,
    ) -> tuple[bytes, bytes]:
        """Cache exact network inputs, never RNG state or sampled chance outcomes.

        Deduplication also covers the leaf batch. Adapters reused across moves
        are invalidated when parameter/buffer versions change.
        """
        gs, ss, ls = ENC.D_GLOBAL * 4, ENC.NUM_SOURCES * ENC.D_SOURCE * 4, A.NUM_ACTIONS
        keys = [
            bytes([num_players])
            + bytes(gb[i * gs : (i + 1) * gs])
            + bytes(sb[i * ss : (i + 1) * ss])
            + bytes(lb[i * ls : (i + 1) * ls])
            for i in range(len(g))
        ]
        resolved: dict[bytes, tuple[bytes, bytes]] = {}
        missing: dict[bytes, int] = {}
        for i, key in enumerate(keys):
            if key in self.cache:
                resolved[key] = self.cache[key]
                self.cache.move_to_end(key)
            elif key not in missing:
                missing[key] = i
        self.cache_hits += len(keys) - len(missing)
        if missing:
            indices = list(missing.values())
            policy, value = self.model(
                g[indices], s[indices], legal[indices], num_players
            )
            policy = (
                policy.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
            )
            value = (
                value.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
            )
            self.inferred_rows += len(indices)
            for i, key in enumerate(missing):
                resolved[key] = (policy[i].tobytes(), value[i].tobytes())
                self.cache[key] = resolved[key]
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return (
            b"".join(resolved[k][0] for k in keys),
            b"".join(resolved[k][1] for k in keys),
        )


def _inference_adapter(net: Any, cfg: SearchConfig) -> NativeInferenceAdapter:
    """Reuse leaf evaluations across moves until an optimizer/checkpoint update.

    The cache belongs to the network, not a global model ID. Tensor version
    counters catch standard in-place optimizer steps and load_state_dict calls.
    Proxy models (including parallel workers) get a search-local cache instead.
    """
    from ..eval.inference import InferenceModel

    owner = (
        net
        if isinstance(net, AzulNet)
        else net.net
        if isinstance(net, InferenceModel)
        else None
    )
    if owner is not None:
        owner.eval()
    key = None
    if cfg.inference_cache_size and owner is not None:
        try:
            key = (
                cfg.inference_cache_size,
                cfg.inference_batch_size,
                tuple(
                    (id(t), t._version, t.device, t.dtype)
                    for t in (*owner.parameters(), *owner.buffers())
                ),
            )
        except RuntimeError:
            pass  # Inference tensors without version counters cannot persist a cache.
        cached = getattr(owner, "_native_search_cache", None)
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]
    model = net
    if isinstance(net, AzulNet):
        model = InferenceModel(
            net, str(next(net.parameters()).device), batch_size=cfg.inference_batch_size
        )
    adapter = NativeInferenceAdapter(model, cfg.inference_cache_size)
    if key is not None:
        owner._native_search_cache = (key, adapter)
    return adapter


@torch.no_grad()
def native_tree_act(
    engine: GameEngine, net: Any, cfg: SearchConfig, perf: PerfCounters | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(engine, GameEngine):
        raise TypeError("the Rust tree core requires GameEngine")
    seed = (
        cfg.seed if cfg.seed is not None else int(torch.randint(2**63 - 1, ()).item())
    )
    adapter = _inference_adapter(net, cfg)
    hits_before, inferred_before = adapter.cache_hits, adapter.inferred_rows
    result = _native_module().gumbel_tree(
        engine.state_dict()["snapshots"],
        adapter,
        num_simulations=cfg.num_simulations,
        temperature=cfg.temperature,
        q_scale=cfg.q_scale,
        root_noise_scale=cfg.root_noise_scale,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_mix=cfg.dirichlet_mix,
        reward_mode=cfg.reward_mode,
        seed=seed,
        max_root_candidates=cfg.max_root_candidates,
        max_depth=cfg.max_depth,
        chance_samples=cfg.chance_samples,
        move_deadline_ms=(
            None
            if cfg.move_deadline_s is None
            else max(1, int(cfg.move_deadline_s * 1000))
        ),
    )
    actions = torch.tensor(result["actions"], dtype=torch.long, device=engine.device)
    policies_np = (
        np.frombuffer(result["policies"], dtype=np.float32)
        .reshape(engine.batch_size, A.NUM_ACTIONS)
        .copy()
    )
    policies = torch.from_numpy(policies_np).to(engine.device)
    if perf is not None:
        metrics = {
            "inference_cache_hits": adapter.cache_hits - hits_before,
            "inferred_rows": adapter.inferred_rows - inferred_before,
            "simulations_per_root": result["simulations_min"],
            "simulations_max_per_root": result["simulations_max"],
            "max_depth": result["max_depth"],
            "nodes": result["nodes"],
            "search_refill_children": result["refill_children"],
            "search_terminal_children": result["terminal_children"],
        }
        for key, value in metrics.items():
            perf.add_count(f"native_tree_{key}", float(value))
        perf.add_count("native_tree_elapsed_s", float(result["elapsed_s"]))
    return actions, policies
