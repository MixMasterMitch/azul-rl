"""Low-level CPU simulator backed by persistent Rust state.

Production uses GameEngine in engine.py for CPU/CUDA observations. Full snapshots are private
simulator data, not inputs to Astra's public-information tactical search.
"""
from __future__ import annotations

import importlib
from typing import Any, Sequence

import torch

from . import actions as A
from .batched_engine import BatchedEngine


def _native_module() -> Any:
    message = "The Rust game engine is required. Install/rebuild it with: python -m pip install ./native/astra"
    try:
        module = importlib.import_module("azul_astra")
    except ImportError as exc:
        raise ImportError(message) from exc
    if not hasattr(module, "BatchEngine") or not hasattr(module.BatchEngine, "state_buffers"):
        raise ImportError(message)
    return module


def _cpu_list(values: Sequence[Any] | torch.Tensor) -> list[Any]:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError("RustEngine inputs must be on CPU; transfer explicitly before calling")
        return values.tolist()
    return list(values)


def _tensor_buffer(data: bytearray, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    # torch.frombuffer retains the bytearray owner. Empty buffers need a separate path.
    if not data:
        return torch.empty(shape, dtype=dtype, device="cpu")
    return torch.frombuffer(data, dtype=dtype).reshape(shape)


class RustEngine:
    """A homogeneous batch of 2–4-player games, held entirely in native CPU memory.

    Seeds select per-game SplitMix64 streams, not PyTorch/Python RNG streams.
    Cloning, selecting and repeating copy RNG state exactly. Use ``reseed`` or
    ``expand(..., game_seeds=...)`` for independent hypothetical refill outcomes.
    """

    def __init__(
        self,
        batch_size: int,
        num_players: int = 2,
        seed: int = 0,
        game_seeds: Sequence[int] | None = None,
    ) -> None:
        self._native = _native_module().BatchEngine(
            batch_size, num_players, seed,
            None if game_seeds is None else list(game_seeds),
        )

    @classmethod
    def _wrap(cls, native: Any) -> RustEngine:
        engine = cls.__new__(cls)
        engine._native = native
        return engine

    @property
    def batch_size(self) -> int:
        return int(self._native.batch_size)

    @property
    def num_players(self) -> int:
        return int(self._native.num_players)

    @property
    def num_factories(self) -> int:
        return A.num_factories_for_players(self.num_players)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def current_player(self) -> torch.Tensor:
        return torch.tensor(self._native.current_player, dtype=torch.int8, device="cpu")

    @property
    def ended(self) -> torch.Tensor:
        return torch.tensor(self._native.ended, dtype=torch.bool, device="cpu")

    @property
    def scores(self) -> torch.Tensor:
        return torch.tensor(self._native.scores, dtype=torch.int16, device="cpu").reshape(self.batch_size, 4)

    def get_winners(self) -> torch.Tensor:
        return torch.tensor(self._native.get_winners(), dtype=torch.int8, device="cpu")

    def final_values(self, reward_mode: str = "binary") -> torch.Tensor:
        """Training outcomes in absolute seat order, matching agent.env.outcomes."""
        return _tensor_buffer(
            self._native.final_values_buffer(reward_mode), torch.float32, (self.batch_size, 4),
        )

    def total_tile_count(self) -> torch.Tensor:
        return torch.tensor(self._native.total_tile_count(), dtype=torch.int32, device="cpu")

    def legal_action_mask(self) -> torch.Tensor:
        return _tensor_buffer(self._native.legal_mask_buffer(), torch.bool, (self.batch_size, A.NUM_ACTIONS))

    def encode_state_with_legal(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return current-player-relative global (B,275), source (B,10,5), legal (B,300)."""
        g, s, legal = self._native.encode_buffers()
        return (
            _tensor_buffer(g, torch.float32, (self.batch_size, 275)),
            _tensor_buffer(s, torch.float32, (self.batch_size, 10, 5)),
            _tensor_buffer(legal, torch.bool, (self.batch_size, 300)),
        )

    def encode_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        g, s, _ = self.encode_state_with_legal()
        return g, s

    def step(
        self,
        actions: Sequence[int] | torch.Tensor,
        *,
        finalize_round: bool = True,
        draw_uniforms: Sequence[Sequence[float]] | torch.Tensor | None = None,
    ) -> None:
        """Apply one legal action per live game; terminal rows are ignored.

        Optional float32 draw uniforms have shape (B, num_factories*4). They are
        consumed by refill slot and do not advance the RNG. This permits exact
        cross-engine comparisons despite different default RNG algorithms.
        All input is validated before any game in the batch changes.
        """
        self._native.step(
            _cpu_list(actions), finalize_round,
            None if draw_uniforms is None else _cpu_list(draw_uniforms),
        )

    def finalize_round(
        self, *, draw_uniforms: Sequence[Sequence[float]] | torch.Tensor | None = None,
    ) -> None:
        self._native.finalize_round(None if draw_uniforms is None else _cpu_list(draw_uniforms))

    def clone(self) -> RustEngine:
        return self._wrap(self._native.clone())

    def index_select(self, indices: Sequence[int] | torch.Tensor) -> RustEngine:
        return self._wrap(self._native.index_select(_cpu_list(indices)))

    def repeat_interleave(self, repeats: int) -> RustEngine:
        return self._wrap(self._native.repeat_interleave(repeats))

    def reseed(self, seeds: Sequence[int] | torch.Tensor) -> None:
        self._native.reseed(_cpu_list(seeds))

    def expand(
        self,
        actions: Sequence[Sequence[int]] | torch.Tensor,
        *,
        game_seeds: Sequence[int] | torch.Tensor | None = None,
    ) -> RustEngine:
        """Copy and advance B×K children, ordered parent-major, without changing parents."""
        return self._wrap(self._native.expand(
            _cpu_list(actions), None if game_seeds is None else _cpu_list(game_seeds),
        ))

    def snapshots(self) -> list[list[int]]:
        """Versioned full snapshots including private inventory, floor slots and RNG."""
        return self._native.snapshots()

    @classmethod
    def from_snapshots(cls, num_players: int, snapshots: Sequence[Sequence[int]]) -> RustEngine:
        return cls._wrap(_native_module().BatchEngine.from_snapshots(num_players, [list(s) for s in snapshots]))

    @classmethod
    def from_batched(cls, engine: BatchedEngine, *, seed: int = 0) -> RustEngine:
        """Copy CPU engine state. Future draws use fresh Rust RNGs seeded by seed+row.

        PyTorch RNG bytes cannot be imported as SplitMix64 state. To compare
        future refills exactly, pass common draw_uniforms to both engines.
        """
        if engine.device.type != "cpu":
            raise ValueError("from_batched requires a CPU engine")
        expected_active = torch.arange(4, device="cpu") < engine.num_players
        if not torch.equal(engine.active_mask, expected_active.expand(engine.batch_size, -1)):
            raise ValueError("RustEngine requires contiguous active seats for num_players")
        from ..eval.heuristic_astra import snapshot

        data = []
        for i in range(engine.batch_size):
            row = snapshot(engine, i)
            row[0] = 2
            row.extend(engine.bag[i].tolist())
            row.extend(engine.box_lid[i].tolist())
            for p in range(engine.num_players):
                row.extend(engine.floor_tiles[i, p].tolist())
                row.extend(engine.floor_slots[i, p].tolist())
            rng = (seed + i) % (1 << 64)
            row.extend([rng >> 32, rng & 0xFFFFFFFF])
            data.append(row)
        return cls.from_snapshots(engine.num_players, data)

    def to_batched(self, *, seed: int = 0) -> BatchedEngine:
        """Copy state into a CPU reference engine; seed fresh PyTorch future draws.

        For exact native continuation/checkpointing use snapshots(), which
        preserves the Rust RNG. This conversion is for compatibility and QA,
        not the per-turn training path.
        """
        e = BatchedEngine(self.batch_size, self.num_players, device="cpu", seed=seed)
        for i, data in enumerate(self.snapshots()):
            n = self.num_players
            e.current_player[i], e.first_player[i] = data[2], data[3]
            e.center_first[i], e.ended[i] = bool(data[4]), bool(data[5])
            e.factory_tiles[i] = torch.tensor(data[7:52], dtype=torch.int8, device="cpu").reshape(9, 5)
            e.center_tiles[i] = torch.tensor(data[52:57], dtype=torch.int8, device="cpu")
            for p in range(n):
                board = data[57 + p * 14:57 + (p + 1) * 14]
                wall, score, floor, marker = board[:4]
                e.wall[i, p] = torch.tensor([bool(wall & (1 << k)) for k in range(25)], device="cpu").reshape(5, 5)
                e.scores[i, p], e.floor_count[i, p], e.floor_first[i, p] = score, floor, bool(marker)
                e.pattern_count[i, p] = torch.tensor(board[4:9], dtype=torch.int8, device="cpu")
                e.pattern_color[i, p] = torch.tensor(board[9:14], dtype=torch.int8, device="cpu")
            offset = 57 + n * 14
            e.bag[i] = torch.tensor(data[offset:offset + 5], dtype=torch.int8, device="cpu")
            e.box_lid[i] = torch.tensor(data[offset + 5:offset + 10], dtype=torch.int8, device="cpu")
            for p in range(n):
                base = offset + 10 + p * 12
                e.floor_tiles[i, p] = torch.tensor(data[base:base + 5], dtype=torch.int8, device="cpu")
                e.floor_slots[i, p] = torch.tensor(data[base + 5:base + 12], dtype=torch.int8, device="cpu")
        return e
