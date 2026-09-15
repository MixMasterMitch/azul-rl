"""Default engine: persistent Rust simulation with CPU/CUDA tensor observations.

BatchedEngine in batched_engine.py remains the explicit PyTorch reference.
State tensor attributes here are snapshots for inspection, not writable state.
"""
from __future__ import annotations

import base64
from typing import Any, Sequence

import torch

from . import actions as A
from . import batched_engine as reference
from .rust_engine import RustEngine, _tensor_buffer

MAX_PLAYERS = A.MAX_PLAYERS
MAX_FACTORIES = A.MAX_FACTORIES
NUM_COLORS = A.NUM_COLORS
NUM_ACTIONS = A.NUM_ACTIONS
FLOOR_SIZE = A.FLOOR_SIZE
NO_WINNER = reference.NO_WINNER
SHARED_VICTORY = reference.SHARED_VICTORY
_STATE_TENSOR_ATTRS = reference._STATE_TENSOR_ATTRS

_LAYOUT = {
    "factory_tiles": (torch.int8, (9, 5)), "center_tiles": (torch.int8, (5,)),
    "center_first": (torch.bool, ()), "pattern_count": (torch.int8, (4, 5)),
    "pattern_color": (torch.int8, (4, 5)), "wall": (torch.bool, (4, 5, 5)),
    "floor_count": (torch.int8, (4,)), "floor_tiles": (torch.int8, (4, 5)),
    "floor_slots": (torch.int8, (4, 7)), "floor_first": (torch.bool, (4,)),
    "scores": (torch.int16, (4,)), "bag": (torch.int8, (5,)), "box_lid": (torch.int8, (5,)),
    "current_player": (torch.int8, ()), "first_player": (torch.int8, ()),
    "active_mask": (torch.bool, (4,)), "ended": (torch.bool, ()),
}


def _cpu(values: Any) -> Any:
    return values.cpu() if isinstance(values, torch.Tensor) else values


def _copy_rng(rng: torch.Generator) -> torch.Generator:
    return torch.Generator(device="cpu").set_state(rng.get_state())


class GameEngine:
    """Rust rules/RNG on CPU; ``device`` selects observation/inference tensors."""

    def __init__(self, batch_size: int, num_players: int = 2,
                 device: torch.device | str = "cpu", seed: int | None = None,
                 game_seeds: Sequence[int] | None = None) -> None:
        if seed is None:
            seed = int(torch.randint(2**63 - 1, (), device="cpu"))
        self.native = RustEngine(batch_size, num_players, seed, game_seeds)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("GameEngine supports CPU and CUDA observations")
        self._legacy_rng: torch.Generator | None = None
        self._legacy_game_rngs: list[torch.Generator] | None = None
        self._invalidate()

    def _invalidate(self) -> None:
        self._tensors: dict[str, torch.Tensor] = {}
        self._buffers: dict[str, bytearray] | None = None
        self._observation: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._cpu_view: GameEngine | None = None

    @classmethod
    def _wrap(cls, native: RustEngine, device: torch.device | str = "cpu") -> GameEngine:
        engine = cls.__new__(cls)
        engine.native, engine.device = native, torch.device(device)
        engine._legacy_rng = None
        engine._legacy_game_rngs = None
        engine._invalidate()
        return engine

    @property
    def batch_size(self) -> int:
        return self.native.batch_size

    @property
    def num_players(self) -> int:
        return self.native.num_players

    @property
    def num_factories(self) -> int:
        return self.native.num_factories

    def __getattr__(self, name: str) -> torch.Tensor:
        if name not in _LAYOUT:
            raise AttributeError(name)
        if name not in self._tensors:
            if name in {"current_player", "ended", "scores"}:
                value = getattr(self.native, name)
            else:
                if self._buffers is None:
                    self._buffers = self.native._native.state_buffers()
                dtype, shape = _LAYOUT[name]
                value = _tensor_buffer(self._buffers[name], dtype, (self.batch_size, *shape))
            self._tensors[name] = value.to(self.device)
        return self._tensors[name]

    def encode_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._observation is None:
            self._observation = tuple(t.to(self.device) for t in self.native.encode_state_with_legal())
        return self._observation[0], self._observation[1]

    def legal_action_mask(self) -> torch.Tensor:
        if self._observation is not None:
            return self._observation[2]
        return self.native.legal_action_mask().to(self.device)

    def final_values(self, reward_mode: str = "binary") -> torch.Tensor:
        return self.native.final_values(reward_mode).to(self.device)

    def get_winners(self) -> torch.Tensor:
        return self.native.get_winners().to(self.device)

    def total_tile_count(self) -> torch.Tensor:
        return self.native.total_tile_count().to(self.device)

    def round_done(self) -> torch.Tensor:
        return torch.tensor(self.native._native.round_done(), dtype=torch.bool, device=self.device)

    def public_snapshot(self, game_idx: int) -> list[int]:
        return self.native._native.public_snapshot(game_idx)

    def step(self, actions: Sequence[int] | torch.Tensor, *, finalize_round: bool = True,
             draw_uniforms: Sequence[Sequence[float]] | torch.Tensor | None = None) -> None:
        # Legacy saves retain their original PyTorch draw streams, but all rules run in Rust.
        legacy = self._legacy_rng is not None and draw_uniforms is None
        self.native.step(_cpu(actions), finalize_round=finalize_round and not legacy,
                         draw_uniforms=_cpu(draw_uniforms))
        self._invalidate()
        if legacy and finalize_round:
            self.finalize_round()

    def finalize_round(self, *, draw_uniforms: Sequence[Sequence[float]] | torch.Tensor | None = None) -> None:
        if self._legacy_rng is not None and draw_uniforms is None:
            counts = self.native._native.refill_counts()
            tape = torch.zeros((self.batch_size, self.num_factories * 4), dtype=torch.float32)
            for slot in range(max(counts, default=0)):
                rows = [i for i, count in enumerate(counts) if count > slot]
                if self._legacy_game_rngs is None:
                    tape[rows, slot] = torch.rand(len(rows), generator=self._legacy_rng)
                else:
                    for row in rows:
                        tape[row, slot] = torch.rand((), generator=self._legacy_game_rngs[row])
            draw_uniforms = tape
        self.native.finalize_round(draw_uniforms=_cpu(draw_uniforms))
        self._invalidate()

    def clone(self) -> GameEngine:
        other = self._wrap(self.native.clone(), self.device)
        if self._legacy_rng is not None:
            other._legacy_rng = _copy_rng(self._legacy_rng)
        if self._legacy_game_rngs is not None:
            other._legacy_game_rngs = [_copy_rng(rng) for rng in self._legacy_game_rngs]
        return other

    def __deepcopy__(self, memo: dict[int, Any]) -> GameEngine:
        other = self.clone()
        memo[id(self)] = other
        return other

    def index_select(self, indices: Sequence[int] | torch.Tensor) -> GameEngine:
        ids = _cpu(indices)
        other = self._wrap(self.native.index_select(ids), self.device)
        if self._legacy_rng is not None:
            other._legacy_rng = _copy_rng(self._legacy_rng)
        if self._legacy_game_rngs is not None:
            other._legacy_game_rngs = [_copy_rng(self._legacy_game_rngs[int(i)]) for i in ids]
        return other

    def repeat_interleave(self, repeats: int) -> GameEngine:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        if self._legacy_rng is not None:
            return self.index_select(torch.arange(self.batch_size).repeat_interleave(repeats))
        return self._wrap(self.native.repeat_interleave(repeats), self.device)

    def reseed(self, seed: int) -> None:
        self.native.reseed([(seed + i) % (1 << 64) for i in range(self.batch_size)])
        self._legacy_rng = self._legacy_game_rngs = None
        self._cpu_view = None

    def expand(self, actions: torch.Tensor, *, seed: int) -> GameEngine:
        count = actions.numel()
        return self._wrap(self.native.expand(actions.cpu(),
                          game_seeds=[(seed + i) % (1 << 64) for i in range(count)]), self.device)

    @classmethod
    def concatenate(cls, engines: Sequence[GameEngine], *, seed: int) -> GameEngine:
        if not engines:
            raise ValueError("At least one batch is required")
        native = engines[0].native.index_select([])
        for engine in engines:
            native._native.append(engine.native._native)
        result = cls._wrap(native, "cpu")
        result.reseed(seed)  # Search chance samples are independent of live game draws.
        return result

    def cpu_view(self) -> GameEngine:
        if self.device.type == "cpu":
            return self
        if self._cpu_view is None:
            self._cpu_view = self.clone()
            self._cpu_view.device = torch.device("cpu")
        return self._cpu_view

    @classmethod
    def from_batched(cls, engine: reference.BatchedEngine, *, preserve_rng: bool = False,
                     seed: int = 0) -> GameEngine:
        if engine.device.type != "cpu":
            raise ValueError("Import a CPU reference engine")
        result = cls._wrap(RustEngine.from_batched(engine, seed=seed))
        if preserve_rng:
            result._legacy_rng = _copy_rng(engine._rng)
            if engine._game_rngs is not None:
                result._legacy_game_rngs = [_copy_rng(rng) for rng in engine._game_rngs]
        return result

    def state_dict(self) -> dict[str, Any]:
        def pack(rng: torch.Generator) -> str:
            return base64.b64encode(bytes(rng.get_state().tolist())).decode()
        return {"backend": "rust", "version": 1, "num_players": self.num_players,
                "snapshots": self.native.snapshots(),
                "legacy_rng": pack(self._legacy_rng) if self._legacy_rng is not None else None,
                "legacy_game_rngs": [pack(rng) for rng in self._legacy_game_rngs]
                    if self._legacy_game_rngs is not None else None}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], device: str = "cpu") -> GameEngine:
        if state.get("backend") != "rust" or state.get("version") != 1:
            raise ValueError("Unsupported engine checkpoint")
        result = cls._wrap(RustEngine.from_snapshots(state["num_players"], state["snapshots"]), device)
        def unpack(value: str) -> torch.Generator:
            data = base64.b64decode(value, validate=True)
            return torch.Generator(device="cpu").set_state(torch.tensor(list(data), dtype=torch.uint8))
        if state.get("legacy_rng") is not None:
            result._legacy_rng = unpack(state["legacy_rng"])
        if state.get("legacy_game_rngs") is not None:
            if result._legacy_rng is None or len(state["legacy_game_rngs"]) != result.batch_size:
                raise ValueError("Invalid legacy per-game RNGs")
            result._legacy_game_rngs = [unpack(value) for value in state["legacy_game_rngs"]]
        return result


# Keeps the established constructor spelling at production call sites.
BatchedEngine = GameEngine
