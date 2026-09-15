"""Astra's Python adapter. Native code sees only the public current-round state."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import importlib
from importlib.resources import files
import json
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..env.engine import BatchedEngine


@dataclass(frozen=True)
class AstraWeights:
    partial: float = 1.0
    bonus: float = 1.0
    adjacency: float = 0.25
    line_lock: float = 0.65
    initiative: float = 0.6
    safety: float = 0.35
    opponent: float = 1.0
    urgency: float = 1.0
    column_prior: float = 0.0
    bonus_link: float = 0.0
    field_pressure: float = 0.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or not 0 <= v <= 100 for v in asdict(self).values()):
            raise ValueError("Astra weights must be finite and in [0, 100]")
        if self.urgency <= 0:
            raise ValueError("urgency must be positive")
        if self.column_prior > 1 or self.bonus_link > 1 or self.field_pressure > 1:
            raise ValueError("column_prior, bonus_link and field_pressure must be at most one")


@dataclass(frozen=True)
class AstraConfig:
    nodes: int = 4000
    center_nodes: int = 0
    terminal_nodes: int = 0
    time_ms: int = 2000
    depth: int = 32
    width: int = 12
    rollout: bool = False
    weights: AstraWeights = field(default_factory=AstraWeights)

    def __post_init__(self) -> None:
        for name, lo, hi in (("nodes", 1, 100_000_000), ("center_nodes", 0, 100_000_000),
                             ("terminal_nodes", 0, 100_000_000), ("time_ms", 1, 2000),
                             ("depth", 0, 64), ("width", 0, 300)):
            v = getattr(self, name)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        if not isinstance(self.weights, AstraWeights):
            raise ValueError("weights must be AstraWeights")
        if type(self.rollout) is not bool:
            raise ValueError("rollout must be boolean")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AstraConfig:
        values = dict(data)
        if "weights" in values:
            values["weights"] = AstraWeights(**values["weights"])
        return cls(**values)


class AstraUnavailableError(RuntimeError):
    """The optional native extension is missing, incompatible, or stale."""


def native_module() -> Any:
    try:
        module = importlib.import_module("azul_astra")
    except ImportError as exc:
        raise AstraUnavailableError(
            "Astra requires its Rust extension. Install it with "
            "python -m pip install ./native/astra (Rust 1.90.0 required), "
            "or install a prebuilt azul-astra wheel."
        ) from exc
    if getattr(module, "CONFIG_VERSION", None) != 3:
        raise AstraUnavailableError("Astra's installed native extension is out of date; rebuild or reinstall ./native/astra")
    return module


@lru_cache(maxsize=3)
def production_config(num_players: int) -> AstraConfig:
    """Load a versioned, independently selected configuration for this table."""
    if num_players not in (2, 3, 4):
        raise ValueError("Astra supports 2, 3, or 4 players")
    data = json.loads(files("agent.eval").joinpath("astra_configs/production.json").read_text())
    return AstraConfig.from_dict(data["configs"][str(num_players)])


def snapshot(engine: BatchedEngine, game_idx: int) -> list[int]:
    """v1: seven header ints, 45 factory counts, 5 center counts, 14 ints/player.

    Header: version, players, current seat, next starter, center marker, ended,
    resolved. Per board: wall bits, score, floor count, marker, 5 line counts,
    5 line colors. The engine always exposes an unresolved live round or an
    ended game. No bag, lid, floor colors, or RNG is copied.
    """
    native_snapshot = getattr(engine, "public_snapshot", None)
    if native_snapshot is not None:
        return native_snapshot(game_idx)
    if not 0 <= game_idx < engine.batch_size:
        raise IndexError("game_idx outside engine batch")
    b = game_idx
    n = engine.num_players
    ended = int(engine.ended[b].item())
    data = [1, n, int(engine.current_player[b]), int(engine.first_player[b]),
            int(engine.center_first[b]), ended, ended]
    data.extend(engine.factory_tiles[b].flatten().tolist())
    data.extend(engine.center_tiles[b].tolist())
    walls = engine.wall[b, :n].reshape(n, 25).tolist()
    scores = engine.scores[b, :n].tolist()
    floors = engine.floor_count[b, :n].tolist()
    markers = engine.floor_first[b, :n].tolist()
    counts = engine.pattern_count[b, :n].tolist()
    colors = engine.pattern_color[b, :n].tolist()
    for p in range(n):
        wall = sum(1 << i for i, present in enumerate(walls[p]) if present)
        data.extend([wall, scores[p], floors[p], int(markers[p]), *counts[p], *colors[p]])
    return data


def pending_game_end(state: list[int]) -> bool:
    """A completed pattern line already guarantees this round ends the game."""
    if len(state) < 2 or state[1] not in (2, 3, 4) or len(state) != 57 + 14 * state[1]:
        return False  # The native boundary reports invalid snapshot dimensions.
    return any(((state[57 + 14 * p] >> (5 * r)) & 31).bit_count() == 4
               and state[57 + 14 * p + 4 + r] == r + 1
               for p in range(state[1]) for r in range(5))


def native_options(config: AstraConfig, state: list[int]) -> dict[str, Any]:
    """Apply the optional public-state budget policy before the one native call."""
    options = asdict(config)
    terminal_nodes = options.pop("terminal_nodes")
    if terminal_nodes and pending_game_end(state):
        options["nodes"] = max(options["nodes"], terminal_nodes)
        options["center_nodes"] = max(options["nodes"], options["center_nodes"])
    options["weights"] = list(options["weights"].values())
    return options


class HeuristicAstraBot:
    """Hand-coded Rust search with the standard scalar Bot interface."""

    def __init__(self, seed: int | None = None,
                 config: AstraConfig | dict[str, Any] | None = None) -> None:
        self.seed = 0 if seed is None else int(seed) % (1 << 64)
        if isinstance(config, dict):
            config = AstraConfig.from_dict(config)
        if config is not None and not isinstance(config, AstraConfig):
            raise ValueError("config must be AstraConfig, a configuration dictionary, or None")
        self.config = config

    def configuration(self, num_players: int) -> AstraConfig:
        return self.config if self.config is not None else production_config(num_players)

    def analyze(self, engine: BatchedEngine, game_idx: int) -> dict[str, Any]:
        state = snapshot(engine, game_idx)
        cfg = native_options(self.configuration(engine.num_players), state)
        result = native_module().analyze(state, seed=self.seed, **cfg)
        action = int(result["action"])
        if not bool(engine.legal_action_mask()[game_idx, action]):
            raise RuntimeError(f"Astra returned illegal action {action}")
        return result

    def select_action(self, engine: BatchedEngine, game_idx: int) -> int:
        return int(self.analyze(engine, game_idx)["action"])
