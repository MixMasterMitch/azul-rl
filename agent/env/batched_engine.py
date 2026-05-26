"""Batched Azul environment on a single device.

Represents B parallel games with a fixed maximum player count P (default 4)
using padded tensors. Active seats are tracked by `active_mask`.

State tensor shapes (all on the device provided in the constructor):
- `factory_tiles`    (B, MAX_FACTORIES, NUM_COLORS)  int8   # tile counts per factory
- `center_tiles`     (B, NUM_COLORS)                 int8   # tile counts in center
- `center_first`     (B,)                            bool   # first-player marker in center
- `pattern_count`    (B, P, 5)                       int8   # tiles placed in each pattern line
- `pattern_color`    (B, P, 5)                       int8   # color of tiles (-1 if empty)
- `wall`             (B, P, 5, 5)                    bool   # placed tiles on wall grid
- `floor_count`      (B, P)                          int8   # items in floor line
- `floor_tiles`      (B, P, NUM_COLORS)              int8   # tile colors in floor
- `floor_slots`      (B, P, FLOOR_SIZE)              int8   # ordered floor (-1 empty, 0-4 color, 5 marker)
- `floor_first`      (B, P)                          bool   # floor has first-player marker
- `scores`           (B, P)                          int16  # current scores
- `bag`              (B, NUM_COLORS)                  int8   # tiles in bag
- `box_lid`          (B, NUM_COLORS)                 int8   # tiles in box lid (discard)
- `current_player`   (B,)                            int8
- `first_player`     (B,)                            int8   # who goes first next round
- `active_mask`      (B, P)                          bool
- `ended`            (B,)                            bool

The engine handles the full game loop:
1. Factory offer phase: players take turns picking tiles
2. Wall-tiling phase: automatic scoring when all tiles are taken
3. Next round setup: refill factories from bag
"""

from __future__ import annotations

from typing import Optional

import torch

from . import actions as A
from . import tiles as T

MAX_PLAYERS: int = A.MAX_PLAYERS
MAX_FACTORIES: int = A.MAX_FACTORIES
NUM_COLORS: int = A.NUM_COLORS
NUM_ACTIONS: int = A.NUM_ACTIONS
FLOOR_SIZE: int = A.FLOOR_SIZE
FLOOR_PENALTIES: list[int] = A.FLOOR_PENALTIES

def _rng_device_for(device: torch.device) -> str:
    """PyTorch Generator device must match multinomial input (CUDA or CPU only)."""
    return "cuda" if device.type == "cuda" else "cpu"


_STATE_TENSOR_ATTRS = (
    "factory_tiles",
    "center_tiles",
    "center_first",
    "pattern_count",
    "pattern_color",
    "wall",
    "floor_count",
    "floor_tiles",
    "floor_slots",
    "floor_first",
    "scores",
    "bag",
    "box_lid",
    "current_player",
    "first_player",
    "active_mask",
    "ended",
)


class BatchedEngine:
    """Vectorized Azul game engine running B parallel games."""

    def __init__(
        self,
        batch_size: int,
        num_players: int = 2,
        device: torch.device | str = "cpu",
        seed: Optional[int] = None,
    ):
        assert 2 <= num_players <= 4
        self.batch_size = batch_size
        self.num_players = num_players
        self.num_factories = A.num_factories_for_players(num_players)
        self.device = torch.device(device)
        self._rng = torch.Generator(device=_rng_device_for(self.device))
        if seed is not None:
            self._rng.manual_seed(seed)

        self._init_state()

    def _sample_bag_color(self, counts: torch.Tensor) -> int:
        """Draw one color index from a length-5 count vector on the engine device."""
        probs = counts.float()
        if self.device.type == "cpu":
            probs = probs.cpu()
        return int(torch.multinomial(probs, 1, generator=self._rng).item())

    def _init_state(self) -> None:
        B = self.batch_size
        P = MAX_PLAYERS
        dev = self.device

        self.factory_tiles = torch.zeros((B, MAX_FACTORIES, NUM_COLORS), dtype=torch.int8, device=dev)
        self.center_tiles = torch.zeros((B, NUM_COLORS), dtype=torch.int8, device=dev)
        self.center_first = torch.ones((B,), dtype=torch.bool, device=dev)
        self.pattern_count = torch.zeros((B, P, 5), dtype=torch.int8, device=dev)
        self.pattern_color = torch.full((B, P, 5), -1, dtype=torch.int8, device=dev)
        self.wall = torch.zeros((B, P, 5, 5), dtype=torch.bool, device=dev)
        self.floor_count = torch.zeros((B, P), dtype=torch.int8, device=dev)
        self.floor_tiles = torch.zeros((B, P, NUM_COLORS), dtype=torch.int8, device=dev)
        self.floor_slots = torch.full((B, P, FLOOR_SIZE), -1, dtype=torch.int8, device=dev)
        self.floor_first = torch.zeros((B, P), dtype=torch.bool, device=dev)
        self.scores = torch.zeros((B, P), dtype=torch.int16, device=dev)
        self.bag = torch.full((B, NUM_COLORS), T.TILES_PER_COLOR, dtype=torch.int8, device=dev)
        self.box_lid = torch.zeros((B, NUM_COLORS), dtype=torch.int8, device=dev)
        self.current_player = torch.zeros((B,), dtype=torch.int8, device=dev)
        self.first_player = torch.zeros((B,), dtype=torch.int8, device=dev)
        self.active_mask = torch.zeros((B, P), dtype=torch.bool, device=dev)
        self.active_mask[:, :self.num_players] = True
        self.ended = torch.zeros((B,), dtype=torch.bool, device=dev)

        self._fill_factories()

    def _next_floor_slot(self, b: int, player: int) -> int:
        """Return the next empty floor slot index, or -1 if full."""
        for i in range(FLOOR_SIZE):
            if self.floor_slots[b, player, i].item() == -1:
                return i
        return -1

    def _place_on_floor(self, b: int, player: int, count: int, slot_value: int) -> int:
        """Place up to `count` items on the floor. Returns number actually placed."""
        placed = 0
        for _ in range(count):
            idx = self._next_floor_slot(b, player)
            if idx < 0:
                break
            self.floor_slots[b, player, idx] = slot_value
            self.floor_count[b, player] += 1
            if slot_value != A.FLOOR_MARKER:
                self.floor_tiles[b, player, slot_value] += 1
            placed += 1
        return placed

    def _fill_factories(self) -> None:
        """Fill all factory displays with 4 tiles each from the bag."""
        B = self.batch_size
        for b in range(B):
            for f in range(self.num_factories):
                for _ in range(T.TILES_PER_FACTORY):
                    available = self.bag[b].clone()
                    total = available.sum().item()
                    if total == 0:
                        # Refill bag from box lid
                        self.bag[b] = self.box_lid[b].clone()
                        self.box_lid[b].zero_()
                        available = self.bag[b].clone()
                        total = available.sum().item()
                        if total == 0:
                            break
                    color = self._sample_bag_color(available)
                    self.bag[b, color] -= 1
                    self.factory_tiles[b, f, color] += 1

    def index_select(self, indices: torch.Tensor) -> "BatchedEngine":
        """Return a new engine containing only the selected batch rows."""
        n = int(indices.numel())
        other = BatchedEngine.__new__(BatchedEngine)
        other.batch_size = n
        other.num_players = self.num_players
        other.num_factories = self.num_factories
        other.device = self.device
        other._rng = torch.Generator(device=_rng_device_for(self.device))
        other._rng.set_state(self._rng.get_state())
        idx = indices.to(device=self.device, dtype=torch.long)
        for attr in _STATE_TENSOR_ATTRS:
            setattr(other, attr, getattr(self, attr).index_select(0, idx))
        return other

    def repeat_interleave(self, repeats: int) -> "BatchedEngine":
        """Repeat each game state ``repeats`` times along the batch dimension (B -> B*repeats)."""
        if repeats <= 0:
            raise ValueError(f"repeats must be positive, got {repeats}")
        other = BatchedEngine.__new__(BatchedEngine)
        other.batch_size = self.batch_size * repeats
        other.num_players = self.num_players
        other.num_factories = self.num_factories
        other.device = self.device
        other._rng = torch.Generator(device=_rng_device_for(self.device))
        other._rng.set_state(self._rng.get_state())
        for attr in _STATE_TENSOR_ATTRS:
            setattr(other, attr, getattr(self, attr).repeat_interleave(repeats, dim=0))
        return other

    def clone(self) -> "BatchedEngine":
        """Deep copy of the engine state."""
        new = BatchedEngine.__new__(BatchedEngine)
        new.batch_size = self.batch_size
        new.num_players = self.num_players
        new.num_factories = self.num_factories
        new.device = self.device
        new._rng = torch.Generator(device=_rng_device_for(self.device))
        new._rng.set_state(self._rng.get_state())
        for attr in _STATE_TENSOR_ATTRS:
            setattr(new, attr, getattr(self, attr).clone())
        return new

    def legal_action_mask(self) -> torch.Tensor:
        """Returns (B, NUM_ACTIONS) bool mask of legal actions."""
        B = self.batch_size
        dev = self.device
        mask = torch.zeros((B, NUM_ACTIONS), dtype=torch.bool, device=dev)

        cp = self.current_player.long()  # (B,)

        for b in range(B):
            if self.ended[b]:
                continue
            p = cp[b].item()
            self._compute_legal_for_game(b, p, mask[b])

        return mask

    def _compute_legal_for_game(
        self, b: int, player: int, out: torch.Tensor
    ) -> None:
        """Compute legal actions for a single game/player into `out`."""
        for source in range(self.num_factories + 1):
            # Get tile counts at this source
            if source < self.num_factories:
                tiles = self.factory_tiles[b, source]
            else:
                tiles = self.center_tiles[b]

            for color in range(NUM_COLORS):
                if tiles[color].item() <= 0:
                    continue

                # This color exists at this source; check each target
                for target in range(A.NUM_TARGETS):
                    if target == A.FLOOR_TARGET:
                        # Always legal to dump to floor
                        idx = A.encode_action(source, color, target)
                        out[idx] = True
                    else:
                        # Target is pattern line `target`
                        row = target
                        line_color = self.pattern_color[b, player, row].item()
                        line_count = self.pattern_count[b, player, row].item()
                        line_capacity = row + 1

                        # Check: line is empty or same color
                        if line_color != -1 and line_color != color:
                            continue
                        # Check: line is not already full
                        if line_count >= line_capacity:
                            continue
                        # Check: wall row doesn't already have this color
                        wall_col = A.wall_column_for_color(row, color)
                        if self.wall[b, player, row, wall_col].item():
                            continue

                        idx = A.encode_action(source, color, target)
                        out[idx] = True

    def step(self, actions: torch.Tensor) -> None:
        """Apply one action per game in the batch.

        actions: (B,) int64 tensor of action indices.
        """
        B = self.batch_size
        cp = self.current_player.long()

        for b in range(B):
            if self.ended[b]:
                continue
            action = actions[b].item()
            player = cp[b].item()
            self._apply_action(b, player, action)

        # Check if the round (factory offer phase) is over
        self._check_round_end()

    def _apply_action(self, b: int, player: int, action: int) -> None:
        """Apply a single action for one game."""
        source, color, target = A.decode_action(action)

        # Determine how many tiles of this color at source
        if source < self.num_factories:
            num_picked = self.factory_tiles[b, source, color].item()
            # Move remaining tiles to center
            for c in range(NUM_COLORS):
                if c != color:
                    self.center_tiles[b, c] += self.factory_tiles[b, source, c]
            # Clear factory
            self.factory_tiles[b, source].zero_()
        else:
            # Picking from center
            num_picked = self.center_tiles[b, color].item()
            self.center_tiles[b, color] = 0
            # First player to pick from center gets the first-player marker
            if self.center_first[b]:
                self.center_first[b] = False
                self.floor_first[b, player] = True
                self._place_on_floor(b, player, 1, A.FLOOR_MARKER)

        # Place tiles
        if target == A.FLOOR_TARGET:
            # All tiles go to floor
            placed_floor = self._place_on_floor(b, player, num_picked, color)
            overflow = num_picked - placed_floor
            # Overflow goes to box lid
            self.box_lid[b, color] += overflow
        else:
            # Target is a pattern line
            row = target
            line_capacity = row + 1
            current_count = self.pattern_count[b, player, row].item()
            space_available = line_capacity - current_count

            placed_in_line = min(num_picked, space_available)
            excess = num_picked - placed_in_line

            self.pattern_count[b, player, row] += placed_in_line
            self.pattern_color[b, player, row] = color

            # Excess goes to floor
            if excess > 0:
                placed_floor = self._place_on_floor(b, player, excess, color)
                overflow = excess - placed_floor
                self.box_lid[b, color] += overflow

        # Advance to next player
        self._advance_player(b)

    def _advance_player(self, b: int) -> None:
        """Move to the next active player."""
        cp = self.current_player[b].item()
        for i in range(1, self.num_players + 1):
            next_p = (cp + i) % self.num_players
            if self.active_mask[b, next_p]:
                self.current_player[b] = next_p
                return

    def _check_round_end(self) -> None:
        """Check if factory offer phase is complete (all sources empty)."""
        B = self.batch_size
        for b in range(B):
            if self.ended[b]:
                continue
            factories_empty = (self.factory_tiles[b, :self.num_factories].sum().item() == 0)
            center_empty = (self.center_tiles[b].sum().item() == 0)
            if factories_empty and center_empty:
                self._do_wall_tiling(b)

    def _do_wall_tiling(self, b: int) -> None:
        """Execute wall-tiling phase for one game: score tiles, apply penalties,
        check game end, and prepare next round."""
        game_over = False

        for player in range(self.num_players):
            # Move completed pattern lines to wall
            for row in range(5):
                count = self.pattern_count[b, player, row].item()
                capacity = row + 1
                if count < capacity:
                    continue
                # Line is complete - move tile to wall
                color = self.pattern_color[b, player, row].item()
                col = A.wall_column_for_color(row, color)
                self.wall[b, player, row, col] = True

                # Score this placement
                points = self._score_placement(b, player, row, col)
                self.scores[b, player] += points

                # Clear pattern line; return excess tiles to box lid
                tiles_to_return = count - 1  # one tile went to wall
                self.box_lid[b, color] += tiles_to_return
                self.pattern_count[b, player, row] = 0
                self.pattern_color[b, player, row] = -1

            # Apply floor penalties
            floor_n = self.floor_count[b, player].item()
            penalty = 0
            for i in range(min(floor_n, FLOOR_SIZE)):
                penalty += FLOOR_PENALTIES[i]
            self.scores[b, player] = max(0, self.scores[b, player].item() + penalty)

            # Return floor tiles to box lid
            for c in range(NUM_COLORS):
                self.box_lid[b, c] += self.floor_tiles[b, player, c]
            self.floor_tiles[b, player].zero_()
            self.floor_slots[b, player].fill_(-1)
            self.floor_count[b, player] = 0

            # Determine first player for next round
            if self.floor_first[b, player]:
                self.first_player[b] = player
            self.floor_first[b, player] = False

            # Check if this player completed a horizontal row
            for row in range(5):
                if self.wall[b, player, row].all():
                    game_over = True

        if game_over:
            self._end_game(b)
        else:
            self._prepare_next_round(b)

    def _score_placement(self, b: int, player: int, row: int, col: int) -> int:
        """Score a single tile placement on the wall."""
        wall = self.wall[b, player]
        h_count = 1  # the placed tile itself
        v_count = 1

        # Count horizontal neighbors
        for c in range(col - 1, -1, -1):
            if wall[row, c]:
                h_count += 1
            else:
                break
        for c in range(col + 1, 5):
            if wall[row, c]:
                h_count += 1
            else:
                break

        # Count vertical neighbors
        for r in range(row - 1, -1, -1):
            if wall[r, col]:
                v_count += 1
            else:
                break
        for r in range(row + 1, 5):
            if wall[r, col]:
                v_count += 1
            else:
                break

        if h_count == 1 and v_count == 1:
            return 1  # isolated tile
        score = 0
        if h_count > 1:
            score += h_count
        if v_count > 1:
            score += v_count
        return score

    def _end_game(self, b: int) -> None:
        """Apply end-game bonuses and mark game as ended."""
        for player in range(self.num_players):
            bonus = 0
            wall = self.wall[b, player]

            # +2 for each complete horizontal row
            for row in range(5):
                if wall[row].all():
                    bonus += 2

            # +7 for each complete vertical column
            for col in range(5):
                if wall[:, col].all():
                    bonus += 7

            # +10 for each color with all 5 tiles placed
            for color in range(NUM_COLORS):
                all_placed = True
                for row in range(5):
                    col = A.wall_column_for_color(row, color)
                    if not wall[row, col]:
                        all_placed = False
                        break
                if all_placed:
                    bonus += 10

            self.scores[b, player] += bonus

        self.ended[b] = True

    def _prepare_next_round(self, b: int) -> None:
        """Set up the next round: refill factories, reset center."""
        self.center_first[b] = True
        self.current_player[b] = self.first_player[b]

        # Fill factories
        for f in range(self.num_factories):
            for _ in range(T.TILES_PER_FACTORY):
                total = self.bag[b].sum().item()
                if total == 0:
                    # Refill from box lid
                    self.bag[b] = self.box_lid[b].clone()
                    self.box_lid[b].zero_()
                    total = self.bag[b].sum().item()
                    if total == 0:
                        return  # no tiles left anywhere
                color = self._sample_bag_color(self.bag[b])
                self.bag[b, color] -= 1
                self.factory_tiles[b, f, color] += 1

    def get_winners(self) -> torch.Tensor:
        """Returns (B,) tensor with the winning player index for ended games, -1 otherwise."""
        B = self.batch_size
        winners = torch.full((B,), -1, dtype=torch.int8, device=self.device)
        for b in range(B):
            if self.ended[b]:
                # Highest score wins; tiebreak = most complete horizontal rows
                best_score = -1
                best_rows = -1
                best_player = 0
                for p in range(self.num_players):
                    score = self.scores[b, p].item()
                    rows = sum(1 for r in range(5) if self.wall[b, p, r].all())
                    if score > best_score or (score == best_score and rows > best_rows):
                        best_score = score
                        best_rows = rows
                        best_player = p
                winners[b] = best_player
        return winners
