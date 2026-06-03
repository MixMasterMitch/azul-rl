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

Winner resolution (official rules): highest score, then most complete horizontal
rows; if still tied, shared victory (no single winner).
"""

from __future__ import annotations

from typing import Optional

import torch

from . import actions as A
from . import tiles as T

# get_winners() return values
NO_WINNER = -1  # game not finished
SHARED_VICTORY = -2  # tie after score + row tiebreakers (official shared win)

TOTAL_TILES: int = T.TOTAL_TILES

MAX_PLAYERS: int = A.MAX_PLAYERS
MAX_FACTORIES: int = A.MAX_FACTORIES
NUM_COLORS: int = A.NUM_COLORS
NUM_ACTIONS: int = A.NUM_ACTIONS
FLOOR_SIZE: int = A.FLOOR_SIZE
FLOOR_PENALTIES: list[int] = A.FLOOR_PENALTIES
_ACTIONS_PER_SOURCE: int = NUM_COLORS * A.NUM_TARGETS

def _rng_device_for(device: torch.device) -> str:
    """PyTorch Generator device must match multinomial input (CUDA or CPU only)."""
    return "cuda" if device.type == "cuda" else "cpu"


def _cumulative_floor_penalties(device: torch.device) -> torch.Tensor:
    """Indexed by floor_count (0..FLOOR_SIZE); matches scalar sum(FLOOR_PENALTIES[:n])."""
    cum = [0]
    total = 0
    for p in FLOOR_PENALTIES:
        total += p
        cum.append(total)
    while len(cum) < FLOOR_SIZE + 1:
        cum.append(cum[-1])
    return torch.tensor(cum, dtype=torch.int16, device=device)


def _score_completed_placements_batch(
    wall: torch.Tensor,
    row: int,
    col: torch.Tensor,
) -> torch.Tensor:
    """Vectorized Azul placement score for one row with variable columns."""
    n = wall.shape[0]
    batch_idx = torch.arange(n, device=wall.device)

    h_count = torch.ones((n,), dtype=torch.int16, device=wall.device)
    active = torch.ones((n,), dtype=torch.bool, device=wall.device)
    for step in range(1, 5):
        c = col - step
        active = active & (c >= 0)
        hit = active & wall[batch_idx, row, c.clamp(0, 4)]
        h_count += hit.to(torch.int16)
        active = hit

    active = torch.ones((n,), dtype=torch.bool, device=wall.device)
    for step in range(1, 5):
        c = col + step
        active = active & (c < 5)
        hit = active & wall[batch_idx, row, c.clamp(0, 4)]
        h_count += hit.to(torch.int16)
        active = hit

    v_count = torch.ones((n,), dtype=torch.int16, device=wall.device)
    active = torch.ones((n,), dtype=torch.bool, device=wall.device)
    for step in range(1, 5):
        r = row - step
        if r < 0:
            break
        hit = active & wall[batch_idx, r, col]
        v_count += hit.to(torch.int16)
        active = hit

    active = torch.ones((n,), dtype=torch.bool, device=wall.device)
    for step in range(1, 5):
        r = row + step
        if r >= 5:
            break
        hit = active & wall[batch_idx, r, col]
        v_count += hit.to(torch.int16)
        active = hit

    connected = (h_count > 1) | (v_count > 1)
    linked = torch.where(h_count > 1, h_count, torch.zeros_like(h_count)) + torch.where(
        v_count > 1, v_count, torch.zeros_like(v_count)
    )
    return torch.where(connected, linked, torch.ones_like(linked))


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

    def _fill_factories(self) -> None:
        """Fill all factory displays with 4 tiles each from the bag."""
        rows = torch.arange(self.batch_size, device=self.device, dtype=torch.long)
        self._fill_factories_batch(rows)

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

        # Build per-source tile counts for the fixed action space layout.
        # Active factories occupy sources 0..num_factories-1; center is source num_factories.
        counts = torch.zeros((B, A.NUM_SOURCES, NUM_COLORS), dtype=torch.int16, device=dev)
        if self.num_factories > 0:
            counts[:, : self.num_factories] = self.factory_tiles[:, : self.num_factories].to(torch.int16)
        counts[:, self.num_factories] = self.center_tiles.to(torch.int16)

        has_color = counts > 0  # (B, NUM_SOURCES, C)

        # Pull the acting player's per-row state for each game in batch.
        cp = self.current_player.to(torch.long)  # (B,)
        b_idx = torch.arange(B, device=dev)
        line_color = self.pattern_color[b_idx, cp].to(torch.int16)  # (B, 5)
        line_count = self.pattern_count[b_idx, cp].to(torch.int16)  # (B, 5)
        wall_cp = self.wall[b_idx, cp]  # (B, 5, 5) bool

        # For each (row, color), check whether that color can be placed on that pattern row.
        # Conditions:
        # - pattern line is empty OR already same color
        # - pattern line not full
        # - wall row does not already contain that color
        capacities = torch.arange(1, 6, device=dev, dtype=torch.int16)  # (5,)
        compat = (line_color == -1).unsqueeze(-1) | (line_color.unsqueeze(-1) == torch.arange(NUM_COLORS, device=dev, dtype=torch.int16))
        not_full = (line_count < capacities.unsqueeze(0)).unsqueeze(-1)  # (B, 5, 1)

        # wall_col[row, color] = (color + row) % 5, shape (5, 5)
        rc = torch.arange(5, device=dev, dtype=torch.long)
        cc = torch.arange(NUM_COLORS, device=dev, dtype=torch.long)
        wall_col = (cc.unsqueeze(0) + rc.unsqueeze(1)) % 5  # (5, 5)

        # wall_has[b, row, color] = wall_cp[b, row, wall_col[row, color]]
        wall_has = wall_cp.gather(
            2,
            wall_col.unsqueeze(0).expand(B, -1, -1),
        )  # (B, 5, 5)

        legal_row_color = compat & not_full & (~wall_has)  # (B, 5, 5)

        # Assemble full action mask in (source, color, target) layout.
        legal_sct = torch.zeros(
            (B, A.NUM_SOURCES, NUM_COLORS, A.NUM_TARGETS), dtype=torch.bool, device=dev
        )

        # Floor target always legal if the color exists at the source.
        legal_sct[..., A.FLOOR_TARGET] = has_color

        # Pattern line targets: require both "has_color at source" and row/color legality.
        # Broadcast legal_row_color (B, row, color) to (B, source, color, row).
        legal_src_color = has_color.unsqueeze(-1)  # (B, S, C, 1)
        legal_color_row = legal_row_color.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C, 5)
        legal_sct[..., : A.NUM_PATTERN_LINES] = legal_src_color & legal_color_row

        # Ended games have no legal actions.
        if self.ended.any():
            legal_sct = legal_sct & (~self.ended).view(B, 1, 1, 1)

        # Flatten into action-index order: source-major, then color, then target.
        return legal_sct.reshape(B, NUM_ACTIONS)

    @staticmethod
    def _decode_actions(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode flat action indices into (source, color, target) tensors."""
        source = actions // _ACTIONS_PER_SOURCE
        rem = actions % _ACTIONS_PER_SOURCE
        color = rem // A.NUM_TARGETS
        target = rem % A.NUM_TARGETS
        return source, color, target

    def _place_on_floor_batch(
        self,
        sel: torch.Tensor,
        players: torch.Tensor,
        counts: torch.Tensor,
        slot_values: torch.Tensor,
    ) -> torch.Tensor:
        """Place up to ``counts`` items on the floor for selected games. Returns placed counts."""
        sel = sel.reshape(-1).to(torch.long)
        players = players.reshape(-1).to(torch.long)
        counts = counts.reshape(-1)
        slot_values = slot_values.reshape(-1).to(torch.long)
        if sel.numel() == 0:
            return torch.zeros(0, dtype=torch.int16, device=self.device)

        floor_count = self.floor_count[sel, players]
        space = (FLOOR_SIZE - floor_count).clamp_min(0)
        placed = torch.minimum(counts.to(torch.int16), space)

        for k in range(FLOOR_SIZE):
            can = placed > k
            if not can.any():
                break
            slot_idx = (floor_count + k)[can].to(torch.long)
            self.floor_slots[sel[can], players[can], slot_idx] = slot_values[can].to(
                torch.int8
            )

        self.floor_count[sel, players] += placed
        is_tile = slot_values != A.FLOOR_MARKER
        tile_sel = is_tile & (placed > 0)
        if tile_sel.any():
            b = sel[tile_sel]
            p = players[tile_sel]
            colors = slot_values[tile_sel].long()
            self.floor_tiles.index_put_(
                (b, p, colors),
                placed[tile_sel].to(torch.int8),
                accumulate=True,
            )
        return placed

    def total_tile_count(self) -> torch.Tensor:
        """Per-game count of colored tiles (bag, lid, factories, center, pattern, wall, floor)."""
        t = self.bag.sum(dim=1, dtype=torch.int32)
        t = t + self.box_lid.sum(dim=1, dtype=torch.int32)
        t = t + self.factory_tiles[:, : self.num_factories].sum(dim=(1, 2), dtype=torch.int32)
        t = t + self.center_tiles.sum(dim=1, dtype=torch.int32)
        t = t + self.pattern_count.sum(dim=(1, 2), dtype=torch.int32)
        t = t + self.wall.sum(dim=(1, 2, 3), dtype=torch.int32)
        t = t + self.floor_tiles.sum(dim=(1, 2), dtype=torch.int32)
        return t

    def _add_to_box_lid(
        self,
        sel: torch.Tensor,
        colors: torch.Tensor,
        amounts: torch.Tensor,
    ) -> None:
        """Add ``amounts`` tiles of ``colors`` to box lid for selected batch rows."""
        if sel.numel() == 0:
            return
        sel = sel.reshape(-1).to(torch.long)
        colors = colors.reshape(-1).to(torch.long)
        amounts = amounts.reshape(-1).to(torch.int8)
        self.box_lid.index_put_((sel, colors), amounts, accumulate=True)

    def _pick_from_factory_vectorized(
        self,
        sel: torch.Tensor,
        sources: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """Pick one color from a factory; move leftovers to center. Returns tiles picked."""
        sel = sel.reshape(-1).to(torch.long)
        sources = sources.reshape(-1).to(torch.long)
        colors = colors.reshape(-1).to(torch.long)
        num_picked = self.factory_tiles[sel, sources, colors].to(torch.int16)

        fac_rows = self.factory_tiles[sel, sources, :].to(torch.int16)
        to_center = fac_rows.clone()
        to_center.scatter_(1, colors.unsqueeze(1), 0)
        self.center_tiles[sel] = (self.center_tiles[sel].to(torch.int16) + to_center).to(
            torch.int8
        )
        self.factory_tiles[sel, sources, :] = 0
        return num_picked

    def _pick_from_center_vectorized(
        self,
        sel: torch.Tensor,
        players: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """Pick one color from the center; place first-player marker when applicable."""
        sel = sel.reshape(-1).to(torch.long)
        players = players.reshape(-1).to(torch.long)
        colors = colors.reshape(-1).to(torch.long)
        num_picked = self.center_tiles[sel, colors].to(torch.int16)
        self.center_tiles[sel, colors] = 0

        fp = self.center_first[sel]
        if fp.any():
            fp_sel = sel[fp]
            fp_players = players[fp]
            self.center_first[fp_sel] = False
            self.floor_first[fp_sel, fp_players] = True
            ones = torch.ones(fp_sel.numel(), dtype=torch.int16, device=self.device)
            markers = torch.full(
                (fp_sel.numel(),), A.FLOOR_MARKER, dtype=torch.long, device=self.device
            )
            self._place_on_floor_batch(fp_sel, fp_players, ones, markers)
        return num_picked

    def _advance_player_batch(self, sel: torch.Tensor) -> None:
        """Move to the next active player for selected games."""
        sel = sel.reshape(-1).to(torch.long)
        if sel.numel() == 0:
            return
        cp = self.current_player[sel].to(torch.long)
        updated = torch.zeros(sel.numel(), dtype=torch.bool, device=self.device)
        for i in range(1, self.num_players + 1):
            cand = (cp + i) % self.num_players
            can = (~updated) & self.active_mask[sel, cand]
            if can.any():
                self.current_player[sel[can]] = cand[can].to(torch.int8)
                updated = updated | can

    def _step_offer_vectorized(self, actions: torch.Tensor) -> None:
        """Batched factory-offer phase (pick, place, advance)."""
        B = self.batch_size
        dev = self.device
        actions = actions.to(device=dev, dtype=torch.long).reshape(B)

        active = ~self.ended
        if not active.any():
            return

        source, color, target = self._decode_actions(actions)
        b_idx = torch.arange(B, device=dev)
        players = self.current_player.to(torch.long)

        num_picked = torch.zeros(B, dtype=torch.int16, device=dev)

        pick_factory = active & (source < self.num_factories)
        if pick_factory.any():
            sel = b_idx[pick_factory]
            picked = self._pick_from_factory_vectorized(
                sel, source[pick_factory], color[pick_factory]
            )
            num_picked[pick_factory] = picked

        pick_center = active & (source == self.num_factories)
        if pick_center.any():
            sel = b_idx[pick_center]
            picked = self._pick_from_center_vectorized(
                sel, players[pick_center], color[pick_center]
            )
            num_picked[pick_center] = picked

        to_floor = active & (target == A.FLOOR_TARGET)
        if to_floor.any():
            sel = b_idx[to_floor]
            p = players[to_floor]
            c = color[to_floor]
            n = num_picked[to_floor]
            placed = self._place_on_floor_batch(sel, p, n, c)
            overflow = n - placed
            has_overflow = overflow > 0
            if has_overflow.any():
                self._add_to_box_lid(sel[has_overflow], c[has_overflow], overflow[has_overflow])

        to_pattern = active & (target < A.NUM_PATTERN_LINES)
        if to_pattern.any():
            sel = b_idx[to_pattern]
            p = players[to_pattern]
            row = target[to_pattern].to(torch.long)
            c = color[to_pattern]
            n = num_picked[to_pattern]

            line_cap = (row + 1).to(torch.int16)
            cur = self.pattern_count[sel, p, row].to(torch.int16)
            space = (line_cap - cur).clamp_min(0)
            placed = torch.minimum(n, space)
            excess = n - placed

            self.pattern_count[sel, p, row] = (
                self.pattern_count[sel, p, row].to(torch.int16) + placed
            ).to(torch.int8)
            self.pattern_color[sel, p, row] = c.to(torch.int8)

            has_excess = excess > 0
            if has_excess.any():
                ex_sel = sel[has_excess]
                ex_p = p[has_excess]
                ex_c = c[has_excess]
                ex_n = excess[has_excess]
                placed_floor = self._place_on_floor_batch(ex_sel, ex_p, ex_n, ex_c)
                overflow = ex_n - placed_floor
                has_overflow = overflow > 0
                if has_overflow.any():
                    self._add_to_box_lid(
                        ex_sel[has_overflow], ex_c[has_overflow], overflow[has_overflow]
                    )

        self._advance_player_batch(b_idx[active])

    def step(self, actions: torch.Tensor, *, finalize_round: bool = True) -> None:
        """Apply one action per game in the batch.

        actions: (B,) int64 tensor of action indices.
        finalize_round: When False, skip wall-tiling / next-round setup after the
            factory-offer phase. Used for MCTS child expansion where finalizing
            thousands of hypothetical states dominates cost.
        """
        self._step_offer_vectorized(actions)

        if finalize_round:
            self._check_round_end_vectorized()

    def _check_round_end_vectorized(self) -> None:
        """Detect finished factory-offer rounds and run wall-tiling per game."""
        factories_empty = self.factory_tiles[:, : self.num_factories].sum(dim=(1, 2)) == 0
        center_empty = self.center_tiles.sum(dim=1) == 0
        round_done = factories_empty & center_empty & (~self.ended)
        if not round_done.any():
            return
        rows = round_done.nonzero(as_tuple=True)[0]
        self._do_wall_tiling_batch(rows)

    def _do_wall_tiling_batch(self, rows: torch.Tensor) -> None:
        """Vectorized wall-tiling, end-game bonuses, and next-round refill."""
        if rows.numel() == 0:
            return

        floor_penalties = _cumulative_floor_penalties(self.device)

        for player in range(self.num_players):
            for row in range(5):
                count = self.pattern_count[rows, player, row].to(torch.int16)
                complete = count >= row + 1
                if not complete.any():
                    continue

                r = rows[complete]
                color = self.pattern_color[r, player, row].to(torch.long)
                col = (color + row) % 5

                self.wall[r, player, row, col] = True
                wall = self.wall[r, player]
                points = _score_completed_placements_batch(wall, row, col)
                self.scores[r, player] += points
                returned = (count[complete] - 1).to(torch.int8)
                self._add_to_box_lid(r, color, returned)
                self.pattern_count.index_put_(
                    (r, torch.full_like(r, player), torch.full_like(r, row)),
                    torch.zeros(r.shape[0], dtype=torch.int8, device=self.device),
                )
                self.pattern_color.index_put_(
                    (r, torch.full_like(r, player), torch.full_like(r, row)),
                    torch.full((r.shape[0],), -1, dtype=torch.int8, device=self.device),
                )

            floor_n = self.floor_count[rows, player].to(torch.long).clamp(0, FLOOR_SIZE)
            updated_scores = self.scores[rows, player] + floor_penalties[floor_n]
            self.scores[rows, player] = updated_scores.clamp_min(0).to(self.scores.dtype)
            self.box_lid[rows] += self.floor_tiles[rows, player]
            player_idx = torch.full_like(rows, fill_value=player, dtype=torch.long)
            self.floor_tiles.index_put_(
                (rows, player_idx),
                torch.zeros((rows.numel(), NUM_COLORS), dtype=torch.int8, device=self.device),
            )
            self.floor_slots.index_put_(
                (rows, player_idx),
                torch.full((rows.numel(), FLOOR_SIZE), -1, dtype=torch.int8, device=self.device),
            )
            self.floor_count.index_put_(
                (rows, player_idx),
                torch.zeros((rows.numel(),), dtype=torch.int8, device=self.device),
            )

            has_first = self.floor_first[rows, player]
            if has_first.any():
                self.first_player[rows[has_first]] = player
            self.floor_first.index_put_(
                (rows, player_idx),
                torch.zeros((rows.numel(),), dtype=torch.bool, device=self.device),
            )

        wall_rows_complete = self.wall[rows, : self.num_players].all(dim=-1)
        game_over = wall_rows_complete.any(dim=(1, 2))
        if game_over.any():
            self._end_game_batch(rows[game_over])

        next_round = rows[~game_over]
        if next_round.numel() > 0:
            self._prepare_next_round_batch(next_round)

    def _end_game_batch(self, rows: torch.Tensor) -> None:
        """Apply end-game bonuses for a subset of batch rows."""
        if rows.numel() == 0:
            return

        nP = self.num_players
        wall = self.wall[rows, :nP]
        bonus = wall.all(dim=-1).sum(dim=-1).to(torch.int16) * 2
        bonus += wall.all(dim=-2).sum(dim=-1).to(torch.int16) * 7

        row_idx = torch.arange(5, device=self.device)
        for color in range(NUM_COLORS):
            col_idx = (row_idx + color) % 5
            bonus += wall[:, :, row_idx, col_idx].all(dim=-1).to(torch.int16) * 10

        self.scores[rows, :nP] += bonus
        self.ended[rows] = True

    def _prepare_next_round_batch(self, rows: torch.Tensor) -> None:
        """Refill factories for round-done games without per-game Python loops."""
        rows = rows.to(device=self.device, dtype=torch.long)
        if rows.numel() == 0:
            return
        self.center_first[rows] = True
        self.current_player[rows] = self.first_player[rows]
        self._fill_factories_batch(rows)

    def _fill_factories_batch(self, rows: torch.Tensor) -> None:
        """Fill factory displays for selected games, refilling each bag from its lid as needed."""
        rows = rows.to(device=self.device, dtype=torch.long)
        if rows.numel() == 0:
            return

        one = torch.ones((rows.numel(),), dtype=torch.int8, device=self.device)
        minus_one = torch.full((rows.numel(),), -1, dtype=torch.int8, device=self.device)
        for slot in range(self.num_factories * T.TILES_PER_FACTORY):
            bag_total = self.bag[rows].sum(dim=1, dtype=torch.int16)
            empty_bag = bag_total == 0
            if empty_bag.any():
                refill_rows = rows[empty_bag]
                self.bag[refill_rows] = self.box_lid[refill_rows]
                self.box_lid[refill_rows] = 0
                bag_total = self.bag[rows].sum(dim=1, dtype=torch.int16)

            can_draw = bag_total > 0
            if not can_draw.any():
                return

            draw_rows = rows[can_draw]
            totals = bag_total[can_draw]
            draws = torch.floor(
                torch.rand(
                    (draw_rows.numel(),),
                    device=self.device,
                    generator=self._rng,
                )
                * totals.to(torch.float32)
            ).to(torch.int16)
            cumulative = self.bag[draw_rows].to(torch.int16).cumsum(dim=1)
            colors = (cumulative <= draws.unsqueeze(1)).sum(dim=1).to(torch.long)

            self.bag.index_put_(
                (draw_rows, colors),
                minus_one[: draw_rows.numel()],
                accumulate=True,
            )
            factories = torch.full(
                (draw_rows.numel(),),
                slot // T.TILES_PER_FACTORY,
                dtype=torch.long,
                device=self.device,
            )
            self.factory_tiles.index_put_(
                (draw_rows, factories, colors),
                one[: draw_rows.numel()],
                accumulate=True,
            )

    def get_winners(self) -> torch.Tensor:
        """Return (B,) winner seat per game.

        ``NO_WINNER`` (-1): game not finished (includes turn-cap stalls).
        ``SHARED_VICTORY`` (-2): ended with official shared win (tied score and rows).
        Otherwise: sole winner seat index.
        """
        B = self.batch_size
        P = self.num_players
        winners = torch.full((B,), NO_WINNER, dtype=torch.int8, device=self.device)
        if not self.ended.any():
            return winners

        scores = self.scores[:, :P].to(torch.int32)
        rows_complete = self.wall[:, :P].all(dim=-1).sum(dim=-1).to(torch.int32)

        max_score = scores.max(dim=-1, keepdim=True).values
        score_best = scores == max_score

        rows_for_best = rows_complete.masked_fill(~score_best, -1)
        max_rows = rows_for_best.max(dim=-1, keepdim=True).values
        at_best = score_best & (rows_complete == max_rows)

        num_at_best = at_best.sum(dim=-1)
        first_best = at_best.to(torch.int64).argmax(dim=-1).to(torch.int8)

        single = self.ended & (num_at_best == 1)
        shared = self.ended & (num_at_best > 1)
        winners = torch.where(single, first_best, winners)
        winners = torch.where(
            shared,
            torch.full((B,), SHARED_VICTORY, dtype=torch.int8, device=self.device),
            winners,
        )
        return winners
