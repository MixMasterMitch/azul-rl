"""Bradley-Terry rating system with per-player-count pairwise records.

Result rows store wins per player count:
    {"a": "ckpt:42", "b": "random", "wins_a_2p": 10, "wins_b_2p": 6, "wins_a_3p": ...}

Rating computation:
1. Fit separate ratings per player count (anchor: random=1000).
2. Calibrate each to a common scale using reference-anchor-derived scales.
3. Combined rating = weighted average of calibrated per-PC ratings.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Optional, Sequence

import torch

RANDOM_ANCHOR_RATING = 1000.0
DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_ANCHORS = {"random": RANDOM_ANCHOR_RATING}
RATING_SCALE = 1000.0
_MIN_PROB = 1e-9

PLAYER_COUNTS = (2, 3, 4)

_CALIBRATION_REFERENCE_ENTITY = "heuristic_opus"

REFERENCE_ENTITIES = frozenset({"random", "heuristic", "heuristic_opus"})

# Fallback when league.json has no ``reference_anchors_per_pc``. Regenerate with:
#   python -m agent.scripts.gen_reference_anchors
DEFAULT_REFERENCE_ANCHORS_PER_PC: dict[int, dict[str, float]] = {
    2: {"random": 1000.0, "heuristic": 2679.6, "heuristic_opus": 3138.6},
    3: {"random": 1000.0, "heuristic": 2454.6, "heuristic_opus": 2845.4},
    4: {"random": 1000.0, "heuristic": 1487.2, "heuristic_opus": 1823.5},
}

# Backward-compatible alias.
REFERENCE_ANCHORS_PER_PC = DEFAULT_REFERENCE_ANCHORS_PER_PC


def reference_anchors_from_manifest(manifest: Mapping | None) -> dict[int, dict[str, float]]:
    """Load per-PC reference anchors from league.json, else use code defaults."""
    if manifest is None:
        return dict(DEFAULT_REFERENCE_ANCHORS_PER_PC)
    raw = manifest.get("reference_anchors_per_pc")
    if not raw:
        return dict(DEFAULT_REFERENCE_ANCHORS_PER_PC)
    out: dict[int, dict[str, float]] = {}
    for pc_key, anchors in raw.items():
        out[int(pc_key)] = {str(k): float(v) for k, v in anchors.items()}
    return out


def is_reference_only_result_row(row: Mapping) -> bool:
    return row["a"] in REFERENCE_ENTITIES and row["b"] in REFERENCE_ENTITIES

_MIN_ANCHORS_NO_PRIOR = 3


@dataclasses.dataclass(frozen=True)
class MatchResult:
    a: str
    b: str
    wins_a: float
    wins_b: float

    @property
    def total_games(self) -> float:
        return self.wins_a + self.wins_b

    @property
    def score_a(self) -> float:
        return self.wins_a


def canonical_match(
    a: str,
    b: str,
    wins_a: float,
    wins_b: float,
) -> MatchResult:
    if a <= b:
        return MatchResult(a=a, b=b, wins_a=wins_a, wins_b=wins_b)
    return MatchResult(a=b, b=a, wins_a=wins_b, wins_b=wins_a)


def add_match_result(
    results: list[dict],
    a: str,
    b: str,
    wins_a: float,
    wins_b: float,
    ties: float = 0.0,
    num_players: int = 2,
) -> None:
    """Record a pairwise result into the results list."""
    if a == b:
        return
    wa_raw = round(wins_a)
    wb_raw = round(wins_b)
    ties_raw = round(ties)
    if wa_raw + wb_raw + ties_raw <= 0:
        return

    if a <= b:
        ca, cb = a, b
        wa, wb = wa_raw, wb_raw
    else:
        ca, cb = b, a
        wa, wb = wb_raw, wa_raw

    pc = num_players
    key_a = f"wins_a_{pc}p"
    key_b = f"wins_b_{pc}p"
    key_t = f"ties_{pc}p"

    for row in results:
        if row["a"] == ca and row["b"] == cb:
            row[key_a] = row.get(key_a, 0) + wa
            row[key_b] = row.get(key_b, 0) + wb
            if ties_raw > 0:
                row[key_t] = row.get(key_t, 0) + ties_raw
            return

    new_row: dict = {"a": ca, "b": cb, key_a: wa, key_b: wb}
    if ties_raw > 0:
        new_row[key_t] = ties_raw
    results.append(new_row)


def winrate_vs_anchor(
    results: Sequence[dict],
    entity: str,
    anchor: str,
    pc: int,
) -> tuple[float, int]:
    if entity == anchor:
        return (0.0, 0)
    key_a = f"wins_a_{pc}p"
    key_b = f"wins_b_{pc}p"
    key_t = f"ties_{pc}p"
    entity_score = 0.0
    total = 0
    for row in results:
        a = row.get("a")
        b = row.get("b")
        if a == entity and b == anchor:
            wa = int(row.get(key_a, 0))
            wb = int(row.get(key_b, 0))
            ties = int(row.get(key_t, 0))
            entity_score += wa + 0.5 * ties
            total += wa + wb + ties
        elif a == anchor and b == entity:
            wa = int(row.get(key_a, 0))
            wb = int(row.get(key_b, 0))
            ties = int(row.get(key_t, 0))
            entity_score += wb + 0.5 * ties
            total += wa + wb + ties
    return (entity_score, total)


def _extract_pc_results(
    results: Sequence[dict], pc: int
) -> list[MatchResult]:
    key_a = f"wins_a_{pc}p"
    key_b = f"wins_b_{pc}p"
    key_t = f"ties_{pc}p"
    out: list[MatchResult] = []
    for row in results:
        wa = float(row.get(key_a, 0))
        wb = float(row.get(key_b, 0))
        ties = float(row.get(key_t, 0))
        effective_a = wa + 0.5 * ties
        effective_b = wb + 0.5 * ties
        if effective_a + effective_b <= 0:
            continue
        out.append(MatchResult(a=row["a"], b=row["b"], wins_a=effective_a, wins_b=effective_b))
    return out


def expected_score(rating_a: torch.Tensor, rating_b: torch.Tensor) -> torch.Tensor:
    logits = (rating_a - rating_b) * (math.log(10.0) / RATING_SCALE)
    return torch.sigmoid(logits)


def fit_ratings_for_pc(
    results: Sequence[dict],
    pc: int,
    anchors: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
    max_iter: int = 200,
    prior_sigma: float = 600.0,
    use_reference_anchors: bool = True,
    reference_anchors_per_pc: Mapping[int, Mapping[str, float]] | None = None,
) -> dict[str, float]:
    ref_table = (
        dict(DEFAULT_REFERENCE_ANCHORS_PER_PC)
        if reference_anchors_per_pc is None
        else dict(reference_anchors_per_pc)
    )
    base_anchors = dict(DEFAULT_ANCHORS if anchors is None else anchors)
    if use_reference_anchors:
        ref = ref_table.get(pc, {})
        for k, v in ref.items():
            base_anchors.setdefault(k, v)
    anchors_map = base_anchors
    initial_map = {} if initial is None else dict(initial)

    matches = _extract_pc_results(results, pc)
    if not matches:
        return dict(anchors_map)

    participants: set[str] = set(anchors_map)
    for m in matches:
        participants.add(m.a)
        participants.add(m.b)

    free_ids = sorted(pid for pid in participants if pid not in anchors_map)
    ratings = dict(anchors_map)
    if not free_ids:
        return ratings

    use_prior = len(anchors_map) < _MIN_ANCHORS_NO_PRIOR

    init_values = [
        float(initial_map.get(pid, DEFAULT_INITIAL_RATING))
        for pid in free_ids
    ]
    params = torch.nn.Parameter(torch.tensor(init_values, dtype=torch.float64))
    opt = torch.optim.LBFGS(
        [params],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )
    index = {pid: i for i, pid in enumerate(free_ids)}
    anchor_tensors = {
        pid: torch.tensor(value, dtype=torch.float64)
        for pid, value in anchors_map.items()
    }
    prior_mean = torch.tensor(DEFAULT_INITIAL_RATING, dtype=torch.float64)
    prior_var = prior_sigma**2

    def _rating(pid: str) -> torch.Tensor:
        if pid in anchor_tensors:
            return anchor_tensors[pid]
        return params[index[pid]]

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float64)
        for match in matches:
            prob_a = expected_score(_rating(match.a), _rating(match.b)).clamp(
                _MIN_PROB, 1.0 - _MIN_PROB
            )
            score_a = torch.tensor(match.score_a, dtype=torch.float64)
            score_b = torch.tensor(match.total_games - match.score_a, dtype=torch.float64)
            loss = loss - score_a * torch.log(prob_a) - score_b * torch.log1p(-prob_a)
        if use_prior:
            loss = loss + 0.5 * ((params - prior_mean) ** 2).sum() / prior_var
        loss.backward()
        return loss

    opt.step(closure)
    out = dict(anchors_map)
    solved = params.detach().cpu().tolist()
    for pid, value in zip(free_ids, solved, strict=True):
        out[pid] = float(value)
    return out


def calibration_scales_for(
    reference_anchors_per_pc: Mapping[int, Mapping[str, float]] | None = None,
) -> dict[int, float]:
    ref_table = (
        dict(DEFAULT_REFERENCE_ANCHORS_PER_PC)
        if reference_anchors_per_pc is None
        else dict(reference_anchors_per_pc)
    )
    ref_entity = _CALIBRATION_REFERENCE_ENTITY
    baseline_pc = 2
    anchor_rating = RANDOM_ANCHOR_RATING

    baseline_ref = ref_table.get(baseline_pc, {}).get(ref_entity)
    if baseline_ref is None:
        return {pc: 1.0 for pc in PLAYER_COUNTS}
    baseline_diff = baseline_ref - anchor_rating
    if baseline_diff <= 0:
        return {pc: 1.0 for pc in PLAYER_COUNTS}

    scales: dict[int, float] = {}
    for pc in PLAYER_COUNTS:
        ref_raw = ref_table.get(pc, {}).get(ref_entity)
        if ref_raw is None or (ref_raw - anchor_rating) <= 0:
            scales[pc] = 1.0
        else:
            scales[pc] = baseline_diff / (ref_raw - anchor_rating)
    return scales


CALIBRATION_SCALE: dict[int, float] = calibration_scales_for()


def calibrate_rating(
    raw: float,
    pc: int,
    calibration_scale: Mapping[int, float] | None = None,
) -> float:
    scales = CALIBRATION_SCALE if calibration_scale is None else calibration_scale
    return RANDOM_ANCHOR_RATING + (raw - RANDOM_ANCHOR_RATING) * scales[pc]


def _count_games_per_entity_per_pc(
    results: Sequence[dict],
) -> dict[str, dict[int, float]]:
    counts: dict[str, dict[int, float]] = {}
    for row in results:
        a, b = row["a"], row["b"]
        for pc in PLAYER_COUNTS:
            wa = float(row.get(f"wins_a_{pc}p", 0))
            wb = float(row.get(f"wins_b_{pc}p", 0))
            ties = float(row.get(f"ties_{pc}p", 0))
            pairwise = wa + wb + ties
            if pairwise <= 0:
                continue
            actual = pairwise / (pc - 1)
            counts.setdefault(a, {}).setdefault(pc, 0.0)
            counts[a][pc] += actual
            counts.setdefault(b, {}).setdefault(pc, 0.0)
            counts[b][pc] += actual
    return counts


def compute_ratings(
    results: Sequence[dict],
    anchors: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
    use_reference_anchors: bool = True,
    reference_anchors_per_pc: Mapping[int, Mapping[str, float]] | None = None,
) -> dict[str, dict]:
    ref_table = (
        dict(DEFAULT_REFERENCE_ANCHORS_PER_PC)
        if reference_anchors_per_pc is None
        else dict(reference_anchors_per_pc)
    )
    cal_scale = calibration_scales_for(ref_table)

    per_pc: dict[int, dict[str, float]] = {}
    for pc in PLAYER_COUNTS:
        per_pc[pc] = fit_ratings_for_pc(
            results,
            pc,
            anchors=anchors,
            initial=initial,
            use_reference_anchors=use_reference_anchors,
            reference_anchors_per_pc=ref_table,
        )

    games_per_entity_pc = _count_games_per_entity_per_pc(results)

    all_entities: set[str] = set()
    for pc_ratings in per_pc.values():
        all_entities.update(pc_ratings.keys())

    out: dict[str, dict] = {}
    for entity in all_entities:
        entry: dict = {}
        calibrated_sum = 0.0
        weight_sum = 0.0
        for pc in PLAYER_COUNTS:
            raw = per_pc[pc].get(entity)
            if raw is not None:
                entry[f"rating_{pc}p"] = raw
                cal = calibrate_rating(raw, pc, cal_scale)
                entry[f"calibrated_{pc}p"] = cal
                weight = games_per_entity_pc.get(entity, {}).get(pc, 0.0)
                if weight <= 0:
                    weight = 1.0
                calibrated_sum += cal * weight
                weight_sum += weight
        if weight_sum > 0:
            entry["rating"] = calibrated_sum / weight_sum
        else:
            entry["rating"] = None
        out[entity] = entry
    return out


def fit_anchored_ratings(
    results: Sequence[dict],
    anchors: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
    max_iter: int = 200,
) -> dict[str, float]:
    has_new_format = any(
        any(k.startswith("wins_a_") and k.endswith("p") for k in row)
        for row in results
    )

    if has_new_format:
        ratings_data = compute_ratings(results, anchors=anchors, initial=initial)
        return {
            entity: data["rating"]
            for entity, data in ratings_data.items()
            if data["rating"] is not None
        }

    anchors_map = dict(DEFAULT_ANCHORS if anchors is None else anchors)
    initial_map = {} if initial is None else dict(initial)

    participants: set[str] = set(anchors_map)
    clean_results: list[MatchResult] = []
    for row in results:
        a_str = str(row["a"])
        b_str = str(row["b"])
        if a_str == b_str:
            continue
        wa = float(row.get("wins_a", 0.0)) + 0.5 * float(row.get("ties", 0.0))
        wb = float(row.get("wins_b", 0.0)) + 0.5 * float(row.get("ties", 0.0))
        if wa + wb <= 0:
            continue
        if a_str <= b_str:
            clean_results.append(MatchResult(a=a_str, b=b_str, wins_a=wa, wins_b=wb))
        else:
            clean_results.append(MatchResult(a=b_str, b=a_str, wins_a=wb, wins_b=wa))
        participants.add(a_str)
        participants.add(b_str)

    free_ids = sorted(pid for pid in participants if pid not in anchors_map)
    ratings = dict(anchors_map)
    if not free_ids:
        return ratings

    init_values = [
        float(initial_map.get(pid, DEFAULT_INITIAL_RATING))
        for pid in free_ids
    ]
    params = torch.nn.Parameter(torch.tensor(init_values, dtype=torch.float64))
    opt = torch.optim.LBFGS(
        [params],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )
    index = {pid: i for i, pid in enumerate(free_ids)}
    anchor_tensors = {
        pid: torch.tensor(value, dtype=torch.float64)
        for pid, value in anchors_map.items()
    }

    def _rating(pid: str) -> torch.Tensor:
        if pid in anchor_tensors:
            return anchor_tensors[pid]
        return params[index[pid]]

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float64)
        for match in clean_results:
            prob_a = expected_score(_rating(match.a), _rating(match.b)).clamp(
                _MIN_PROB, 1.0 - _MIN_PROB
            )
            score_a = torch.tensor(match.score_a, dtype=torch.float64)
            score_b = torch.tensor(match.total_games - match.score_a, dtype=torch.float64)
            loss = loss - score_a * torch.log(prob_a) - score_b * torch.log1p(-prob_a)
        loss.backward()
        return loss

    opt.step(closure)
    out = dict(anchors_map)
    solved = params.detach().cpu().tolist()
    for pid, value in zip(free_ids, solved, strict=True):
        out[pid] = float(value)
    return out


# Backward-compatible alias used by human_rating store.
def fit_ratings(
    results: Sequence[dict],
    anchors: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
    sigma: float = 600.0,
) -> dict[str, float]:
    del sigma
    return fit_anchored_ratings(results, anchors=anchors, initial=initial)
