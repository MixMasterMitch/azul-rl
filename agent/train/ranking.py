"""Bradley-Terry rating system anchored to random=1000."""

from __future__ import annotations

import math
from typing import Optional

import torch

DEFAULT_ANCHORS = {"random": 1000.0, "heuristic": 1200.0}


def add_match_result(
    results: list[dict],
    a: str,
    b: str,
    wins_a: float,
    wins_b: float,
    ties: float = 0.0,
) -> None:
    """Append or accumulate a pairwise result row."""
    for r in results:
        if r["a"] == a and r["b"] == b:
            r["wins_a"] = r.get("wins_a", 0) + wins_a
            r["wins_b"] = r.get("wins_b", 0) + wins_b
            if ties:
                r["ties"] = r.get("ties", 0) + ties
            return
        if r["a"] == b and r["b"] == a:
            r["wins_a"] = r.get("wins_a", 0) + wins_b
            r["wins_b"] = r.get("wins_b", 0) + wins_a
            if ties:
                r["ties"] = r.get("ties", 0) + ties
            return
    row: dict = {"a": a, "b": b, "wins_a": wins_a, "wins_b": wins_b}
    if ties:
        row["ties"] = ties
    results.append(row)


def fit_anchored_ratings(
    results: list[dict],
    anchors: dict[str, float] | None = None,
    sigma: float = 600.0,
) -> dict[str, float]:
    """Fit Bradley-Terry ratings with optional tie support in results rows."""
    return fit_ratings(results, anchors=anchors, sigma=sigma)


def fit_ratings(
    results: list[dict],
    anchors: dict[str, float] | None = None,
    sigma: float = 600.0,
) -> dict[str, float]:
    """Fit Bradley-Terry ratings from pairwise results.

    Args:
        results: list of {"a": name, "b": name, "wins_a": int, "wins_b": int}
        anchors: fixed ratings (e.g. {"random": 1000.0})
        sigma: Gaussian prior standard deviation

    Returns:
        dict mapping entity name to fitted rating.
    """
    if anchors is None:
        anchors = {"random": 1000.0}

    # Collect all entities
    entities = set()
    for r in results:
        entities.add(r["a"])
        entities.add(r["b"])

    free_entities = sorted(entities - set(anchors.keys()))
    all_entities = sorted(anchors.keys()) + free_entities

    if not free_entities:
        return dict(anchors)

    # Map entity name to index
    idx = {name: i for i, name in enumerate(all_entities)}
    N = len(all_entities)
    n_anchors = len(anchors)

    # Initialize ratings
    ratings = torch.zeros(N, dtype=torch.float64)
    for name, rating in anchors.items():
        ratings[idx[name]] = rating
    for name in free_entities:
        ratings[idx[name]] = 1500.0  # initial guess

    # L-BFGS optimization on free parameters
    free_ratings = ratings[n_anchors:].clone().requires_grad_(True)

    def get_all_ratings():
        anchor_vals = torch.tensor(
            [anchors[name] for name in sorted(anchors.keys())],
            dtype=torch.float64,
        )
        return torch.cat([anchor_vals, free_ratings])

    optimizer = torch.optim.LBFGS([free_ratings], lr=1.0, max_iter=100)

    def closure():
        optimizer.zero_grad()
        all_r = get_all_ratings()

        # Negative log-likelihood
        nll = torch.tensor(0.0, dtype=torch.float64)
        for r in results:
            i_a = idx[r["a"]]
            i_b = idx[r["b"]]
            ra = all_r[i_a]
            rb = all_r[i_b]
            # BT model: P(a beats b) = sigmoid((ra - rb) / scale)
            scale = 1000.0 / math.log(10)
            logit = (ra - rb) / scale
            log_pa = torch.nn.functional.logsigmoid(logit)
            log_pb = torch.nn.functional.logsigmoid(-logit)
            ties = r.get("ties", 0)
            nll = (
                nll
                - r.get("wins_a", 0) * log_pa
                - r.get("wins_b", 0) * log_pb
                - 0.5 * ties * (log_pa + log_pb)
            )

        # Gaussian prior
        prior = 0.5 * (free_ratings - 1500.0).pow(2).sum() / (sigma ** 2)
        loss = nll + prior
        loss.backward()
        return loss

    optimizer.step(closure)

    all_r = get_all_ratings()
    return {name: all_r[idx[name]].item() for name in all_entities}
