"""Ranking evaluation.

The question a recommender answers is "which ten artists do I show this user?", so it
is judged on its top-k lists, not on how well it reproduces play counts.

Protocol used throughout: the model sees a user's past, recommends artists the user
has *not* played yet, and a recommendation is a hit if the user goes on to play that
artist in the held-out future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


def top_k(scores: np.ndarray, seen: sparse.csr_matrix, k: int) -> np.ndarray:
    """Top-``k`` unseen artists per user, best first. Returns ``(n_users, k)`` indices."""
    scores = np.array(scores, dtype=np.float32, copy=True)
    scores[seen.nonzero()] = -np.inf
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


@dataclass
class Report:
    """Per-user metric values (so that confidence intervals can be bootstrapped)."""

    precision: np.ndarray
    recall: np.ndarray
    ndcg: np.ndarray
    hit: np.ndarray
    coverage: float  # share of the catalogue that appears in at least one list
    median_listeners: float  # how mainstream the recommended artists are
    k: int

    def summary(self) -> dict:
        out = {
            name: float(getattr(self, name).mean()) for name in ("precision", "recall", "ndcg", "hit")
        }
        out.update(coverage=self.coverage, median_listeners=self.median_listeners)
        return out

    def confidence_interval(self, metric: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
        """95 % bootstrap interval of the mean over users."""
        values = getattr(self, metric)
        rng = np.random.default_rng(seed)
        means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(
    recommended: np.ndarray,
    relevant: sparse.csr_matrix,
    listeners: np.ndarray,
) -> Report:
    """Score top-k lists against the artists each user went on to play.

    Users with no relevant artist are left out of the averages. ``listeners`` is the
    number of training users per artist, used to describe how popular the
    recommendations are.
    """
    k = recommended.shape[1]
    n_relevant = np.diff(relevant.indptr)
    users = np.flatnonzero(n_relevant > 0)
    rec = recommended[users]
    hits = np.asarray(relevant[users[:, None], rec].todense(), dtype=np.float64)  # (n_eval, k) 0/1

    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = hits @ discount
    ideal = np.cumsum(discount)[np.minimum(n_relevant[users], k) - 1]
    n_hits = hits.sum(axis=1)
    return Report(
        precision=n_hits / k,
        recall=n_hits / n_relevant[users],
        ndcg=dcg / ideal,
        hit=(n_hits > 0).astype(np.float64),
        coverage=float(len(np.unique(rec)) / relevant.shape[1]),
        median_listeners=float(np.median(listeners[rec])),
        k=k,
    )


def popularity_penalty(scores: np.ndarray, listeners: np.ndarray, beta: float) -> np.ndarray:
    """Divide every score by ``listeners ** beta`` to push recommendations down the long tail.

    ``beta = 0`` leaves the ranking untouched; larger values trade accuracy for novelty.
    Only positive scores are rescaled, so an artist the model rejects is never promoted.
    """
    if beta == 0:
        return scores
    return np.where(scores > 0, scores / np.power(listeners + 1.0, beta), scores)
