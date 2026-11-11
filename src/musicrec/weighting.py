"""From play counts to confidence.

A play count is not a rating. Someone with 2 000 plays of one band does not like it
a thousand times more than a band they played twice, and a band everybody plays
says little about any one listener. These functions turn a sparse matrix of play
counts (users x artists) into a matrix of *confidence* values that a model can use.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def linear(plays: sparse.csr_matrix) -> sparse.csr_matrix:
    """Confidence proportional to the play count (Hu, Koren & Volinsky, 2008)."""
    return plays.astype(np.float32)


def log(plays: sparse.csr_matrix) -> sparse.csr_matrix:
    """``log(1 + plays)``: the 100th play adds much less than the 2nd."""
    out = plays.astype(np.float32)
    out.data = np.log1p(out.data)
    return out


def bm25(plays: sparse.csr_matrix, k1: float = 1.2, b: float = 0.75) -> sparse.csr_matrix:
    """BM25, the ranking function of text search engines, applied to listening data.

    A user is a "document" and an artist is a "word". Three effects:

    * saturation: the weight grows with the play count but levels off quickly (``k1``);
    * length normalisation: heavy listeners need more plays for the same weight (``b``);
    * rarity (IDF): an artist few people listen to is more informative than a hit.
    """
    plays = plays.tocsr().astype(np.float32)
    n_users = plays.shape[0]
    listeners = np.bincount(plays.indices, minlength=plays.shape[1])
    idf = np.log(n_users) - np.log1p(listeners)

    user_total = np.asarray(plays.sum(axis=1)).ravel()
    length = (1.0 - b) + b * user_total / user_total.mean()
    row_of_entry = np.repeat(np.arange(n_users), np.diff(plays.indptr))

    out = plays.copy()
    tf = plays.data * (k1 + 1.0) / (k1 * length[row_of_entry] + plays.data)
    out.data = (tf * idf[plays.indices]).astype(np.float32)
    out.data = np.maximum(out.data, 0.0)  # an artist heard by everyone carries no weight
    return out


TRANSFORMS = {"linear": linear, "log": log, "bm25": bm25}
