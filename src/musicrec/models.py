"""Recommendation models.

Every model exposes ``scores(train)``, a dense ``(n_users, n_artists)`` array where a
higher value means "recommend this artist to this user first".

* :class:`Popularity` - the same chart for everyone (baseline).
* :class:`ItemKNN`    - "people who listen to X also listen to Y" (strong classic baseline).
* :class:`ImplicitALS` - matrix factorisation for implicit feedback, written from scratch.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class Popularity:
    """Rank artists by how many users listen to them. Not personalised."""

    def fit(self, train: sparse.csr_matrix) -> Popularity:
        self.listeners_ = np.bincount(train.indices, minlength=train.shape[1]).astype(np.float32)
        return self

    def scores(self, train: sparse.csr_matrix) -> np.ndarray:
        return np.tile(self.listeners_, (train.shape[0], 1))


class ItemKNN:
    """Item-based nearest neighbours with cosine similarity.

    Two artists are similar if the same users listen to both. A user's score for
    an artist is the summed similarity between that artist and the user's own artists,
    weighted by how much the user listens to each of them.
    """

    def __init__(self, n_neighbors: int = 100, chunk: int = 2000):
        self.n_neighbors = n_neighbors
        self.chunk = chunk

    def fit(self, weighted: sparse.csr_matrix) -> ItemKNN:
        dense = weighted.toarray().astype(np.float32)  # users x artists
        norms = np.linalg.norm(dense, axis=0)
        unit = dense / np.where(norms > 0, norms, 1.0)
        n_items, k = dense.shape[1], min(self.n_neighbors, dense.shape[1] - 1)

        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.chunk):
            sim = unit[:, start : start + self.chunk].T @ unit  # (chunk, n_items)
            sim[np.arange(sim.shape[0]), np.arange(start, start + sim.shape[0])] = 0.0  # not itself
            top = np.argpartition(sim, -k, axis=1)[:, -k:]
            rows.append(np.repeat(np.arange(start, start + sim.shape[0]), k))
            cols.append(top.ravel())
            vals.append(np.take_along_axis(sim, top, axis=1).ravel())
        self.similarity_ = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
        )
        self._weighted = weighted
        return self

    def scores(self, train: sparse.csr_matrix) -> np.ndarray:
        # entry (u, i) = sum over the user's artists j of weight(u, j) * similarity(i, j)
        return np.asarray((self._weighted @ self.similarity_.T).todense(), dtype=np.float32)


class ImplicitALS:
    """Matrix factorisation for implicit feedback, solved by alternating least squares.

    Every user ``u`` gets a vector ``x_u`` and every artist ``i`` a vector ``y_i`` such
    that ``x_u . y_i`` is close to 1 if the user listens to the artist and close to 0
    otherwise. Observed pairs count more: their squared error is multiplied by
    ``1 + alpha * confidence``. Unobserved pairs still count, with weight 1, which is
    what lets the model learn from the artists a user does *not* play.

        minimise  sum_{u,i} w_ui * (p_ui - x_u . y_i)^2  +  reg * (sum ||x_u||^2 + sum ||y_i||^2)

    With the artist vectors fixed, each user vector is the solution of a small linear
    system, and vice versa, so the algorithm alternates between the two (Hu, Koren and
    Volinsky, 2008). Each sweep costs time proportional to the number of observed pairs,
    not to the size of the full matrix.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 1.0,
        alpha: float = 10.0,
        iterations: int = 15,
        cg_steps: int = 3,
        seed: int = 0,
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.cg_steps = cg_steps
        self.seed = seed

    def fit(self, confidence: sparse.csr_matrix, track_loss: bool = False) -> ImplicitALS:
        """``confidence`` is users x artists; zero means "never played"."""
        by_user = (confidence.tocsr() * self.alpha).astype(np.float32)
        by_item = by_user.T.tocsr()
        rng = np.random.default_rng(self.seed)
        n_users, n_items = by_user.shape
        self.user_factors_ = (rng.standard_normal((n_users, self.factors)) * 0.01).astype(np.float32)
        self.item_factors_ = (rng.standard_normal((n_items, self.factors)) * 0.01).astype(np.float32)
        self.loss_ = []
        for _ in range(self.iterations):
            self.user_factors_ = _solve_side(
                by_user, self.item_factors_, self.user_factors_, self.regularization, self.cg_steps
            )
            self.item_factors_ = _solve_side(
                by_item, self.user_factors_, self.item_factors_, self.regularization, self.cg_steps
            )
            if track_loss:
                self.loss_.append(self.loss(by_user))
        return self

    def loss(self, extra_weight: sparse.csr_matrix) -> float:
        """Value of the objective (``extra_weight`` = alpha * confidence)."""
        X, Y = self.user_factors_.astype(np.float64), self.item_factors_.astype(np.float64)
        # all pairs with weight 1 and target 0:  sum (x.y)^2 = trace(X^T X  Y^T Y)
        total = np.sum((X.T @ X) * (Y.T @ Y))
        coo = extra_weight.tocoo()
        pred = np.einsum("ij,ij->i", X[coo.row], Y[coo.col])
        # observed pairs: replace the (0 - pred)^2 term by (1 + a) * (1 - pred)^2
        total += np.sum((1.0 + coo.data) * (1.0 - pred) ** 2 - pred**2)
        return float(total + self.regularization * (np.sum(X * X) + np.sum(Y * Y)))

    def scores(self, train: sparse.csr_matrix | None = None) -> np.ndarray:
        return self.user_factors_ @ self.item_factors_.T

    def similar_items(self, item: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """The ``n`` artists whose vectors point in the most similar direction (cosine)."""
        unit = _unit_rows(self.item_factors_)
        sim = unit @ unit[item]
        sim[item] = -np.inf
        top = np.argsort(-sim)[:n]
        return top, sim[top]


def _solve_side(
    extra_weight: sparse.csr_matrix, fixed: np.ndarray, start: np.ndarray, reg: float, cg_steps: int
) -> np.ndarray:
    """Update every row vector given the other side's vectors.

    For row ``u`` with observed columns ``S`` and extra weights ``a`` the new vector solves

        (Y^T Y  +  reg * I  +  Y_S^T diag(a) Y_S) x_u  =  Y_S^T (1 + a)

    Instead of factorising one matrix per row, all the systems are solved together with
    a few conjugate-gradient steps (Takacs et al., 2011), starting from the previous
    vectors. One step needs only ``A_u p_u`` for every row, and that is two sparse
    products: cost proportional to the number of observed pairs, with no Python loop.
    """
    Y = fixed.astype(np.float32)
    gram = Y.T @ Y + reg * np.eye(Y.shape[1], dtype=np.float32)
    rows = np.repeat(np.arange(extra_weight.shape[0]), np.diff(extra_weight.indptr))
    cols = extra_weight.indices
    correction = extra_weight.copy()

    def apply(P: np.ndarray) -> np.ndarray:
        """Row-wise ``A_u p_u``."""
        # p_u . y_i for every observed pair, scaled by its extra weight, then summed back per row
        correction.data = extra_weight.data * np.einsum("ij,ij->i", P[rows], Y[cols])
        return P @ gram + correction @ Y

    one_plus = extra_weight.copy()
    one_plus.data = one_plus.data + 1.0
    X = start.astype(np.float32).copy()
    R = one_plus @ Y - apply(X)
    P = R.copy()
    rs = np.einsum("ij,ij->i", R, R)
    for _ in range(cg_steps):
        AP = apply(P)
        step = rs / np.maximum(np.einsum("ij,ij->i", P, AP), 1e-30)
        X += step[:, None] * P
        R -= step[:, None] * AP
        rs_new = np.einsum("ij,ij->i", R, R)
        P = R + (rs_new / np.maximum(rs, 1e-30))[:, None] * P
        rs = rs_new
    return X


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1.0)
