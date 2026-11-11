"""Load the prepared interaction tables as sparse user x artist matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


@dataclass
class Split:
    """One train/evaluate split.

    ``train``    plays in the past (skip-weighted), users x artists
    ``relevant`` 1 where the user played, in the held-out future, an artist that is
                 absent from ``train`` - the artists a recommender should have found
    """

    train: sparse.csr_matrix
    relevant: sparse.csr_matrix

    @property
    def listeners(self) -> np.ndarray:
        """Number of training users per artist."""
        return np.bincount(self.train.indices, minlength=self.train.shape[1]).astype(np.float32)


@dataclass
class Dataset:
    interactions: pd.DataFrame
    artists: pd.DataFrame
    n_users: int
    n_artists: int

    def matrix(self, periods: tuple[str, ...], column: str = "weighted_plays") -> sparse.csr_matrix:
        """Sum of ``column`` over the given periods, as a users x artists matrix."""
        rows = self.interactions[self.interactions["period"].isin(periods)]
        m = sparse.coo_matrix(
            (rows[column].to_numpy(np.float32), (rows["user"].to_numpy(), rows["artist"].to_numpy())),
            shape=(self.n_users, self.n_artists),
        ).tocsr()
        m.eliminate_zeros()  # pairs whose every play was a skip
        return m

    def split(self, stage: str, column: str = "weighted_plays", min_plays: float = 1.0) -> Split:
        """``"validation"``: fit -> val (for tuning). ``"test"``: fit + val -> test (final numbers)."""
        past, future = {"validation": (("fit",), ("val",)), "test": (("fit", "val"), ("test",))}[stage]
        train = self.matrix(past, column)
        # ground truth always uses skip-weighted plays, whatever the model was trained on
        ahead = self.matrix(future, "weighted_plays")
        ahead.data = (ahead.data >= min_plays).astype(np.float32)
        ahead.eliminate_zeros()
        known = self.matrix(past, "plays")
        known.data[:] = 1.0
        relevant = (ahead - ahead.multiply(known)).tocsr()
        relevant.eliminate_zeros()
        return Split(train, relevant)

    def find_artist(self, name: str) -> int:
        """Index of the most-played artist with this name (case-insensitive)."""
        match = self.artists[self.artists["name"].str.lower() == name.lower().strip()]
        if match.empty:
            raise KeyError(f"no artist named {name!r}")
        return int(match["plays"].idxmax())


def from_tables(interactions: pd.DataFrame, artists: pd.DataFrame) -> Dataset:
    return Dataset(interactions, artists, int(interactions["user"].max()) + 1, len(artists))


def load(directory: str | Path = DEFAULT_DIR) -> Dataset:
    directory = Path(directory)
    interactions = pd.read_parquet(directory / "interactions.parquet")
    artists = pd.read_parquet(directory / "artists.parquet")
    return from_tables(interactions, artists)
