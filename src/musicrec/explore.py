"""Browse the artist space a trained model has learned."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import ImplicitALS, _unit_rows
from .prepare import normalise_name


def similar_artists(model: ImplicitALS, artists: pd.DataFrame, artist: int, n: int = 10) -> pd.DataFrame:
    """The ``n`` artists closest to ``artist`` in the latent space."""
    idx, sim = model.similar_items(artist, n)
    out = artists.loc[idx, ["name", "listeners", "plays"]].copy()
    out.insert(1, "similarity", sim)
    return out.reset_index(drop=True)


def hidden_gems(
    model: ImplicitALS,
    artists: pd.DataFrame,
    artist: int,
    n: int = 10,
    max_listeners: int = 25,
    min_listeners: int = 8,
) -> pd.DataFrame:
    """Little-known artists that sit close to ``artist`` in the latent space.

    Candidates have between ``min_listeners`` and ``max_listeners`` listeners: few
    enough to be a discovery, enough for their vector to be trustworthy. Entries
    that are just a misspelt or duplicated copy of a better-known artist are dropped.
    """
    unit = _unit_rows(model.item_factors_)
    sim = unit @ unit[artist]

    norm = normalise_name(artists["name"])
    most_listened_with_name = artists.groupby(norm.to_numpy())["listeners"].transform("max")
    is_duplicate = artists["listeners"] < most_listened_with_name
    candidate = artists["listeners"].between(min_listeners, max_listeners) & ~is_duplicate
    candidate.loc[artist] = False

    idx = candidate.index[candidate.to_numpy()]
    top = idx[np.argsort(-sim[idx])[:n]]
    out = artists.loc[top, ["name", "listeners", "plays"]].copy()
    out.insert(1, "similarity", sim[top])
    return out.reset_index(drop=True)
