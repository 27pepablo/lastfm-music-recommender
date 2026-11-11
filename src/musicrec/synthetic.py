"""A small synthetic listening log with known structure.

Used by the tests and the quick demo so that everything runs without downloading
the real dataset. Users belong to hidden "taste groups"; each group listens mostly
to its own artists, a few artists are popular with everyone, and some plays are
skipped after a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def listening_log(
    n_users: int = 120,
    n_groups: int = 4,
    artists_per_group: int = 40,
    n_hits: int = 8,
    plays_per_user: int = 600,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a raw log with the columns of the Last.fm file read by ``prepare.read_raw_log``.

    Artist names encode the truth: ``g2_a17`` belongs to taste group 2, ``hit_3`` is
    liked by everybody.
    """
    rng = np.random.default_rng(seed)
    group_artists = [[f"g{g}_a{a}" for a in range(artists_per_group)] for g in range(n_groups)]
    hits = [f"hit_{h}" for h in range(n_hits)]
    # within a group, artist popularity is long-tailed
    tail = 1.0 / np.arange(1, artists_per_group + 1)
    tail /= tail.sum()

    frames = []
    for u in range(n_users):
        group = u % n_groups
        # the user's own ranking of the group's artists: the group's long tail, reshuffled a little.
        # Plays are drawn from it, so artists far down the list tend to turn up late in the history.
        taste = tail * rng.lognormal(0.0, 1.0, artists_per_group)
        own = np.array(group_artists[group])[
            rng.choice(artists_per_group, size=plays_per_user, p=taste / taste.sum())
        ]
        other_group = (group + 1 + rng.integers(0, n_groups - 1, size=plays_per_user)) % n_groups
        other_artist = rng.integers(0, artists_per_group, plays_per_user)
        other = np.array([group_artists[g][a] for g, a in zip(other_group, other_artist, strict=True)])
        # 75 % own group, 20 % hits that everybody plays, 5 % anything else
        source = rng.random(plays_per_user)
        names = np.where(
            source < 0.75, own, np.where(source < 0.95, rng.choice(hits, plays_per_user), other)
        )

        # full listens last 2-6 minutes, about 4 % of plays are skipped within 20 seconds
        skipped = rng.random(plays_per_user) < 0.04
        duration = np.where(
            skipped, rng.integers(2, 20, plays_per_user), rng.integers(120, 360, plays_per_user)
        )
        start = pd.Timestamp("2008-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 10**6), unit="s")
        frames.append(
            pd.DataFrame(
                {
                    "user_id": f"user_{u:04d}",
                    "timestamp": start + pd.to_timedelta(np.cumsum(duration), unit="s"),
                    "artist_mbid": None,
                    "artist_name": names,
                }
            )
        )
    log = pd.concat(frames, ignore_index=True)
    log["timestamp"] = log["timestamp"].astype("datetime64[s, UTC]")
    for column in ("user_id", "artist_mbid", "artist_name"):
        log[column] = log[column].astype("category")
    return log
