"""Turn the raw Last.fm 1K listening log into compact user-artist interaction tables.

The raw file has one row per *play*: user, timestamp, artist, track (19 million rows).
The recommender works at the user-artist level, so this module

1. reads the log,
2. estimates which plays were skipped from the time until the user's next play,
3. merges duplicated artist entries,
4. splits every user's history in time (past -> train, future -> test),
5. keeps users and artists with enough history, and
6. counts plays per (user, artist, period).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["user_id", "timestamp", "artist_mbid", "artist_name", "track_mbid", "track_name"]

# A play followed by another play after `gap` seconds gets this weight.
# Shorter than 30 s is a skip; between 30 and 90 s is a partial listen.
SKIP_BREAKPOINTS = (30.0, 60.0, 90.0)
SKIP_WEIGHTS = (0.0, 0.5, 0.8, 1.0)

# Every user's plays are cut, in time order, into three periods.
PERIODS = ("fit", "val", "test")
FIT_END, VAL_END = 0.70, 0.80  # first 70 % | next 10 % | last 20 %

MIN_ARTISTS_PER_USER = 50
MIN_LISTENERS_PER_ARTIST = 5


@dataclass
class Prepared:
    """Output of :func:`prepare`."""

    interactions: pd.DataFrame  # user, artist, period, plays, weighted_plays
    artists: pd.DataFrame  # artist (index), name, mbid, plays, listeners
    users: pd.DataFrame  # user (index), user_id, n_plays
    stats: dict  # numbers for the data section of the README
    gap_histogram: pd.DataFrame  # seconds, plays: distribution of time to next play


def read_raw_log(path: str | Path) -> pd.DataFrame:
    """Read the raw tab-separated log, keeping only the columns the recommender needs."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=RAW_COLUMNS, block_size=64 << 20),
        # Track names contain stray quote characters, so quoting must be off.
        parse_options=pacsv.ParseOptions(
            delimiter="\t", quote_char=False, invalid_row_handler=lambda row: "skip"
        ),
        convert_options=pacsv.ConvertOptions(
            include_columns=["user_id", "timestamp", "artist_mbid", "artist_name"],
            column_types={
                "user_id": pa.dictionary(pa.int32(), pa.string()),
                "timestamp": pa.timestamp("s", tz="UTC"),
                "artist_mbid": pa.dictionary(pa.int32(), pa.string()),
                "artist_name": pa.dictionary(pa.int32(), pa.string()),
            },
            strings_can_be_null=True,
        ),
    )
    return table.to_pandas()


def skip_weight(gap_seconds: np.ndarray) -> np.ndarray:
    """Weight of a play given the seconds until the same user's next play.

    The log has no track durations, so the gap is the only evidence of whether a
    track was heard or skipped. The last play of a user (gap = NaN) counts in full.
    """
    weights = np.asarray(SKIP_WEIGHTS, dtype=np.float32)
    bucket = np.searchsorted(SKIP_BREAKPOINTS, gap_seconds, side="right")
    return np.where(np.isnan(gap_seconds), 1.0, weights[bucket]).astype(np.float32)


def normalise_name(names: pd.Series) -> pd.Series:
    """Lower-case, trim and collapse whitespace, so that spelling variants compare equal."""
    return names.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)


def canonical_artists(plays: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Give every play one artist key and return ``(key per play, artist table)``.

    The same artist can appear under a MusicBrainz id in some rows and with only a
    name in others. Plays without an id are attached to the most-played id that
    has the same normalised name; if there is none, the name itself is the key.
    """
    pairs = (
        plays.groupby(["artist_mbid", "artist_name"], observed=True, dropna=False)
        .size()
        .rename("plays")
        .reset_index()
    )
    pairs["artist_mbid"] = pairs["artist_mbid"].astype("string")
    pairs["artist_name"] = pairs["artist_name"].astype("string")
    pairs = pairs[pairs["artist_name"].notna()].copy()
    pairs["norm"] = normalise_name(pairs["artist_name"])
    pairs = pairs[pairs["norm"] != ""]

    with_id = pairs[pairs["artist_mbid"].notna()]
    best_id_for_name = (
        with_id.groupby(["norm", "artist_mbid"])["plays"]
        .sum()
        .reset_index()
        .sort_values("plays", ascending=False)
        .drop_duplicates("norm")
        .set_index("norm")["artist_mbid"]
    )
    # Plain Python strings here: when no row has an id at all, the id column has no
    # string type to speak of, and filling it with names fails on recent pandas.
    id_for_name = best_id_for_name.to_dict()
    ids, norms = pairs["artist_mbid"].astype(object), pairs["norm"].astype(object)
    pairs["key"] = [
        mbid if isinstance(mbid, str) else id_for_name.get(norm, "name:" + norm)
        for mbid, norm in zip(ids, norms, strict=True)
    ]

    # Display name of an artist: its most common spelling.
    display = (
        pairs.groupby(["key", "artist_name"])["plays"]
        .sum()
        .reset_index()
        .sort_values("plays", ascending=False)
        .drop_duplicates("key")
        .set_index("key")["artist_name"]
    )
    keys = pd.Index(pairs["key"].unique())
    pairs["artist"] = keys.get_indexer(pairs["key"])

    lookup = pairs.set_index(["artist_mbid", "artist_name"])["artist"]
    per_play = pd.MultiIndex.from_arrays(
        [plays["artist_mbid"].astype("string"), plays["artist_name"].astype("string")]
    )
    artist_of_play = lookup.reindex(per_play).to_numpy(dtype="float64")

    table = pd.DataFrame({"key": keys, "name": display.reindex(keys).to_numpy()})
    table["mbid"] = table["key"].where(~table["key"].str.startswith("name:"))
    return artist_of_play, table


def prepare(raw: pd.DataFrame) -> Prepared:
    """Run the whole preparation on a raw log (see the module docstring)."""
    stats: dict = {"raw_plays": int(len(raw)), "raw_users": int(raw["user_id"].nunique())}

    plays = raw.dropna(subset=["user_id", "timestamp"])
    plays = plays.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    user_codes = plays["user_id"].cat.remove_unused_categories().cat.codes.to_numpy()

    # ---- 1. skip weights, from the gap to the user's next play (on the full stream)
    seconds = plays["timestamp"].dt.as_unit("s").astype("int64").to_numpy()
    gap = np.full(len(plays), np.nan)
    same_user = user_codes[1:] == user_codes[:-1]
    gap[:-1][same_user] = (seconds[1:] - seconds[:-1])[same_user]
    weight = skip_weight(gap)

    capped = gap[~np.isnan(gap) & (gap <= 600)].astype(int)
    gap_histogram = pd.DataFrame(
        {"seconds": np.arange(601), "plays": np.bincount(capped, minlength=601)}
    )
    stats["share_skipped"] = float((weight == 0).mean())
    stats["share_partial"] = float(((weight > 0) & (weight < 1)).mean())
    stats["median_gap_seconds"] = float(np.nanmedian(gap))

    # ---- 2. one key per artist
    artist_of_play, artist_table = canonical_artists(plays)
    stats["raw_artist_entries"] = int(
        plays.groupby(["artist_mbid", "artist_name"], observed=True, dropna=False).ngroups
    )
    stats["artists_after_merging"] = int(len(artist_table))

    # ---- 3. position of each play in its user's history -> period
    plays_per_user = np.bincount(user_codes)
    first_row = np.concatenate([[0], np.cumsum(plays_per_user)[:-1]])
    position = (np.arange(len(plays)) - first_row[user_codes]) / plays_per_user[user_codes]
    period = np.digitize(position, [FIT_END, VAL_END]).astype(np.int8)  # 0 fit, 1 val, 2 test

    events = pd.DataFrame(
        {"user": user_codes, "artist": artist_of_play, "period": period, "weight": weight}
    ).dropna(subset=["artist"])
    events["artist"] = events["artist"].astype(np.int32)

    # ---- 4. filters, decided on the training window (fit + val) only
    train = events[events["period"] < 2]
    artists_per_user = train.groupby("user")["artist"].nunique()
    listeners = train.groupby("artist")["user"].nunique()
    keep_users = artists_per_user.index[artists_per_user >= MIN_ARTISTS_PER_USER]
    keep_artists = listeners.index[listeners >= MIN_LISTENERS_PER_ARTIST]
    events = events[events["user"].isin(keep_users) & events["artist"].isin(keep_artists)]
    stats["plays_kept"] = int(len(events))
    stats["share_plays_kept"] = float(len(events) / len(plays))

    # ---- 5. aggregate to (user, artist, period) and re-index users and artists densely
    interactions = (
        events.groupby(["user", "artist", "period"])
        .agg(plays=("weight", "size"), weighted_plays=("weight", "sum"))
        .reset_index()
    )
    user_index = pd.Index(np.sort(interactions["user"].unique()))
    artist_index = pd.Index(np.sort(interactions["artist"].unique()))
    interactions["user"] = user_index.get_indexer(interactions["user"]).astype(np.int32)
    interactions["artist"] = artist_index.get_indexer(interactions["artist"]).astype(np.int32)
    interactions["period"] = pd.Categorical.from_codes(interactions["period"], PERIODS)
    interactions["plays"] = interactions["plays"].astype(np.int32)

    user_ids = plays["user_id"].cat.remove_unused_categories().cat.categories
    users = pd.DataFrame({"user_id": user_ids[user_index], "n_plays": plays_per_user[user_index]})
    users.index.name = "user"

    artists = artist_table.iloc[artist_index].reset_index(drop=True)
    totals = interactions.groupby("artist").agg(plays=("plays", "sum"), listeners=("user", "nunique"))
    artists = artists.join(totals)
    artists.index.name = "artist"

    stats.update(
        users=int(len(users)),
        artists=int(len(artists)),
        user_artist_pairs=int(interactions.groupby(["user", "artist"]).ngroups),
        first_play=str(plays["timestamp"].min()),
        last_play=str(plays["timestamp"].max()),
    )
    return Prepared(interactions, artists, users, stats, gap_histogram)


def save(prepared: Prepared, out_dir: str | Path) -> None:
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared.interactions.to_parquet(out / "interactions.parquet", index=False)
    prepared.artists.to_parquet(out / "artists.parquet")
    prepared.users.to_parquet(out / "users.parquet")
    prepared.gap_histogram.to_parquet(out / "gap_histogram.parquet", index=False)
    (out / "stats.json").write_text(json.dumps(prepared.stats, indent=2) + "\n", encoding="utf-8")
