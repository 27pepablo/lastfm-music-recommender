import numpy as np
import pandas as pd
import pytest

from musicrec import prepare, synthetic


def make_log(rows):
    """rows: (user, 'HH:MM:SS', mbid or None, name)"""
    log = pd.DataFrame(rows, columns=["user_id", "timestamp", "artist_mbid", "artist_name"])
    log["timestamp"] = pd.to_datetime("2009-01-01 " + log["timestamp"], utc=True).astype(
        "datetime64[s, UTC]"
    )
    for column in ("user_id", "artist_mbid", "artist_name"):
        log[column] = log[column].astype("category")
    return log


def test_skip_weight_buckets():
    gaps = np.array([0, 29, 30, 59, 60, 89, 90, 240, 86400, np.nan])
    expected = [0.0, 0.0, 0.5, 0.5, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
    assert prepare.skip_weight(gaps) == pytest.approx(expected)


def test_normalise_name():
    names = pd.Series(["  The  Beatles ", "the beatles", "THE BEATLES"])
    assert prepare.normalise_name(names).nunique() == 1


def test_plays_without_id_join_the_artist_with_the_same_name():
    log = make_log(
        [
            ("u1", "10:00:00", "id-radiohead", "Radiohead"),
            ("u1", "10:05:00", "id-radiohead", "Radiohead"),
            ("u1", "10:10:00", None, "radiohead "),  # no id, sloppy spelling -> same artist
            ("u1", "10:15:00", None, "Unknown Band"),  # no id, no match -> its own artist
            ("u1", "10:20:00", "id-sakamoto", "坂本龍一"),
            ("u1", "10:25:00", "id-sakamoto", "Ryuichi Sakamoto"),  # same id, two spellings
        ]
    )
    artist_of_play, table = prepare.canonical_artists(log)
    assert len(table) == 3
    assert artist_of_play[0] == artist_of_play[1] == artist_of_play[2]
    assert artist_of_play[4] == artist_of_play[5]
    assert table.loc[int(artist_of_play[0]), "name"] == "Radiohead"  # most common spelling
    assert table["mbid"].isna().sum() == 1  # only "Unknown Band" has no id


@pytest.fixture(scope="module")
def prepared():
    return prepare.prepare(synthetic.listening_log())


def test_periods_follow_time_order_and_fractions(prepared):
    plays = prepared.interactions.groupby("period", observed=True)["plays"].sum()
    share = plays / plays.sum()
    assert share["fit"] == pytest.approx(0.70, abs=0.02)
    assert share["val"] == pytest.approx(0.10, abs=0.02)
    assert share["test"] == pytest.approx(0.20, abs=0.02)


def test_filters_and_dense_indices(prepared):
    it, artists, users = prepared.interactions, prepared.artists, prepared.users
    assert sorted(it["user"].unique()) == list(range(len(users)))
    assert sorted(it["artist"].unique()) == list(range(len(artists)))
    train = it[it["period"] != "test"]
    assert train.groupby("artist")["user"].nunique().min() >= prepare.MIN_LISTENERS_PER_ARTIST
    assert train.groupby("user")["artist"].nunique().min() >= prepare.MIN_ARTISTS_PER_USER


def test_skips_lower_the_weighted_count(prepared):
    it = prepared.interactions
    assert (it["weighted_plays"] <= it["plays"] + 1e-6).all()
    # the synthetic log skips about 4 % of plays
    assert prepared.stats["share_skipped"] == pytest.approx(0.04, abs=0.01)
    assert it["weighted_plays"].sum() < it["plays"].sum()


def test_gap_is_not_carried_across_users():
    # u1's last play is followed 5 s later by a play of u2: that must not count as a skip of u1's track.
    # u2 then skips its first track after 5 s.
    rows = []
    for user, first in (("u1", "10:00:00"), ("u2", "10:04:05")):
        rows += [(user, first, None, "a")]
    rows.insert(1, ("u1", "10:04:00", None, "b"))
    rows.append(("u2", "10:04:10", None, "b"))
    log = make_log(rows)
    log = log.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    users = log["user_id"].cat.codes.to_numpy()
    seconds = log["timestamp"].dt.as_unit("s").astype("int64").to_numpy()
    gap = np.full(len(log), np.nan)
    same = users[1:] == users[:-1]
    gap[:-1][same] = (seconds[1:] - seconds[:-1])[same]
    assert prepare.skip_weight(gap) == pytest.approx([1.0, 1.0, 0.0, 1.0])
