import numpy as np
import pytest
from scipy import sparse

from musicrec import dataset, explore, prepare, synthetic, weighting
from musicrec import evaluation as ev
from musicrec.models import ImplicitALS, ItemKNN, Popularity, _solve_side


@pytest.fixture(scope="module")
def data():
    prepared = prepare.prepare(synthetic.listening_log())
    return dataset.from_tables(prepared.interactions, prepared.artists)


@pytest.fixture(scope="module")
def split(data):
    return data.split("test")


@pytest.fixture(scope="module")
def als(split):
    return ImplicitALS(factors=8, regularization=1.0, alpha=5.0, iterations=20).fit(
        weighting.log(split.train), track_loss=True
    )


def ndcg(scores, split):
    return ev.evaluate(ev.top_k(scores, split.train, 10), split.relevant, split.listeners).summary()[
        "ndcg"
    ]


# ---------------------------------------------------------------- weighting


def test_log_weighting():
    plays = sparse.csr_matrix(np.array([[0.0, 1.0, 9.0]]))
    assert weighting.log(plays).toarray() == pytest.approx(np.log1p([[0.0, 1.0, 9.0]]))


def test_bm25_by_hand():
    # 3 users x 2 artists; artist 0 is played by everyone, artist 1 by one user
    plays = sparse.csr_matrix(np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 2.0]]))
    out = weighting.bm25(plays, k1=1.2, b=0.75).toarray()
    idf_rare = np.log(3) - np.log1p(1)
    length = 0.25 + 0.75 * 4 / (8 / 3)  # user 2 has 4 plays, the average is 8/3
    assert out[2, 1] == pytest.approx(idf_rare * 2 * 2.2 / (1.2 * length + 2))
    assert out[2, 1] > out[2, 0]  # the rare artist says more about the user than the hit
    assert (out[:, 0] == 0).all()  # log(3) - log(1 + 3) < 0 is clipped: no information


def test_transforms_keep_the_sparsity_pattern(split):
    for transform in weighting.TRANSFORMS.values():
        out = transform(split.train)
        assert out.shape == split.train.shape
        assert out.nnz == split.train.nnz
        assert np.isfinite(out.data).all() and (out.data >= 0).all()


# ---------------------------------------------------------------------- ALS


def test_conjugate_gradient_matches_direct_solution():
    rng = np.random.default_rng(0)
    extra = sparse.random(30, 50, density=0.2, random_state=1, format="csr", dtype=np.float32) * 5
    Y = rng.standard_normal((50, 6)).astype(np.float32)
    reg = 0.7
    solved = _solve_side(extra.tocsr(), Y, np.zeros((30, 6), np.float32), reg, cg_steps=50)

    dense, Y64 = extra.toarray().astype(np.float64), Y.astype(np.float64)
    for u in range(30):
        S = dense[u] > 0
        A = Y64.T @ Y64 + reg * np.eye(6) + (Y64[S] * dense[u, S][:, None]).T @ Y64[S]
        expected = np.linalg.solve(A, Y64[S].T @ (1 + dense[u, S]))
        assert solved[u] == pytest.approx(expected, abs=2e-3)


def test_loss_matches_brute_force(als, split):
    extra = (weighting.log(split.train) * als.alpha).toarray().astype(np.float64)
    X, Y = als.user_factors_.astype(np.float64), als.item_factors_.astype(np.float64)
    target = (extra > 0).astype(np.float64)
    brute = np.sum((1 + extra) * (target - X @ Y.T) ** 2) + als.regularization * (
        np.sum(X**2) + np.sum(Y**2)
    )
    assert als.loss(sparse.csr_matrix(extra)) == pytest.approx(brute, rel=1e-6)


def test_loss_decreases(als):
    assert als.loss_[-1] < 0.8 * als.loss_[0]
    # each sweep minimises the objective approximately, so allow a tiny tolerance
    assert all(b <= a * 1.001 for a, b in zip(als.loss_, als.loss_[1:], strict=False))


def test_same_seed_same_model(split):
    a = ImplicitALS(factors=4, iterations=3, seed=3).fit(split.train)
    b = ImplicitALS(factors=4, iterations=3, seed=3).fit(split.train)
    assert np.array_equal(a.item_factors_, b.item_factors_)


def test_personalised_models_beat_popularity(als, split):
    popularity = ndcg(Popularity().fit(split.train).scores(split.train), split)
    knn = ndcg(ItemKNN(20).fit(weighting.log(split.train)).scores(split.train), split)
    assert ndcg(als.scores(), split) > 1.5 * popularity
    assert knn > 1.5 * popularity


def test_latent_space_recovers_the_hidden_taste_groups(als, data):
    names = data.artists["name"]
    group_artists = names[names.str.startswith("g")]
    agree = []
    for artist in group_artists.index[:40]:
        neighbours = explore.similar_artists(als, data.artists, artist, n=5)["name"]
        agree.append((neighbours.str.split("_").str[0] == names[artist].split("_")[0]).mean())
    assert np.mean(agree) > 0.9


def test_hidden_gems_are_little_known(als, data):
    seed = data.find_artist("g0_a0")
    ceiling = int(data.artists["listeners"].quantile(0.4))  # bottom 40 % of artists by audience
    gems = explore.hidden_gems(als, data.artists, seed, n=5, max_listeners=ceiling, min_listeners=5)
    assert len(gems) == 5
    assert gems["listeners"].between(5, ceiling).all()
    assert gems["name"].str.startswith("g0_").mean() >= 0.8  # and they share the seed's taste group
    assert (gems["similarity"].diff().dropna() <= 1e-9).all()  # sorted, most similar first


# -------------------------------------------------------------------- others


def test_popularity_scores_are_listener_counts(split):
    scores = Popularity().fit(split.train).scores(split.train)
    assert scores.shape == split.train.shape
    assert np.array_equal(scores[0], split.listeners)


def test_knn_keeps_k_neighbours_and_no_self_similarity(split):
    knn = ItemKNN(n_neighbors=7).fit(weighting.log(split.train))
    assert (np.diff(knn.similarity_.indptr) <= 7).all()
    assert knn.similarity_.diagonal().max() == 0.0
    assert knn.similarity_.max() <= 1.0 + 1e-5


def test_find_artist(data):
    assert data.artists.loc[data.find_artist("HIT_0"), "name"] == "hit_0"
    with pytest.raises(KeyError):
        data.find_artist("nobody")
