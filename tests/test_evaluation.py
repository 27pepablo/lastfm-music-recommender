import numpy as np
import pytest
from scipy import sparse

from musicrec import dataset, prepare, synthetic
from musicrec import evaluation as ev


def test_top_k_skips_seen_and_orders_best_first():
    scores = np.array([[0.9, 0.8, 0.1, 0.5, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5]])
    seen = sparse.csr_matrix(np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]))
    assert ev.top_k(scores, seen, 3).tolist() == [[1, 4, 3], [3, 2, 1]]
    assert scores[0, 0] == 0.9  # the input is not modified


def test_metrics_by_hand():
    # user 0: relevant {1, 2}; recommended [1, 5, 2] -> hits at ranks 1 and 3
    # user 1: relevant {0};    recommended [3, 4, 5] -> no hit
    # user 2: nothing relevant -> left out of the averages
    relevant = sparse.csr_matrix(np.array([[0, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))
    recommended = np.array([[1, 5, 2], [3, 4, 5], [0, 1, 2]])
    listeners = np.array([10, 20, 30, 40, 50, 60])
    report = ev.evaluate(recommended, relevant, listeners)

    assert report.precision.tolist() == pytest.approx([2 / 3, 0.0])
    assert report.recall.tolist() == pytest.approx([1.0, 0.0])
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    ideal = 1 / np.log2(2) + 1 / np.log2(3)
    assert report.ndcg.tolist() == pytest.approx([dcg / ideal, 0.0])
    assert report.hit.tolist() == [1.0, 0.0]
    assert report.coverage == pytest.approx(5 / 6)  # artists 1, 2, 3, 4, 5 appear in the two lists
    assert report.median_listeners == pytest.approx(np.median([20, 60, 30, 40, 50, 60]))
    assert report.summary()["precision"] == pytest.approx(1 / 3)


def test_perfect_ranking_has_ndcg_one():
    relevant = sparse.csr_matrix(np.array([[1, 1, 0, 0]]))
    report = ev.evaluate(np.array([[0, 1]]), relevant, np.ones(4))
    assert report.ndcg[0] == pytest.approx(1.0)


def test_confidence_interval_contains_the_mean():
    relevant = sparse.csr_matrix((np.random.default_rng(0).random((200, 30)) < 0.2).astype(float))
    recommended = np.tile(np.arange(5), (200, 1))
    report = ev.evaluate(recommended, relevant, np.ones(30))
    lo, hi = report.confidence_interval("precision")
    assert lo < report.precision.mean() < hi


def test_popularity_penalty():
    scores = np.array([[1.0, 1.0, -0.5]])
    listeners = np.array([99.0, 0.0, 99.0])
    assert ev.popularity_penalty(scores, listeners, 0.0) is scores
    out = ev.popularity_penalty(scores, listeners, 0.5)
    assert out[0].tolist() == pytest.approx([0.1, 1.0, -0.5])  # hit pushed down, negative untouched


def test_relevant_items_are_new_to_the_user():
    prepared = prepare.prepare(synthetic.listening_log())
    data = dataset.from_tables(prepared.interactions, prepared.artists)
    for stage in ("validation", "test"):
        split = data.split(stage)
        assert split.train.multiply(split.relevant).nnz == 0  # never both known and "to discover"
        assert split.relevant.nnz > 0
        assert set(np.unique(split.relevant.data)) == {1.0}
    # the test stage trains on more history than the validation stage
    assert data.split("test").train.sum() > data.split("validation").train.sum()
