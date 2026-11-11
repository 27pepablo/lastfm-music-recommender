"""Choose hyperparameters on the validation split (fit on the first 70 % of each
user's history, check on the next 10 %). The test period is never touched here.

    python scripts/02_tune.py

Writes ``results/tuning.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from musicrec import dataset, weighting
from musicrec import evaluation as ev
from musicrec.models import ImplicitALS, ItemKNN

ROOT = Path(__file__).resolve().parents[1]
K = 10
ALPHA_RANGE = {"linear": (0.01, 10.0), "log": (0.3, 100.0), "bm25": (0.3, 100.0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=20, help="random ALS configurations per weighting")
    parser.add_argument("--data", type=Path, default=dataset.DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "tuning.csv")
    args = parser.parse_args()

    split = dataset.load(args.data).split("validation")
    listeners = split.listeners
    rows = []

    def record(model: str, transform: str, scores: np.ndarray, seconds: float, **params) -> None:
        summary = ev.evaluate(ev.top_k(scores, split.train, K), split.relevant, listeners).summary()
        rows.append({"model": model, "weighting": transform, **params, **summary, "seconds": seconds})
        print(f"{model:4s} {transform:6s} {params}  ndcg={summary['ndcg']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    weighted = {name: fn(split.train) for name, fn in weighting.TRANSFORMS.items()}

    for name, matrix in weighted.items():
        for n_neighbors in (20, 50, 100, 200, 400):
            start = time.perf_counter()
            scores = ItemKNN(n_neighbors).fit(matrix).scores(split.train)
            record("knn", name, scores, time.perf_counter() - start, n_neighbors=n_neighbors)

    rng = np.random.default_rng(0)
    for name, matrix in weighted.items():
        lo, hi = ALPHA_RANGE[name]
        for _ in range(args.trials):
            params = {
                "factors": int(rng.choice([32, 64, 128])),
                "regularization": float(np.round(10 ** rng.uniform(-1, 2), 3)),
                "alpha": float(np.round(10 ** rng.uniform(np.log10(lo), np.log10(hi)), 3)),
            }
            start = time.perf_counter()
            scores = ImplicitALS(iterations=15, **params).fit(matrix).scores()
            record("als", name, scores, time.perf_counter() - start, **params)


if __name__ == "__main__":
    main()
