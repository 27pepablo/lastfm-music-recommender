"""Final evaluation on the held-out test period.

Takes the best configuration of every model from ``results/tuning.csv`` (chosen on
validation data only), retrains it on each user's first 80 % of plays and scores its
recommendations against the new artists the user played in the last 20 %.

    python scripts/03_evaluate.py

Writes ``results/metrics.json`` and ``results/model.npz`` (the final ALS factors).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from musicrec import dataset, weighting
from musicrec import evaluation as ev
from musicrec.models import ImplicitALS, ItemKNN, Popularity

ROOT = Path(__file__).resolve().parents[1]
K = 10
BETAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
ALS_PARAMS = ("factors", "regularization", "alpha")


def best_rows(tuning: pd.DataFrame) -> pd.DataFrame:
    """Best validation NDCG for every (model, weighting)."""
    return tuning.sort_values("ndcg", ascending=False).drop_duplicates(["model", "weighting"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=dataset.DEFAULT_DIR)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--seeds", type=int, default=5, help="ALS restarts, to measure seed noise")
    args = parser.parse_args()

    data = dataset.load(args.data)
    split = data.split("test")
    listeners = split.listeners
    chosen = best_rows(pd.read_csv(args.results / "tuning.csv"))

    def measure(scores: np.ndarray, target=split) -> dict:
        report = ev.evaluate(ev.top_k(scores, target.train, K), target.relevant, target.listeners)
        out = report.summary()
        for metric in ("precision", "recall", "ndcg", "hit"):
            out[f"{metric}_ci"] = report.confidence_interval(metric)
        out["users_evaluated"] = int(len(report.ndcg))
        per_user.append(report.ndcg)
        return out

    def paired_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 5000) -> dict:
        """Mean per-user NDCG difference a - b with a 95 % bootstrap interval (same users in both)."""
        diff = a - b
        draws = np.random.default_rng(0).integers(0, len(diff), size=(n_boot, len(diff)))
        means = diff[draws].mean(axis=1)
        return {
            "difference": float(diff.mean()),
            "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "share_of_resamples_above_zero": float((means > 0).mean()),
        }

    def als_params(row: pd.Series) -> dict:
        return {name: (int(row[name]) if name == "factors" else float(row[name])) for name in ALS_PARAMS}

    models: dict[str, dict] = {}
    per_user: list[np.ndarray] = []  # NDCG of every user for the most recent call to measure()
    ndcg_by_model: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    models["Random"] = {"metrics": measure(rng.random(split.train.shape, dtype=np.float32))}
    models["Popularity"] = {"metrics": measure(Popularity().fit(split.train).scores(split.train))}
    ndcg_by_model["Popularity"] = per_user[-1]

    for _, row in chosen[chosen["model"] == "knn"].iterrows():
        matrix = weighting.TRANSFORMS[row["weighting"]](split.train)
        scores = ItemKNN(int(row["n_neighbors"])).fit(matrix).scores(split.train)
        name = f"Item-kNN ({row['weighting']})"
        models[name] = {
            "params": {"n_neighbors": int(row["n_neighbors"])},
            "validation_ndcg": row["ndcg"],
            "metrics": measure(scores),
        }
        ndcg_by_model[name] = per_user[-1]
        print(f"{name}: ndcg={models[name]['metrics']['ndcg']:.4f}")

    fitted = {}
    for _, row in chosen[chosen["model"] == "als"].iterrows():
        params = als_params(row)
        matrix = weighting.TRANSFORMS[row["weighting"]](split.train)
        per_seed = []
        for seed in range(args.seeds):
            model = ImplicitALS(iterations=15, seed=seed, **params).fit(matrix)
            per_seed.append(measure(model.scores()))
            if seed == 0:
                fitted[row["weighting"]] = model
                ndcg_by_model[f"ALS ({row['weighting']})"] = per_user[-1]
        name = f"ALS ({row['weighting']})"
        models[name] = {
            "params": params,
            "validation_ndcg": row["ndcg"],
            "metrics": per_seed[0],
            "ndcg_per_seed": [m["ndcg"] for m in per_seed],
        }
        print(f"{name}: ndcg per seed = {np.round(models[name]['ndcg_per_seed'], 4)}")

    # ------------------------------------------------ best ALS: the model used from here on
    best_weighting = max(fitted, key=lambda w: models[f"ALS ({w})"]["validation_ndcg"])
    best = fitted[best_weighting]
    best_params = models[f"ALS ({best_weighting})"]["params"]
    print(f"selected on validation: ALS ({best_weighting}) {best_params}")

    # ------------------------------------------------ is the difference real? (same users, paired)
    selected_ndcg = ndcg_by_model[f"ALS ({best_weighting})"]
    best_knn = max(
        (n for n in models if n.startswith("Item-kNN")), key=lambda n: models[n]["validation_ndcg"]
    )
    comparisons = {
        other: paired_difference(selected_ndcg, ndcg_by_model[other])
        for other in ("Popularity", best_knn)
    }
    print("paired comparisons:", comparisons)

    # ------------------------------------------------ does down-weighting skips help?
    raw_split = data.split("test", column="plays")
    transform = weighting.TRANSFORMS[best_weighting]
    skip_ablation = {"skip-weighted plays": [], "raw plays": []}
    for seed in range(args.seeds):
        for label, target in (("skip-weighted plays", split), ("raw plays", raw_split)):
            model = ImplicitALS(iterations=15, seed=seed, **best_params).fit(transform(target.train))
            skip_ablation[label].append(measure(model.scores(), target)["ndcg"])
    print("skip ablation:", {k: float(np.mean(v)) for k, v in skip_ablation.items()})

    # ------------------------------------------------ accuracy against novelty
    penalty = []
    for beta in BETAS:
        result = measure(ev.popularity_penalty(best.scores(), listeners, beta))
        penalty.append({"beta": beta, **result})
        print(f"beta={beta}: ndcg={result['ndcg']:.4f} coverage={result['coverage']:.3f}")

    # ------------------------------------------------ how fast does it train?
    loss_curve = (
        ImplicitALS(iterations=15, seed=0, **best_params)
        .fit(transform(split.train), track_loss=True)
        .loss_
    )

    args.results.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.results / "model.npz", user_factors=best.user_factors_, item_factors=best.item_factors_
    )
    output = {
        "k": K,
        "protocol": {
            "users": data.n_users,
            "artists": data.n_artists,
            "train_pairs": int(split.train.nnz),
            "new_artists_to_find": int(split.relevant.nnz),
            "median_new_artists_per_user": float(np.median(np.diff(split.relevant.indptr))),
        },
        "models": models,
        "selected": {"weighting": best_weighting, "params": best_params},
        "selected_vs": comparisons,
        "skip_ablation": skip_ablation,
        "popularity_penalty": penalty,
        "loss_curve": loss_curve,
    }
    (args.results / "metrics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.results / 'metrics.json'}")


if __name__ == "__main__":
    main()
