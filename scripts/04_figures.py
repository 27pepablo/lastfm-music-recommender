"""Draw the figures of the README and write the example tables.

    python scripts/04_figures.py

Needs ``data/processed/`` (script 01) and ``results/`` (scripts 02 and 03).
Writes ``docs/figures/*.png`` and ``results/examples.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from musicrec import dataset, explore, weighting
from musicrec import evaluation as ev
from musicrec.models import ImplicitALS, Popularity, _unit_rows

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, ACCENT, BASE = "#1b1f23", "#8a9099", "#d1495b", "#3a7ca5"
MAP_ARTISTS = 1500
MAP_CLUSTERS = 12
# where to write each model's name around its point in the accuracy/variety plot (points)
LABEL_OFFSETS = {
    "ALS (bm25)": (-9, 5),
    "ALS (linear)": (9, 5),
    "Item-kNN (log)": (-9, -6),
    "ALS (log)": (9, -5),
}
SEEDS = ("Radiohead", "Daft Punk", "Miles Davis", "Metallica", "Johnny Cash", "Jay-Z")

plt.rcParams.update(
    {
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
    }
)


def load_model(results: Path) -> ImplicitALS:
    saved = np.load(results / "model.npz")
    model = ImplicitALS(factors=saved["item_factors"].shape[1])
    model.user_factors_, model.item_factors_ = saved["user_factors"], saved["item_factors"]
    return model


# ------------------------------------------------------------------ taste map


def taste_map(model: ImplicitALS, artists: pd.DataFrame, path: Path) -> None:
    """2-D map of the best-known artists, coloured by cluster of their latent vectors."""
    top = artists["listeners"].nlargest(MAP_ARTISTS).index.to_numpy()
    vectors = _unit_rows(model.item_factors_[top])
    xy = TSNE(metric="cosine", init="pca", perplexity=30, random_state=0).fit_transform(vectors)
    n_clusters = min(MAP_CLUSTERS, len(top) // 5)
    cluster = KMeans(n_clusters, n_init=10, random_state=0).fit_predict(vectors)
    listeners = artists.loc[top, "listeners"].to_numpy()
    names = artists.loc[top, "name"].to_numpy()

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[[0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 5, 9]]
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.scatter(
        xy[:, 0], xy[:, 1], c=colors[cluster], s=6 + 60 * (listeners / listeners.max()) ** 1.5,
        alpha=0.75, linewidths=0,
    )  # fmt: skip
    # name the most listened artists of every cluster, skipping names that would overlap
    span = xy.max(axis=0) - xy.min(axis=0)
    placed: list[np.ndarray] = []
    for rank in range(10):
        for c in range(n_clusters):
            members = np.flatnonzero(cluster == c)
            if rank >= len(members):
                continue
            i = members[np.argsort(-listeners[members])[rank]]
            width = (
                0.0055 * len(names[i]) + 0.012
            )  # rough half-width of the text, as a share of the plot
            clash = any(
                abs(xy[i, 0] - q[0]) / span[0] < width + q[2] and abs(xy[i, 1] - q[1]) / span[1] < 0.024
                for q in placed
            )
            if clash:
                continue
            placed.append(np.array([xy[i, 0], xy[i, 1], width]))
            ax.annotate(
                names[i], xy[i], fontsize=8.5, ha="center", va="center", color=INK, weight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            )  # fmt: skip
    ax.text(
        0, 1.07, f"The {len(top):,} most listened artists, placed by who listens to them",
        transform=ax.transAxes, va="bottom", fontsize=15, weight="bold",
    )  # fmt: skip
    ax.text(
        0, 1.03, "Each dot is an artist. Nearby dots share listeners. "
        "Colours are clusters found without any genre information.",
        transform=ax.transAxes, va="bottom", color=MUTED, fontsize=10.5,
    )  # fmt: skip
    ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------- data overview


def data_overview(data: dataset.Dataset, processed: Path, path: Path) -> None:
    artists = data.artists
    gaps = pd.read_parquet(processed / "gap_histogram.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # 1. long tail
    ax = axes[0]
    plays = np.sort(artists["plays"].to_numpy())[::-1]
    share = np.cumsum(plays) / plays.sum()
    rank = np.arange(1, len(plays) + 1)
    ax.loglog(rank, plays, color=BASE, lw=2)
    top1 = max(1, len(plays) // 100)
    ax.axvspan(1, top1, color=ACCENT, alpha=0.12, lw=0)
    ax.text(
        1.3, plays[-1] * 3,
        f"the top 1 % of artists\n({top1:,}) get {share[top1 - 1]:.0%}\nof all plays",
        color=ACCENT, fontsize=9.5, va="bottom",
    )  # fmt: skip
    ax.set_xlabel("artist rank (most played first)")
    ax.set_ylabel("plays")
    ax.set_title("A few artists get most of the plays", loc="left", weight="bold")

    # 2. time to next play
    ax = axes[1]
    seconds, count = gaps["seconds"].to_numpy(), gaps["plays"].to_numpy()
    ax.fill_between(seconds, count / 1e3, step="mid", color=BASE, alpha=0.85, lw=0)
    ax.axvspan(0, 30, color=ACCENT, alpha=0.25, lw=0)
    ax.axvspan(30, 90, color=ACCENT, alpha=0.10, lw=0)
    ymax = count[30:].max() / 1e3
    ax.text(15, ymax * 0.97, "skip", color=ACCENT, ha="center", va="top", fontsize=9, rotation=90)
    ax.text(60, ymax * 0.97, "partial", color=ACCENT, ha="center", va="top", fontsize=9, rotation=90)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_xlabel("seconds until the same user's next play")
    ax.set_ylabel("plays (thousands)")
    ax.set_title("How long a track was played", loc="left", weight="bold")

    # 3. play count -> confidence
    ax = axes[2]
    counts = np.unique(np.round(np.logspace(0, 3, 60)))
    column = sparse.csr_matrix(counts.reshape(1, -1).astype(np.float32))
    curves = {
        "linear": weighting.linear(column).toarray().ravel(),
        "log": weighting.log(column).toarray().ravel(),
        # BM25 saturation only (one user of average activity, rarity factor left out)
        "bm25": counts * 2.2 / (1.2 + counts),
    }
    for (name, values), color in zip(curves.items(), (MUTED, BASE, ACCENT), strict=True):
        ax.plot(counts, values / values[0], lw=2.2, color=color, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("times a user played an artist")
    ax.set_ylabel("weight relative to a single play")
    ax.set_title("Three ways to read a play count", loc="left", weight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -------------------------------------------------------------------- results


def results_figure(metrics: dict, path: Path) -> None:
    order = sorted(
        (name for name in metrics["models"] if name != "Random"),
        key=lambda name: metrics["models"][name]["metrics"]["ndcg"],
    )
    values = [metrics["models"][name]["metrics"] for name in order]
    best = f"ALS ({metrics['selected']['weighting']})"  # chosen on validation, not on these numbers
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    ndcg = np.array([v["ndcg"] for v in values])
    ci = np.array([v["ndcg_ci"] for v in values])
    colors = [ACCENT if name == best else (MUTED if name == "Popularity" else BASE) for name in order]
    ax.barh(order, ndcg, xerr=[ndcg - ci[:, 0], ci[:, 1] - ndcg], color=colors, ecolor=INK, capsize=3)
    for y, value in enumerate(ndcg):
        ax.text(ci[y, 1] + 0.002, y, f"{value:.3f}", va="center", fontsize=9.5)
    ax.set_xlabel(f"NDCG@{metrics['k']} on the test period (bars: 95 % interval over users)")
    ax.set_title("Accuracy: finding the new artists a user will play", loc="left", weight="bold")
    ax.set_xlim(0, ci[:, 1].max() * 1.15)

    ax = axes[1]
    for name, v in zip(order, values, strict=True):
        color = ACCENT if name == best else (MUTED if name == "Popularity" else BASE)
        ax.scatter(100 * v["coverage"], v["ndcg"], s=90, color=color, zorder=3)
        dx, dy = LABEL_OFFSETS.get(name, (9, 0))
        ax.annotate(
            name, (100 * v["coverage"], v["ndcg"]), fontsize=9, va="center",
            xytext=(dx, dy), textcoords="offset points", ha="right" if dx < 0 else "left",
        )  # fmt: skip
    ax.set_xlabel("catalogue coverage (% of artists recommended to someone)")
    ax.set_ylabel(f"NDCG@{metrics['k']}")
    ax.set_title("Accuracy and variety", loc="left", weight="bold")
    ax.grid(alpha=0.2)
    ax.set_xlim(left=0)
    ax.margins(x=0.25, y=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def novelty_figure(metrics: dict, path: Path) -> None:
    sweep = pd.DataFrame(metrics["popularity_penalty"])
    popularity = metrics["models"]["Popularity"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, column, label, scale in (
        (axes[0], "coverage", "catalogue coverage (% of artists recommended to someone)", 100),
        (axes[1], "median_listeners", "median audience of a recommended artist (listeners)", 1),
    ):
        ax.plot(scale * sweep[column], sweep["ndcg"], "-o", color=ACCENT, lw=2, zorder=3)
        for _, row in sweep.iterrows():
            if column == "median_listeners" and row["beta"] == 0.05:
                continue  # too close to its neighbours on this axis
            ax.annotate(
                f"β={row['beta']:g}", (scale * row[column], row["ndcg"]),
                xytext=(6, 5), textcoords="offset points", fontsize=8.5,
            )  # fmt: skip
        ax.axhline(popularity["ndcg"], color=MUTED, ls="--", lw=1.2)
        ax.set_xlabel(label)
        ax.grid(alpha=0.2)
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[0].text(
        0.02, popularity["ndcg"], "popularity baseline", color=MUTED, fontsize=9,
        ha="left", va="bottom", transform=axes[0].get_yaxis_transform(),
    )  # fmt: skip
    axes[0].set_ylabel(f"NDCG@{metrics['k']}")
    axes[0].set_title(
        "Trading accuracy for variety with a popularity penalty", loc="left", weight="bold"
    )
    axes[1].set_title("... and for less mainstream artists", loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------- examples


def markdown_table(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns)
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy())
    return head + "\n" + body


def examples(model: ImplicitALS, data: dataset.Dataset, path: Path) -> None:
    artists = data.artists
    lines = ["# Examples produced by the final model", ""]

    lines += ["## Similar artists", ""]
    table = {}
    for seed in SEEDS:
        try:
            table[seed] = explore.similar_artists(model, artists, data.find_artist(seed), n=8)[
                "name"
            ].tolist()
        except KeyError:
            continue
    lines += [markdown_table(pd.DataFrame(table)), ""]

    lines += ["## Hidden gems (8 to 25 listeners)", ""]
    table = {}
    for seed in SEEDS:
        try:
            gems = explore.hidden_gems(model, artists, data.find_artist(seed), n=8)
        except KeyError:
            continue
        table[seed] = [
            f"{name} ({listeners})"
            for name, listeners in zip(gems["name"], gems["listeners"], strict=True)
        ]
    lines += [markdown_table(pd.DataFrame(table)), ""]

    # one listener: what they play, what a chart would suggest, what the model suggests
    split = data.split("test")
    recommended = ev.top_k(model.scores(), split.train, 10)
    chart = ev.top_k(Popularity().fit(split.train).scores(split.train), split.train, 10)
    hits = np.asarray(split.relevant[np.arange(data.n_users)[:, None], recommended].todense()).sum(
        axis=1
    )
    lines += ["## Three listeners", ""]
    ranked = np.argsort(-hits)
    for user in ranked[[len(ranked) // 100, len(ranked) // 15, len(ranked) // 4]]:
        history = split.train[user]
        favourites = history.indices[np.argsort(-history.data)[:10]]
        played_later = set(split.relevant[user].indices)

        def mark(items, played_later=played_later):
            return [artists.loc[i, "name"] + (" ✔" if i in played_later else "") for i in items]

        frame = pd.DataFrame(
            {
                "most played so far": artists.loc[favourites, "name"].tolist(),
                "chart (popularity)": mark(chart[user]),
                "model (ALS)": mark(recommended[user]),
            }
        )
        lines += [f"### Listener {user}", "", markdown_table(frame), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=dataset.DEFAULT_DIR)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = dataset.load(args.data)
    metrics = json.loads((args.results / "metrics.json").read_text(encoding="utf-8"))
    model = load_model(args.results)

    data_overview(data, args.data, args.out / "data_overview.png")
    results_figure(metrics, args.out / "results.png")
    novelty_figure(metrics, args.out / "accuracy_vs_novelty.png")
    examples(model, data, args.results / "examples.md")
    taste_map(model, data.artists, args.out / "taste_map.png")
    print(f"figures in {args.out}, examples in {args.results / 'examples.md'}")


if __name__ == "__main__":
    main()
