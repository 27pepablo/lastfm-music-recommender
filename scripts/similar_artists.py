"""Ask the trained model for artists similar to one you know. No dataset needed.

    python scripts/similar_artists.py "Daft Punk"
    python scripts/similar_artists.py "miles davis" --n 15

Uses ``results/artist_vectors.npz``, the latent vectors of the 5 000 most listened
artists, which ships with the repository.

    python scripts/similar_artists.py --export    # rebuild that file from results/model.npz
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import numpy as np
import pandas as pd

from musicrec import explore
from musicrec.models import ImplicitALS

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "results" / "artist_vectors.npz"
N_EXPORTED = 5000


def export() -> None:
    from musicrec import dataset

    artists = dataset.load().artists
    factors = np.load(ROOT / "results" / "model.npz")["item_factors"]
    top = artists["listeners"].nlargest(N_EXPORTED).index.to_numpy()
    np.savez_compressed(
        VECTORS,
        name=artists.loc[top, "name"].to_numpy(dtype=str),
        listeners=artists.loc[top, "listeners"].to_numpy(np.int32),
        plays=artists.loc[top, "plays"].to_numpy(np.int64),
        vectors=factors[top].astype(np.float16),
    )
    print(f"wrote {VECTORS} ({VECTORS.stat().st_size / 1e6:.1f} MB)")


def load() -> tuple[ImplicitALS, pd.DataFrame]:
    saved = np.load(VECTORS)
    model = ImplicitALS(factors=saved["vectors"].shape[1])
    model.item_factors_ = saved["vectors"].astype(np.float32)
    artists = pd.DataFrame({column: saved[column] for column in ("name", "listeners", "plays")})
    return model, artists


def find(artists: pd.DataFrame, query: str) -> int:
    lower = artists["name"].str.lower()
    exact = artists[lower == query.lower().strip()]
    if not exact.empty:
        return int(exact["listeners"].idxmax())
    close = difflib.get_close_matches(query.lower(), lower.tolist(), n=5, cutoff=0.6)
    hint = ", ".join(artists.loc[lower.isin(close), "name"]) or "nothing similar"
    raise SystemExit(f"No artist called {query!r} among the {len(artists):,} in the model. Try: {hint}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artist", nargs="?", help="artist name (case-insensitive)")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--export", action="store_true", help="rebuild results/artist_vectors.npz")
    args = parser.parse_args()
    if args.export:
        export()
        return
    if not args.artist:
        parser.error("give an artist name")

    model, artists = load()
    seed = find(artists, args.artist)
    name, listeners = artists.loc[seed, "name"], artists.loc[seed, "listeners"]
    print(f"\n{name}  ({listeners} of 960 listeners)\n")

    similar = explore.similar_artists(model, artists, seed, args.n)
    gems = explore.hidden_gems(model, artists, seed, args.n, max_listeners=40, min_listeners=15)
    table = pd.DataFrame(
        {
            "listeners who play this also play": similar["name"],
            "  listeners": similar["listeners"],
            "    lesser-known, same taste": gems["name"],
            " listeners": gems["listeners"],
        }
    )
    table.index = np.arange(1, len(table) + 1)
    print(table.to_string())


if __name__ == "__main__":
    main()
