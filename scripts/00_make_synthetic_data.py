"""Create a small synthetic dataset so the whole pipeline can run without Last.fm.

    python scripts/00_make_synthetic_data.py --out demo/data
    python scripts/02_tune.py     --data demo/data --out demo/results/tuning.csv --trials 3
    python scripts/03_evaluate.py --data demo/data --results demo/results --seeds 1
    python scripts/04_figures.py  --data demo/data --results demo/results --out demo/figures

The synthetic listeners belong to hidden taste groups, so there is a right answer for
the models to find. The continuous-integration job runs exactly these four commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from musicrec import prepare, synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--users", type=int, default=200)
    args = parser.parse_args()
    prepared = prepare.prepare(synthetic.listening_log(n_users=args.users))
    prepare.save(prepared, args.out)
    print(f"{prepared.stats['users']} users, {prepared.stats['artists']} artists -> {args.out}")


if __name__ == "__main__":
    main()
