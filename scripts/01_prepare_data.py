"""Download the Last.fm 1K dataset (optional) and build the interaction tables.

    python scripts/01_prepare_data.py --download          # fetch ~640 MB, then prepare
    python scripts/01_prepare_data.py --raw path/to/userid-timestamp-artid-artname-traid-traname.tsv

Writes ``data/processed/``: a few MB of parquet files that the other scripts read.
Reading and preparing the 19 million plays takes about a minute and 3 GB of RAM.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
import urllib.request
from pathlib import Path

from musicrec import prepare

ROOT = Path(__file__).resolve().parents[1]
URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
LOG_NAME = "userid-timestamp-artid-artname-traid-traname.tsv"


def download(raw_dir: Path) -> Path:
    """Fetch and unpack the official archive; return the path of the listening log."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "lastfm-dataset-1K.tar.gz"
    if not archive.exists():
        print(f"downloading {URL} ...")
        # the server rejects clients that do not send a browser-like User-Agent
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
            total, done = int(response.headers.get("Content-Length", 0)), 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"  {100 * done / total:5.1f} %", end="\r", flush=True)
        partial.rename(archive)
    with tarfile.open(archive) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(LOG_NAME))
        member.name = LOG_NAME  # extract flat, whatever folder the archive uses
        tar.extract(member, raw_dir)
    return raw_dir / LOG_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, help="path to the listening log (.tsv)")
    parser.add_argument("--download", action="store_true", help="download into data/raw first")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()

    if args.download:
        log_path = download(ROOT / "data" / "raw")
    elif args.raw:
        log_path = args.raw
    else:
        log_path = ROOT / "data" / "raw" / LOG_NAME
    if not log_path.exists():
        parser.error(f"{log_path} not found: pass --raw PATH or --download")

    start = time.perf_counter()
    raw = prepare.read_raw_log(log_path)
    print(f"read {len(raw):,} plays in {time.perf_counter() - start:.0f} s")

    start = time.perf_counter()
    prepared = prepare.prepare(raw)
    prepare.save(prepared, args.out)
    print(f"prepared in {time.perf_counter() - start:.0f} s -> {args.out}")
    print(json.dumps(prepared.stats, indent=2))


if __name__ == "__main__":
    main()
