"""WP-18 measurement 2, step 2: fetch the window's images with the WP-04 fetcher, unmodified.

`experiments/wp04/fetch.py` hard-codes its pool and output paths as module constants. This
wrapper re-points those constants (read at call time) and runs the same code, so every
politeness limit is exactly the WP-04 one: i.redd.it 4->16 workers with 0.1 s jittered pacing,
other hosts 2 workers at ~1 req/s, full stop on 429/403 bursts. Output: tar shards + index
under /mnt/d/memes-multisub/ (ADR 0015 layout; no loose files).

    uv run python -m experiments.wp18.m2_fetch [--limit N]
"""
import argparse
import asyncio
from pathlib import Path

from experiments.wp04 import fetch
from experiments.wp18.common import OUT

fetch.POOL = OUT / "m2_pool.parquet"
fetch.METRICS = OUT / "m2_fetch_metrics.jsonl"
fetch.OUT_ROOT = Path("/mnt/d/memes-multisub")
fetch.IMAGES_DIR = fetch.OUT_ROOT / "images"
fetch.INDEX_PARTS = fetch.OUT_ROOT / "index" / "parts"
fetch.SHARD_PREFIX = "multisub-2025-11"
# Size cap lowered from 30 MB to 10 MB to keep the fetch near 16 GB instead of 29 GB (r/whenthe
# is 97% GIF, mean ~7 MB). See expectations.md, deviation of 2026-11-27 12:30.
fetch.MAX_IMAGE_BYTES = 10 * 1024 * 1024

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    asyncio.run(fetch.run(ap.parse_args().limit))
