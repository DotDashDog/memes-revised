"""WP-04 Method §1: draw the pre-registered uniform pool.

Registration (WP-04 Log, 2026-07-27): calendar year 2019 UTC, uniform draw of
N0 = 50,000 image posts from all hosts, seed 42, dedup by post_id. Run once;
refuses to overwrite an existing pool so the draw stays frozen.

Usage:
    uv run python -m experiments.wp04.draw_pool
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SUBMISSIONS = Path(
    "/home/linus/memes-revised/dataset_saves/reddit/submissions-2025-12/submissions.parquet"
)
OUT_DIR = Path("/home/linus/memes-revised/experiments/WP-04")

WINDOW_START = 1546300800  # 2019-01-01T00:00:00Z
WINDOW_END = 1577836800  # 2020-01-01T00:00:00Z
POOL_SIZE = 50_000
SEED = 42

COLUMNS = [
    "post_id",
    "img_url",
    "created_utc",
    "removed_by_category",
    "over_18",
    "score",
    "num_comments",
    "title",
]


def main() -> None:
    pool_path = OUT_DIR / "pool.parquet"
    if pool_path.exists():
        sys.exit(f"{pool_path} already exists — the draw is frozen; refusing to redraw.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(SUBMISSIONS, columns=COLUMNS)
    mask = (
        pc.and_(
            pc.and_(
                pc.is_valid(table["img_url"]),
                pc.not_equal(table["img_url"], ""),
            ),
            pc.and_(
                pc.greater_equal(table["created_utc"], WINDOW_START),
                pc.less(table["created_utc"], WINDOW_END),
            ),
        )
    )
    window = table.filter(mask)

    df = window.to_pandas()
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="post_id", keep="first")
    population = len(df)
    df = df.sort_values("post_id").reset_index(drop=True)  # deterministic order pre-draw

    rng = np.random.default_rng(SEED)
    idx = rng.choice(population, size=POOL_SIZE, replace=False)
    pool = df.iloc[np.sort(idx)].reset_index(drop=True)

    pool["host"] = pool["img_url"].str.extract(r"https?://([^/]+)/", expand=False)

    pq.write_table(pa.Table.from_pandas(pool, preserve_index=False), pool_path)
    digest = hashlib.sha256(pool_path.read_bytes()).hexdigest()

    host_counts = pool["host"].value_counts()
    meta = {
        "registration": "WP-04 Log 2026-07-27",
        "window_utc": [WINDOW_START, WINDOW_END],
        "window_label": "calendar year 2019 UTC",
        "population_filter": "img_url non-null and non-empty; dedup by post_id",
        "population_size": population,
        "population_rows_before_dedup": n_before_dedup,
        "pool_size": POOL_SIZE,
        "seed": SEED,
        "source_parquet": str(SUBMISSIONS),
        "pool_sha256": digest,
        "drawn_at": datetime.now(timezone.utc).isoformat(),
        "host_counts_top10": {h: int(c) for h, c in host_counts.head(10).items()},
        "n_removed_flagged": int(
            (pool["removed_by_category"].notna() & (pool["removed_by_category"] != "")).sum()
        ),
    }
    (OUT_DIR / "pool_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"population (2019 image posts, deduped): {population:,}")
    print(f"pool: {POOL_SIZE:,} rows -> {pool_path}")
    print(f"sha256: {digest}")
    print("top hosts:")
    for h, c in host_counts.head(10).items():
        print(f"  {h:30s} {c:>7,}")
    print(f"removed_by_category flagged: {meta['n_removed_flagged']:,}")


if __name__ == "__main__":
    main()
