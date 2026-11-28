"""WP-18 measurement 2, step 3: dHash every fetched image (same hash as WP-04's pool).

Streams the tar shards sequentially (USB platter: sequential reads only) and reuses
`experiments.wp04.dhash_pool.hash_one` so hashes are bit-comparable with
experiments/WP-04/dhash.parquet (the 2019 r/memes background control).

    uv run python -m experiments.wp18.m2_hash
"""
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from experiments.wp04.dhash_pool import hash_one
from experiments.wp18.common import OUT

ROOT = Path("/mnt/d/memes-multisub")


def one(name: str, body: bytes) -> dict:
    row = hash_one(name, body)
    row["ext"] = name.rsplit(".", 1)[1]
    row["n_bytes"] = len(body)
    return row


def main() -> None:
    idx = pq.read_table(ROOT / "index" / "shard_index.parquet").to_pandas()
    idx.to_parquet(OUT / "m2_shard_index.parquet", index=False)  # NVMe copy for analysis
    ok = idx[idx.fetch_status == "ok"]
    sha = dict(zip(ok.post_id, ok.sha256))
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for shard in sorted((ROOT / "images").glob("*.tar")):
            futs = []
            with tarfile.open(shard, mode="r") as tf:
                for m in tf:
                    futs.append(pool.submit(one, m.name, tf.extractfile(m).read()))
            rows.extend(f.result() for f in futs)
            print(shard.name, len(futs), flush=True)
    df = pd.DataFrame(rows)
    df["sha256"] = df.post_id.map(sha)
    df["dhash"] = df.dhash.astype("uint64")
    df.to_parquet(OUT / "m2_dhash.parquet", index=False)
    print(len(df), "members; decode failures:", int((~df.decode_ok).sum()),
          "; missing sha:", int(df.sha256.isna().sum()))


if __name__ == "__main__":
    main()
