"""WP-18 m2: verify the tar shards against the fetch index after the unplanned shutdown.

Checks: every shard is a complete tar (iterable to its end-of-archive), no .part shards remain,
every index row with a shard_id points at bytes whose sha256 equals the recorded one, and the
index covers every post in the pool exactly once. Writes m2_shard_verify.json.

    uv run python -m experiments.wp18.m2_verify_shards
"""
import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd

from experiments.wp18.common import OUT

ROOT = Path("/mnt/d/memes-multisub")


def main() -> None:
    idx = pd.read_parquet(ROOT / "index/shard_index.parquet")
    copy = pd.read_parquet(OUT / "m2_shard_index.parquet")
    pool = pd.read_parquet(OUT / "m2_pool.parquet")
    res = {"index_rows": len(idx), "pool_rows": len(pool),
           "index_post_ids_unique": bool(idx.post_id.is_unique),
           "pool_covered": bool(set(pool.post_id) == set(idx.post_id)),
           "wp18_copy_same_rows": bool(
               copy.sort_values("post_id")[["post_id", "fetch_status", "sha256"]].reset_index(drop=True)
               .equals(idx.sort_values("post_id")[["post_id", "fetch_status", "sha256"]].reset_index(drop=True))),
           "part_files": [p.name for p in (ROOT / "images").glob("*.part")],
           "status_counts": idx.fetch_status.value_counts().to_dict(), "shards": {}}
    bad = []
    for shard, g in idx[idx.shard_id.notna()].groupby("shard_id"):
        p = ROOT / "images" / f"{shard}.tar"
        with tarfile.open(p) as tf:
            n_members = sum(1 for _ in tf)  # raises on truncation
        with open(p, "rb") as f:
            for r in g.drop_duplicates(["offset"]).itertuples():
                f.seek(int(r.offset))
                if hashlib.sha256(f.read(int(r.length))).hexdigest() != r.sha256:
                    bad.append(r.post_id)
        res["shards"][shard] = {"bytes": p.stat().st_size, "tar_members": n_members,
                                "index_rows": len(g), "unique_offsets": int(g.offset.nunique())}
        print(shard, res["shards"][shard], flush=True)
    res["sha_mismatches"] = len(bad)
    res["sha_mismatch_examples"] = bad[:20]
    (OUT / "m2_shard_verify.json").write_text(json.dumps(res, indent=1, default=int))
    print(json.dumps({k: v for k, v in res.items() if k != "shards"}, indent=1, default=int))


if __name__ == "__main__":
    main()
