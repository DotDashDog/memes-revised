"""WP-04 Method §2 (stage 1): dhash every fetched pool image.

Streams each ADR 0015 tar shard sequentially (the /mnt/d drive is a 5400 rpm
USB disk — sequential reads only), decodes with PIL (cv2.imdecode has no GIF
support), and computes the same 64-bit dHash as `experiments/wp01/dhash.py`
(grayscale, cv2 INTER_AREA resize to 9x8, row-wise brightness gradient).

Output: experiments/WP-04/dhash.parquet
    post_id | sha256 | dhash (uint64) | width | height | n_frames | decode_ok

One row per *unique stored image* (first post_id of each tar member); the
shard index maps the remaining post_ids to the same sha256.

Usage:
    uv run python -m experiments.wp04.dhash_pool
"""

from __future__ import annotations

import io
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

IMAGES_DIR = Path("/mnt/d/memes/images")
INDEX = Path("/mnt/d/memes/index/shard_index.parquet")
OUT = Path("/home/linus/memes-revised/experiments/WP-04/dhash.parquet")

Image.MAX_IMAGE_PIXELS = 200_000_000  # a few legit panorama memes exceed PIL's default


def dhash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hash_one(member_name: str, body: bytes) -> dict:
    post_id = member_name.rsplit(".", 1)[0]
    row = {
        "post_id": post_id,
        "dhash": 0,
        "width": 0,
        "height": 0,
        "n_frames": 1,
        "decode_ok": False,
    }
    try:
        img = Image.open(io.BytesIO(body))
        row["n_frames"] = getattr(img, "n_frames", 1)
        gray = np.asarray(img.convert("L"))
        row.update(
            dhash=dhash64(gray), width=img.width, height=img.height, decode_ok=True
        )
    except Exception:
        pass
    return row


def main() -> None:
    idx = pq.read_table(
        INDEX, columns=["post_id", "shard_id", "fetch_status", "sha256"]
    ).to_pandas()
    ok = idx[idx.fetch_status == "ok"]
    sha_by_pid = dict(zip(ok.post_id, ok.sha256))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for shard in sorted(IMAGES_DIR.glob("*.tar")):
            futures = []
            with tarfile.open(shard, mode="r") as tf:
                for member in tf:
                    body = tf.extractfile(member).read()
                    futures.append(pool.submit(hash_one, member.name, body))
            done = [f.result() for f in futures]
            rows.extend(done)
            print(f"{shard.name}: {len(done):,} members hashed")

    df = pd.DataFrame(rows)
    df["sha256"] = df.post_id.map(sha_by_pid)
    df["dhash"] = df["dhash"].astype("uint64")
    n_bad = int((~df.decode_ok).sum())
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), OUT)
    print(f"{len(df):,} unique images hashed -> {OUT}  (decode failures: {n_bad})")


if __name__ == "__main__":
    main()
