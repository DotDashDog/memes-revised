"""Contact sheets of dhash families at several radii, to check the radius by eye.

For each radius T in {2, 4, 6}: the three largest components (first 12 members)
and eight randomly chosen components of size 2-5 that are *new at this radius*
(i.e. not already a single component at the next smaller radius), so each sheet
shows what raising the radius adds.

Usage: uv run python -m experiments.wp04.radius_montage
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

WP04 = Path("/home/linus/memes-revised/experiments/WP-04")
IMAGES = Path("/mnt/d/memes/images")
INDEX = Path("/mnt/d/memes/index/shard_index.parquet")
SEED = 42
THUMB = 150


def load(idx: pd.DataFrame, post_ids: list[str]) -> dict[str, Image.Image]:
    out = {}
    rows = idx.loc[post_ids].sort_values(["shard_id", "offset"])
    for pid, r in rows.iterrows():
        with open(IMAGES / f"{r.shard_id}.tar", "rb") as f:
            f.seek(int(r.offset))
            body = f.read(int(r.length))
        try:
            im = Image.open(io.BytesIO(body)).convert("RGB")
            im.thumbnail((THUMB, THUMB))
            out[pid] = im
        except Exception:
            out[pid] = Image.new("RGB", (THUMB, THUMB), "red")
    return out


def sheet(rows: list[tuple[str, list[str]]], idx: pd.DataFrame, path: Path) -> None:
    ncol = max(len(m) for _, m in rows)
    W, H = ncol * (THUMB + 4) + 110, len(rows) * (THUMB + 4)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for y, (label, members) in enumerate(rows):
        d.text((4, y * (THUMB + 4) + 60), label, fill="black")
        ims = load(idx, members)
        for x, pid in enumerate(members):
            canvas.paste(ims[pid], (110 + x * (THUMB + 4), y * (THUMB + 4)))
    canvas.save(path)


def main() -> None:
    fam = pd.read_parquet(WP04 / "families.parquet").drop_duplicates("sha256")
    idx = pd.read_parquet(INDEX).set_index("post_id")
    rng = np.random.default_rng(SEED)
    prev = {2: "fam_t0", 4: "fam_t2", 6: "fam_t4"}
    for T in (2, 4, 6):
        col = f"fam_t{T}"
        groups = fam.groupby(col).post_id.apply(list)
        sizes = groups.apply(len).sort_values(ascending=False)
        rows = []
        for comp in sizes.index[:3]:
            m = groups[comp]
            pick = list(rng.choice(m, size=min(12, len(m)), replace=False))
            rows.append((f"T={T} big\nn={len(m)}", pick))
        # components of size 2-5 that are not one component at the smaller radius
        sub = fam[fam[col].isin(sizes[(sizes >= 2) & (sizes <= 5)].index)]
        new = sub.groupby(col)[prev[T]].nunique()
        new = new[new > 1].index.to_numpy()
        for comp in rng.choice(new, size=min(8, len(new)), replace=False):
            rows.append((f"T={T} new\nn={len(groups[comp])}", groups[comp]))
        sheet(rows, idx, WP04 / "radius-check" / f"radius_T{T}.png")
        print(T, "new small comps:", len(new))


if __name__ == "__main__":
    main()
