"""WP-04 Method §2 (stage 2): dhash near-duplicate graph -> family-size distribution.

Builds connected components over the fetched pool at several Hamming radii
(T = 0, 2, 4, 6, 8, 10 on the 64-bit dHash; byte-identical sha256 images are
always merged) and writes:

- experiments/WP-04/families.parquet       one row per pool post that has a
  decoded image: post_id | sha256 | component id per radius (fam_t0 ... fam_t10)
- experiments/WP-04/family_size_distribution.json   per-radius distributions
- experiments/WP-04/plots/family_sizes.png          log-log size histograms +
  pairwise-distance histogram (the radius-choice evidence)

The dhash components are *scaffolding* (template families, appearance-level);
the identity criterion for the audit is the human family_id layer (WP-04 §4).

Usage:
    uv run python -m experiments.wp04.families
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WP04 = Path("/home/linus/memes-revised/experiments/WP-04")
INDEX = Path("/mnt/d/memes/index/shard_index.parquet")
RADII = [0, 2, 4, 6, 8, 10]


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def pairs_within(hashes: np.ndarray, radius: int, chunk: int = 2048):
    """Yield (i, j) index pairs with hamming(h_i, h_j) <= radius, i < j."""
    n = len(hashes)
    for start in range(0, n, chunk):
        block = hashes[start : start + chunk, None] ^ hashes[None, :]
        dist = np.bitwise_count(block)
        ii, jj = np.nonzero(dist <= radius)
        for i, j in zip(ii + start, jj):
            if i < j:
                yield int(i), int(j)


def main() -> None:
    dh = pd.read_parquet(WP04 / "dhash.parquet")
    dh = dh[dh.decode_ok].reset_index(drop=True)
    idx = pq.read_table(
        INDEX, columns=["post_id", "fetch_status", "sha256"]
    ).to_pandas()
    ok = idx[idx.fetch_status == "ok"]

    # unique dhash values; identical hashes collapse before the O(n^2) sweep
    uniq_hashes, inv = np.unique(dh["dhash"].to_numpy(), return_inverse=True)
    n_uniq = len(uniq_hashes)
    print(f"{len(ok):,} ok posts, {dh.shape[0]:,} decoded unique images, "
          f"{n_uniq:,} distinct dhash values")

    # pairwise distance histogram over a subsample (radius-choice evidence)
    rng = np.random.default_rng(42)
    sub = uniq_hashes[rng.choice(n_uniq, size=min(4000, n_uniq), replace=False)]
    d = np.bitwise_count(sub[:, None] ^ sub[None, :])
    tri = d[np.triu_indices_from(d, k=1)]

    max_r = max(RADII)
    uf = UnionFind(n_uniq)
    edges_by_r: dict[int, list[tuple[int, int]]] = {r: [] for r in RADII}
    for i, j in pairs_within(uniq_hashes, max_r):
        dist = int(np.bitwise_count(uniq_hashes[i] ^ uniq_hashes[j]))
        for r in RADII:
            if dist <= r:
                edges_by_r[r].append((i, j))

    # components per radius (edges are nested: T=0 subset of T=2 subset of ...)
    fam_cols: dict[str, np.ndarray] = {}
    dist_summary: dict[str, dict] = {}
    for r in RADII:
        uf = UnionFind(n_uniq)
        for i, j in edges_by_r[r]:
            uf.union(i, j)
        roots = np.array([uf.find(i) for i in range(n_uniq)])
        # map every image (via inverse) to its hash's component
        comp = roots[inv]
        fam_cols[f"fam_t{r}"] = comp
        sizes = Counter(Counter(comp).values())
        total_fams = sum(sizes.values())
        multi = {k: v for k, v in sizes.items() if k >= 2}
        dist_summary[f"t{r}"] = {
            "n_families": total_fams,
            "n_singletons": sizes.get(1, 0),
            "n_multi_families": sum(multi.values()),
            "n_posts_in_multi": sum(k * v for k, v in multi.items()),
            "largest": max(sizes) if sizes else 0,
            "size_hist": {str(k): v for k, v in sorted(sizes.items())},
        }
        print(f"T={r}: {total_fams:,} families, "
              f"{sum(multi.values()):,} with >=2 members, largest {max(sizes)}")

    out = dh[["post_id", "sha256", "width", "height"]].copy()
    for col, vals in fam_cols.items():
        out[col] = vals
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False),
                   WP04 / "families.parquet")
    (WP04 / "family_size_distribution.json").write_text(
        json.dumps(dist_summary, indent=2))

    plots = WP04 / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(tri, bins=np.arange(0, 66) - 0.5, log=True, color="#4477aa")
    axes[0].set_xlabel("pairwise dhash Hamming distance (4k subsample)")
    axes[0].set_ylabel("pair count (log)")
    axes[0].set_title("Distance histogram — where is the near-dup valley?")
    for r in RADII:
        hist = dist_summary[f"t{r}"]["size_hist"]
        ks = np.array([int(k) for k in hist])
        vs = np.array([hist[k] for k in hist])
        axes[1].plot(ks, vs, marker="o", ms=3, lw=1, label=f"T={r}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("family size")
    axes[1].set_ylabel("number of families")
    axes[1].set_title("Family-size distribution vs Hamming radius")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plots / "family_sizes.png", dpi=150)
    print(f"wrote {plots / 'family_sizes.png'}")


if __name__ == "__main__":
    main()
