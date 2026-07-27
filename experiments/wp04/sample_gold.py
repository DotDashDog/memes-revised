"""WP-04 Method §3: draw the two-stratum gold sample.

Run ONLY after the design (n_seed, n_bg, cap, radius, CI target) is approved
and registered in the WP-04 Log. Sampling unit = unique stored image (byte
duplicates were collapsed at fetch; the shard index maps units back to every
pool post sharing the bytes). Refuses to overwrite an existing sample.

- Family stratum: n_seed units drawn uniformly, expanded with their dhash
  siblings (component at the chosen radius) up to cap members per family —
  every drawn meme brings its family, giving within-cluster support. Seed
  inclusion is proportional to family size by construction; reported per WP.
- Background stratum: n_bg units drawn uniformly from the remainder — the
  window into the corpus's true (mostly singleton) size distribution.

Outputs: <out>/sample.parquet (post_id | sha256 | stratum | dhash_component)
and <out>/units.parquet (unit_id | post_id | dhash_component) for the
labeling server.

Usage:
    uv run python -m experiments.wp04.sample_gold \
        --radius 4 --n-seed 250 --n-bg 250 --cap 12 --out experiments/WP-04/sample-001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WP04 = Path("/home/linus/memes-revised/experiments/WP-04")
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, required=True)
    ap.add_argument("--n-seed", type=int, required=True)
    ap.add_argument("--n-bg", type=int, required=True)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "sample.parquet").exists():
        sys.exit(f"{out}/sample.parquet exists — sample is frozen; refusing to redraw.")
    out.mkdir(parents=True, exist_ok=True)

    fam = pd.read_parquet(WP04 / "families.parquet")
    col = f"fam_t{args.radius}"
    fam = fam[["post_id", "sha256", col]].rename(columns={col: "dhash_component"})

    rng = np.random.default_rng(SEED)
    n = len(fam)
    members_by_comp = fam.groupby("dhash_component").indices

    chosen: dict[int, int] = {}  # row index -> stratum (0 family, 1 background)
    seeds = rng.choice(n, size=args.n_seed, replace=False)
    for s in seeds:
        comp = fam.iloc[s].dhash_component
        idxs = list(members_by_comp[comp])
        if len(idxs) > args.cap:
            others = [i for i in idxs if i != s]
            idxs = [s] + list(rng.choice(others, size=args.cap - 1, replace=False))
        for i in idxs:
            chosen.setdefault(int(i), 0)

    remaining = np.setdiff1d(np.arange(n), np.fromiter(chosen, dtype=int))
    for i in rng.choice(remaining, size=args.n_bg, replace=False):
        chosen[int(i)] = 1

    rows = fam.iloc[sorted(chosen)].copy()
    rows["stratum"] = [
        "family" if chosen[i] == 0 else "background" for i in sorted(chosen)
    ]
    rows.to_parquet(out / "sample.parquet", index=False)

    units = rows.rename(columns={"sha256": "unit_id"})[
        ["unit_id", "post_id", "dhash_component"]
    ]
    units.to_parquet(out / "units.parquet", index=False)

    comp_sizes = rows[rows.stratum == "family"].groupby("dhash_component").size()
    meta = {
        "seed": SEED,
        "radius": args.radius,
        "n_seed": args.n_seed,
        "n_bg": args.n_bg,
        "cap": args.cap,
        "n_units_total": len(rows),
        "n_family_stratum": int((rows.stratum == "family").sum()),
        "n_background": int((rows.stratum == "background").sum()),
        "n_seeded_families": int(len(comp_sizes)),
        "seeded_family_size_hist": {
            str(k): int(v) for k, v in comp_sizes.value_counts().sort_index().items()
        },
    }
    (out / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
