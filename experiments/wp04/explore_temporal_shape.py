"""Exploratory (not registered): how concentrated in time are templates versus
meaning-level families, within the 2019 pool?

Templates: perceptual-hash groups (radius 4) over all 26k fetched images, size >= 8.
Families: agent-labelled meaning families in the 797-image gold sample, size >= 6
(labels not validated by the user on hard cases; see WP-04 log).
Statistic: number of days containing the middle 80% of a group's posts
(10th to 90th percentile of post date). A year-round format gives ~290 days.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

W = Path("/home/linus/memes-revised/experiments/WP-04")
pool = pd.read_parquet(W / "pool.parquet")
tcol = "created_utc"
day = (pool.set_index("post_id")[tcol] - 1546300800) / 86400.0

fam = pd.read_parquet(W / "families.parquet").drop_duplicates("sha256")
g = fam.groupby("fam_t4").post_id.apply(list)
tpl = {f"hash group {k} (n={len(v)})": day[v].to_numpy() for k, v in g.items() if len(v) >= 8}

lab = pd.read_parquet(W / "sample-001/labels.parquet")
gf = lab.groupby("family_id").post_id.apply(list)
mean = {f"{k} (n={len(v)})": day[v].to_numpy() for k, v in gf.items() if len(v) >= 6}
gt = lab.groupby("template_id").post_id.apply(list)
ltpl = {f"{k} (n={len(v)})": day[v].to_numpy() for k, v in gt.items() if len(v) >= 6}

def span(d): return float(np.percentile(d, 90) - np.percentile(d, 10))
rows = []
for kind, dd in (("template (hash group, whole pool)", tpl), ("template (labelled, gold sample)", ltpl),
                 ("meaning family (labelled, gold sample)", mean)):
    for name, d in dd.items():
        rows.append({"kind": kind, "name": name, "n": len(d), "span80_days": span(d), "median_day": float(np.median(d))})
df = pd.DataFrame(rows)
df.to_csv(W / "sample-001/temporal_shape.csv", index=False)
print(df.groupby("kind").span80_days.describe()[["count", "25%", "50%", "75%"]].round(0))
print(df[df.kind.str.startswith("meaning")].sort_values("span80_days").to_string(index=False))
print(df[df.kind.str.startswith("template (lab")].sort_values("span80_days").to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
for ax, (title, dd) in zip(axes, (("Templates: same base image, any caption\n(largest hash groups in the 26k-image pool)", tpl),
                                  ("Meaning families: same joke or trend, any image\n(agent labels, 797-image sample)", mean))):
    items = sorted(dd.items(), key=lambda kv: np.median(kv[1]))[-22:] if len(dd) > 22 else sorted(dd.items(), key=lambda kv: np.median(kv[1]))
    if len(dd) > 22:
        items = sorted(sorted(dd.items(), key=lambda kv: -len(kv[1]))[:22], key=lambda kv: np.median(kv[1]))
    for y, (name, d) in enumerate(items):
        ax.plot(d, np.full(len(d), y), "|", ms=12, color="#1f77b4")
        ax.plot([np.percentile(d, 10), np.percentile(d, 90)], [y, y], color="#ff7f0e", lw=3, alpha=0.5)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _ in items], fontsize=8)
    ax.set_xlim(0, 365)
    ax.set_xlabel("Day of 2019 the post was made")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
fig.suptitle("When were each group's posts made? Each tick is one post; the orange bar holds the middle 80% of them.\n"
             "A short bar is a burst (one rise and fall); a bar spanning the year is a format in continuous use.", fontsize=11)
fig.tight_layout()
fig.savefig(W / "sample-001/plots/temporal_shape.png", dpi=120)
