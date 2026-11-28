"""Plots for the ingest review. Reads monthly_counts.csv and compare_results.json.

    uv run python experiments/wp16/review/plots.py
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

OUT = "experiments/WP-16/review"
INK, MUTED, GRID = "#191919", "#666666", "#e4e4e4"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

rows = list(csv.DictReader(open(f"{OUT}/monthly_counts.csv")))
x = np.array([r["month"] for r in rows], dtype="M8[M]").astype("M8[D]").astype(object)
raw = np.array([int(r["raw_archive"]) for r in rows]); par = np.array([int(r["parquet"]) for r in rows])
d = par - raw
nbad = int((d != 0).sum())

fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[3, 1.3])
a.plot(x, raw, color=BLUE, lw=2, label="raw archive (reviewer's own count; line joins across 22 empty months before 2011)")
a.plot(x, par, ls="none", marker="o", ms=3.5, mfc="white", mec=ORANGE, mew=1, label="ingested parquet")
a.set_yscale("log"); a.set_ylabel("posts per month (log scale)")
a.grid(True, which="major", color=GRID, lw=0.6); a.set_axisbelow(True)
a.legend(frameon=False, loc="lower right")
if nbad == 0:
    title = (f"The ingested table has exactly as many posts as the raw archive in every one of the "
             f"{len(rows)} months\n({raw.sum():,} posts in each; largest monthly difference: 0)")
else:
    title = (f"The ingested table differs from the raw archive in {nbad} of {len(rows)} months "
             f"(largest difference {int(np.abs(d).max()):,} posts)")
fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
b.axhline(0, color=MUTED, lw=0.8)
b.plot(x, d, ls="none", marker="o", ms=3.5, color=ORANGE)
lim = max(5, int(np.abs(d).max() * 1.3))
b.set_ylim(-lim, lim)
b.set_ylabel("parquet minus raw\n(posts)"); b.set_xlabel("month the post was created (UTC)")
b.grid(True, color=GRID, lw=0.6); b.set_axisbelow(True)
b.xaxis.set_major_locator(mdates.YearLocator(2)); b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
b.text(0.995, 0.08, "A faithful ingest puts every point on the zero line.", transform=b.transAxes,
       ha="right", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(f"{OUT}/monthly_counts_parquet_vs_raw.png", dpi=150)

# ---- what the img_url column contains, by year
# img_url content comes from summary_stats.py (same classification as compare.py)
res = json.load(open(f"{OUT}/summary_stats.json"))
ky = {int(y): c for y, c in res["img_url_content"]["kinds_by_year"].items() if int(y) >= 2016}
kinds = [("i.redd.it direct image", BLUE), ("other direct image (imgur / image extension)", ORANGE),
         ("reddit gallery page (images not in parquet)", AQUA), ("v.redd.it video", YELLOW),
         ("the post's own page (text post)", MAGENTA), ("other link", GREEN), ("no URL (null)", "#9c9c9c")]
years = sorted(ky)
tot = np.array([sum(ky[y].values()) for y in years])
fig, ax = plt.subplots(figsize=(10, 5.8))
left = np.zeros(len(years))
for k, c in kinds:
    v = np.array([ky[y].get(k, 0) for y in years]) / tot * 100
    ax.barh([str(y) for y in years], v, left=left, color=c, edgecolor="white", linewidth=1.5, height=0.72, label=k)
    for i, (l, w) in enumerate(zip(left, v)):
        if w >= 6:
            ax.text(l + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    left += v
for i, t in enumerate(tot):
    ax.text(101, i, f"{t:,} posts", va="center", fontsize=8.5, color=MUTED)
ax.set_xlim(0, 100); ax.invert_yaxis()
ax.set_xlabel("share of that year's posts (%)"); ax.set_ylabel("year the post was created (UTC)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
kt = res["img_url_content"]["kinds_total"]; n = sum(kt.values())
direct = kt.get("i.redd.it direct image", 0) + kt.get("other direct image (imgur / image extension)", 0)
fig.suptitle(f"The column named img_url is the post's link, whatever it points to: {direct/n*100:.0f}% of all rows "
             f"are direct image links,\nbut in 2022-2024 a quarter to almost half are text posts, videos and other links "
             f"(years before 2016 omitted: under 1% of rows)",
             x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 0.93, 0.97))
fig.savefig(f"{OUT}/img_url_content_by_year.png", dpi=150, bbox_inches="tight")
print("plots written", nbad)
