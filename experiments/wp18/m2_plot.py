"""WP-18 measurement 2 plots + contact sheet of matched pairs.

    uv run python -m experiments.wp18.m2_plot
"""
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from experiments.wp18.common import COLORS, OUT, SEED, SUBS

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
SEQ = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cce3fd", "#84b9f3", "#257de0", "#0e3d7e"])
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
ROOT = Path("/mnt/d/memes-multisub")
Image.MAX_IMAGE_PIXELS = 200_000_000


def overlap_figure(res, titles):
    T = str(res["primary_radius"])
    r = res["by_radius"][T]
    labels = [f"r/{s}" for s in SUBS]
    M = 100 * np.array([[r["matrix_row_has_dup_in_col"][a][b] for b in SUBS] for a in SUBS])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 8.6), gridspec_kw={"width_ratios": [1.05, 1]})
    fig.suptitle(titles["overlap_title"], x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.94, titles["overlap_subtitle"], ha="left", va="top", fontsize=10, color=MUTED)
    off = M.copy()
    np.fill_diagonal(off, np.nan)
    im = ax.imshow(off, cmap=SEQ, vmin=0, vmax=max(1.0, np.nanmax(off)))
    for i in range(7):
        for j in range(7):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#eeeeee", lw=0))
                ax.text(j, i, f"({M[i, j]:.1f})", ha="center", va="center", color=MUTED, fontsize=9)
            else:
                dark = off[i, j] > 0.6 * im.get_clim()[1]
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if dark else INK, fontsize=10)
    ax.set_xticks(range(7), labels, rotation=30, ha="right")
    ax.set_yticks(range(7), [f"{l}  (n={r['per_community'][s]['n']:,})" for l, s in zip(labels, SUBS)])
    ax.set_xlabel("...has a near-duplicate image in this community")
    ax.set_ylabel("share (%) of this community's fetched image posts that...")
    ax.set_title("Pairwise overlap, % of row community's posts\n(grey diagonal, in brackets: reposts "
                 "inside the same community)", loc="left", fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("% of row community's posts")
    cb.outline.set_visible(False)

    y = np.arange(7)[::-1]
    val = np.array([100 * r["per_community"][s]["cross_any_other"] for s in SUBS])
    lo = np.array([100 * r["per_community"][s]["cross_any_other_ci95"][0] for s in SUBS])
    hi = np.array([100 * r["per_community"][s]["cross_any_other_ci95"][1] for s in SUBS])
    bg = np.array([100 * r["per_community"][s]["background_2019_pool"] for s in SUBS])
    same = np.array([100 * r["per_community"][s]["same_community_repost"] for s in SUBS])
    bx.barh(y, val, height=0.5, color=[COLORS[s] for s in SUBS])
    bx.errorbar(val, y, xerr=[val - lo, hi - val], fmt="none", ecolor=INK, elinewidth=1.4, capsize=4)
    bx.plot(bg, y, marker="|", ms=22, mew=2.5, ls="", color="#c0392b",
            label="control: same test against 26k r/memes images from 2019\n(matches here are chance collisions or evergreen templates)")
    bx.plot(same, y, marker="D", ms=7, ls="", color="white", mec=INK, mew=1.4,
            label="for scale: near-duplicate of another post in the SAME community")
    for yi, v, h_ in zip(y, val, hi):
        bx.text(h_ + 0.08, yi, f"{v:.2f}%", va="center", fontsize=10, color=INK)
    p = r["pooled"]
    bx.axvline(100 * p["cross_any_other"], color=MUTED, lw=1, ls=(0, (4, 3)))
    bx.text(100 * p["cross_any_other"], 6.62, f" all seven pooled: {100 * p['cross_any_other']:.2f}% "
            f"(95% CI {100 * p['cross_any_other_ci95'][0]:.2f}-{100 * p['cross_any_other_ci95'][1]:.2f})",
            fontsize=9, color=MUTED, va="bottom")
    bx.set_yticks(y, labels)
    bx.set_ylim(-0.6, 7.0)
    bx.set_xlabel("% of the community's fetched image posts with a near-duplicate\nin ANY other of the seven "
                  "communities (whiskers: 95% interval, resampling the 28 days)")
    bx.grid(True, axis="x", color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=9, ncol=1)
    bx.set_title("Any-other-community overlap per community", loc="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "m2_crosspost_overlap.png", dpi=130)
    plt.close(fig)


def radius_figure(res, titles):
    radii = sorted(int(k) for k in res["by_radius"])
    fig, ax = plt.subplots(figsize=(9, 5.4))
    series = [("cross_any_other", "near-duplicate in another community", "#257de0"),
              ("same_community_repost", "near-duplicate in the same community", "#f47230"),
              ("background_2019_pool", "control: match in 2019 r/memes pool", "#6b6b6b")]
    for key, lab, col in series:
        v = [100 * res["by_radius"][str(T)]["pooled"][key] for T in radii]
        lo = [100 * res["by_radius"][str(T)]["pooled"][key + "_ci95"][0] for T in radii]
        hi = [100 * res["by_radius"][str(T)]["pooled"][key + "_ci95"][1] for T in radii]
        ax.fill_between(radii, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(radii, v, marker="o", ms=7, lw=2, color=col, mec="white", mew=1.2)
        ax.text(radii[-1] + 0.2, v[-1], lab, color=INK, va="center", fontsize=9.5)
    ax.axvline(res["primary_radius"], color=MUTED, lw=0.8, ls=":")
    ax.text(res["primary_radius"], ax.get_ylim()[1], " pre-registered radius", color=MUTED, fontsize=9, va="top")
    ax.set_xlim(-0.4, 17)
    ax.set_xticks(radii)
    ax.set_xlabel("match radius: maximum number of differing bits out of 64 in the image hash")
    ax.set_ylabel("share of fetched image posts with a match (%)")
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(titles["radius_title"], loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "m2_overlap_vs_radius.png", dpi=130)
    plt.close(fig)


def contact_sheet(n_pairs=30):
    cp = pd.read_parquet(OUT / "m2_cross_pairs.parquet")
    if not len(cp):
        return
    idx = pd.read_parquet(OUT / "m2_shard_index.parquet").set_index("post_id")
    pick = cp.sample(min(n_pairs, len(cp)), random_state=SEED).reset_index(drop=True)
    pick.to_csv(OUT / "m2_contact_sheet_pairs.csv", index=False)

    def load(pid):
        r = idx.loc[pid]
        with open(ROOT / "images" / f"{r.shard_id}.tar", "rb") as f:
            f.seek(int(r.offset))
            im = Image.open(io.BytesIO(f.read(int(r.length))))
            im.seek(0)
            im = im.convert("RGB")
            im.thumbnail((300, 300))
            return im

    cols = 5
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4.4, rows * 2.6))
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in pick.iterrows():
        for side, (pid, comm) in enumerate([(r.post_i, r.comm_i), (r.post_j, r.comm_j)]):
            ax = axes[k // cols, (k % cols) * 2 + side]
            ax.imshow(load(pid))
            ax.set_title(f"#{k + 1}{'ab'[side]} r/{comm}" + (f"  d={r.dist}" if side else ""), fontsize=8)
    fig.suptitle(f"{len(pick)} randomly drawn cross-community matched pairs at Hamming <= 4 "
                 f"(seed {SEED}); d = hash bits differing", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "m2_contact_sheet.png", dpi=95)
    plt.close(fig)


def main():
    res = json.loads((OUT / "m2_summary.json").read_text())
    tp = OUT / "m2_plot_titles.json"
    titles = json.loads(tp.read_text()) if tp.exists() else {
        "overlap_title": "DRAFT", "overlap_subtitle": "", "radius_title": "DRAFT"}
    overlap_figure(res, titles)
    radius_figure(res, titles)
    contact_sheet()
    print("wrote plots")


if __name__ == "__main__":
    main()
