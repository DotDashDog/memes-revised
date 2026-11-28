"""WP-18 measurement 3 plots: confusion matrix (main deliverable) + accuracy vs baselines.

    uv run python -m experiments.wp18.m3_plot
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from experiments.wp18.common import OUT, SUBS

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
SEQ = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cce3fd", "#84b9f3", "#257de0", "#0e3d7e"])
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
KEYS = ["p_img_ocr (image + extracted text)", "p_vlm_only (image only)"]
NICE = {"p_img_ocr (image + extracted text)": "Embedding: image + text read off the image",
        "p_vlm_only (image only)": "Embedding: image only",
        "extracted-text vector only": "Embedding of the text read off the image, alone",
        "file metadata only (size, shape, format)": "Baseline: file size, shape and format only",
        "timestamp only (day, hour, weekday)": "Baseline: posting time only",
        "label-permutation control (p_img_ocr)": "Control: same pipeline, community labels shuffled"}


def main():
    res = json.loads((OUT / "m3_summary.json").read_text())
    tp = OUT / "m3_plot_titles.json"
    titles = json.loads(tp.read_text()) if tp.exists() else {"cm_title": "DRAFT", "cm_subtitle": "",
                                                              "acc_title": "DRAFT"}
    R = res["results"]
    labels = [f"r/{s}" for s in SUBS]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6))
    fig.suptitle(titles["cm_title"], x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.945, titles["cm_subtitle"], ha="left", va="top", fontsize=10, color=MUTED)
    for ax, key in zip(axes, KEYS):
        cm = np.array(R[key]["confusion_rows_true_cols_pred"], dtype=float)
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(pct, cmap=SEQ, vmin=0, vmax=100)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center",
                        color="white" if pct[i, j] > 55 else INK,
                        fontsize=11, fontweight="bold" if i == j else "normal")
        ax.axhline(3.5, color=INK, lw=1.2)
        ax.axvline(3.5, color=INK, lw=1.2)
        ax.set_xticks(range(7), labels, rotation=30, ha="right")
        ax.set_yticks(range(7), labels)
        ax.set_xlabel("community the classifier guessed")
        ax.set_ylabel("community the meme was actually posted in")
        a, ci = R[key]["accuracy"], R[key]["ci95"]
        ax.set_title(f"{NICE[key]}\ncorrect {100 * a:.1f}% (95% CI {100 * ci[0]:.1f}-{100 * ci[1]:.1f}); "
                     f"guessing would give {100 * res['chance']:.1f}%", loc="left", fontsize=10.5, pad=22)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.text(1.5, -0.62, "long-running set", ha="center", color=MUTED, fontsize=9)
        ax.text(5, -0.62, "newer communities", ha="center", color=MUTED, fontsize=9)
    cb = fig.colorbar(im, cax=fig.add_axes([0.925, 0.14, 0.012, 0.62]))
    cb.set_label("% of that row's held-out memes (each row sums to 100)")
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.76, bottom=0.14, wspace=0.28)
    fig.savefig(OUT / "m3_confusion_matrix.png", dpi=130)
    plt.close(fig)

    order = ["p_img_ocr (image + extracted text)", "p_vlm_only (image only)", "extracted-text vector only",
             "file metadata only (size, shape, format)", "timestamp only (day, hour, weekday)",
             "label-permutation control (p_img_ocr)"]
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        a, ci = 100 * R[k]["accuracy"], [100 * c for c in R[k]["ci95"]]
        col = "#257de0" if k in KEYS else "#9a9a9a"
        ax.barh(yi, a, height=0.55, color=col)
        ax.errorbar([a], [yi], xerr=[[a - ci[0]], [ci[1] - a]], fmt="none", ecolor=INK, capsize=4, elinewidth=1.4)
        ax.plot(100 * np.array(R[k]["fold_accuracies"]), [yi] * 5, ls="", marker="o", ms=4, color="white",
                mec=INK, mew=0.9, zorder=3)
        ax.text(ci[1] + 1, yi, f"{a:.1f}%", va="center", fontsize=10)
    ax.axvline(100 * res["chance"], color="#c0392b", lw=1.2, ls=(0, (4, 3)))
    ax.text(100 * res["chance"] + 0.5, len(order) - 0.45, "chance = majority class = 14.3%", color="#c0392b",
            fontsize=9, va="bottom")
    ax.set_yticks(y, [NICE[k] for k in order])
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("held-out accuracy naming 1 of 7 communities (%); whiskers = 95% bootstrap interval, "
                  "dots = the 5 folds")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(titles["acc_title"], loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "m3_accuracy_vs_baselines.png", dpi=130)
    plt.close(fig)
    print("wrote plots")


if __name__ == "__main__":
    main()
