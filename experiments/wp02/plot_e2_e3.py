"""Plots for E2 (bins) and E3 (overdispersion / spikes)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from experiments.wp02.plot_e1 import NICE, SHAPE_DESC, WIN_LABEL

BASE = Path("experiments/WP-02")
KEY = ["growth_rate", "peak_time", "log_peak_rate", "log_total", "log_R0", "log_gamma"]


def plot_e2():
    out = BASE / "e2-bins"
    df = pd.read_csv(out / "summary.csv")
    e1 = pd.read_csv(BASE / "e1-plots" / "e1_summary_merged.csv")
    combos = df[["shape", "window", "n_obs_expected"]].drop_duplicates().itertuples(index=False)
    ref_rows = []
    for shape, win, n in combos:
        r = e1[(e1["shape"] == shape) & (e1.window == win) & (e1.n_obs_expected == n)].copy()
        ref_rows.append(r)
    ref = pd.concat(ref_rows)
    both = pd.concat([df, ref[df.columns.intersection(ref.columns)]])
    both.to_csv(out / "summary_with_native.csv", index=False)
    combos = list(df[["shape", "window", "n_obs_expected"]].drop_duplicates().itertuples(index=False))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(combos)))
    for value, fname, ylab, title in (
        ("width_ratio", "e2_width_vs_bins.png", "median 90% posterior interval width\nrelative to daily bins (1 = no loss)",
         "Coarser time bins cost little until there are about as many bins as parameters (4): interval width relative to daily bins"),
        ("cov90", "e2_coverage_vs_bins.png", "fraction of replicates whose 90% interval covers the truth",
         "Coverage of 90% intervals vs number of bins (solid: full posterior, dashed: Hessian/Laplace). Grey band = honest"),
    ):
        fig, axes = plt.subplots(1, len(KEY), figsize=(24, 4.8))
        for ax, t in zip(axes, KEY):
            for col, (shape, win, n) in zip(cols, combos):
                for method, ls in (("posterior", "-"), ("laplace", "--")):
                    s = both[(both["shape"] == shape) & (both.window == win) & (both.n_obs_expected == n) & (both.target == t) & (both.method == method)].sort_values("n_bins")
                    if value == "width_ratio":
                        if method == "laplace":
                            continue
                        yv = s.width90_median / s.width90_median.iloc[-1]
                    else:
                        yv = s.cov90
                    ax.plot(s.n_bins, yv, ls, marker="o", color=col, label=f"{SHAPE_DESC[shape]}, {WIN_LABEL[win]}, {n} posts" if method == "posterior" else None)
            ax.set_xscale("log"); ax.set_xticks([4, 8, 16, 44, 106]); ax.set_xticklabels(["4", "8", "16", "44", "106"])
            ax.set_xlabel("number of equal-width bins across the window\n(rightmost point of each line = daily bins)")
            ax.set_title(NICE[t], fontsize=10); ax.grid(alpha=0.3)
            if value == "width_ratio":
                ax.axhline(1.25, color="k", ls=":", lw=1); ax.set_yscale("log")
            else:
                ax.axhspan(0.848, 0.952, color="0.85", zorder=0); ax.set_ylim(0, 1.02)
        axes[0].set_ylabel(ylab)
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(title + ("  (dotted: +25%)" if value == "width_ratio" else ""), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)


def plot_e3():
    out = BASE / "e3-noise-spikes"
    df = pd.read_csv(out / "summary.csv")

    def gen_label(r):
        kw = json.loads(r.gen_kwargs)
        if r.generator == "poisson":
            return "Poisson\n(control)"
        if r.generator == "nb":
            return f"overdispersed\nk={kw['nb_k']:.0f}"
        return "Poisson +\nrepost spikes"

    df["gen"] = [gen_label(r) for r in df.itertuples()]
    df["combo"] = [f"{SHAPE_DESC[r.shape]}, full curve, {r.n_obs_expected} posts" for r in df.itertuples()]
    methods = [("laplace", "Poisson fit, Hessian interval", "#b2182b"), ("posterior", "Poisson fit, full posterior", "#ef8a62"),
               ("quasi_poisson", "Poisson fit, interval × √(Pearson χ²/dof)", "#67a9cf"), ("nb_laplace", "negative-binomial fit", "#2166ac")]
    targets = ["growth_rate", "peak_time", "log_peak_rate", "log_total", "log_R0"]
    combos = list(dict.fromkeys(df.combo))
    for value, fname, ylab, title in (
        ("cov90", "e3_coverage.png", "fraction of 300 replicates whose\n90% interval covers the truth",
         "With realistic noise, Poisson-based intervals are overconfident; in the grey band = honest 90% intervals"),
        ("bias_median", "e3_bias.png", "median (estimate − truth)\n(log targets: natural-log units)",
         "Bias of point estimates under overdispersion and repost spikes (0 = unbiased)"),
    ):
        fig, axes = plt.subplots(len(combos), len(targets), figsize=(22, 4.2 * len(combos)), squeeze=False)
        for i, combo in enumerate(combos):
            for j, t in enumerate(targets):
                ax = axes[i, j]
                sub = df[(df.combo == combo) & (df.target == t)]
                gens = list(dict.fromkeys(sub.gen))
                for m, (method, lab, col) in enumerate(methods):
                    s = sub[sub.method == method].set_index("gen").reindex(gens)
                    ax.bar(np.arange(len(gens)) + (m - 1.5) * 0.2, s[value], width=0.2, color=col, label=lab)
                ax.set_xticks(range(len(gens))); ax.set_xticklabels(gens, fontsize=8)
                if value == "cov90":
                    ax.axhspan(0.848, 0.952, color="0.85", zorder=0); ax.axhline(0.9, color="k", lw=0.8); ax.set_ylim(0, 1.02)
                else:
                    ax.axhline(0, color="k", lw=0.8)
                ax.set_title(f"{NICE[t]}\n{combo}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(ylab)
        axes[0, 0].legend(fontsize=7, loc="lower left")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    for which in sys.argv[1:]:
        {"e2": plot_e2, "e3": plot_e3}[which]()
