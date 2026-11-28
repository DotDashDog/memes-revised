"""Plots for E1 (counts x window). Reads summary.csv (+ rerun overrides) and per-cell npz."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

sys.path.append(str(Path(__file__).resolve().parents[2]))
from experiments.wp02.calib import TARGETS

RUN = Path("experiments/WP-02/e1-window-counts")
RERUN = Path("experiments/WP-02/e1-rerun-v2")
OUT = Path("experiments/WP-02/e1-plots")
ORDER = ["growth_rate", "log_init_rate", "peak_time", "log_peak_rate", "log_total",
         "log_R0", "log_beta", "log_gamma", "log_scale", "logit_i0"]
NICE = {
    "growth_rate": "initial growth rate (per day)",
    "log_init_rate": "initial post rate s·I0 (log)",
    "peak_time": "peak time (days)",
    "log_peak_rate": "peak height, posts/day (log)",
    "log_total": "final size, total posts (log)",
    "log_R0": "R0 = β/γ (log)",
    "log_beta": "β, infection rate (log)",
    "log_gamma": "γ, recovery rate (log)",
    "log_scale": "s, scale (log)",
    "logit_i0": "I0, initial fraction (logit)",
}
WIN_LABEL = {"0.7xpeak": "ends at 0.7 × peak time", "1.0xpeak": "ends at the peak",
             "1.5xpeak": "ends at 1.5 × peak time", "full": "full curve (99% of posts)"}
WIN_COL = {"0.7xpeak": "#b2182b", "1.0xpeak": "#ef8a62", "1.5xpeak": "#67a9cf", "full": "#2166ac"}
SHAPE_DESC = {"A": "shape A (R0 = 3)", "B": "shape B (R0 = 1.5)", "C": "shape C (R0 = 10)",
              "D": "shape D (R0 = 0.86, pure decay)", "E": "shape E (R0 = 1, at threshold)"}
BAND = (0.855, 0.945)


def usable_width(target, truth_row):
    """Width of a 90% interval that I call 'narrow': factor 2 either way for log targets,
    +-50% of the true value for growth rate and peak time."""
    if target == "growth_rate":
        return abs(truth_row["growth_true"])
    if target == "peak_time":
        return truth_row["t_peak"] if truth_row["t_peak"] > 0 else 1.0
    return 2 * np.log(2)


def load():
    df = pd.read_csv(RUN / "summary.csv")
    df["posterior_source"] = "v1"
    if (RERUN / "summary.csv").exists():
        re = pd.read_csv(RERUN / "summary.csv")
        re["posterior_source"] = "v2-rerun"
        key = ["cell", "target", "method"]
        df = df.set_index(key)
        re = re[re.method == "posterior"].set_index(key)
        df.loc[re.index, re.columns] = re
        df = df.reset_index()
    tt = Path("experiments/WP-02/e1-analysis/trust_table.csv")
    if tt.exists():  # resolved posterior (v1 / v2 / v2-rerun / v3) from analyse_e1.py takes precedence
        t = pd.read_csv(tt).set_index(["cell", "target"])
        m = df.method == "posterior"
        keys = list(zip(df.loc[m, "cell"], df.loc[m, "target"]))
        df.loc[m, "cov90"] = t.loc[keys, "cov"].to_numpy()
        df.loc[m, "width90_median"] = t.loc[keys, "width"].to_numpy()
        df.loc[m, "posterior_source"] = "resolved"
    growth = df[df.target == "growth_rate"].groupby("cell").truth.first()
    df["growth_true"] = df.cell.map(growth)
    df["usable_width"] = [usable_width(t, r) for t, r in zip(df.target, df.to_dict("records"))]
    return df


def fig_lines(df, shape, value, fname, title, ylab, log_y):
    sub = df[df["shape"] == shape]
    fig, axes = plt.subplots(2, 5, figsize=(21, 8.2), sharex=True)
    for ax, target in zip(axes.ravel(), ORDER):
        for win in sub.window.unique():
            for method, ls in (("posterior", "-"), ("laplace", "--")):
                s = sub[(sub.window == win) & (sub.target == target) & (sub.method == method)].sort_values("n_obs_expected")
                if value == "width90_median" and method == "laplace":
                    continue
                ax.plot(s.n_obs_expected, s[value], ls, color=WIN_COL[win], marker="o", ms=4, lw=1.8,
                        alpha=1.0 if method == "posterior" else 0.75)
        if value == "width90_median":
            ax.axhline(sub[sub.target == target].usable_width.iloc[0], color="k", lw=1, ls=":")
        else:
            ax.axhspan(*BAND, color="0.85", zorder=0)
            ax.axhline(0.9, color="k", lw=0.8)
            ax.set_ylim(0, 1.02)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(NICE[target], fontsize=11)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("expected posts inside the observed window")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylab)
    handles = [plt.Line2D([], [], color=WIN_COL[w], lw=2, label="window " + WIN_LABEL[w]) for w in sub.window.unique()]
    if value != "width90_median":
        handles += [plt.Line2D([], [], color="k", lw=2, ls="-", label="full posterior"),
                    plt.Line2D([], [], color="k", lw=2, ls="--", label="Hessian (Laplace) interval; no interval = miss"),
                    plt.Rectangle((0, 0), 1, 1, color="0.85", label="90% ± 3 binomial s.e. (400 replicates)")]
    else:
        handles += [plt.Line2D([], [], color="k", lw=1, ls=":", label="'narrow' line: factor 2 either way (log targets) / ±50% of truth")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def verdict(row):
    """0 miscalibrated (covers < 85.5%), 1 honest but wide, 2 honest and narrow, 3 over-covers and wide."""
    if row.cov90 < BAND[0]:
        return 0
    narrow = row.width90_median <= row.usable_width
    return 2 if narrow else 1


def fig_verdict(df, method, fname, title):
    shapes = ["A", "B", "C", "D", "E"]
    cmap = ListedColormap(["#d73027", "#fee08b", "#1a9850"])
    fig, axes = plt.subplots(1, 5, figsize=(24, 6.2), gridspec_kw=dict(width_ratios=[24, 24, 24, 6, 6]))
    for ax, shape in zip(axes, shapes):
        sub = df[(df["shape"] == shape) & (df.method == method)]
        wins = list(sub.window.unique())
        counts = sorted(sub.n_obs_expected.unique())
        mat = np.full((len(ORDER), len(wins) * len(counts)), np.nan)
        for i, t in enumerate(ORDER):
            for a, w in enumerate(wins):
                for b, n in enumerate(counts):
                    r = sub[(sub.target == t) & (sub.window == w) & (sub.n_obs_expected == n)]
                    if len(r):
                        r = r.iloc[0]
                        mat[i, a * len(counts) + b] = verdict(r)
                        ax.text(a * len(counts) + b, i, f"{100*r.cov90:.0f}", ha="center", va="center", fontsize=6.5)
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(wins) * len(counts)))
        ax.set_xticklabels([str(n) for _ in wins for n in counts], rotation=90, fontsize=7)
        for a, w in enumerate(wins):
            ax.text(a * len(counts) + (len(counts) - 1) / 2, -0.75, WIN_LABEL[w].replace(" (99% of posts)", ""), ha="center", fontsize=8)
            if a:
                ax.axvline(a * len(counts) - 0.5, color="k", lw=1.5)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([NICE[t] for t in ORDER] if shape == "A" else [], fontsize=9)
        ax.set_title(SHAPE_DESC[shape] + "\n", fontsize=10)
        ax.set_xlabel("expected observed posts")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in (
        ("#1a9850", "trustworthy: 90% interval covers truth ≥ 85.5% of the time AND is narrow (factor 2 / ±50%)"),
        ("#fee08b", "honest but wide: covers, interval too wide to be a useful label"),
        ("#d73027", "misleading: covers the truth < 85.5% of the time (number in cell = coverage %)"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def fig_examples():
    meta = json.load((RUN / "cells.json").open())["cells"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    for i, shape in enumerate("ABC"):
        for j, win in enumerate(WIN_LABEL):
            c = next(c for c in meta if c["shape"] == shape and c["window"] == win and c["n_obs_expected"] == 100)
            d = np.load(RUN / f"cell-{c['cell']:03d}.npz")
            ax = axes[i, j]
            t = np.arange(c["t_end"]) + 0.5
            for k in range(3):
                ax.plot(t, d["y_first"][k], "o-", ms=3, lw=0.7, alpha=0.7)
            ax.plot(t, d["mu_true"], "k-", lw=2, label="true expected posts/day")
            ax.axvline(c["t_peak"], color="0.5", ls="--", label="true peak time")
            ax.set_title(f"{SHAPE_DESC[shape]}, window {WIN_LABEL[win]}\n(sees {100*c['frac_observed']:.0f}% of all posts)", fontsize=9)
            ax.set_xlabel("day"); ax.set_ylabel("posts per day")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("What the simulated data look like at ~100 observed posts: three example datasets (colours) per shape and window", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "e1_example_data.png", dpi=120)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    df["verdict"] = [verdict(r) for r in df.itertuples()]
    df.to_csv(OUT / "e1_summary_merged.csv", index=False)
    fig_examples()
    for shape in "ABC":
        fig_lines(df, shape, "width90_median", f"e1_width_{shape}.png",
                  f"{SHAPE_DESC[shape]}: how wide the 90% posterior interval is, by observed posts and window. "
                  "Below the dotted line = narrow enough to be a useful label.",
                  "median width of 90% posterior interval\n(log targets: natural-log units)", True)
        fig_lines(df, shape, "cov90", f"e1_coverage_{shape}.png",
                  f"{SHAPE_DESC[shape]}: how often the stated 90% interval contains the truth. In the grey band = honest.",
                  "fraction of 400 replicates covered", False)
    fig_verdict(df, "posterior", "e1_verdict_posterior.png",
                "Full posterior: which targets can be trusted, by curve shape, observation window and observed post count")
    fig_verdict(df, "laplace", "e1_verdict_laplace.png",
                "Hessian (Laplace) intervals from the maximum-likelihood fit: same grid. Red = stated 90% interval is misleading")


if __name__ == "__main__":
    main()
