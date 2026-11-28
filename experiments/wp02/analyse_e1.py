"""E1 analysis beyond summary.csv: conditional calibration, flag rules F1-F3, practical trust map.

Inputs: experiments/WP-02/e1-window-counts (all cells) and experiments/WP-02/e1-rerun-v2 (v2 posterior
replacements for v1 cells with poor importance-sampling ESS). Convention (same as plot_e1): for a rerun
cell the posterior comes from the rerun file, the Laplace fit from the original file.
Outputs into experiments/WP-02/e1-analysis/.
Usage: uv run python experiments/wp02/analyse_e1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.append(str(Path(__file__).resolve().parents[2]))
from experiments.wp02.calib import TARGETS
from experiments.wp02.plot_e1 import NICE, ORDER, SHAPE_DESC, WIN_LABEL

RUN = Path("experiments/WP-02/e1-window-counts")
RERUN = Path("experiments/WP-02/e1-rerun-v2")
RERUN3 = Path("experiments/WP-02/e1-rerun-v3")
OUT = Path("experiments/WP-02/e1-analysis")
Z90 = 1.6448536269514722
SUMMARIES = ["growth_rate", "log_init_rate", "peak_time", "log_peak_rate", "log_total", "log_R0"]


def usable(target, truth, t_peak):
    if target == "growth_rate":
        return abs(truth)
    if target == "peak_time":
        return t_peak if t_peak > 0 else 1.0
    return 2 * np.log(2)


def load_cells():
    cells = json.load((RUN / "cells.json").open())["cells"]
    out = []
    for c in cells:
        p = RUN / f"cell-{c['cell']:03d}.npz"
        if not p.exists():
            continue
        d = dict(np.load(p))
        src = "v1" if "proposal" not in d else str(d["proposal"])
        rp = RERUN / f"cell-{c['cell']:03d}.npz"
        if rp.exists():
            r = dict(np.load(rp))
            assert np.array_equal(r["n_obs"], d["n_obs"]), c["cell"]
            d["v1_post_q05"], d["v1_post_q95"], d["v1_post_pit"] = d["post_q05"], d["post_q95"], d["post_pit"]
            for k in r:
                if k.startswith("post_") or k == "ess":
                    d[k] = r[k]
            d["v2rerun_pois_pit_obs"], d["v2rerun_pois_se_obs"] = r["pois_pit_obs"], r["pois_se_obs"]
            src = "v2-rerun"
        p3 = RERUN3 / f"cell-{c['cell']:03d}.npz"
        r = dict(np.load(p3)) if p3.exists() else None
        # adaptive proposal, validated on control cell 33 (compare_v2_v3.json); adopted only where it
        # lowers the share of replicates with ESS < 200 (it made subcritical cells 77, 81 worse)
        if r is not None and np.mean(r["ess"] < 200) < np.mean(d["ess"] < 200):
            assert np.array_equal(r["n_obs"], d["n_obs"]), c["cell"]
            if "v1_post_pit" not in d:
                d["v1_post_q05"], d["v1_post_q95"], d["v1_post_pit"] = d["post_q05"], d["post_q95"], d["post_pit"]
            for k in r:
                if k.startswith("post_") or k == "ess":
                    d[k] = r[k]
            src = "v3"
        out.append((c, d, src))
    return out


def per_replicate_tables(cells):
    """Long table: one row per (cell, replicate, target)."""
    rows = []
    for c, d, src in cells:
        has = d["n_obs"] >= 1
        n = len(has)
        f1 = d["pois_pd_obs"] & ~d["pois_boundary"] & (d["pois_newton_dec"] < 1e-4)
        tercile = pd.qcut(pd.Series(d["n_obs"]).rank(method="first"), 3, labels=[0, 1, 2]).to_numpy()
        for j, t in enumerate(TARGETS):
            truth = d["truth"][j]
            pw = d["post_q95"][:, j] - d["post_q05"][:, j]
            se = d["pois_se_obs"][:, j]
            lap_valid = np.isfinite(se) & (se > 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                lap_pit = np.where(lap_valid, ndtr((truth - d["pois_est"][:, j]) / se), np.nan)
            rows.append(pd.DataFrame(dict(
                cell=c["cell"], shape=c["shape"], window=c["window"], n_exp=c["n_obs_expected"],
                total_posts=c["total_posts"], frac_obs=c["frac_observed"], t_peak=c["t_peak"], src=src,
                rep=np.arange(n), has=has, n_obs=d["n_obs"], tercile=tercile, target=t, truth=truth,
                usable=usable(t, truth, c["t_peak"]),
                post_cov=(d["post_pit"][:, j] >= 0.05) & (d["post_pit"][:, j] <= 0.95),
                post_cov50=(d["post_pit"][:, j] >= 0.25) & (d["post_pit"][:, j] <= 0.75),
                post_width=pw, post_err=d["post_med"][:, j] - truth,
                lap_cov=(lap_pit >= 0.05) & (lap_pit <= 0.95), lap_valid=lap_valid, lap_width=2 * Z90 * se,
                lap_err=d["pois_est"][:, j] - truth, f1=f1, p_peak=d["post_p_peak_in_window"], ess=d["ess"],
                mle_peak_in_window=d["pois_peak_in_window"],
            )))
    return pd.concat(rows, ignore_index=True)


def binom_se(p, n):
    return np.sqrt(np.clip(p * (1 - p), 1e-4, None) / np.maximum(n, 1))


def conditional_calibration(long):
    g = long[long.has].groupby(["cell", "shape", "window", "n_exp", "target"])
    cell = g.agg(post_cov=("post_cov", "mean"), lap_cov=("lap_cov", "mean"), n=("rep", "size")).reset_index()
    low = long[long.has & (long.tercile == 0)].groupby(["cell", "target"]).agg(
        post_cov_low=("post_cov", "mean"), lap_cov_low=("lap_cov", "mean"), n_low=("rep", "size")).reset_index()
    cc = cell.merge(low, on=["cell", "target"])
    cc.to_csv(OUT / "conditional_calibration.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, m, lab in ((axes[0], "post", "full posterior"), (axes[1], "lap", "Hessian (Laplace) interval")):
        for t in SUMMARIES:
            s = cc[cc.target == t]
            ax.scatter(s[f"{m}_cov"], s[f"{m}_cov_low"], s=14, alpha=0.7, label=NICE[t])
        ax.plot([0, 1], [0, 1], "k-", lw=0.8)
        ax.axhspan(0.9 - 3 * binom_se(0.9, 133), 0.9 + 3 * binom_se(0.9, 133), color="0.88", zorder=0)
        ax.set_xlabel(f"coverage of 90% {lab}, all replicates in the cell")
        ax.set_ylabel("coverage among the third of replicates\nwith the fewest realised posts")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.set_title(lab)
    axes[0].legend(fontsize=7, loc="lower right")
    d = cc.post_cov_low - cc.post_cov
    fig.suptitle("Does coverage hold for the unluckiest (lowest-count) datasets in each cell? On the diagonal = yes.\n"
                 f"Posterior: median shift {100*np.median(d):+.1f} pp, 5th pct {100*np.quantile(d, .05):+.1f} pp. Grey = 90% ± 3 s.e. for 133 replicates",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "e1_conditional_calibration.png", dpi=130)
    plt.close(fig)
    return cc


def flags(long):
    L = long[long.has & long.target.isin(SUMMARIES)].copy()
    L["f2"] = L.p_peak > 0.9
    L["f3"] = L.post_width <= L.usable
    L["f2_mle"] = L.mle_peak_in_window  # exploratory, not pre-registered
    res = []
    for flag, method in (("f1", "lap"), ("f2", "post"), ("f3", "post"), ("f2_mle", "post"), ("f2_and_f3", "post"), ("none", "post"), ("none", "lap")):
        if flag == "f2_and_f3":
            L[flag] = L.f2 & L.f3
        if flag == "none":
            L[flag] = True
        for (shape_grp, t), s in L.groupby([L["shape"].isin(list("ABC")).map({True: "ABC (grows then decays)", False: "DE (no growth)"}), "target"]):
            p = s[s[flag]]
            cov = p[f"{method}_cov"].mean() if len(p) else np.nan
            # cell-clustered s.e.: coverage computed per cell then s.d. over cells / sqrt(n cells)
            per_cell = p.groupby("cell")[f"{method}_cov"].mean()
            res.append(dict(flag=flag, method=method, group=shape_grp, target=t, pass_rate=s[flag].mean(), n_pass=len(p),
                            n_cells_with_pass=p.cell.nunique(), cov90_pooled=cov,
                            worst_cell_cov=per_cell[p.groupby("cell").size() >= 40].min() if len(p) else np.nan,
                            frac_pass_narrow=(p[f"{method}_width"] <= p.usable).mean() if len(p) else np.nan,
                            med_abs_err=p[f"{method}_err"].abs().median() if len(p) else np.nan))
    df = pd.DataFrame(res)
    df.to_csv(OUT / "flag_evaluation.csv", index=False)
    # per-cell table for F2/F3 by window
    per = L.groupby(["shape", "window", "n_exp", "target"]).apply(lambda s: pd.Series(dict(
        pass_f1=s.f1.mean(), lap_cov_f1=s.loc[s.f1, "lap_cov"].mean() if s.f1.any() else np.nan,
        pass_f2=s.f2.mean(), post_cov_f2=s.loc[s.f2, "post_cov"].mean() if s.f2.any() else np.nan,
        pass_f3=s.f3.mean(), post_cov_f3=s.loc[s.f3, "post_cov"].mean() if s.f3.any() else np.nan,
        post_cov=s.post_cov.mean(), lap_cov=s.lap_cov.mean())), include_groups=False).reset_index()
    per.to_csv(OUT / "flag_per_cell.csv", index=False)
    return df, per


def fig_flags(df):
    sub = df[df.group.str.startswith("ABC")]
    labels = {("none", "lap"): "Hessian interval, all fits", ("f1", "lap"): "Hessian interval, fits passing F1\n(Hessian PD, off-bound, converged)",
              ("none", "post"): "posterior, all fits", ("f2", "post"): "posterior, fits passing F2\n(P(peak inside window) > 0.9)",
              ("f3", "post"): "posterior, fits passing F3\n(own 90% width below 'narrow' line)", ("f2_and_f3", "post"): "posterior, F2 and F3"}
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    keys = list(labels)
    x = np.arange(len(SUMMARIES))
    for k, key in enumerate(keys):
        s = sub[(sub.flag == key[0]) & (sub.method == key[1])].set_index("target").reindex(SUMMARIES)
        axes[0].bar(x + (k - 2.5) * 0.13, s.cov90_pooled, 0.13, label=labels[key])
        axes[1].bar(x + (k - 2.5) * 0.13, s.pass_rate, 0.13)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([NICE[t] for t in SUMMARIES], rotation=20, ha="right", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0].axhspan(0.855, 0.945, color="0.85", zorder=0); axes[0].axhline(0.9, color="k", lw=0.8)
    axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("coverage of 90% intervals among fits that pass the flag\n(pooled over all shape A-C cells)")
    axes[1].set_ylabel("fraction of all simulated fits that pass"); axes[1].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_title("Honesty of intervals among passing fits (grey band = honest)")
    axes[1].set_title("How many fits pass")
    fig.suptitle("Which reliability flag picks out trustworthy fits? (shapes A-C, all windows and counts pooled)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "e1_flags.png", dpi=130)
    plt.close(fig)


def trust_map(long):
    """Per target: verdict on the (total posts of whole curve, observed fraction) plane."""
    L = long[long.has]
    agg = L.groupby(["cell", "shape", "window", "n_exp", "target"]).agg(
        total_posts=("total_posts", "first"), frac_obs=("frac_obs", "first"), cov=("post_cov", "mean"),
        width=("post_width", "median"), usable=("usable", "first"), lap_cov=("lap_cov", "mean"),
        lap_width=("lap_width", "median"), lap_valid=("lap_valid", "mean"), ess_med=("ess", "median"),
        frac_ess_lt200=("ess", lambda e: np.mean(e < 200))).reset_index()
    agg["verdict"] = np.where(agg["cov"] < 0.855, 0, np.where(agg.width <= agg.usable, 2, 1))
    lap_cov_all = agg.lap_cov  # NaN interval counted as a miss already
    agg["lap_verdict"] = np.where(lap_cov_all < 0.855, 0, np.where(agg.lap_width <= agg.usable, 2, 1))
    agg.to_csv(OUT / "trust_table.csv", index=False)
    cols = {0: "#d73027", 1: "#fee08b", 2: "#1a9850"}
    mk = {"A": "o", "B": "s", "C": "^", "D": "v", "E": "D"}
    for method, vcol, fname in (("posterior", "verdict", "e1_trust_map_posterior.png"), ("Hessian/Laplace", "lap_verdict", "e1_trust_map_laplace.png")):
        fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True, sharey=True)
        for ax, t in zip(axes.ravel(), ORDER):
            s = agg[(agg.target == t)]
            for shp, ss in s.groupby("shape"):
                ax.scatter(ss.frac_obs, ss.total_posts, c=[cols[v] for v in ss[vcol]], marker=mk[shp], s=70, edgecolor="k", lw=0.5)
            ax.set_xscale("logit"); ax.set_yscale("log")
            ax.set_xticks([0.1, 0.25, 0.5, 0.8, 0.99]); ax.set_xticklabels(["10%", "25%", "50%", "80%", "99%"])
            ax.set_title(NICE[t], fontsize=11); ax.grid(alpha=0.3)
        for ax in axes[1]:
            ax.set_xlabel("fraction of the meme's eventual posts already observed")
        for ax in axes[:, 0]:
            ax.set_ylabel("total posts the whole curve will have")
        handles = [plt.Line2D([], [], ls="", marker="s", ms=10, color=cols[v], label=l) for v, l in
                   ((2, "trustworthy (covers ≥85.5% and narrow)"), (1, "honest but too wide"), (0, "misleading (covers <85.5%)"))]
        handles += [plt.Line2D([], [], ls="", marker=mk[s], color="0.5", mec="k", label=SHAPE_DESC[s]) for s in mk]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False)
        fig.suptitle(f"Where can each SIR quantity be trusted? {method} 90% intervals, by meme size and how much of it has been seen\n"
                     "(each marker = 400 simulated datasets with known truth)", fontsize=13)
        fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        fig.savefig(OUT / fname, dpi=120)
        plt.close(fig)
    return agg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    print("cells loaded", len(cells), {s: sum(1 for _, _, x in cells if x == s) for s in ("v1", "v2", "v2-rerun", "v3")})
    long = per_replicate_tables(cells)
    long.to_parquet(OUT / "per_replicate.parquet") if False else None
    cc = conditional_calibration(long)
    df, per = flags(long)
    fig_flags(df)
    agg = trust_map(long)
    # numerical error of the posterior: v1 vs v2 for rerun cells
    rows = []
    for c, d, src in cells:
        if src not in ("v2-rerun", "v3") or "v1_post_pit" not in d:
            continue
        for j, t in enumerate(TARGETS):
            w1 = np.median(d["v1_post_q95"][:, j] - d["v1_post_q05"][:, j])
            w2 = np.median(d["post_q95"][:, j] - d["post_q05"][:, j])
            c1 = np.mean((d["v1_post_pit"][:, j] >= .05) & (d["v1_post_pit"][:, j] <= .95))
            c2 = np.mean((d["post_pit"][:, j] >= .05) & (d["post_pit"][:, j] <= .95))
            rows.append(dict(src=src, cell=c["cell"], shape=c["shape"], window=c["window"], n_exp=c["n_obs_expected"], target=t,
                             width_v1=w1, width_v2=w2, cov_v1=c1, cov_v2=c2))
    pd.DataFrame(rows).to_csv(OUT / "posterior_v1_vs_v2.csv", index=False)
    print(df[df.group.str.startswith("ABC")].pivot_table(index=["flag", "method"], columns="target", values="cov90_pooled").round(3).to_string())


if __name__ == "__main__":
    main()
