"""Plot E0(a) from experiments/WP-02/e0-first-pass-recheck/nonpd_refit.csv (no recomputation).
Output: experiments/WP-02/e0-plots/nonpd_refit.png"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("experiments/WP-02/e0-first-pass-recheck/nonpd_refit.csv")
out = Path("experiments/WP-02/e0-plots"); out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(11, 5.2))
xs = np.arange(len(df))
col = np.where(df.refit_on_bound, "#d73027", np.where(df.refit_min_eig > 1e-6, "#1a9850", "#fee08b"))
ax.bar(xs, np.maximum(df.nll_improvement, 1e-4), color=col, edgecolor="k")
ax.set_yscale("log"); ax.axhline(0.01, color="k", ls=":", lw=1)
ax.set_xticks(xs); ax.set_xticklabels([f"{r.case_id}\n{int(r.counts)} posts\nR0={r.R0_true:.2f}" for r in df.itertuples()], fontsize=7)
ax.set_ylabel("log-likelihood gained by refitting\nfrom a data-based start (nats, log scale)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in (
    ("#d73027", "refit runs to a parameter bound (likelihood still rising along a ridge)"),
    ("#1a9850", "refit reaches an interior optimum with positive-definite Hessian"),
    ("#fee08b", "interior, Hessian still singular or indefinite"))], fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
n_moved = int(((df.nll_improvement > 0.01) | df.refit_on_bound).sum())
ax.set_title(f"First-pass 'non-positive Hessian' fits: {n_moved} of {len(df)} were not at an interior maximum (refit gains > 0.01 nats, dotted,\n"
             f"or runs to a bound: {int(df.refit_on_bound.sum())}). 10 of 12 have under 70 posts; the other two are subcritical / at threshold.", fontsize=10)
fig.tight_layout(); fig.savefig(out / "nonpd_refit.png", dpi=140)
print(n_moved, int(df.refit_on_bound.sum()))
