"""E1c: compare posteriors with I0 lower bound logit -14 vs -20 (same data). Output in e1-prior-box-check/."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from experiments.wp02.calib import TARGETS
from experiments.wp02.plot_e1 import NICE

B = Path("experiments/WP-02/e1-prior-box-check")
CELLS = {53: "C, ends at 0.7×peak, 3000 posts", 4: "A, ends at 0.7×peak, 1000 posts", 10: "A, ends at the peak, 1000 posts", 22: "A, full curve, 1000 posts (control)"}
TG = ["growth_rate", "log_init_rate", "peak_time", "log_peak_rate", "log_total", "log_R0"]
res = {}
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
x = np.arange(len(TG))
for k, (cell, lab) in enumerate(CELLS.items()):
    fa, fb = B / "logit_i0_lo_-14" / f"cell-{cell:03d}.npz", B / "logit_i0_lo_-20" / f"cell-{cell:03d}.npz"
    if not (fa.exists() and fb.exists()):
        continue
    a, b = np.load(fa), np.load(fb)
    r = {}
    for t in TG:
        j = TARGETS.index(t)
        wa = np.median(a["post_q95"][:, j] - a["post_q05"][:, j])
        wb = np.median(b["post_q95"][:, j] - b["post_q05"][:, j])
        cov = lambda d: float(np.mean((d["post_pit"][:, j] >= .05) & (d["post_pit"][:, j] <= .95)))
        r[t] = dict(width_m14=float(wa), width_m20=float(wb), ratio=float(wb / wa), cov_m14=cov(a), cov_m20=cov(b),
                    median_shift_over_width=float(np.median(np.abs(b["post_med"][:, j] - a["post_med"][:, j])) / wa))
    res[cell] = dict(label=lab, ess_m14=float(np.median(a["ess"])), ess_m20=float(np.median(b["ess"])), targets=r)
    axes[0].bar(x + (k - 1.5) * 0.2, [r[t]["ratio"] for t in TG], 0.2, label=lab)
    axes[1].bar(x + (k - 1.5) * 0.2, [r[t]["cov_m20"] - r[t]["cov_m14"] for t in TG], 0.2)
json.dump(res, (B / "prior_box_comparison.json").open("w"), indent=1)
for ax in axes:
    ax.set_xticks(x); ax.set_xticklabels([NICE[t] for t in TG], rotation=20, ha="right", fontsize=9); ax.grid(alpha=.3, axis="y")
axes[0].axhline(1, color="k", lw=.8); axes[0].axhline(1.3, color="k", ls=":", lw=1)
axes[0].set_ylabel("median 90% posterior width with I0 ≥ 2e-9\n÷ width with I0 ≥ 8e-7 (same data)")
axes[0].set_title("Widening the prior's lower bound on I0: how much wider do intervals get?\n(1 = data decide; dotted = +30% pre-registered line)", fontsize=10)
axes[1].axhline(0, color="k", lw=.8)
axes[1].set_ylabel("change in coverage of 90% interval\n(I0 ≥ 2e-9 minus I0 ≥ 8e-7)")
axes[1].set_title("…and how their coverage changes", fontsize=10)
axes[0].legend(fontsize=8)
fig.suptitle("Prior-bound sensitivity: only shape A observed to 0.7×peak changes (+53-65% width, coverage down 7-8 pp);\n"
             "shape C at 0.7×peak, shape A at the peak, and the full-curve control do not move", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(B / "prior_box_check.png", dpi=130)
for c, v in res.items():
    print(c, v["label"], {t: (round(z["ratio"], 2), round(z["cov_m14"], 2), round(z["cov_m20"], 2)) for t, z in v["targets"].items()})
