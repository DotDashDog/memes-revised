"""E0(b) figure: recompute the case-052 posterior with the first (300k) cloud of run_e0_recheck.py
(same seed -> should reproduce cloud_300000 in case052_posterior.json) and plot it.
Output: experiments/WP-02/e0-plots/case052_posterior.png and case052_reproduced.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.wp02 import calib as C
from experiments.wp02.run_e0_recheck import regenerate, SEED, _wq

OUT = Path("experiments/WP-02/e0-plots"); OUT.mkdir(parents=True, exist_ok=True)
grid = pd.read_csv("experiments/WP-02/grid-sweep-001/hessian_grid.csv")
edges = np.arange(0.0, 61.0)
forward = C.make_forward(edges)
row = grid[grid.case_id == "case-052"].iloc[0]
y = regenerate(row, edges)
truth3 = np.array([np.log(row.true_beta), np.log(row.true_gamma), np.log(row.true_i0 / (1 - row.true_i0))])
theta4 = np.array([truth3[0], truth3[1], np.log(row.true_scale), truth3[2]])
fisher, _ = C.expected_fisher(theta4, edges)
w, v = np.linalg.eigh(fisher)
cov3 = ((v / np.maximum(w, 1e-4)) @ v.T)[np.ix_([0, 1, 3], [0, 1, 3])]
p, log_pq = C.build_cloud_v2(np.random.default_rng([SEED, 0]), truth3, cov3, 300_000, prior_frac=0.6)
mm = forward(p)
ll = y @ np.log(mm).T - y.sum() * np.log(mm.sum(axis=1))
lw = ll + log_pq; wts = np.exp(lw - lw.max()); wts /= wts.sum()
i0 = 1 / (1 + np.exp(-p[:, 2])); high = i0 > 0.1
res = dict(ess=float(1 / np.sum(wts**2)), mass_i0_gt_0p1=float(wts[high].sum()),
           max_ll_high=float(ll[high].max()), max_ll_low=float(ll[~high].max()))
json.dump(res, (OUT / "case052_reproduced.json").open("w"), indent=1)
print(res)
idx = np.random.default_rng(SEED).choice(len(p), 4000, p=wts)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
labels = ["log β (infection rate, per day)", "log γ (recovery rate, per day)", "logit I0 (initial infected fraction)"]
for ax, (a, b) in zip(axes[:2], ((0, 1), (0, 2))):
    hi = i0[idx] > 0.1
    ax.scatter(p[idx[~hi], a], p[idx[~hi], b], s=3, alpha=0.3, color="#1f6fb2", label="posterior draws, I0 ≤ 0.1")
    ax.scatter(p[idx[hi], a], p[idx[hi], b], s=3, alpha=0.3, color="#d95f02", label="posterior draws, I0 > 0.1")
    ax.plot(truth3[a], truth3[b], "k*", ms=16, label="true parameters")
    ax.set_xlim(*C.BOX3[a]); ax.set_ylim(*C.BOX3[b])
    ax.set_xlabel(labels[a] + "\n(axes span the whole prior range)"); ax.set_ylabel(labels[b])
    ax.legend(loc="lower left", fontsize=8, markerscale=3)
t = np.arange(60) + 0.5
sub = idx[:300]
m3 = forward(p[sub]); s_draw = y.sum() / m3.sum(axis=1)
cols = np.where(i0[sub] > 0.1, "#d95f02", "#1f6fb2")
for k in range(len(sub)):
    axes[2].plot(t, m3[k] * s_draw[k], color=cols[k], alpha=0.06)
axes[2].plot(t, y, "ko", ms=3, label="simulated posts per day (577 in total)")
axes[2].set_xlabel("day"); axes[2].set_ylabel("posts per day"); axes[2].legend(fontsize=8)
axes[2].set_title("300 posterior curves vs the data: both groups fit", fontsize=10)
fig.suptitle(f"First-pass example 'case-052' (β = γ = 1, I0 = 0.003): {100*res['mass_i0_gt_0p1']:.0f}% of the flat-prior posterior has I0 > 0.1 (orange), "
             f"far from the truth (star).\nBest log-likelihood there is {res['max_ll_high']-res['max_ll_low']:+.2f} nats vs the I0 ≤ 0.1 region: "
             "the data cannot tell 'small outbreak at threshold' from 'large decaying start'.", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "case052_posterior.png", dpi=130)
