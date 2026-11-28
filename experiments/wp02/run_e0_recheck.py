"""E0: re-examine two first-pass claims (non-positive Hessians; case-052 ridge posterior)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.wp02 import calib as C
from experiments.wp02.sir import SirParameters, simulate_counts

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/WP-02/e0-first-pass-recheck")
SEED = 20261130


def regenerate(row, edges):
    params = SirParameters(beta=row.true_beta, gamma=row.true_gamma, scale=row.true_scale, i0=row.true_i0)
    counts, _ = simulate_counts(jax.random.PRNGKey(int(row.seed)), params, jnp.asarray(edges))
    return np.asarray(counts, float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    grid = pd.read_csv("experiments/WP-02/grid-sweep-001/hessian_grid.csv")
    edges = np.arange(0.0, 61.0)
    forward = C.make_forward(edges)
    fitter = C.make_fitter(edges, n_iter=200)

    # shared prior-wide cloud for data-scored starts
    pts = rng.uniform(C.BOX3[:, 0], C.BOX3[:, 1], size=(200_000, 3))
    m = forward(pts)
    logm, log_big_m = np.log(m), np.log(m.sum(axis=1))

    # ---- (a) the 12 non-positive-Hessian cases ---------------------------------
    bad = grid[~grid.hessian_positive_definite]
    ys, starts, old = [], [], []
    for row in bad.itertuples():
        y = regenerate(row, edges)
        assert y.sum() == row.observed_counts_total, (row.case_id, y.sum(), row.observed_counts_total)
        ys.append(y)
        n = max(y.sum(), 1e-9)
        k = np.argmax(y @ logm.T - n * log_big_m)
        starts.append([pts[k, 0], pts[k, 1], np.log(n) - log_big_m[k], pts[k, 2]])
        old.append([np.log(row.mle_beta), np.log(row.mle_gamma), np.log(row.mle_scale), np.log(row.mle_i0 / (1 - row.mle_i0))])
    ys, starts, old = np.array(ys), np.array(starts), np.array(old)
    zero = jnp.zeros(len(ys))
    rows = []
    res_new = [np.asarray(a) for a in fitter(jnp.asarray(starts), jnp.asarray(ys), zero)]
    res_old = [np.asarray(a) for a in fitter(jnp.asarray(np.clip(old, C.BOX4[:, 0], C.BOX4[:, 1])), jnp.asarray(ys), zero)]
    for i, row in enumerate(bad.itertuples()):
        best = res_new if res_new[1][i] <= res_old[1][i] else res_old
        theta, nll_new, hess = best[0][i], best[1][i], best[2][i]
        eig = np.linalg.eigvalsh(0.5 * (hess + hess.T))
        on_bound = bool(np.any((theta < C.BOX4[:, 0] + 1e-4) | (theta > C.BOX4[:, 1] - 1e-4)))
        rows.append(dict(case_id=row.case_id, counts=row.observed_counts_total, R0_true=row.true_beta / row.true_gamma,
                         nll_first_pass=row.neg_log_likelihood, nll_refit=float(nll_new),
                         nll_improvement=float(row.neg_log_likelihood - nll_new),
                         first_pass_min_eig=row.hessian_min_eigenvalue, refit_min_eig=float(eig[0]),
                         refit_on_bound=on_bound, refit_newton_dec=float(best[4][i]),
                         refit_beta=float(np.exp(theta[0])), refit_gamma=float(np.exp(theta[1])),
                         refit_scale=float(np.exp(theta[2])), refit_i0=float(1 / (1 + np.exp(-theta[3])))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "nonpd_refit.csv", index=False)
    print(df.to_string())
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(df))
    col = np.where(df.refit_on_bound, "#d73027", np.where(df.refit_min_eig > 1e-6, "#1a9850", "#fee08b"))
    ax.bar(xs, np.maximum(df.nll_improvement, 1e-4), color=col, edgecolor="k")
    ax.set_yscale("log"); ax.axhline(0.01, color="k", ls=":", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([f"{r.case_id}\n{int(r.counts)} posts\nR0={r.R0_true:.2f}" for r in df.itertuples()], fontsize=7)
    ax.set_ylabel("log-likelihood gained by refitting (nats)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in (
        ("#d73027", "refit runs to a parameter bound (likelihood keeps rising along a ridge)"),
        ("#1a9850", "refit finds an interior optimum with a positive-definite Hessian"),
        ("#fee08b", "interior, Hessian still singular/indefinite"))], fontsize=8, loc="upper left")
    n_moved = int(((df.nll_improvement > 0.01) | df.refit_on_bound).sum())
    ax.set_title(f"{n_moved} of {len(df)} first-pass fits flagged 'non-positive Hessian' were not at an interior maximum: refitting gains > 0.01 nats\n"
                 f"(dotted line) or runs to a parameter bound ({int(df.refit_on_bound.sum())} cases). Bars: likelihood gained; colour: where the refit ends.", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "nonpd_refit.png", dpi=140); plt.close(fig)

    # ---- (b) case-052 posterior ---------------------------------------------------
    row = grid[grid.case_id == "case-052"].iloc[0]
    y = regenerate(row, edges)
    truth3 = np.array([np.log(row.true_beta), np.log(row.true_gamma), np.log(row.true_i0 / (1 - row.true_i0))])
    theta4_true = np.array([truth3[0], truth3[1], np.log(row.true_scale), truth3[2]])
    fisher, _ = C.expected_fisher(theta4_true, edges)
    w, v = np.linalg.eigh(fisher)
    cov3 = ((v / np.maximum(w, 1e-4)) @ v.T)[np.ix_([0, 1, 3], [0, 1, 3])]
    out = {}
    for rep, n_cloud in enumerate((300_000, 600_000)):  # two independent clouds: numerical repeatability
        p, log_pq = C.build_cloud_v2(np.random.default_rng([SEED, rep]), truth3, cov3, n_cloud, prior_frac=0.6)
        mm = forward(p)
        ll = y @ np.log(mm).T - y.sum() * np.log(mm.sum(axis=1))
        logw = ll + log_pq
        wts = np.exp(logw - logw.max())
        wts /= wts.sum()
        i0 = 1 / (1 + np.exp(-p[:, 2]))
        high = i0 > 0.1
        out[f"cloud_{n_cloud}"] = dict(
            ess=float(1 / np.sum(wts**2)), mass_i0_gt_0p1=float(wts[high].sum()),
            max_profile_loglik_i0_gt_0p1=float(ll[high].max()), max_profile_loglik_i0_le_0p1=float(ll[~high].max()),
            profile_loglik_at_truth=float(y @ np.log(forward(truth3[None])[0]) - y.sum() * np.log(forward(truth3[None])[0].sum())),
            q05_q50_q95=dict(zip(("log_beta", "log_gamma", "logit_i0"),
                                 [[float(x) for x in _wq(p[:, c], wts)] for c in range(3)])),
        )
        keep = (p, wts, ll)
    json.dump(dict(seed=SEED, case="case-052", counts=float(y.sum()), results=out), (OUT / "case052_posterior.json").open("w"), indent=1)
    print(json.dumps(out, indent=1))

    p, wts, ll = keep
    idx = np.random.default_rng(SEED).choice(len(p), 4000, p=wts)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = ["log β", "log γ", "logit I0"]
    for ax, (a, b) in zip(axes[:2], ((0, 1), (0, 2))):
        ax.scatter(p[idx, a], p[idx, b], s=3, alpha=0.3, color="#1f6fb2", label="posterior draws (importance sampling)")
        ax.plot(truth3[a], truth3[b], "k*", ms=16, label="true parameters")
        ax.set_xlim(*C.BOX3[a]); ax.set_ylim(*C.BOX3[b])
        ax.set_xlabel(labels[a] + "  (axis spans the full prior range)"); ax.set_ylabel(labels[b])
        ax.legend(loc="lower left", fontsize=8)
    t = np.arange(60) + 0.5
    mm = forward(p[idx[:300]])
    s_draw = y.sum() / mm.sum(axis=1)
    axes[2].plot(t, (mm * s_draw[:, None]).T, color="#1f6fb2", alpha=0.05)
    axes[2].plot(t, y, "ko", ms=3, label="simulated posts per day (case-052)")
    axes[2].set_xlabel("day"); axes[2].set_ylabel("posts per day"); axes[2].legend(fontsize=8)
    axes[2].set_title("300 posterior curves (blue) vs data", fontsize=10)
    mass = out["cloud_600000"]["mass_i0_gt_0p1"]
    fig.suptitle(f"First-pass example case-052 (β=γ=1, 577 posts): {100*mass:.0f}% of the flat-prior posterior sits at I0 > 0.1, "
                 "far from the truth;\nall of it fits the curve. If the posterior identified the parameters, draws would cluster on the star.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "case052_posterior.png", dpi=140)


def _wq(x, w, qs=(0.05, 0.5, 0.95)):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return [x[o][np.searchsorted(c, q)] for q in qs]


if __name__ == "__main__":
    main()
