"""E1 posterior recomputation with an adaptive importance-sampling proposal ("v3").

For cells where the v2 cloud still leaves > 10% of replicates with effective sample size < 200.
Stage 1 reproduces the v2 cloud exactly (same seed and random stream as run_cell). Stage 2 builds
one shared cloud per cell from the stage-1 posteriors of all replicates: an equal-weight mixture of
Gaussians in psi = (growth rate, log gamma, logit I0) coordinates, one per replicate, centred on that
replicate's stage-1 posterior mean with sd inflated x2 (covariance x4) and a diagonal floor, plus a
20% uniform-prior defensive component. The target posterior is unchanged (flat box prior, s
integrated out); only the quadrature changes. The data are regenerated from the recorded seed and
checked against the stored file.

Usage: uv run python experiments/wp02/run_e1_v3.py CELL [CELL ...]
Writes experiments/WP-02/e1-rerun-v3/cell-NNN.npz with post_* arrays, ess, ess_stage1, config.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from experiments.wp02 import calib as C

RUN = Path("experiments/WP-02/e1-window-counts")
OUT = Path("experiments/WP-02/e1-rerun-v3")
N_CLOUD1 = 120_000
N_CLOUD2 = 120_000
N_DRAWS = 4000


def to_psi(t):
    beta, gam = np.exp(t[..., 0]), np.exp(t[..., 1])
    i0 = 1 / (1 + np.exp(-t[..., 2]))
    return np.stack([beta * (1 - i0) - gam, t[..., 1], t[..., 2]], axis=-1), beta * (1 - i0)


def from_psi(p):
    gam = np.exp(np.clip(p[..., 1], -50, 50))
    i0 = 1 / (1 + np.exp(-np.clip(p[..., 2], -50, 50)))
    with np.errstate(invalid="ignore", divide="ignore"):
        logb = np.log((p[..., 0] + gam) / (1 - i0))
    return np.stack([logb, p[..., 1], p[..., 2]], axis=-1)


def weights(y, n_obs, logm, log_big_m, log_pq):
    loglik = y @ logm.T - n_obs[:, None] * log_big_m[None, :]
    logw = loglik + log_pq[None, :]
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=1, keepdims=True)


def build_v3(rng, pts1, w1, n_points, prior_frac=0.2, infl=2.0):
    lo, hi = C.BOX3[:, 0], C.BOX3[:, 1]
    vol = np.prod(hi - lo)
    psi1, _ = to_psi(pts1)
    means = w1 @ psi1  # (R, 3)
    covs = np.einsum("rk,rki,rkj->rij", w1, psi1[None] - means[:, None], psi1[None] - means[:, None])
    floor = np.diag([1e-4, 1e-3, 1e-3])
    covs = infl**2 * (covs + floor)
    chols = np.linalg.cholesky(covs)
    inv = np.linalg.inv(covs)
    logdet = np.linalg.slogdet(covs)[1]
    n_rep = len(means)
    n_prior = int(round(prior_frac * n_points))
    got = []
    need = n_points - n_prior
    while sum(len(g) for g in got) < need:
        comp = rng.integers(0, n_rep, size=need)
        x = means[comp] + np.einsum("kij,kj->ki", chols[comp], rng.standard_normal((need, 3)))
        t = from_psi(x)
        keep = np.all(np.isfinite(t), axis=1)
        keep[keep] &= np.all((t[keep] > lo) & (t[keep] < hi), axis=1)
        got.append(t[keep])
    gauss_pts = np.concatenate(got)[:need]
    pts = np.concatenate([rng.uniform(lo, hi, size=(n_prior, 3)), gauss_pts])
    # mixture density in theta coordinates. Gaussian part is truncated to the box: the truncation
    # constant per component is estimated by Monte Carlo (acceptance rate).
    acc = np.empty(n_rep)
    for r in range(n_rep):
        x = means[r] + rng.standard_normal((4000, 3)) @ chols[r].T
        t = from_psi(x)
        ok = np.all(np.isfinite(t), axis=1)
        ok[ok] &= np.all((t[ok] > lo) & (t[ok] < hi), axis=1)
        acc[r] = max(ok.mean(), 1e-4)
    # Sampling scheme: each component chosen with prob 1/R, then rejected outside box -> component r
    # contributes with relative frequency acc_r / sum(acc), density N_r / acc_r inside the box.
    # So the Gaussian-part density = sum_r (acc_r/sum acc) * N_r(x)/acc_r = sum_r N_r(x) / sum(acc).
    psi, jac = to_psi(pts)
    dens_g = np.zeros(len(pts))
    for start in range(0, n_rep, 50):
        sl = slice(start, start + 50)
        diff = psi[None, :, :] - means[sl, None, :]
        maha = np.einsum("rki,rij,rkj->rk", diff, inv[sl], diff)
        dens_g += np.exp(-0.5 * maha - 0.5 * logdet[sl, None] - 1.5 * np.log(2 * np.pi)).sum(axis=0)
    dens_g = dens_g / acc.sum() * jac
    dens = prior_frac / vol + (1 - prior_frac) * dens_g
    return pts, -np.log(dens)


def main():
    global OUT
    import os
    lo_logit = os.environ.get("WP02_BOX_LOGIT_I0_LO")  # prior-box sensitivity check (e1-prior-box-check)
    if lo_logit is not None:
        C.BOX3[2, 0] = float(lo_logit)
        C.BOX4[3, 0] = float(lo_logit)
        OUT = Path(f"experiments/WP-02/e1-prior-box-check/logit_i0_lo_{float(lo_logit):g}")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {c["cell"]: c for c in json.load((RUN / "cells.json").open())["cells"]}
    for cell in map(int, sys.argv[1:]):
        t0 = time.time()
        c = meta[cell]
        d = np.load(RUN / f"cell-{cell:03d}.npz")
        seed = int(d["seed"])
        rng = np.random.default_rng(seed)
        edges = np.arange(0.0, c["t_end"] + 1.0)
        theta4_true = d["theta4_true"]
        theta3_true = theta4_true[[0, 1, 3]]
        y = C.simulate(rng, d["mu_true"], len(d["n_obs"]))
        assert np.array_equal(y.sum(axis=1), d["n_obs"]) and np.array_equal(y[:8], d["y_first"])
        n_obs = y.sum(axis=1)
        fisher, _ = C.expected_fisher(theta4_true, edges)
        w, v = np.linalg.eigh(fisher)
        cov3 = ((v / np.maximum(w, 1e-4)) @ v.T)[np.ix_([0, 1, 3], [0, 1, 3])]
        forward = C.make_forward(edges)
        pts1, lpq1 = C.build_cloud_v2(rng, theta3_true, cov3, N_CLOUD1)
        m1 = forward(pts1)
        w1 = weights(y, n_obs, np.log(m1), np.log(m1.sum(axis=1)), lpq1)
        ess1 = 1 / (w1**2).sum(axis=1)
        rng2 = np.random.default_rng([seed, 3])
        pts, lpq = build_v3(rng2, pts1, w1, N_CLOUD2)
        del w1, m1
        m = forward(pts)
        logm, log_big_m = np.log(m), np.log(m.sum(axis=1))
        truth = d["truth"]
        d_cloud = C.derived(pts[:, 0], pts[:, 1], pts[:, 2], 0.0)
        scale_cols = np.array([t in ("log_scale", "log_peak_rate", "log_total", "log_init_rate") for t in C.TARGETS])
        n_rep, n_t = len(y), len(C.TARGETS)
        out = {k: np.full((n_rep, n_t), np.nan) for k in ("post_pit", "post_q05", "post_q25", "post_med", "post_q75", "post_q95")}
        out["ess"] = np.full(n_rep, np.nan)
        out["post_p_peak_in_window"] = np.full(n_rep, np.nan)
        out["post_p_supercritical"] = np.full(n_rep, np.nan)
        i_peak, i_growth = C.TARGETS.index("peak_time"), C.TARGETS.index("growth_rate")
        for start in range(0, n_rep, 100):
            sl = slice(start, min(start + 100, n_rep))
            wts = weights(y[sl], n_obs[sl], logm, log_big_m, lpq)
            out["ess"][sl] = 1 / (wts**2).sum(axis=1)
            for local, r in enumerate(range(sl.start, sl.stop)):
                idx = rng2.choice(len(pts), size=N_DRAWS, p=wts[local])
                log_s = np.log(rng2.gamma(n_obs[r], 1.0, size=N_DRAWS)) - log_big_m[idx]
                draws = d_cloud[idx] + scale_cols[None, :] * log_s[:, None]
                out["post_pit"][r] = (draws < truth).mean(axis=0) + 0.5 * (draws == truth).mean(axis=0)
                q = np.quantile(draws, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
                for name, row in zip(("post_q05", "post_q25", "post_med", "post_q75", "post_q95"), q):
                    out[name][r] = row
                out["post_p_peak_in_window"][r] = np.mean(draws[:, i_peak] < edges[-1])
                out["post_p_supercritical"][r] = np.mean(draws[:, i_growth] > 0)
        np.savez_compressed(OUT / f"cell-{cell:03d}.partial.npz", seed=seed, box3=C.BOX3, seed_stage2=np.array([seed, 3]), proposal="v3",
                            n_cloud1=N_CLOUD1, n_cloud2=N_CLOUD2, n_obs=n_obs, ess_stage1=ess1, truth=truth, **out)
        (OUT / f"cell-{cell:03d}.partial.npz").replace(OUT / f"cell-{cell:03d}.npz")
        print(f"cell {cell} ess stage1 med {np.median(ess1):.0f} frac<200 {np.mean(ess1<200):.2f} | v3 med {np.median(out['ess']):.0f} "
              f"frac<200 {np.mean(out['ess']<200):.2f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
