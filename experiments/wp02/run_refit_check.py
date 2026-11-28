"""Optimiser-convergence check for E1: refit selected cells with 10x more scoring iterations.

Regenerates each cell's datasets from its recorded seed, restarts the fit from the stored
40-iteration estimate, runs 300 more iterations (first 150 replicates) and recomputes Laplace coverage.
Usage: uv run python experiments/wp02/run_refit_check.py CELL_ID
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
import jax.numpy as jnp

from experiments.wp02 import calib as C

RUN = Path("experiments/WP-02/e1-window-counts")
OUT = Path("experiments/WP-02/e1-refit-check")


def main():
    cell = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    c = next(c for c in json.load((RUN / "cells.json").open())["cells"] if c["cell"] == cell)
    d = np.load(RUN / f"cell-{cell:03d}.npz")
    edges = np.arange(0.0, c["t_end"] + 1.0)
    rng = np.random.default_rng(int(d["seed"]))
    y = C.simulate(rng, d["mu_true"], len(d["n_obs"]))
    assert np.array_equal(y.sum(axis=1), d["n_obs"]) and np.array_equal(y[:8], d["y_first"])
    n_use = 150
    fitter = C.make_fitter(edges, n_iter=300)
    sub = {k: d[k][:n_use] for k in ('pois_theta','n_obs','pois_nll','pois_newton_dec','pois_boundary','pois_pd_obs','pois_pit_obs','pois_se_obs')}
    y = y[:n_use]
    d = dict(sub, truth=d['truth'], seed=d['seed'])
    theta, cur, hess, fisher, dec, mu = (np.asarray(a) for a in fitter(jnp.asarray(d["pois_theta"]), jnp.asarray(y), jnp.zeros(len(y))))
    res = C._laplace_summary(theta, hess, fisher, dec, mu, y, d["truth"], d["n_obs"], edges[-1])
    out = dict(cell=cell, shape=c["shape"], window=c["window"], n_obs_expected=c["n_obs_expected"],
               nll_improvement_quantiles=dict(zip(("q50", "q90", "max"), np.quantile(d["pois_nll"] - cur, [0.5, 0.9, 1.0]).tolist())),
               frac_converged_40=float(np.mean(d["pois_newton_dec"] < 1e-4)), frac_converged_340=float(np.mean(dec < 1e-4)),
               frac_boundary_40=float(d["pois_boundary"].mean()), frac_boundary_340=float(res["boundary"].mean()),
               frac_pd_40=float(d["pois_pd_obs"].mean()), frac_pd_340=float(res["pd_obs"].mean()), targets={})
    for j, t in enumerate(C.TARGETS):
        cov = lambda p: float(np.mean((p >= 0.05) & (p <= 0.95)))
        out["targets"][t] = dict(cov90_40=cov(d["pois_pit_obs"][:, j]), cov90_340=cov(res["pit_obs"][:, j]),
                                 width_40=float(np.nanmedian(3.29 * d["pois_se_obs"][:, j])), width_340=float(np.nanmedian(3.29 * res["se_obs"][:, j])))
    json.dump(out, (OUT / f"cell-{cell:03d}.json").open("w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
