"""Collapse per-cell npz files of a calibration run into summary.csv (one row per cell x target x method)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from experiments.wp02.calib import TARGETS

Z90 = 1.6448536269514722


def _cov(pit, lo, hi):
    return np.mean((pit >= lo) & (pit <= hi))  # NaN pit (no interval produced) counts as a miss


def summarise(run_dir: Path) -> pd.DataFrame:
    meta = json.load((run_dir / "cells.json").open())
    rows = []
    for c in meta["cells"]:
        path = run_dir / f"cell-{c['cell']:03d}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        has = d["n_obs"] >= 1
        n = int(has.sum())
        base = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in c.items()}
        base.update(n_rep=len(has), n_with_data=n, frac_zero_count=1 - has.mean(),
                    frac_ess_lt100=float(np.mean(d["ess"][has] < 100)), ess_median=float(np.median(d["ess"][has])))
        methods = {"posterior": None, "laplace": "pois"}
        if "nb_est" in d:
            methods["nb_laplace"] = "nb"
            methods["quasi_poisson"] = "pois"
        for j, t in enumerate(TARGETS):
            truth = d["truth"][j]
            for method, tag in methods.items():
                row = dict(base, target=t, method=method, truth=truth)
                if method == "posterior":
                    pit = d["post_pit"][has, j]
                    est = d["post_med"][has, j]
                    width = d["post_q95"][has, j] - d["post_q05"][has, j]
                    valid = np.ones(n, bool)
                else:
                    kind = "exp" if method == "nb_laplace" else "obs"
                    est = d[f"{tag}_est"][has, j]
                    se = d[f"{tag}_se_{kind}"][has, j].copy()
                    if method == "quasi_poisson":
                        se = se * np.sqrt(np.maximum(d["pois_pearson_phi"][has], 1.0))
                    valid = np.isfinite(se) & (se > 0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        z = (truth - est) / se
                    from scipy.special import ndtr
                    pit = np.where(valid, ndtr(z), np.nan)
                    width = 2 * Z90 * se
                    row.update(frac_pd=float(d[f"{tag}_pd_{kind}"][has].mean()),
                               frac_boundary=float(d[f"{tag}_boundary"][has].mean()),
                               frac_converged=float(np.mean(d[f"{tag}_newton_dec"][has] < 1e-4)))
                row.update(
                    cov90=_cov(pit, 0.05, 0.95), cov50=_cov(pit, 0.25, 0.75),
                    cov90_valid=_cov(pit[valid], 0.05, 0.95) if valid.any() else np.nan,
                    frac_valid=float(valid.mean()),
                    width90_median=float(np.nanmedian(width[valid])) if valid.any() else np.nan,
                    width90_p90=float(np.nanquantile(width[valid], 0.9)) if valid.any() else np.nan,
                    abs_err_median=float(np.median(np.abs(est - truth))),
                    abs_err_p90=float(np.quantile(np.abs(est - truth), 0.9)),
                    bias_median=float(np.median(est - truth)),
                )
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "summary.csv", index=False)
    return df


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg, len(summarise(Path(arg))))
