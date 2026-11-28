"""Compare v3 (adaptive) posterior against v2 for E1 cells. Usage: uv run python experiments/wp02/compare_v3.py CELL ..."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from experiments.wp02.calib import TARGETS

B = Path("experiments/WP-02")
res = {}
for cell in map(int, sys.argv[1:]):
    f2 = B / "e1-rerun-v2" / f"cell-{cell:03d}.npz"
    if not f2.exists():
        f2 = B / "e1-window-counts" / f"cell-{cell:03d}.npz"
    a, b = np.load(f2), np.load(B / "e1-rerun-v3" / f"cell-{cell:03d}.npz")
    assert np.array_equal(a["n_obs"], b["n_obs"])
    r = {"ess_v2_median": float(np.median(a["ess"])), "ess_v3_median": float(np.median(b["ess"])),
         "frac_ess_lt200_v2": float(np.mean(a["ess"] < 200)), "frac_ess_lt200_v3": float(np.mean(b["ess"] < 200)), "targets": {}}
    for j, t in enumerate(TARGETS):
        w2 = a["post_q95"][:, j] - a["post_q05"][:, j]
        shift = {q: float(np.median(np.abs(b[q][:, j] - a[q][:, j]) / np.maximum(w2, 1e-12))) for q in ("post_q05", "post_med", "post_q95")}
        cov = lambda d: float(np.mean((d["post_pit"][:, j] >= 0.05) & (d["post_pit"][:, j] <= 0.95)))
        r["targets"][t] = dict(cov_v2=cov(a), cov_v3=cov(b), width_v2=float(np.median(w2)),
                               width_v3=float(np.median(b["post_q95"][:, j] - b["post_q05"][:, j])),
                               median_abs_quantile_shift_over_v2_width=shift)
    res[cell] = r
    print(cell, json.dumps({k: v for k, v in r.items() if k != "targets"}))
    for t, v in r["targets"].items():
        print(f"   {t:14s} cov {v['cov_v2']:.3f}->{v['cov_v3']:.3f} width {v['width_v2']:.3f}->{v['width_v3']:.3f} shift q05/med/q95 "
              + "/".join(f"{x:.3f}" for x in v["median_abs_quantile_shift_over_v2_width"].values()))
out = B / "e1-rerun-v3" / "compare_v2_v3.json"
old = json.load(out.open()) if out.exists() else {}
old.update({str(k): v for k, v in res.items()})
json.dump(old, out.open("w"), indent=1)
