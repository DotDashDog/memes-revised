"""E2 (number of bins) and E3 (overdispersion / repost spikes) calibration runs.

Usage: uv run python experiments/wp02/run_e2_e3.py --experiment e2 --shard 0 --n-shards 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.wp02 import calib as C
from experiments.wp02.run_e1_window_counts import SHAPES, shape_info


def _base(shape, win):
    t_peak, t_full, frac = shape_info(shape)
    t_end = t_full if win == "full" else int(round(float(win.split("x")[0]) * t_peak))
    return t_peak, t_end, float(frac[t_end])


def cells_e2():
    """Reduced design (see expectations file): native daily-bin reference comes from E1."""
    cells = []
    combos = [("A", w, n) for w in ("1.0xpeak", "full") for n in (100, 1000)] + [("C", "full", 1000)]
    for shape, win, n in combos:
        t_peak, t_end, frac = _base(shape, win)
        for n_bins in (4, 8, 16):
            cells.append(dict(shape=shape, window=win, t_end=t_end, n_bins=n_bins, t_peak=t_peak,
                              frac_observed=frac, n_obs_expected=n, total_posts=n / frac,
                              generator="poisson", gen_kwargs={}, fit_nb=False, **SHAPES[shape]))
    return cells


def cells_e3():
    """Reduced design (see expectations file)."""
    cells = []
    all_gens = [("poisson", {}), ("nb", {"nb_k": 5.0}), ("nb", {"nb_k": 1.0}), ("spikes", {"spike_prob": 0.05, "spike_size": "peak"})]
    combos = [("A", "full", 100, all_gens), ("A", "full", 1000, all_gens), ("C", "full", 1000, [all_gens[1], all_gens[3]])]
    for shape, win, n, gens in combos:
        t_peak, t_end, frac = _base(shape, win)
        for gen, kw in gens:
            cells.append(dict(shape=shape, window=win, t_end=t_end, n_bins=t_end, t_peak=t_peak,
                              frac_observed=frac, n_obs_expected=n, total_posts=n / frac,
                              generator=gen, gen_kwargs=kw, fit_nb=True, **SHAPES[shape]))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["e2", "e3"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-rep", type=int, default=300)
    ap.add_argument("--n-cloud", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--only-cells", default="", help="comma-separated cell ids; empty = all (added after 2nd machine restart)")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = {"e2": 20261128, "e3": 20261129}[args.experiment]
    if args.output_dir is None:
        args.output_dir = Path("experiments/WP-02") / {"e2": "e2-bins", "e3": "e3-noise-spikes"}[args.experiment]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = cells_e2() if args.experiment == "e2" else cells_e3()
    for i, c in enumerate(cells):
        c["cell"] = i
    if args.shard == 0 and not (args.output_dir / "cells.json").exists():
        with (args.output_dir / "cells.json").open("w") as fh:
            json.dump(dict(config=vars(args) | {"output_dir": str(args.output_dir)}, cells=cells), fh, indent=1)
    only = {int(x) for x in args.only_cells.split(",") if x}
    run = [c for c in cells if not only or c["cell"] in only]
    for k, c in enumerate(run):
        if k % args.n_shards != args.shard:
            continue
        path = args.output_dir / f"cell-{c['cell']:03d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        edges = np.linspace(0.0, c["t_end"], c["n_bins"] + 1)
        seed = int(np.random.SeedSequence([args.seed, c["cell"]]).generate_state(1)[0])
        kw = dict(c["gen_kwargs"])
        if kw.get("spike_size") == "peak":
            # burst as large as the expected count of the tallest bin
            scale = C.scale_for_total(c["beta"], c["gamma"], c["i0"], c["total_posts"])
            th = np.array([np.log(c["beta"]), np.log(c["gamma"]), np.log(scale), np.log(c["i0"] / (1 - c["i0"]))])
            kw["spike_size"] = float(C.expected_fisher(th, edges)[1].max())
        out = C.run_cell(beta=c["beta"], gamma=c["gamma"], i0=c["i0"], total_posts=c["total_posts"],
                         bin_edges=edges, n_rep=args.n_rep, seed=seed, n_cloud=args.n_cloud,
                         generator=c["generator"], gen_kwargs=kw, fit_nb=c["fit_nb"], proposal="v2")
        tmp = path.with_name(path.stem + '.partial.npz'); np.savez_compressed(tmp, seed=seed, **out); tmp.replace(path)
        print(f"cell {c['cell']} {c['shape']} {c['window']} n={c['n_obs_expected']} bins={c['n_bins']} "
              f"{c['generator']} {c['gen_kwargs']} ess_med={np.nanmedian(out['ess']):.0f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
