"""E1: calibration of SIR targets vs observed post count and observation window.

Usage: uv run python experiments/wp02/run_e1_window_counts.py --shard 0 --n-shards 4
Writes one npz per cell plus cells.json into the output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp

from experiments.wp02 import calib as C
from experiments.wp02.sir import solve_sir

SHAPES = {
    "A": dict(beta=0.6, gamma=0.2, i0=0.003),
    "B": dict(beta=1.0, gamma=0.67, i0=0.003),
    "C": dict(beta=0.5, gamma=0.05, i0=0.003),
    "D": dict(beta=0.3, gamma=0.35, i0=0.003),
    "E": dict(beta=1.0, gamma=1.0, i0=0.003),
}
WINDOWS = ("0.7xpeak", "1.0xpeak", "1.5xpeak", "full")
COUNTS = (10, 30, 100, 300, 1000, 3000)


def shape_info(shape):
    p = SHAPES[shape]
    th = jnp.array([np.log(p["beta"]), np.log(p["gamma"]), 0.0, np.log(p["i0"] / (1 - p["i0"]))])
    ts = np.arange(0, 2001.0)
    cum = np.asarray(solve_sir(th, jnp.asarray(ts)))[:, 3]
    d = C.derived(np.log(p["beta"]), np.log(p["gamma"]), np.log(p["i0"] / (1 - p["i0"])), 0.0)
    area = float(np.exp(d[C.TARGETS.index("log_total")]))
    frac = cum / area
    t_peak = float(d[C.TARGETS.index("peak_time")])
    t_full = int(ts[np.argmax(frac >= 0.99)])
    return t_peak, t_full, frac


def build_cells():
    cells = []
    for shape in SHAPES:
        t_peak, t_full, frac = shape_info(shape)
        windows = WINDOWS if shape in "ABC" else ("full",)
        for win in windows:
            if win == "full":
                t_end = t_full if shape in "ABC" else 60
            else:
                t_end = int(round(float(win.split("x")[0]) * t_peak))
            for n in COUNTS:
                cells.append(
                    dict(
                        shape=shape, window=win, t_end=t_end, n_bins=t_end, t_peak=t_peak,
                        frac_observed=float(frac[t_end]), n_obs_expected=n,
                        total_posts=n / float(frac[t_end]), **SHAPES[shape],
                    )
                )
    for i, c in enumerate(cells):
        c["cell"] = i
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-rep", type=int, default=400)
    ap.add_argument("--n-cloud", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--proposal", default="v1")
    ap.add_argument("--only-cells", default="", help="comma-separated cell ids; empty = all")
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/WP-02/e1-window-counts"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    only = {int(x) for x in args.only_cells.split(",") if x}
    if only:
        cells_run = [c for c in cells if c["cell"] in only]
        for k, c in enumerate(cells_run):
            c["_slot"] = k
    else:
        cells_run = cells
        for c in cells_run:
            c["_slot"] = c["cell"]
    cfg_name = "cells.json" if not (args.output_dir / "cells.json").exists() else f"cells-{args.proposal}-{args.n_cloud}.json"
    if args.shard == 0:
        with (args.output_dir / cfg_name).open("w") as fh:
            json.dump(dict(config=vars(args) | {"output_dir": str(args.output_dir)}, cells=cells), fh, indent=1)
    cache = {}
    for c in cells_run:
        if c["_slot"] % args.n_shards != args.shard:
            continue
        path = args.output_dir / f"cell-{c['cell']:03d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        edges = np.arange(0.0, c["t_end"] + 1.0)
        if c["t_end"] not in cache:
            cache[c["t_end"]] = (C.make_forward(edges), C.make_fitter(edges))
        fwd, fit = cache[c["t_end"]]
        seed = int(np.random.SeedSequence([args.seed, c["cell"]]).generate_state(1)[0])
        out = C.run_cell(
            beta=c["beta"], gamma=c["gamma"], i0=c["i0"], total_posts=c["total_posts"],
            bin_edges=edges, n_rep=args.n_rep, seed=seed, n_cloud=args.n_cloud, forward=fwd, fitter=fit, proposal=args.proposal,
        )
        tmp = path.with_name(path.stem + '.partial.npz'); np.savez_compressed(tmp, seed=seed, proposal=args.proposal, n_cloud=args.n_cloud, **out); tmp.replace(path)
        print(f"cell {c['cell']} {c['shape']} {c['window']} n={c['n_obs_expected']} t_end={c['t_end']} "
              f"frac={c['frac_observed']:.3f} ess_med={np.nanmedian(out['ess']):.0f} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
