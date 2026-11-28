"""E6-E8: comparison with the upstream R code (computationalmedia/sir-hawkes).

R lives outside the project (user-level micromamba env); paths are passed in.
Stages:
  export     pick tweet cascades from upstream Dataset/News and write one CSV each
  identity   E6: upstream objective vs my reading of it vs the textbook likelihood
  fit        E8: upstream fitSeries vs JAX fit_hawkesn vs exact grid, on the tweet cascades
  tutorial   E7b: JAX / grid fits on the cascades the R tutorial simulated, vs the R fits

  JAX_PLATFORMS=cpu uv run python experiments/wp12/upstream_compare.py <stage> \
      --upstream ~/wp12-ext/upstream/sir-hawkes --rscript ~/wp12-ext/renv35/bin/Rscript
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "WP-12" / "nov_upstream"
RDIR = Path(__file__).resolve().parent / "r_upstream"

from experiments.wp12 import exact_profile as ep  # noqa: E402


def upstream_nll_as_read(times, K, theta, N):
    """My reading of upstream neg.log.likelihood (history includes the seed(s) at time 0)."""
    t = np.asarray(times, dtype=np.float64)
    n = len(t)
    start = int(np.argmax(t > 0))
    s = 0.0
    for i in range(start, n):
        incl = t <= t[i]  # includes event i itself and ties
        Nt = min(incl.sum(), N)
        lam = K * (1 - Nt / N) * theta * np.exp(-theta * (t[i] - t[incl])).sum()
        s += math.log(lam if lam != 0 else np.finfo(float).eps)
    integ = 0.0
    for i in range(n):
        terms = np.exp(-theta * (t[i:] - t[i]))
        avail = N - np.arange(i + 1, n)  # R: N - i:(n-1), 1-based
        if (avail < 0).any():
            return float("nan")
        integ += ((avail / N) * (terms[:-1] - terms[1:])).sum()
    return K * integ - s


def textbook_ll(times, K, theta, N):
    """Textbook HawkesN log-lik, seeds = events at time 0, window ends at last event."""
    t = np.asarray(times, dtype=np.float64)
    I0 = int((t <= 0).sum())
    return ep.loglik_point(t[I0:], float(t[-1]), I0, K, theta, N)


def read_cascade(cid):
    with open(OUT / "cascades" / f"{cid}.csv") as fh:
        return np.array([float(r["time"]) for r in csv.DictReader(fh)])


def stage_export(upstream: Path, n_pick=30):
    (OUT / "cascades").mkdir(parents=True, exist_ok=True)
    idx = list(csv.reader(gzip.open(upstream / "Dataset/News/index.csv.gz", "rt")))[1:]
    data = gzip.open(upstream / "Dataset/News/data.csv.gz", "rt").read().splitlines()[1:]
    picked = []
    for k, row in enumerate(idx):
        s, e = int(row[0]), int(row[1])
        n = e - s + 1
        if 50 <= n <= 300:
            t = [float(x.split(",")[0]) for x in data[s - 1:e]]
            cid = f"news{k:05d}"
            with open(OUT / "cascades" / f"{cid}.csv", "w") as fh:
                fh.write("time\n" + "\n".join(repr(x) for x in t) + "\n")
            picked.append(dict(id=cid, index_row=k, n=n, T_seconds=t[-1], n_ties=n - len(set(t)),
                               n_at_time0=sum(1 for x in t if x <= 0)))
            if len(picked) == n_pick:
                break
    (OUT / "cascades_manifest.json").write_text(json.dumps(picked, indent=1))
    print("exported", len(picked), "cascades; sizes", [p["n"] for p in picked])


def run_R(rscript, upstream, script, *args):
    cmd = [str(rscript), str(RDIR / script), str(upstream), *map(str, args)]
    subprocess.run(cmd, check=True)


def stage_identity(upstream, rscript):
    man = json.loads((OUT / "cascades_manifest.json").read_text())[:5]
    rng = np.random.default_rng(20261120)
    pts = []
    for m in man:
        t = read_cascade(m["id"])
        med = float(np.median(np.diff(t)[np.diff(t) > 0]))
        for _ in range(3):
            pts.append(dict(id=m["id"], K=float(rng.uniform(0.5, 3)), theta=float(rng.uniform(0.05, 2) / med),
                            N=float(len(t) * rng.uniform(1.1, 5))))
    pfile = OUT / "identity_points.csv"
    with open(pfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "K", "theta", "N"])
        w.writeheader()
        w.writerows(pts)
    run_R(rscript, upstream, "eval_objective.R", OUT / "cascades", pfile, OUT / "identity_points_R.csv", "eval")
    rows = list(csv.DictReader(open(OUT / "identity_points_R.csv")))
    out = []
    for r in rows:
        t = read_cascade(r["id"])
        K, th, N = float(r["K"]), float(r["theta"]), float(r["N"])
        up_R = float(r["upstream_nll"])
        up_me = upstream_nll_as_read(t, K, th, N)
        tb = -textbook_ll(t, K, th, N)
        # predicted difference: sum_i log((1-i/N)(A_i+1)) - log((1-(i-1)/N) A_i), ties included upstream
        out.append(dict(id=r["id"], n=len(t), upstream_R=up_R, upstream_as_read=up_me, textbook_nll=tb,
                        rel_R_vs_read=abs(up_R - up_me) / abs(up_R), textbook_minus_upstream=tb - up_R))
    res = dict(max_rel_R_vs_as_read=max(o["rel_R_vs_read"] for o in out), rows=out, seed=20261120)
    (OUT / "E6_identity_results.json").write_text(json.dumps(res, indent=1))
    print("E6: max relative |R - as-read| = %.2e" % res["max_rel_R_vs_as_read"])
    print("    textbook nll - upstream nll (log-lik units):",
          [round(o["textbook_minus_upstream"], 2) for o in out])


def jax_fit(times_events, T, I0, **kw):
    from experiments.wp12.hawkesn import fit_hawkesn

    return fit_hawkesn(times_events, T=T, mu_fixed=0.0, I0=I0, **kw)


def fit_one_all(t):
    """JAX (in median-gap time units) + exact grid on one cascade with seeds at time 0.
    Returns estimates with theta in 1/original-time-unit."""
    I0 = int((t <= 0).sum())
    ev = t[I0:]
    T = float(t[-1])
    gaps = np.diff(t)
    unit = float(np.median(gaps[gaps > 0]))
    p, info = jax_fit(ev / unit, T / unit, I0)
    jx = dict(K=p.K, theta=p.theta / unit, N=p.N, success=bool(info["success"]))
    jx["textbook_ll"] = ep.loglik_point(ev, T, I0, jx["K"], jx["theta"], jx["N"])
    th_c = 1.0 / unit
    gf = ep.grid_fit(ev, T, I0, np.exp(np.linspace(math.log(th_c * 1e-5), math.log(th_c * 1e3), 700)))
    gd = dict(K=gf.mle["K"], theta=gf.mle["theta"], N=gf.mle["N"], textbook_ll=gf.ll_max,
              theta_at_edge=gf.mle["theta_at_edge"], N_at_edge=gf.mle["N_at_edge"])
    b20 = ep.box_max(gf, 20.0)
    return jx, gd, b20, unit


def stage_fit(upstream, rscript, ncores=4):
    man = json.loads((OUT / "cascades_manifest.json").read_text())
    rfile = OUT / "E8_upstream_fits_R.csv"
    if not rfile.exists():
        run_R(rscript, upstream, "eval_objective.R", OUT / "cascades", "NONE", rfile, "fit", ncores)
    R = {r["id"]: r for r in csv.DictReader(open(rfile))}
    rows, pts = [], []
    for m in man:
        t = read_cascade(m["id"])
        jx, gd, b20, unit = fit_one_all(t)
        r = R[m["id"]]
        fl = lambda x: float("nan") if x in ("NA", "") else float(x)
        up = dict(K=fl(r["K"]), theta=fl(r["theta"]), N=fl(r["N"]), nll_upstream=fl(r["nll"]),
                  iter=r["iter"], conv=r["conv"], secs=fl(r["secs"]))
        up["textbook_ll"] = textbook_ll(t, up["K"], up["theta"], up["N"]) if np.isfinite(up["K"]) else float("nan")
        rows.append(dict(id=m["id"], n=m["n"], unit_seconds=unit, upstream=up, jax=jx, grid=gd, box20=b20))
        pts.append(dict(id=m["id"], K=jx["K"], theta=jx["theta"], N=jx["N"]))
        print(m["id"], "n=%d" % m["n"], "| R: K=%.3g th=%.3g N=%.4g" % (up["K"], up["theta"], up["N"]),
              "| JAX: K=%.3g th=%.3g N=%.4g" % (jx["K"], jx["theta"], jx["N"]),
              "| grid: K=%.3g th=%.3g N=%.4g" % (gd["K"], gd["theta"], gd["N"]),
              "| textbook ll  R %.2f  JAX %.2f  grid %.2f" % (up["textbook_ll"], jx["textbook_ll"], gd["textbook_ll"]),
              flush=True)
    pfile = OUT / "E8_points_jax.csv"
    with open(pfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "K", "theta", "N"])
        w.writeheader()
        w.writerows(pts)
    run_R(rscript, upstream, "eval_objective.R", OUT / "cascades", pfile, OUT / "E8_points_jax_R.csv", "eval")
    for row, r in zip(rows, csv.DictReader(open(OUT / "E8_points_jax_R.csv"))):
        assert row["id"] == r["id"]
        row["jax"]["nll_upstream"] = float(r["upstream_nll"]) if r["upstream_nll"] not in ("NA", "") else float("nan")
    (OUT / "E8_fit_comparison.json").write_text(json.dumps(rows, indent=1, default=float))


def stage_fit2(upstream, rscript, ncores=4):
    """E8b: upstream fitter from two workable start points (generic scaled; JAX solution)."""
    rows = json.loads((OUT / "E8_fit_comparison.json").read_text())
    pfile = OUT / "E8b_start_points.csv"
    with open(pfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "K", "theta", "N", "label"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(id=r["id"], K=1.0, theta=1.0 / r["unit_seconds"], N=2.0 * r["n"], label="s1_generic"))
            w.writerow(dict(id=r["id"], K=r["jax"]["K"], theta=r["jax"]["theta"], N=r["jax"]["N"], label="s2_jax_solution"))
    rfile = OUT / "E8b_upstream_fits_R.csv"
    if not rfile.exists():
        run_R(rscript, upstream, "eval_objective.R", OUT / "cascades", pfile, rfile, "fit", ncores)
    fl = lambda x: float("nan") if x in ("NA", "") else float(x)
    R = {}
    for r in csv.DictReader(open(rfile)):
        R[(r["id"], r["label"])] = dict(K=fl(r["K"]), theta=fl(r["theta"]), N=fl(r["N"]), nll_upstream=fl(r["nll"]))
    out = []
    for r in rows:
        t = read_cascade(r["id"])
        d = dict(id=r["id"], n=r["n"], jax=r["jax"], grid=r["grid"])
        for lab in ["s1_generic", "s2_jax_solution"]:
            u = R[(r["id"], lab)]
            ok = np.isfinite(u["K"]) and u["N"] >= len(t)
            u["textbook_ll"] = textbook_ll(t, u["K"], u["theta"], u["N"]) if ok else float("nan")
            d[lab] = u
        out.append(d)
    (OUT / "E8b_fit_comparison.json").write_text(json.dumps(out, indent=1, default=float))
    summ = {}
    for lab in ["s1_generic", "s2_jax_solution"]:
        fin = [d for d in out if np.isfinite(d[lab]["K"])]
        def within(q, f):
            return float(np.mean([(d[lab][q] / d["jax"][q] < f) and (d[lab][q] / d["jax"][q] > 1 / f) for d in fin])) if fin else float("nan")
        summ[lab] = dict(n_finite=len(fin), n=len(out),
                         all_three_within_10pct=float(np.mean([all(abs(d[lab][q] / d["jax"][q] - 1) < 0.1 for q in ("K", "theta", "N")) for d in fin])) if fin else float("nan"),
                         K_within_1p5=within("K", 1.5), theta_within_1p5=within("theta", 1.5), N_within_1p5=within("N", 1.5),
                         theta_ratio_median=float(np.median([d[lab]["theta"] / d["jax"]["theta"] for d in fin])) if fin else float("nan"),
                         frac_textbook_worse_than_jax_by_0p1=float(np.mean([(not np.isfinite(d[lab]["textbook_ll"])) or d[lab]["textbook_ll"] < d["jax"]["textbook_ll"] - 0.1 for d in fin])) if fin else float("nan"),
                         median_textbook_loss_vs_jax=float(np.nanmedian([d["jax"]["textbook_ll"] - d[lab]["textbook_ll"] for d in fin])) if fin else float("nan"))
    summ["jax_vs_grid"] = dict(frac_jax_within_0p1_of_grid=float(np.mean([d["grid"]["textbook_ll"] - d["jax"]["textbook_ll"] < 0.1 for d in out])),
                               frac_jax_more_than_1p92_below=float(np.mean([d["grid"]["textbook_ll"] - d["jax"]["textbook_ll"] > 1.92 for d in out])),
                               max_gap=float(max(d["grid"]["textbook_ll"] - d["jax"]["textbook_ll"] for d in out)),
                               frac_grid_N_equals_observed_count=float(np.mean([abs(d["grid"]["N"] - d["n"]) < 0.5 for d in out])),
                               frac_grid_K_gt_20=float(np.mean([d["grid"]["K"] > 20 for d in out])),
                               frac_jax_K_at_bound=float(np.mean([d["jax"]["K"] > 19.9 for d in out])))
    (OUT / "E8b_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(21, 5))
    for a, q, lab in zip(ax[:3], ["K", "theta", "N"], ["K (offspring per tweet)", "theta (1/second)", "N (population size)"]):
        j = np.array([d["jax"][q] for d in out])
        for key, col, mk, name in [("s1_generic", "#f47230", "s", "upstream from a generic start"),
                                   ("s2_jax_solution", "#257de0", "o", "upstream started AT the JAX solution")]:
            u = np.array([d[key][q] for d in out])
            a.scatter(j, u, s=28, color=col, marker=mk, alpha=0.8, label=name)
        lo, hi = np.nanmin(j) * 0.3, np.nanmax(j) * 3
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("this project's JAX fitter: " + lab)
        a.set_ylabel("upstream R fitter: " + lab)
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=8)
    gap = np.array([d["grid"]["textbook_ll"] - d["jax"]["textbook_ll"] for d in out])
    ax[3].bar(np.arange(len(out)), np.clip(gap, 1e-3, None), color="#6a4ce2")
    ax[3].set_yscale("log")
    ax[3].axhline(1.92, color="k", ls="--", lw=1)
    ax[3].set_xlabel("tweet cascade (30 from the upstream News set)")
    ax[3].set_ylabel("JAX log-likelihood below exact maximum\n(floor 0.001; dashed = 1.92)")
    ax[3].set_title("JAX fitter vs exact maximum of its own likelihood:\nwithin 0.1 on %.0f%% of cascades" % (
        100 * summ["jax_vs_grid"]["frac_jax_within_0p1_of_grid"]), fontsize=10)
    ax[3].grid(alpha=0.25)
    fig.suptitle("30 real tweet cascades. The upstream R fitter gave NO usable fit: from the tutorial's start it errored on 29/30; from a generic start it "
                 "errored on 17 and returned the start unchanged on 13;\nstarted at the JAX solution it returned that point unchanged on 28 of 29 "
                 "(so the blue dots sit on the dashed line by construction, not by agreement). Right: JAX fitter vs the exact maximum.", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "E8b_fit_comparison.png", dpi=120)
    plt.close(fig)


def stage_tutorial():
    """E7b: our fits on the R-simulated cascades (sim_history_XX.csv), against fit_hawkesn.csv from R."""
    tdir = OUT / "tutorial_run_R35"
    R = list(csv.DictReader(open(tdir / "fit_hawkesn.csv")))
    rows = []
    for i in range(1, 21):
        t = np.array([float(r["time"]) for r in csv.DictReader(open(tdir / f"sim_history_{i:02d}.csv"))])
        jx, gd, b20, unit = fit_one_all(t)
        r = R[i - 1]
        fl = lambda x: float("nan") if x in ("NA", "") else float(x)
        up = dict(K=fl(r["K"]), theta=fl(r["theta"]), N=fl(r["N"]))
        if np.isfinite(up["K"]):
            up["textbook_ll"] = textbook_ll(t, up["K"], up["theta"], up["N"])
            # E6 follow-up prediction: upstream's own objective keeps falling as theta grows
            up["nll_upstream_at_fit"] = upstream_nll_as_read(t, up["K"], up["theta"], up["N"])
        else:  # upstream fit failed and returned NA
            up["textbook_ll"] = float("nan")
            up["nll_upstream_at_fit"] = float("nan")
        obj = {lab: upstream_nll_as_read(t, 5.0, th, 1300.0) for lab, th in
               [("truth_theta0.2", 0.2), ("theta1e3", 1e3), ("theta1e6", 1e6)]}
        rows.append(dict(i=i, n_total=len(t), upstream=up, jax=jx, grid=gd, box20=b20, upstream_objective=obj))
        print(i, "upstream objective at truth / theta=1e3 / theta=1e6 / its own fit: %.1f / %.1f / %.1f / %.1f" % (
            obj["truth_theta0.2"], obj["theta1e3"], obj["theta1e6"], up["nll_upstream_at_fit"]))
        print(i, "R: K=%.3g th=%.3g N=%.5g ll=%.2f" % (up["K"], up["theta"], up["N"], up["textbook_ll"]),
              "| JAX: K=%.3g th=%.3g N=%.5g ll=%.2f" % (jx["K"], jx["theta"], jx["N"], jx["textbook_ll"]),
              "| grid: K=%.3g th=%.3g N=%.5g ll=%.2f" % (gd["K"], gd["theta"], gd["N"], gd["textbook_ll"]), flush=True)
    (OUT / "E7b_tutorial_cascades_fits.json").write_text(json.dumps(rows, indent=1, default=float))


def stage_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C1, C2, C3 = "#257de0", "#f47230", "#00b894"
    summ = {}
    f7 = OUT / "E7b_tutorial_cascades_fits.json"
    if f7.exists():
        rows = json.loads(f7.read_text())
        tr = dict(K=5.0, theta=0.2, N=1300.0)
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
        x = np.arange(1, len(rows) + 1)
        for a, q in zip(ax[:3], ["K", "theta", "N"]):
            a.plot(x - 0.15, [r["upstream"][q] for r in rows], "s", color=C2, ms=7, label="upstream R fitter")
            a.plot(x + 0.15, [r["jax"][q] for r in rows], "o", color=C1, ms=6, label="this project's JAX fitter")
            a.plot(x, [r["grid"][q] for r in rows], "+", color="#191919", ms=9, label="exact maximum of the textbook likelihood")
            a.axhline(tr[q], color="k", ls="--", lw=1)
            a.text(0.6, tr[q], " true", va="bottom", fontsize=8)
            if q != "N":
                a.set_yscale("log")
            a.set_xlabel("simulated cascade (1-20) from the R tutorial")
            a.set_ylabel({"K": "K (offspring per post)", "theta": "theta (decay rate, 1/time)", "N": "N (population size)"}[q])
            a.grid(alpha=0.25)
        ax[0].legend(fontsize=7.5)
        d_up = np.array([r["grid"]["textbook_ll"] - r["upstream"]["textbook_ll"] for r in rows])
        d_jx = np.array([r["grid"]["textbook_ll"] - r["jax"]["textbook_ll"] for r in rows])
        ax[3].bar(x - 0.2, np.clip(d_up, 1e-3, None), width=0.4, color=C2, label="upstream R solution")
        ax[3].bar(x + 0.2, np.clip(d_jx, 1e-3, None), width=0.4, color=C1, label="JAX solution")
        ax[3].set_yscale("log")
        ax[3].axhline(1.92, color="k", ls="--", lw=1)
        ax[3].set_xlabel("simulated cascade (1-20)")
        ax[3].set_ylabel("log-likelihood below the exact maximum\n(floor 0.001; dashed = 1.92)")
        ax[3].legend(fontsize=7.5)
        ax[3].grid(alpha=0.25)
        med = {k: {q: float(np.nanmedian([r[k][q] for r in rows])) for q in tr} for k in ["upstream", "jax", "grid"]}
        sd = {k: {q: float(np.nanstd([r[k][q] for r in rows], ddof=1)) for q in tr} for k in ["upstream", "jax", "grid"]}
        summ["E7b"] = dict(n_upstream_NA=int(sum(not np.isfinite(r["upstream"]["K"]) for r in rows)), median=med, sd=sd, upstream_ll_deficit=d_up.tolist(), jax_ll_deficit=d_jx.tolist())
        fig.suptitle("The R tutorial's 20 simulated cascades (true K=5, theta=0.2, N=1300), fitted three ways. "
                     "Median K: upstream %.2f, JAX %.2f; median theta: upstream %.3f, JAX %.3f" % (
                         med["upstream"]["K"], med["jax"]["K"], med["upstream"]["theta"], med["jax"]["theta"]), fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "E7b_tutorial_cascades_fits.png", dpi=120)
        plt.close(fig)
    f8 = OUT / "E8_fit_comparison.json"
    if False and f8.exists():  # upstream gave NA on 29/30 from the tutorial start; stage fit2 (E8b) makes the plot
        rows = json.loads(f8.read_text())
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        def gr(d):
            return d["theta"] * (d["K"] - 1)
        specs = [("K", lambda d: d["K"], "K (offspring per tweet)"), ("theta", lambda d: d["theta"], "theta (1/second)"),
                 ("N", lambda d: d["N"], "N (population size)")]
        agree = {}
        for a, (q, f, lab) in zip(ax[:3], specs):
            u = np.array([f(r["upstream"]) for r in rows]); j = np.array([f(r["jax"]) for r in rows])
            a.scatter(u, j, s=30, color=C1)
            lo, hi = min(u.min(), j.min()) * 0.5, max(u.max(), j.max()) * 2
            a.plot([lo, hi], [lo, hi], "k--", lw=1)
            a.set_xscale("log"); a.set_yscale("log")
            a.set_xlabel("upstream R fitter: " + lab); a.set_ylabel("this project's JAX fitter: " + lab)
            ratio = j / u
            agree[q] = dict(within_10pct=float(np.mean(np.abs(ratio - 1) < 0.1)),
                            within_factor_1p5=float(np.mean((ratio < 1.5) & (ratio > 1 / 1.5))))
            a.set_title("%s: within 10%% in %.0f%% of cascades,\nwithin a factor 1.5 in %.0f%%" % (
                q, 100 * agree[q]["within_10pct"], 100 * agree[q]["within_factor_1p5"]), fontsize=10)
            a.grid(alpha=0.25)
        tb_up = np.array([r["upstream"]["textbook_ll"] for r in rows]); tb_jx = np.array([r["jax"]["textbook_ll"] for r in rows])
        tb_gd = np.array([r["grid"]["textbook_ll"] for r in rows])
        up_up = np.array([r["upstream"]["nll_upstream"] for r in rows]); up_jx = np.array([r["jax"].get("nll_upstream", np.nan) for r in rows])
        x = np.arange(len(rows))
        d = tb_jx - tb_up
        d = np.where(np.isfinite(d), d, 1e4)
        ax[3].bar(x, np.sign(d) * np.log10(1 + np.abs(d)), color=C1)
        ax[3].set_xlabel("tweet cascade (30 from the upstream News set)")
        ax[3].set_ylabel("signed log10(1 + |difference|) of textbook log-likelihood:\nJAX solution minus upstream solution (positive = JAX better)")
        ax[3].grid(alpha=0.25)
        summ["E8"] = dict(agreement=agree,
                          frac_jax_better_textbook=float(np.mean(tb_jx > tb_up + 0.1)),
                          frac_upstream_better_textbook=float(np.mean(tb_up > tb_jx + 0.1)),
                          frac_jax_within_0p1_of_grid=float(np.mean(tb_gd - tb_jx < 0.1)),
                          frac_upstream_better_on_upstream_objective=float(np.nanmean(up_up < up_jx - 0.1)),
                          median_textbook_ll_gain_jax_over_upstream=float(np.median(d)),
                          n=len(rows))
        ax[3].set_title("Scored by the textbook likelihood, JAX is better in %.0f%%,\nupstream better in %.0f%%" % (
            100 * summ["E8"]["frac_jax_better_textbook"], 100 * summ["E8"]["frac_upstream_better_textbook"]), fontsize=10)
        fig.suptitle("Same 30 tweet cascades, upstream R fitter (tutorial start point) vs this project's JAX fitter: dots on the dashed line = same answer",
                     fontsize=11.5)
        fig.tight_layout()
        fig.savefig(OUT / "E8_fit_comparison.png", dpi=120)
        plt.close(fig)
    (OUT / "E7b_E8_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)[:6000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--upstream", type=Path, default=Path.home() / "wp12-ext/upstream/sir-hawkes")
    ap.add_argument("--rscript", type=Path, default=Path.home() / "wp12-ext/renv35/bin/Rscript")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "export":
        stage_export(a.upstream)
    elif a.stage == "identity":
        stage_identity(a.upstream, a.rscript)
    elif a.stage == "fit":
        stage_fit(a.upstream, a.rscript)
    elif a.stage == "fit2":
        stage_fit2(a.upstream, a.rscript)
    elif a.stage == "tutorial":
        stage_tutorial()
    elif a.stage == "plots":
        stage_plots()
