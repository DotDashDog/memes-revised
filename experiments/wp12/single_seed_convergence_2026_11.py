"""E11: does the HawkesN simulator converge to the mean-field (SIR) curve when started from ONE post?

Three separate claims (see experiments/WP-12/expectations-2026-11.md, E11):
  (a) take-off probability -> 1 - q, q = exp(-K(1-q))
  (b) shape after time-alignment, conditional on take-off: sup deviation ~ N^(-1/2)
  (c) timing (the alignment shift) keeps an O(1/r) spread for every N

  uv run python experiments/wp12/single_seed_convergence_2026_11.py [n_runs]
Outputs: experiments/WP-12/nov_single_seed_{results.json, shape.png, scaling.png}
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "WP-12"

from experiments.wp12 import exact_profile as ep  # noqa: E402

THETA = 0.4
KS = [1.5, 3.0]
NS = [250, 1000, 4000, 16000, 64000]
LEVEL = 0.10  # alignment level, fraction of N
BASE_SEED = 20261140
COLS = ["#257de0", "#f47230", "#00b894", "#6a4ce2", "#ec4647"]


def mean_field(K, theta, N):
    """Cumulative fraction C(t) = 1 - s(t) of the HawkesN mean-field started from one seed.
    ds/dt = -s a ; da/dt = theta a (K s - 1) ; a = K theta * (kernel sum)/N."""
    r = theta * (K - 1)
    T = (math.log(N) + 40) / r + 60 / theta
    sol = solve_ivp(lambda t, y: [-y[0] * y[1], theta * y[1] * (K * y[0] - 1)], [0, T],
                    [1 - 1 / N, K * theta / N], rtol=1e-10, atol=1e-14, dense_output=True, method="LSODA")
    C_inf = 1 - sol.y[0, -1]
    t_level = brentq(lambda t: 1 - sol.sol(t)[0] - LEVEL, 0, T)
    t_end = brentq(lambda t: 1 - sol.sol(t)[0] - (LEVEL + 0.999 * (C_inf - LEVEL)), t_level, T)
    return sol, C_inf, t_level, t_end


def poisson_extinction(K):
    q = 0.0
    for _ in range(2000):
        q = math.exp(-K * (1 - q))
    return q


def run_cell(K, N, n_runs, seed_seq):
    rng = np.random.default_rng(seed_seq)
    sol, C_inf, t_level, t_end = mean_field(K, THETA, N)
    s_grid = np.linspace(0, t_end - t_level, 300)
    C_mf = 1 - sol.sol(t_level + s_grid)[0]
    n_level = int(math.ceil(LEVEL * N)) - 1  # index into events of the post that brings the count to LEVEL*N (seed counts 1)
    finals, taus, sups, curves = [], [], [], []
    # un-aligned: evaluate on absolute time grid
    t_abs = np.linspace(0, t_end * 1.5, 400)
    C_mf_abs = 1 - sol.sol(t_abs)[0]
    abs_sum = np.zeros_like(t_abs)
    for _ in range(n_runs):
        ev = ep.simulate_hawkesn_inversion(K, THETA, float(N), 1, 1e12, rng)
        tot = len(ev) + 1
        finals.append(tot)
        if tot <= 0.5 * C_inf * N:
            continue
        tau = float(ev[n_level - 1])  # time at which cumulative count (incl. seed) reaches ceil(LEVEL*N)
        taus.append(tau)
        C_run = (1 + np.searchsorted(ev, tau + s_grid, side="right")) / N
        sups.append(float(np.abs(C_run - C_mf).max()))
        curves.append(C_run)
        abs_sum += (1 + np.searchsorted(ev, t_abs, side="right")) / N
    finals = np.array(finals)
    took = finals > 0.5 * C_inf * N
    curves = np.array(curves)
    mean_aligned = curves.mean(axis=0)
    mean_unaligned = abs_sum / max(took.sum(), 1)
    return dict(
        K=K, N=N, n_runs=n_runs, n_takeoff=int(took.sum()), p_takeoff=float(took.mean()),
        mf_final_fraction=float(C_inf), mf_t_level=float(t_level),
        final_fraction_takeoff_mean=float(finals[took].mean() / N),
        tau=np.array(taus).tolist(), sup_per_run=sups,
        sup_aligned_mean=float(np.abs(mean_aligned - C_mf).max()),
        sup_unaligned_mean=float(np.abs(mean_unaligned - C_mf_abs).max()),
        _plot=dict(s_grid=s_grid, C_mf=C_mf, example_runs=curves[:25], mean_aligned=mean_aligned,
                   t_abs=t_abs, C_mf_abs=C_mf_abs, mean_unaligned=mean_unaligned),
    )


def slope_ci(Ns, per_run_lists, rng, n_boot=2000):
    x = np.log(np.array(Ns, dtype=float))
    med = np.array([np.median(v) for v in per_run_lists])
    sl = np.polyfit(x, np.log(med), 1)[0]
    boots = []
    for _ in range(n_boot):
        m = [np.median(rng.choice(v, size=len(v), replace=True)) for v in per_run_lists]
        boots.append(np.polyfit(x, np.log(m), 1)[0])
    return float(sl), [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def main(n_runs=400):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = {}
    for ki, K in enumerate(KS):
        for ni, N in enumerate(NS):
            c = run_cell(K, N, n_runs, np.random.SeedSequence([BASE_SEED, ki, ni]))
            cells[(K, N)] = c
            print("K=%.1f N=%6d takeoff %.3f (n=%d)  median sup/run %.4f  aligned-mean sup %.4f  unaligned-mean sup %.4f  "
                  "tau mean %.2f sd %.2f (MF %.2f)" % (
                      K, N, c["p_takeoff"], c["n_takeoff"], np.median(c["sup_per_run"]), c["sup_aligned_mean"],
                      c["sup_unaligned_mean"], np.mean(c["tau"]), np.std(c["tau"]), c["mf_t_level"]), flush=True)
    rng = np.random.default_rng(BASE_SEED + 1)
    summary = {}
    for K in KS:
        sl, ci = slope_ci(NS, [cells[(K, N)]["sup_per_run"] for N in NS], rng)
        q = poisson_extinction(K)
        summary[str(K)] = dict(
            slope_median_sup_per_run=sl, slope_ci95=ci,
            slope_aligned_mean=float(np.polyfit(np.log(NS), np.log([cells[(K, N)]["sup_aligned_mean"] for N in NS]), 1)[0]),
            takeoff_theory_hawkesn=1 - q, takeoff_theory_sir=1 - 1 / K, r=THETA * (K - 1),
            tau_sd={str(N): float(np.std(cells[(K, N)]["tau"])) for N in NS},
            tau_mean_minus_mf={str(N): float(np.mean(cells[(K, N)]["tau"]) - cells[(K, N)]["mf_t_level"]) for N in NS},
        )
        print("K=%.1f slope %.3f CI %s ; aligned-mean slope %.3f" % (K, sl, ci, summary[str(K)]["slope_aligned_mean"]))

    # ---- plot 1: shapes ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for row, K in enumerate(KS):
        for col, N in enumerate([250, 4000, 64000]):
            ax = axes[row, col]
            p = cells[(K, N)]["_plot"]
            for cr in p["example_runs"]:
                ax.plot(p["s_grid"], cr, color="#257de0", lw=0.6, alpha=0.35)
            ax.plot([], [], color="#257de0", lw=0.8, label="25 simulated cascades that took off")
            ax.plot(p["s_grid"], p["mean_aligned"], color="#f47230", lw=2, label="mean of all take-off runs")
            ax.plot(p["s_grid"], p["C_mf"], "k--", lw=1.6, label="mean-field (SIR) curve")
            ax.set_title("K = %.1f, N = %d: median gap to mean-field %.3f" % (K, N, np.median(cells[(K, N)]["sup_per_run"])),
                         fontsize=10)
            ax.set_xlabel("time after the cascade reaches 10% of N (model time units)")
            ax.set_ylabel("cumulative posts / N")
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Started from one post and lined up at the 10% mark, simulated cascades close in on the mean-field curve as N grows",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "nov_single_seed_shape.png", dpi=120)
    plt.close(fig)

    # ---- plot 2: scaling, take-off, timing ----
    fig, ax = plt.subplots(1, 4, figsize=(21, 5))
    for K, c, mk in zip(KS, ["#257de0", "#f47230"], ["o", "s"]):
        med = [np.median(cells[(K, N)]["sup_per_run"]) for N in NS]
        lo = [np.quantile(cells[(K, N)]["sup_per_run"], 0.1) for N in NS]
        hi = [np.quantile(cells[(K, N)]["sup_per_run"], 0.9) for N in NS]
        ax[0].errorbar(NS, med, yerr=[np.array(med) - lo, np.array(hi) - med], color=c, marker=mk, capsize=3,
                       label="K=%.1f single runs (median, 10-90%%); slope %.2f" % (K, summary[str(K)]["slope_median_sup_per_run"]))
        ax[0].plot(NS, [cells[(K, N)]["sup_aligned_mean"] for N in NS], color=c, marker=mk, ls=":", mfc="white",
                   label="K=%.1f mean of runs; slope %.2f" % (K, summary[str(K)]["slope_aligned_mean"]))
        pt = [cells[(K, N)]["p_takeoff"] for N in NS]
        se = [math.sqrt(p * (1 - p) / n_runs) for p in pt]
        ax[1].errorbar(NS, pt, yerr=1.96 * np.array(se), color=c, marker=mk, capsize=3, label="K=%.1f simulated (95%% CI)" % K)
        ax[1].axhline(summary[str(K)]["takeoff_theory_hawkesn"], color=c, ls="--", lw=1.2)
        ax[1].axhline(summary[str(K)]["takeoff_theory_sir"], color=c, ls=":", lw=1.2)
        r = THETA * (K - 1)
        ax[2].plot(NS, [np.std(cells[(K, N)]["tau"]) * r for N in NS], color=c, marker=mk, label="K=%.1f" % K)
        ax[3].plot(NS, [cells[(K, N)]["sup_unaligned_mean"] for N in NS], color=c, marker=mk, label="K=%.1f" % K)
    ref = np.array(NS, dtype=float)
    ax[0].plot(ref, 2.0 / np.sqrt(ref), color="#999", lw=1, label="reference 2/sqrt(N)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_ylabel("largest gap to the mean-field curve after alignment (fraction of N)")
    ax[0].set_title("Shape converges:\ngap shrinks like N^-1/2", fontsize=10)
    ax[0].legend(fontsize=7.5)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("fraction of cascades that take off")
    ax[1].set_title("Take-off probability: dashed = HawkesN theory,\ndotted = SIR theory (1 - 1/K)", fontsize=10)
    ax[1].legend(fontsize=8, loc="lower right")
    ax[2].set_xscale("log"); ax[2].set_ylim(0, None)
    ax[2].set_ylabel("spread (sd) of time to reach 10% of N, in units of 1/growth-rate")
    ax[2].set_title("Timing does not converge:\nspread stays the same at every N", fontsize=10)
    ax[2].legend(fontsize=8)
    ax[3].set_xscale("log"); ax[3].set_ylim(0, None)
    ax[3].set_ylabel("largest gap, mean of take-off runs vs mean-field, NO alignment")
    ax[3].set_title("Without alignment the average curve\nstays off the mean-field curve", fontsize=10)
    ax[3].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("population size N")
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "nov_single_seed_scaling.png", dpi=120)
    plt.close(fig)

    out_cells = []
    for (K, N), c in cells.items():
        c = {k: v for k, v in c.items() if k != "_plot"}
        out_cells.append(c)
    (OUT / "nov_single_seed_results.json").write_text(json.dumps(
        dict(base_seed=BASE_SEED, theta=THETA, level=LEVEL, n_runs=n_runs, summary=summary, cells=out_cells)))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "smoke":   # timing test only; writes nothing
        import time
        t0 = time.time()
        c = run_cell(3.0, 64000, int(sys.argv[1]), np.random.SeedSequence([1, 2, 3]))
        print("smoke: %d runs at N=64000 in %.1fs; takeoff %d" % (int(sys.argv[1]), time.time() - t0, c["n_takeoff"]))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
