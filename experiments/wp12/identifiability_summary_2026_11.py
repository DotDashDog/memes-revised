"""Summarise experiments/WP-12/nov_ident_replicates.json (E5, E5-add): tables + three plots."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "WP-12"
CUT = 1.9207294103470749
Q = ["K", "theta", "N", "beta", "r"]
QN = {"K": "K", "theta": "theta", "N": "N", "beta": "K*theta", "r": "theta*(K-1)\n(growth rate)"}
C1, C2, C3, C4 = "#257de0", "#f47230", "#00b894", "#6a4ce2"


def binom_ci(k, n):
    from scipy.stats import beta

    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main():
    d = json.loads((OUT / "nov_ident_replicates.json").read_text())
    scn = d["scenarios"]
    rows = d["rows"]
    summ = {}
    fig1, ax1 = plt.subplots(1, 4, figsize=(20, 5.4), sharey=True)
    fig2, ax2 = plt.subplots(1, 4, figsize=(20, 5.0), sharey=True)
    fig3, ax3 = plt.subplots(1, 4, figsize=(20, 4.6))
    for si, (name, s) in enumerate(scn.items()):
        R = [r for r in rows if r["scenario"] == name]
        n = len(R)
        tr = dict(K=s["K"], theta=s["theta"], N=s["N"], beta=s["K"] * s["theta"], r=s["theta"] * (s["K"] - 1))
        errs = [r for r in R if "old" not in r]
        R_ok = [r for r in R if "old" in r]

        def derived(K, th, N):
            return dict(K=K, theta=th, N=N, beta=K * th, r=th * (K - 1))

        g = [derived(r["K"], r["theta"], r["N"]) for r in R]
        b20 = [derived(r["box20"]["K"], r["box20"]["theta"], r["box20"]["N"]) for r in R]
        old = [derived(r["old"]["K"], r["old"]["theta"], r["old"]["N"]) for r in R_ok]
        lobe2 = np.array([r["K"] > 20 for r in R])
        gap_lobes = np.array([r["ll_max_grid"] - r["box20"]["ll"] for r in R])
        cov_g = {q: sum(r["cover"][q] for r in R) for q in Q}
        cov_b20 = {q: sum(r["prof_at_truth"][q] >= r["box20"]["ll"] - CUT for r in R) for q in Q}
        cov_b10 = {q: sum(r["prof_at_truth"][q] >= r["box10"]["ll"] - CUT for r in R) for q in Q}
        wald = {q: sum(r["old"]["wald_cover"][q] for r in R_ok) for q in ["K", "theta", "N", "beta"]}
        # existing fitter vs maxima
        best_box = np.array([max(r["box20"]["ll"], r["old"]["ll"], r["old_warm"]["ll"]) for r in R_ok])
        gap_old_box = best_box - np.array([r["old"]["ll"] for r in R_ok])
        gap_old_glob = np.array([max(r["ll_max_grid"], r["old"]["ll"]) - r["old"]["ll"] for r in R_ok])
        pinned = np.array([r["old"]["K"] >= 19.9 for r in R_ok])
        widths = {q: [(r["intervals"][q]["hi"] / r["intervals"][q]["lo"],
                       r["intervals"][q]["hi_open"] or r["intervals"][q]["lo_open"]) for r in R] for q in Q}
        summ[name] = dict(
            n_rep=n, n_fitter_errors=len(errs), events_median=float(np.median([r["n"] for r in R])),
            events_min=int(min(r["n"] for r in R)), events_max=int(max(r["n"] for r in R)),
            frac_global_max_in_second_lobe_K_gt_20=float(lobe2.mean()),
            loglik_advantage_of_global_over_Kle20_median_when_lobe2=float(np.median(gap_lobes[lobe2])) if lobe2.any() else None,
            coverage_global={q: cov_g[q] / n for q in Q},
            coverage_box20={q: cov_b20[q] / n for q in Q},
            coverage_box10={q: cov_b10[q] / n for q in Q},
            coverage_wald_existing_fitter={q: wald[q] / len(R_ok) for q in wald},
            median_ratio_global={q: float(np.median([x[q] / tr[q] for x in g])) for q in Q},
            median_ratio_box20={q: float(np.median([x[q] / tr[q] for x in b20])) for q in Q},
            median_ratio_existing_fitter={q: float(np.median([x[q] / tr[q] for x in old])) for q in Q},
            iqr_ratio_existing_fitter={q: [float(np.quantile([x[q] / tr[q] for x in old], a)) for a in (0.25, 0.75)] for q in Q},
            p05_p95_ratio_existing_fitter={q: [float(np.quantile([x[q] / tr[q] for x in old], a)) for a in (0.05, 0.95)] for q in Q},
            frac_within_20pct_existing_fitter={q: float(np.mean([abs(x[q] / tr[q] - 1) < 0.2 for x in old])) for q in Q},
            existing_fitter=dict(frac_pinned_at_K20=float(pinned.mean()),
                                 frac_gap_gt_0p1_vs_best_Kle20=float((gap_old_box > 0.1).mean()),
                                 max_gap_vs_best_Kle20=float(gap_old_box.max()),
                                 frac_gap_gt_0p1_vs_global=float((gap_old_glob > 0.1).mean()),
                                 frac_gap_gt_1p92_vs_global=float((gap_old_glob > CUT).mean())),
            interval_ratio_median={q: float(np.median([w for w, _ in widths[q]])) for q in Q},
            interval_frac_unbounded={q: float(np.mean([o for _, o in widths[q]])) for q in Q},
        )
        # --- fig1: what the existing fitter returns ---
        ax = ax1[si]
        rng = np.random.default_rng(0)
        for qi, q in enumerate(Q):
            y = np.array([x[q] / tr[q] for x in old])
            xj = qi + rng.uniform(-0.28, 0.28, len(y))
            ax.scatter(xj[~pinned], y[~pinned], s=9, color=C1, alpha=0.55, lw=0)
            ax.scatter(xj[pinned], y[pinned], s=14, color=C2, alpha=0.8, lw=0, marker="^")
            ax.plot([qi - 0.33, qi + 0.33], [np.median(y)] * 2, color="#191919", lw=2)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(0.02, 50)
        ax.set_xticks(range(5)); ax.set_xticklabels([QN[q] for q in Q], fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        ax.set_title("%s: %s\n%d cascades, median %d events; %.0f%% pinned at K=20" % (
            name, s["desc"].split(":")[0], len(R_ok), summ[name]["events_median"], 100 * pinned.mean()), fontsize=9.5)
        # --- fig2: coverage ---
        ax = ax2[si]
        w = 0.26
        for off, cov, col, lab, hatch in [(-w, cov_g, "#666666", "interval from the true global maximum", None),
                                          (0, cov_b20, C1, "interval within the fitter's box (K <= 20)", None),
                                          (w, wald, C2, "existing fitter's own +-1.96 se (Hessian)", "//")]:
            for qi, q in enumerate(Q):
                if q not in cov:
                    continue
                nn = n if cov is not wald else len(R_ok)
                p = cov[q] / nn
                lo, hi = binom_ci(cov[q], nn)
                ax.bar(qi + off, p, width=w * 0.92, color=col, hatch=hatch, edgecolor="white",
                       label=lab if qi == 0 else None)
                ax.plot([qi + off] * 2, [lo, hi], color="#191919", lw=1)
        ax.axhline(0.95, color="k", ls="--", lw=1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(5)); ax.set_xticklabels([QN[q] for q in Q], fontsize=9)
        ax.set_title("%s (%d cascades)" % (name, n), fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        if si == 0:
            ax.set_ylabel("share of cascades whose 95% interval contains the truth")
            ax.legend(fontsize=7.5, loc="lower left")
        # --- fig3: fitter gap ---
        ax = ax3[si]
        gcl = np.clip(gap_old_glob, 1e-4, None)
        ax.hist(np.log10(gcl), bins=np.linspace(-4, 2, 37), color=C1)
        ax.axvline(math.log10(0.1), color="k", ls="--", lw=1)
        ax.axvline(math.log10(CUT), color=C2, ls="--", lw=1)
        ax.set_xlabel("log10 of (true maximum log-lik - fitter's log-lik); <= -4 lumped at left")
        ax.set_ylabel("number of cascades")
        ax.set_title("%s: fitter more than 0.1 below the true maximum in %.0f%%,\nmore than 1.92 (outside the 95%% region) in %.0f%%" % (
            name, 100 * (gap_old_glob > 0.1).mean(), 100 * (gap_old_glob > CUT).mean()), fontsize=9.5)
        ax.grid(alpha=0.25)
    ax1[0].set_ylabel("existing fitter's estimate / true value (log scale; 1 = exact)")
    fig1.suptitle("What the existing fitter returns on 200 simulated cascades per scenario: only the growth rate is reliably near 1. "
                  "Blue = ordinary fit, orange triangle = fit stuck at the K = 20 bound, bar = median", fontsize=11.5)
    fig1.tight_layout(); fig1.savefig(OUT / "nov_ident_recovery.png", dpi=120); plt.close(fig1)
    fig2.suptitle("Do 95% intervals contain the truth 95% of the time (dashed line)?  Thin black line = sampling uncertainty of the share",
                  fontsize=11.5)
    fig2.tight_layout(); fig2.savefig(OUT / "nov_ident_coverage.png", dpi=120); plt.close(fig2)
    fig3.suptitle("How far below the true maximum of the likelihood does the existing fitter stop? (dashed black: 0.1; dashed orange: 1.92)",
                  fontsize=11.5)
    fig3.tight_layout(); fig3.savefig(OUT / "nov_ident_fitter_gap.png", dpi=120); plt.close(fig3)
    (OUT / "nov_ident_replicates_summary.json").write_text(json.dumps(summ, indent=1))
    for name, v in summ.items():
        print("==", name, json.dumps(v, indent=None)[:20000])


if __name__ == "__main__":
    main()
