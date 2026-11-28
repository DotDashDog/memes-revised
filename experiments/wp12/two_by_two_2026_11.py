"""E14: seeds {1, 300} x window {stopped at 85% of mean-field final count, complete}.  Grid only.
uv run python experiments/wp12/two_by_two_2026_11.py
Outputs experiments/WP-12/nov_two_by_two.{jsonl,json,png}"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "WP-12"
from experiments.wp12 import exact_profile as ep  # noqa: E402

K, TH, N, T_FULL = 3.0, 0.4, 1000.0, 75.0
BASE = 20261160
CUT = ep.CHI2_1_95
N_REP = 100


def mf_final_count(I0):
    sol = solve_ivp(lambda t, y: [-y[0] * y[1], TH * y[1] * (K * y[0] - 1)], [0, 400],
                    [1 - I0 / N, K * TH * I0 / N], rtol=1e-10, atol=1e-14, method="LSODA")
    return N * (1 - sol.y[0, -1])


def analyse(ev, T, I0):
    gf = ep.grid_fit(ev, T, I0, np.exp(np.linspace(math.log(TH * 1e-3), math.log(TH * 1e3), 600)))
    pat = ep.profile_at_truth(gf, K, TH, N)
    return dict(n=int(len(ev)), T=float(T), K_hat=gf.mle["K"], theta_hat=gf.mle["theta"], N_hat=gf.mle["N"],
                N_at_edge=gf.mle["N_at_edge"], ll_max=gf.ll_max,
                cover={q: bool(pat[q] >= gf.ll_max - CUT) for q in pat},
                lr_half={q: float(gf.ll_max - pat[q]) for q in pat})


def main():
    ck = OUT / "nov_two_by_two.jsonl"
    rows = []
    if ck.exists():
        for line in ck.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    done = {(r["I0"], r["rep"]) for r in rows}
    with open(ck, "a") as fh:
        for ii, I0 in enumerate([1, 300]):
            level = int(round(0.85 * mf_final_count(I0)))
            for rep in range(N_REP):
                if (I0, rep) in done:
                    continue
                rng = np.random.default_rng(np.random.SeedSequence([BASE, ii, rep]))
                while True:
                    ev = ep.simulate_hawkesn_inversion(K, TH, N, I0, T_FULL, rng)
                    if len(ev) + I0 >= level:
                        break
                t_stop = float(ev[level - I0 - 1])
                row = dict(I0=I0, rep=rep, level=level, final_total=int(len(ev) + I0),
                           stopped=analyse(ev[ev <= t_stop], t_stop, I0), complete=analyse(ev, T_FULL, I0))
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            print("I0", I0, "done", flush=True)
    summ = {}
    for I0 in [1, 300]:
        for w in ["stopped", "complete"]:
            R = [r[w] for r in rows if r["I0"] == I0]
            kh = np.array([r["K_hat"] for r in R]) / K
            summ["I0=%d,%s" % (I0, w)] = dict(
                n=len(R), events_median=float(np.median([r["n"] for r in R])),
                coverage={q: float(np.mean([r["cover"][q] for r in R])) for q in ["K", "theta", "N", "beta", "r"]},
                K_ratio_median=float(np.median(kh)), K_ratio_q05=float(np.quantile(kh, 0.05)),
                K_ratio_q95=float(np.quantile(kh, 0.95)),
                N_ratio_median=float(np.median([r["N_hat"] for r in R]) / N),
                lr_half_K_median=float(np.median([r["lr_half"]["K"] for r in R])))
            print(I0, w, summ["I0=%d,%s" % (I0, w)], flush=True)
    (OUT / "nov_two_by_two.json").write_text(json.dumps(dict(base_seed=BASE, K=K, theta=TH, N=N, summary=summ), indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import beta as betad

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    labels = ["1 seed\nstopped at 85%", "1 seed\nseen to the end", "300 seeds\nstopped at 85%", "300 seeds\nseen to the end"]
    keys = ["I0=1,stopped", "I0=1,complete", "I0=300,stopped", "I0=300,complete"]
    x = np.arange(4)
    for off, q, col, lab, hatch in [(-0.2, "K", "#257de0", "K (offspring per post)", None),
                                    (0.2, "r", "#f47230", "growth rate theta(K-1)", "//")]:
        p = np.array([summ[k]["coverage"][q] for k in keys])
        n = np.array([summ[k]["n"] for k in keys])
        lo = np.array([betad.ppf(0.025, max(pp * nn, 1e-9), nn - pp * nn + 1) for pp, nn in zip(p, n)])
        hi = np.array([betad.ppf(0.975, pp * nn + 1, max(nn - pp * nn, 1e-9)) if pp < 1 else 1.0 for pp, nn in zip(p, n)])
        ax[0].bar(x + off, p, width=0.38, color=col, hatch=hatch, edgecolor="white", label=lab)
        ax[0].errorbar(x + off, p, yerr=[p - lo, hi - p], fmt="none", color="#191919", lw=1)
    ax[0].axhline(0.95, color="k", ls="--", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_ylim(0, 1.02)
    ax[0].set_ylabel("share of 95% intervals containing the truth (dashed = 95%)")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].grid(alpha=0.25, axis="y")
    ax[0].set_title("Interval honesty", fontsize=10)
    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        I0 = 1 if "I0=1," in k else 300
        w = k.split(",")[1]
        kh = np.array([r[w]["K_hat"] for r in rows if r["I0"] == I0]) / K
        ax[1].scatter(i + rng.uniform(-0.25, 0.25, len(kh)), np.clip(kh, 0.2, 200), s=10, color="#257de0", alpha=0.6, lw=0)
        ax[1].plot([i - 0.32, i + 0.32], [np.median(kh)] * 2, color="#191919", lw=2)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set_ylabel("best-fit K / true K (log scale; bar = median; clipped at 200)")
    ax[1].grid(alpha=0.25, axis="y")
    ax[1].set_title("Point estimate of K", fontsize=10)
    fig.suptitle("What breaks the estimate of K: one seed, or watching the cascade to its end? (K=3, theta=0.4, N=1000; 100 cascades per bar)",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "nov_two_by_two.png", dpi=125)


if __name__ == "__main__":
    main()
