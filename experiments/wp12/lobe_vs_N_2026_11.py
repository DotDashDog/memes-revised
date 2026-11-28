"""E12: frequency of the "N = observed count, K huge" lobe as a function of population size.
Grid only.  uv run python experiments/wp12/lobe_vs_N_2026_11.py
Outputs experiments/WP-12/nov_lobe_vs_N.{jsonl,json,png}"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "WP-12"
from experiments.wp12 import exact_profile as ep  # noqa: E402

K, TH, I0 = 1.7, 0.4, 1
NS = [300, 1000, 3000, 10000]
N_REP = 100
FINE = len(sys.argv) > 1 and sys.argv[1] == "fine"   # E12r: 3000-point theta grid, N = 3000 and 10000 only
N_THETA = 3000 if FINE else 400
TAG = "nov_lobe_vs_N_fine" if FINE else "nov_lobe_vs_N"
BASE = 20261150
CUT = ep.CHI2_1_95


def one(ni, N, rep):
    rng = np.random.default_rng(np.random.SeedSequence([BASE, ni, rep]))
    r = TH * (K - 1)
    T = (math.log(N) + 25) / r + 20 / TH
    while True:
        ev = ep.simulate_hawkesn_inversion(K, TH, float(N), I0, T, rng)
        if len(ev) + 1 >= 0.2 * N:
            break
    gf = ep.grid_fit(ev, T, I0, np.exp(np.linspace(math.log(TH * 1e-3), math.log(TH * 1e3), N_THETA)))
    b20 = ep.box_max(gf, 20.0)
    pat = ep.profile_at_truth(gf, K, TH, float(N))
    # K interval inside the K <= 20 box
    Kv = np.exp(np.linspace(math.log(1.01), math.log(20.0), 140))
    pK = ep.profile_generic(gf, Kv, lambda v, t: v + 0 * t)
    iv = ep.interval_from_profile(Kv, pK, ll_max=b20["ll"])
    return dict(N=N, rep=rep, n=int(len(ev)), T=T, K_hat=gf.mle["K"], theta_hat=gf.mle["theta"], N_hat=gf.mle["N"],
                ll_max=gf.ll_max, box20=b20, prof_at_truth=pat, K_iv_box=iv,
                last_event=float(ev[-1]))


def main():
    ck = OUT / (TAG + ".jsonl")
    done = set()
    if ck.exists():
        for line in ck.read_text().splitlines():
            d = json.loads(line)
            done.add((d["N"], d["rep"]))
    with open(ck, "a") as fh:
        for ni, N in enumerate(NS):
            if FINE and N < 3000:
                continue
            for rep in range(N_REP):
                if (N, rep) in done:
                    continue
                fh.write(json.dumps(one(ni, N, rep), default=float) + "\n")
                fh.flush()
            print("N", N, "done", flush=True)
    rows = [json.loads(line) for line in ck.read_text().splitlines()]
    summ = {}
    for N in NS:
        R = [r for r in rows if r["N"] == N]
        if not R:
            continue
        lobe2 = np.array([r["K_hat"] > 20 for r in R])
        adv = np.array([r["ll_max"] - r["box20"]["ll"] for r in R])
        width = np.array([r["K_iv_box"]["hi"] / r["K_iv_box"]["lo"] for r in R])
        hi_open = np.array([r["K_iv_box"]["hi_open"] for r in R])
        summ[N] = dict(
            n=len(R), events_median=float(np.median([r["n"] for r in R])),
            frac_lobe2=float(lobe2.mean()),
            frac_lobe2_wins_by_more_than_1p92=float((adv > CUT).mean()),
            median_adv_when_lobe2=float(np.median(adv[lobe2])) if lobe2.any() else None,
            K_box_interval_ratio_median=float(np.median(width)), K_box_interval_hits_20=float(hi_open.mean()),
            K_box_pinned=float(np.mean([r["box20"]["K_at_bound"] for r in R])),
            cover_global={q: float(np.mean([r["prof_at_truth"][q] >= r["ll_max"] - CUT for r in R])) for q in ["K", "theta", "N", "beta", "r"]},
            cover_box20={q: float(np.mean([r["prof_at_truth"][q] >= r["box20"]["ll"] - CUT for r in R])) for q in ["K", "theta", "N", "beta", "r"]},
            K_box_median=float(np.median([r["box20"]["K"] for r in R])),
        )
        print(N, summ[N])
    (OUT / (TAG + ".json")).write_text(json.dumps(dict(base_seed=BASE, K=K, theta=TH, n_theta=N_THETA, summary=summ), indent=1))
    if FINE:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    f = [summ[N]["frac_lobe2"] for N in NS]
    se = [math.sqrt(max(p * (1 - p), 1e-9) / N_REP) for p in f]
    ax[0].errorbar(NS, f, yerr=1.96 * np.array(se), marker="o", color="#257de0", capsize=3,
                   label="best fit anywhere has K > 20")
    f2 = [summ[N]["frac_lobe2_wins_by_more_than_1p92"] for N in NS]
    ax[0].plot(NS, f2, marker="s", ls="--", color="#f47230", label="...and beats every K <= 20 fit by > 1.92 log-lik units")
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("share of 100 cascades")
    ax[0].set_title("How often the implausible solution\n(N = observed count, K huge) is the best fit", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].plot(NS, [summ[N]["K_box_interval_ratio_median"] for N in NS], marker="o", color="#257de0")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_ylabel("median upper/lower ratio of the\n95% interval for K (K <= 20 imposed)")
    ax[1].set_title("How well K is known if K <= 20 is imposed", fontsize=10)
    for q, c, ls in [("K", "#257de0", "-"), ("N", "#00b894", "-."), ("r", "#ec4647", "--")]:
        ax[2].plot(NS, [summ[N]["cover_box20"][q] for N in NS], marker="o", color=c, ls=ls,
                   label={"K": "K", "N": "N", "r": "growth rate theta(K-1)"}[q] + " (K <= 20 imposed)")
    ax[2].plot(NS, [summ[N]["cover_global"]["r"] for N in NS], marker="x", color="#666", ls=":", label="growth rate, no constraint")
    ax[2].axhline(0.95, color="k", ls="--", lw=1)
    ax[2].set_xscale("log"); ax[2].set_ylim(0, 1.02)
    ax[2].set_ylabel("share of 95% intervals containing the truth")
    ax[2].set_title("Interval coverage (dashed: nominal 95%)", fontsize=10)
    ax[2].legend(fontsize=8, loc="lower right")
    for a in ax:
        a.set_xlabel("population size N (cascade reaches about 0.69 N)")
        a.grid(alpha=0.25)
    fig.suptitle("Single-seed cascades with K = 1.7, fully observed: does a larger population remove the second solution?", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "nov_lobe_vs_N.png", dpi=120)


if __name__ == "__main__":
    main()
