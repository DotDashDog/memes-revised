"""E13: conditional-on-take-off likelihood for the single-seed scenarios M2 and M3 (same cascades as E5).
uv run python experiments/wp12/takeoff_correction_2026_11.py
Outputs experiments/WP-12/nov_takeoff_correction.{jsonl,json,png}"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "WP-12"
from experiments.wp12 import exact_profile as ep  # noqa: E402
from experiments.wp12 import identifiability_2026_11 as idn  # noqa: E402

CUT = ep.CHI2_1_95
M_MIN = 20  # events required for "take-off" in E5 (total progeny incl. seed >= 21)


def log_p_takeoff(K):
    """log P(total progeny of a Poisson(K) branching process >= M_MIN + 1). Borel distribution."""
    n = np.arange(1, M_MIN + 1)
    K = np.atleast_1d(np.asarray(K, dtype=float))
    logp = -K[:, None] * n + (n - 1) * np.log(K[:, None] * n) - gammaln(n + 1)
    return np.log1p(-np.exp(logp).sum(axis=1))


def one(name, rep, Kv):
    s = idn.SCENARIOS[name]
    ev, _ = idn.draw_cascade(name, rep)
    gf = ep.grid_fit(ev, s["T"], s["I0"], idn.theta_grid(s["theta"], 400))
    pK = ep.profile_generic(gf, Kv, lambda v, t: v + 0 * t)
    p_true = float(ep.ll_given_K_theta(gf, np.full(len(gf.ss.thetas), s["K"])).max())
    out = {}
    for tag, corr, corr_true in [("uncond", 0 * Kv, 0.0),
                                 ("cond", -log_p_takeoff(Kv), -float(log_p_takeoff(s["K"])[0]))]:
        p = pK + corr
        i = int(np.argmax(p))
        iv = ep.interval_from_profile(Kv, p)
        out[tag] = dict(K_hat=float(Kv[i]), lr_half_at_truth=float(p[i] - (p_true + corr_true)),
                        covers=bool(p[i] - (p_true + corr_true) <= CUT), lo=iv["lo"], hi=iv["hi"],
                        hi_open=iv["hi_open"])
    return dict(scenario=name, rep=rep, n=int(len(ev)), **out)


def main():
    Kv = np.exp(np.linspace(math.log(1.02), math.log(2000.0), 260))
    ck = OUT / "nov_takeoff_correction.jsonl"
    done = set()
    if ck.exists():
        for line in ck.read_text().splitlines():
            try:
                d = json.loads(line)
                done.add((d["scenario"], d["rep"]))
            except Exception:
                pass
    with open(ck, "a") as fh:
        for name in ["M2", "M3"]:
            for rep in range(1, 201):
                if (name, rep) in done:
                    continue
                fh.write(json.dumps(one(name, rep, Kv)) + "\n")
                fh.flush()
            print(name, "done", flush=True)
    rows = []
    for line in ck.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    summ = {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a, name in zip(ax, ["M2", "M3"]):
        R = [r for r in rows if r["scenario"] == name]
        Kt = idn.SCENARIOS[name]["K"]
        summ[name] = {}
        for tag, col, lab in [("uncond", "#666666", "ordinary likelihood"), ("cond", "#257de0", "likelihood conditional on take-off")]:
            kh = np.array([r[tag]["K_hat"] for r in R]) / Kt
            cov = float(np.mean([r[tag]["covers"] for r in R]))
            summ[name][tag] = dict(n=len(R), coverage_K=cov, median_K_ratio=float(np.median(kh)),
                                   q05=float(np.quantile(kh, 0.05)), q25=float(np.quantile(kh, 0.25)),
                                   q75=float(np.quantile(kh, 0.75)), q95=float(np.quantile(kh, 0.95)),
                                   frac_K_gt_20=float(np.mean(kh * Kt > 20)),
                                   frac_within_20pct=float(np.mean(np.abs(kh - 1) < 0.2)))
            a.hist(np.log10(kh), bins=np.linspace(-0.4, 3.1, 50), histtype="step", lw=2, color=col,
                   ls="-" if tag == "cond" else "--",
                   label="%s: median %.2f x truth, 95%% interval covers in %.0f%%" % (lab, np.median(kh), 100 * cov))
        a.axvline(0, color="k", lw=1)
        a.set_xlabel("log10( best-fit K / true K );  0 = exact")
        a.set_title("%s: %s (200 cascades)" % (name, idn.SCENARIOS[name]["desc"]), fontsize=10)
        a.legend(fontsize=8)
        a.grid(alpha=0.25)
        print(name, summ[name])
    ax[0].set_ylabel("number of cascades")
    fig.suptitle("Does accounting for 'we only look at cascades that took off' repair the estimate of K?", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "nov_takeoff_correction.png", dpi=125)
    (OUT / "nov_takeoff_correction.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
