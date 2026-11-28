"""E9 / E10: the paper's headline figure (final-size distribution, prior vs after observing a prefix)
re-implemented, checked against the upstream R output, and tested against what the processes do.

Upstream method (scripts/functions-size-distribution.R), transcribed:
  * HawkesN (K, theta, N) -> SIR (beta = K*theta, gamma = theta, N).
  * Final-size distribution of the Markovian SIR jump chain started at (S, I).
  * prior: (S, I) = (N - 1, 1).
  * after a prefix of n_seen events ending at time t: S = N - n_seen,
    I = round(sum_i exp(-gamma (t - t_i)))   (expected number still infectious).

Here the chain is solved exactly with an O(N^2) recursion (no 1.7M-state sparse matrix):
at susceptible level s the chance that the next jump is a recovery is
q_s = gamma / (gamma + beta s / N), independent of I; with i infectious the chain is absorbed
(final size N - s) with probability q_s^i, otherwise it moves to level s-1 with
i - k + 1 infectious after k < i recoveries, probability (1 - q_s) q_s^k.

Stages:
  rcheck    E9a  identity vs upstream R output on the R tutorial's cascade (needs nov_upstream/tutorial_run)
  figure    E9b  the prior -> posterior figure for several observed fractions (own simulated cascade)
  truth     E10  is that distribution right for SIR data? for HawkesN data? (Monte-Carlo continuation)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "WP-12"

from experiments.wp12 import exact_profile as ep  # noqa: E402

C1, C2, C3 = "#257de0", "#f47230", "#00b894"


def sir_final_size(beta, gamma, N: int, S0: int, I0: int) -> np.ndarray:
    """P(final size = k), k = 0..N, final size = N - S_infinity (upstream's convention:
    it counts everyone no longer susceptible, including the initial infectious)."""
    out = np.zeros(N + 1)
    f = np.zeros(N + 2)  # distribution over I at the current S level
    f[I0] = 1.0
    for s in range(S0, -1, -1):
        q = gamma / (gamma + beta * s / N)
        imax = N - s  # I cannot exceed N - s
        fi = f[: imax + 1]
        powers = q ** np.arange(imax + 1)
        out[N - s] = float((fi[1:] * powers[1:]).sum()) + (fi[0] if s == S0 and I0 == 0 else 0.0)
        if s == 0:
            break
        # g(j) = sum_{i >= j} f(i) q^{i-j}  via backward IIR:  g(j) = f(j) + q g(j+1)
        g = lfilter([1.0], [1.0, -q], fi[::-1])[::-1]
        # new f'(i') = (1-q) * g(i'-1) for i' >= 2 ... i' = i - k + 1 with k <= i-1 => i' >= 2, i >= i'-1
        newf = np.zeros(N + 2)
        newf[2: imax + 2] = (1 - q) * g[1: imax + 1]
        f = newf
    return out


def upstream_state_from_prefix(times_with_seeds, n_seen, gamma, N):
    t = np.asarray(times_with_seeds[:n_seen])
    I = int(round(float(np.exp(-gamma * (t[-1] - t)).sum())))
    return int(N - n_seen), I


def deterministic_final_size(K, N, I0=1):
    # solve N + (N/R0) log(sinf/(N - I0)) - sinf = 0 on (0, N)   (upstream's formula)
    from scipy.optimize import brentq

    f = lambda s: N + (N / K) * math.log(s / (N - I0)) - s
    return N - brentq(f, 1e-9, N - I0 - 1e-9) if K > 1 else float("nan")


# ---------------------------------------------------------------------------


def stage_rcheck():
    tdir = OUT / "nov_upstream" / "tutorial_run_R35"
    R = list(csv.DictReader(open(tdir / "size_distribution_R.csv")))
    size = np.array([float(r["size"]) for r in R]).astype(int)
    prior_R = np.array([float(r["prior"]) for r in R])
    post_R = np.array([float(r["posterior"]) for r in R])
    meta = dict(l.strip().split("=") for l in open(tdir / "size_distribution_R_meta.txt"))
    seen = int(meta["seenEvents"])
    hist = np.array([float(r["time"]) for r in csv.DictReader(open(tdir / "sim_history_01.csv"))])
    N, K, th = 1300, 5.0, 0.2
    prior = sir_final_size(K * th, th, N, N - 1, 1)
    S, I = upstream_state_from_prefix(hist, seen, th, N)
    post = sir_final_size(K * th, th, N, S, I)
    res = dict(seen=seen, S=S, I=I,
               max_abs_prior=float(np.abs(prior[size] - prior_R).max()),
               max_abs_post=float(np.abs(post[size] - post_R).max()),
               sum_prior_R=float(prior_R.sum()), sum_post_R=float(post_R.sum()),
               sum_prior=float(prior.sum()), sum_post=float(post.sum()),
               order_check=bool((np.diff(size) > 0).all() or (np.diff(size) < 0).all()))
    print("E9a", res)
    (OUT / "nov_size_rcheck.json").write_text(json.dumps(res, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    k = np.arange(N + 1)
    for a, (pm, pr, lab) in zip(ax, [(prior, prior_R, "before anything is observed (one infected)"),
                                     (post, post_R, "after the first %d of %d events" % (seen, len(hist)))]):
        a.plot(size, np.where(pr > 0, pr, np.nan), color=C2, lw=5, alpha=0.55, label="upstream R code (tutorial run)")
        a.plot(k, np.where(pm > 0, pm, np.nan), color="#191919", lw=1.2, label="this project's Python recursion")
        a.set_yscale("log"); a.set_ylim(1e-12, 1)
        a.set_xlabel("final cascade size (number of people ever infected, of N = 1300)")
        a.set_ylabel("probability")
        a.set_title(lab, fontsize=10)
        a.legend(fontsize=8, loc="upper center")
        a.grid(alpha=0.25)
    ax[1].axvline(len(hist), color="#ec4647", lw=1)
    ax[1].text(len(hist) - 5, 1e-10, "size this cascade actually reached: %d" % len(hist), ha="right", fontsize=8)
    fig.suptitle("Paper's headline figure, from the unmodified R tutorial and from our re-implementation: "
                 "largest difference in any probability %.1e (prior), %.1e (posterior)"
                 % (res["max_abs_prior"], res["max_abs_post"]), fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "nov_size_rcheck.png", dpi=125)
    plt.close(fig)


def stage_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, K, th = 1300, 5.0, 0.2
    seed = 20261130
    rng = np.random.default_rng(seed)
    # one stochastic-SIR cascade from a single infected individual that takes off (as in the paper's story)
    while True:
        ev = ep.simulate_sir_gillespie(K * th, th, N, 1, 200.0, rng)
        if len(ev) > 100:
            break
    hist = np.concatenate([[0.0], ev])
    final = len(hist)
    k = np.arange(N + 1)
    prior = sir_final_size(K * th, th, N, N - 1, 1)
    curves = [("nothing observed (1 infected)", "#191919", 2.6, "-", prior)]
    cols = ["#6a4ce2", C1, C3, "#f2a600", C2]
    lss = ["--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
    rows = [dict(seen=1, S=N - 1, I=1, p_small=float(prior[:65].sum()), mode=int(prior[65:].argmax() + 65))]
    for c, ls, n_seen in zip(cols, lss, [5, 20, 65, 325, 650]):
        S, I = upstream_state_from_prefix(hist, n_seen, th, N)
        p = sir_final_size(K * th, th, N, S, I)
        curves.append(("%d events observed; est. %d infectious" % (n_seen, I), c, 1.8, ls, p))
        cdf = np.cumsum(p)
        rows.append(dict(seen=n_seen, S=S, I=I, p_below_100=float(p[:100].sum()), mode=int(p.argmax()),
                         q05=int(np.searchsorted(cdf, 0.05)), q95=int(np.searchsorted(cdf, 0.95))))
    rows[0]["p_below_100"] = float(prior[:100].sum())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, gridspec_kw=dict(width_ratios=[1, 1]))
    for a_, (lo, hi) in zip(ax, [(0, 100), (1255, 1300)]):
        for lab, c, lw, ls, p in curves:
            a_.plot(k, np.where(p > 1e-300, p, np.nan), color=c, lw=lw, ls=ls, label=lab)
        a_.set_xlim(lo, hi)
        a_.set_yscale("log"); a_.set_ylim(1e-9, 1.5)
        a_.grid(alpha=0.25)
        a_.set_xlabel("final cascade size (people ever infected, out of N = 1300)")
    ax[0].set_ylabel("probability (log scale)")
    ax[0].set_title("Left peak: the cascade fizzles (sizes 0-100 shown)", fontsize=10)
    ax[1].set_title("Right peak: nearly everyone is reached (sizes 1255-1300 shown; nothing in between)", fontsize=10)
    ax[1].axvline(final, color="#ec4647", lw=1.2, ls="--")
    ax[1].text(final - 0.5, 3e-9, "this cascade's actual final size: %d" % final, ha="right", fontsize=9)
    ax[0].legend(fontsize=8.5, loc="upper right", title="what has been seen of the cascade")
    fig.suptitle("Final-size distribution (paper's method, K=5, theta=0.2, N=1300): two peaks before any observation; "
                 "P(size < 100) = %.2f at the start,\n%.1e after 5 events, %.1e after 20 events; the right-hand peak does not move"
                 % (rows[0]["p_below_100"], rows[1]["p_below_100"], rows[2]["p_below_100"]), fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "nov_size_prior_to_posterior.png", dpi=125)
    plt.close(fig)
    (OUT / "nov_size_prior_to_posterior.json").write_text(json.dumps(
        dict(seed=seed, N=N, K=K, theta=th, final=final, small_threshold=100, rows=rows), indent=1))
    for r in rows:
        print(r)


def continue_hawkesn(K, theta, N, used, A, rng):
    """Final total count of a HawkesN process continued from (used, kernel sum A)."""
    while True:
        d = 1.0 - used / N
        if d <= 0:
            return used
        cap = d * K * A
        E = rng.exponential()
        if E >= cap:
            return used
        s = -math.log(1.0 - E / cap) / theta
        A = A * math.exp(-theta * s) + 1.0
        used += 1


def continue_sir(beta, gamma, N, S, I, rng):
    while I > 0:
        p_inf = (beta * S / N) / (beta * S / N + gamma)
        if rng.random() < p_inf:
            S -= 1; I += 1
        else:
            I -= 1
    return N - S


def stage_truth_prior(n_mc=20000):
    """E10. Same parameters, one seed. Three questions, each a comparison of distributions:
    (1) prior: SIR chain vs Monte-Carlo SIR vs Monte-Carlo HawkesN;
    (2) after a prefix of 30 events of an SIR cascade: chain from upstream's estimated state vs
        Monte-Carlo continuation of SIR from the TRUE hidden state of that cascade;
    (3) after a prefix of 30 events of a HawkesN cascade: chain from upstream's estimated state vs
        Monte-Carlo continuation of HawkesN from its exact state."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, K, th = 1000, 2.0, 0.4
    beta = K * th
    seeds = dict(prior_sir=20261131, prior_hawkes=20261132, prefix=20261133, cont=20261134)
    res = dict(N=N, K=K, theta=th, n_mc=n_mc, seeds=seeds)
    k = np.arange(N + 1)
    chain_prior = sir_final_size(beta, th, N, N - 1, 1)
    rng = np.random.default_rng(seeds["prior_sir"])
    mc_sir = np.array([continue_sir(beta, th, N, N - 1, 1, rng) for _ in range(n_mc)])
    rng = np.random.default_rng(seeds["prior_hawkes"])
    mc_hk = np.array([continue_hawkesn(K, th, N, 1, 1.0, rng) for _ in range(n_mc)])
    thr = 200  # between the two peaks (see plot)
    q_pois = 0.0
    for _ in range(200):
        q_pois = math.exp(-K * (1 - q_pois))
    res["prior"] = dict(threshold=thr, chain_p_small=float(chain_prior[:thr].sum()),
                        mc_sir_p_small=float((mc_sir < thr).mean()), mc_hawkes_p_small=float((mc_hk < thr).mean()),
                        theory_sir=1 / K, theory_hawkes_poisson_offspring=q_pois,
                        se=float(math.sqrt(0.25 / n_mc)),
                        mc_sir_large_mean=float(mc_sir[mc_sir >= thr].mean()), mc_sir_large_sd=float(mc_sir[mc_sir >= thr].std()),
                        mc_hawkes_large_mean=float(mc_hk[mc_hk >= thr].mean()), mc_hawkes_large_sd=float(mc_hk[mc_hk >= thr].std()),
                        chain_large_mean=float((k[thr:] * chain_prior[thr:]).sum() / chain_prior[thr:].sum()))
    print("prior:", res["prior"])

    res["note"] = "prefix panels are produced by stage truth5 (E10b); the 30-event panels of the first run are in nov_size_truth_check.json (and .png, whose two right-hand panels are uninformative)"
    (OUT / "nov_size_truth_prior.json").write_text(json.dumps(res, indent=1))
    return res, (mc_sir, mc_hk, chain_prior)


def prefix_panels(n_seen, seed_prefix, seed_cont, N=1000, K=2.0, th=0.4, n_cont=4000, thr=200):
    beta = K * th
    k = np.arange(N + 1)
    rng = np.random.default_rng(seed_prefix)
    rngc = np.random.default_rng(seed_cont)
    panels = []
    for kind in ["sir", "hawkesn"]:
        for c in range(8):
            if kind == "sir":
                while True:
                    S, I, t = N - 1, 1, 0.0
                    times = [0.0]
                    while I > 0 and len(times) < n_seen:
                        ri, rr = beta * S * I / N, th * I
                        t += rng.exponential(1 / (ri + rr))
                        if rng.random() < ri / (ri + rr):
                            S -= 1; I += 1; times.append(t)
                        else:
                            I -= 1
                    if len(times) == n_seen:
                        break
                true_state = float(I)
                mc = np.array([continue_sir(beta, th, N, S, I, rngc) for _ in range(n_cont)])
            else:
                while True:
                    ev = ep.simulate_hawkesn_inversion(K, th, N, 1, 1e9, rng)
                    if len(ev) + 1 >= n_seen:
                        break
                times = np.concatenate([[0.0], ev])[:n_seen]
                A = float(np.exp(-th * (times[-1] - np.asarray(times))).sum())
                true_state = A
                mc = np.array([continue_hawkesn(K, th, N, n_seen, A, rngc) for _ in range(n_cont)])
            S_est, I_est = upstream_state_from_prefix(np.asarray(times), n_seen, th, N)
            kernel_mass = float(np.exp(-th * (times[-1] - np.asarray(times))).sum())
            chain = sir_final_size(beta, th, N, S_est, I_est)
            cdf = np.cumsum(chain)
            big = mc >= thr
            panels.append(dict(kind=kind, c=c, n_seen=n_seen, I_est=I_est, kernel_mass=kernel_mass, true_state=true_state,
                               chain_p_small=float(chain[:thr].sum()), mc_p_small=float((mc < thr).mean()),
                               mc_p_small_se=float(math.sqrt(max((mc < thr).mean() * (1 - (mc < thr).mean()), 1e-12) / n_cont)),
                               chain_q=[int(np.searchsorted(cdf / cdf[-1], q)) for q in (0.05, 0.5, 0.95)],
                               chain_q_large=[int(thr + np.searchsorted(np.cumsum(chain[thr:]) / chain[thr:].sum(), q)) for q in (0.05, 0.5, 0.95)],
                               mc_q_large=[float(np.quantile(mc[big], q)) for q in (0.05, 0.5, 0.95)] if big.any() else None))
            print(kind, c, {kk: vv for kk, vv in panels[-1].items() if kk not in ("kind", "c")}, flush=True)
    return panels


def stage_truth5():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, K, th, thr, n_mc = 1000, 2.0, 0.4, 200, 20000
    res, (mc_sir, mc_hk, chain_prior) = stage_truth_prior(n_mc)
    p5 = prefix_panels(5, 20261135, 20261136)
    p30 = prefix_panels(30, 20261133, 20261134)
    res["after_5_events"] = p5
    res["after_30_events"] = p30
    res["seeds_prefix"] = dict(five=[20261135, 20261136], thirty=[20261133, 20261134])
    (OUT / "nov_size_truth_check_E10b.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    bins = np.arange(0, N + 21, 20)
    centers = 0.5 * (bins[:-1] + bins[1:])
    chain_b = np.add.reduceat(chain_prior, bins[:-1])[: len(centers)]
    ax[0].step(centers, np.histogram(mc_sir, bins=bins)[0] / n_mc, where="mid", color=C1, lw=2,
               label="stochastic SIR, simulated (%d runs)" % n_mc)
    ax[0].step(centers, np.histogram(mc_hk, bins=bins)[0] / n_mc, where="mid", color=C2, lw=2, ls="--",
               label="HawkesN, simulated (%d runs)" % n_mc)
    ax[0].plot(centers, chain_b, "k:", lw=1.6, label="paper's method (SIR Markov chain)")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-5, 1.2)
    ax[0].set_xlabel("final cascade size (bins of 20; N = 1000)")
    ax[0].set_ylabel("probability per bin")
    ax[0].legend(fontsize=8, loc="upper center")
    ax[0].set_title("Before any observation. Chance of staying small (< %d):\n"
                    "paper's method %.3f, simulated SIR %.3f, simulated HawkesN %.3f"
                    % (thr, res["prior"]["chain_p_small"], res["prior"]["mc_sir_p_small"], res["prior"]["mc_hawkes_p_small"]),
                    fontsize=10)
    x = np.arange(16)
    lab = ["SIR %d" % (i + 1) for i in range(8)] + ["HkN %d" % (i + 1) for i in range(8)]
    ax[1].errorbar(x - 0.12, [p["mc_p_small"] for p in p5], yerr=[1.96 * p["mc_p_small_se"] for p in p5], fmt="o",
                   color="#191919", capsize=3, label="simulated continuation of the true process (95% CI)")
    ax[1].plot(x[:8] + 0.12, [p["chain_p_small"] for p in p5[:8]], "s", color=C1, ms=8, label="paper's method, SIR-generated cascade")
    ax[1].plot(x[8:] + 0.12, [p["chain_p_small"] for p in p5[8:]], "D", color=C2, ms=8, label="paper's method, HawkesN-generated cascade")
    ax[1].axvline(7.5, color="#999", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab, rotation=60, fontsize=8)
    ax[1].set_ylabel("probability the cascade still fizzles (final size < %d)" % thr)
    ax[1].set_ylim(0, None)
    ax[1].legend(fontsize=7.5)
    ax[1].grid(alpha=0.25, axis="y")
    n_hi = sum(p["chain_p_small"] > p["mc_p_small"] for p in p5[8:])
    ax[1].set_title("After 5 observed events: paper's method above the simulated truth\nin %d of 8 HawkesN cascades" % n_hi, fontsize=10)
    for i, p in enumerate(p30):
        if p["mc_q_large"] is None:
            continue
        ax[2].plot([i - 0.12] * 2, [p["mc_q_large"][0], p["mc_q_large"][2]], color="#191919", lw=3)
        ax[2].plot(i - 0.12, p["mc_q_large"][1], "o", color="#191919")
        col = C1 if i < 8 else C2
        ax[2].plot([i + 0.12] * 2, [p["chain_q_large"][0], p["chain_q_large"][2]], color=col, lw=3)
        ax[2].plot(i + 0.12, p["chain_q_large"][1], "s", color=col)
    ax[2].plot([], [], color="#191919", lw=3, label="simulated continuation (5-95%, median)")
    ax[2].plot([], [], color=C1, lw=3, label="paper's method, SIR-generated")
    ax[2].plot([], [], color=C2, lw=3, label="paper's method, HawkesN-generated")
    ax[2].axvline(7.5, color="#999", lw=1)
    ax[2].set_xticks(x); ax[2].set_xticklabels(lab, rotation=60, fontsize=8)
    ax[2].set_ylabel("final cascade size, given that it takes off")
    ax[2].legend(fontsize=7.5, loc="lower left")
    ax[2].grid(alpha=0.25, axis="y")
    w_ch = np.mean([p["chain_q_large"][2] - p["chain_q_large"][0] for p in p30[8:]])
    w_mc = np.mean([p["mc_q_large"][2] - p["mc_q_large"][0] for p in p30[8:]])
    ax[2].set_title("After 30 observed events: 90%% range of final size. HawkesN cascades:\n"
                    "paper's method %.0f wide, simulated truth %.0f wide" % (w_ch, w_mc), fontsize=10)
    fig.suptitle("Is the paper's final-size distribution right for the process that made the data? "
                 "(K=2, theta=0.4, N=1000, one seed post)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "nov_size_truth_check_E10b.png", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    {"rcheck": stage_rcheck, "figure": stage_figure, "truth5": stage_truth5}[sys.argv[1]]()
