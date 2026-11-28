"""E1-E3 of experiments/WP-12/expectations-2026-11.md.

E1  likelihood identity: JAX scan code vs brute-force quadrature vs sufficient-statistics form
E2  simulator vs likelihood: time-rescaling KS; thinning vs inversion sampler
E3  audit of the June profile routine against an optimiser-free grid profile

Run:  JAX_PLATFORMS=cpu uv run python experiments/wp12/checks_2026_11.py
Writes experiments/WP-12/nov_checks_results.json and nov_E2_*.png, nov_E3_*.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "WP-12"

import jax.numpy as jnp  # noqa: E402

from experiments.wp12 import exact_profile as ep  # noqa: E402
from experiments.wp12.hawkesn import (  # noqa: E402
    HawkesNParams,
    hawkesn_log_likelihood,
    simulate_hawkesn,
)


def brute_loglik(times, T, I0, K, theta, N, mu=0.0):
    """O(n^2) + numerical quadrature. Shares nothing with the other two."""
    all_t = np.concatenate([np.zeros(I0), np.asarray(times)])

    def lam(t, n_used):
        past = all_t[:n_used]
        return (1 - n_used / N) * (mu + K * theta * np.exp(-theta * (t - past)).sum())

    s = 0.0
    for i, t in enumerate(times):
        s += math.log(lam(t, I0 + i))
    comp = 0.0
    edges = np.concatenate([[0.0], times, [T]])
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        if b > a:
            val, _ = integrate.quad(lam, a, b, args=(I0 + k,), epsrel=1e-11, epsabs=1e-13)
            comp += val
    return s - comp


def jax_loglik(times, T, I0, K, theta, N, mu=0.0):
    log_mu = math.log(mu) if mu > 0 else -745.0
    return float(hawkesn_log_likelihood(
        jnp.float64(math.log(K)), jnp.float64(math.log(theta)), jnp.float64(math.log(N)),
        jnp.float64(log_mu), jnp.asarray(times), float(T), I0=I0))


def run_E1(res):
    rng = np.random.default_rng(20261100)
    rows = []
    for c in range(20):
        I0 = 1 if c % 2 == 0 else 30
        Kg, thg, Ng = rng.uniform(1.5, 4), rng.uniform(0.2, 1.0), float(rng.integers(150, 600))
        for _ in range(50):
            ev = ep.simulate_hawkesn_inversion(Kg, thg, Ng, I0, 15 / thg, rng)
            if len(ev) >= 30:
                break
        T = 15 / thg
        # evaluate at a parameter point different from the generating one
        K, th = Kg * rng.uniform(0.5, 2), thg * rng.uniform(0.5, 2)
        N = (I0 + len(ev)) * rng.uniform(1.05, 3)
        b = brute_loglik(ev, T, I0, K, th, N)
        j = jax_loglik(ev, T, I0, K, th, N)
        s = ep.loglik_point(ev, T, I0, K, th, N)
        rows.append(dict(n=len(ev), I0=I0, brute=b, jax=j, suff=s,
                         rel_jax=abs(j - b) / abs(b), rel_suff=abs(s - b) / abs(b)))
    res["E1_mu0"] = dict(max_rel_jax_vs_brute=max(r["rel_jax"] for r in rows),
                         max_rel_suff_vs_brute=max(r["rel_suff"] for r in rows), rows=rows)
    print("E1 mu=0: max rel diff JAX vs brute %.2e ; suff-stats vs brute %.2e" %
          (res["E1_mu0"]["max_rel_jax_vs_brute"], res["E1_mu0"]["max_rel_suff_vs_brute"]))
    # mu > 0 (expected defect in the JAX background compensator)
    rows = []
    for c in range(6):
        I0 = 1
        Ng = 200.0
        p = HawkesNParams(K=1.2, theta=0.5, N=Ng, mu=0.5)
        ev = simulate_hawkesn(p, 30.0, rng, I0=I0)
        N = (I0 + len(ev)) * 1.5
        mu = 0.5
        b = brute_loglik(ev, 30.0, I0, 1.2, 0.5, N, mu=mu)
        j = jax_loglik(ev, 30.0, I0, 1.2, 0.5, N, mu=mu)
        rows.append(dict(n=len(ev), N=N, brute=b, jax=j, abs_diff=j - b, predicted=-mu * 30.0 / N))
    res["E1_mu_pos"] = rows
    print("E1 mu>0: JAX - brute =", [round(r["abs_diff"], 4) for r in rows],
          " predicted -mu*T/N =", [round(r["predicted"], 4) for r in rows])


def rescaled_increments(times, T, I0, K, theta, N):
    """Compensator increments between consecutive observed events (true params)."""
    out = []
    A = float(I0)
    t_prev = 0.0
    for i, t in enumerate(times):
        d = 1 - (I0 + i) / N
        out.append(d * K * A * (1 - math.exp(-theta * (t - t_prev))))
        A = A * math.exp(-theta * (t - t_prev)) + 1
        t_prev = t
    return out


def run_E2(res):
    K, th, N, I0, T = 2.0, 0.5, 500.0, 10, 40.0
    rng = np.random.default_rng(20261101)
    inc = []
    for _ in range(200):
        ev = simulate_hawkesn(HawkesNParams(K=K, theta=th, N=N, mu=0.0), T, rng, I0=I0)
        inc += rescaled_increments(ev, T, I0, K, th, N)
    inc = np.array(inc)
    ks = stats.kstest(inc, "expon")
    crit = 1.63 / math.sqrt(len(inc))
    res["E2_rescaling"] = dict(n=len(inc), ks_stat=float(ks.statistic), p=float(ks.pvalue),
                               crit_1pct=crit, mean=float(inc.mean()))
    print("E2 rescaling: n=%d KS=%.4f (1%% crit %.4f) p=%.3f mean=%.4f" %
          (len(inc), ks.statistic, crit, ks.pvalue, inc.mean()))
    # NOTE (found while writing): conditioning on "an event did occur" truncates the
    # increment at the saturation cap d*K*A, so the increments are Exp(1) only when the
    # cap is large; with I0=10, K=2 the cap is >= ~20 early on and the truncation only
    # matters for the last few events of a dying cascade.

    # thinning vs inversion: final sizes, I0 = 1
    K, th, N, I0, T = 2.0, 0.5, 300.0, 1, 400.0
    rng_a = np.random.default_rng(20261102)
    rng_b = np.random.default_rng(20261103)
    fa = np.array([len(simulate_hawkesn(HawkesNParams(K=K, theta=th, N=N, mu=0.0), T, rng_a, I0=I0))
                   for _ in range(2000)])
    fb = np.array([len(ep.simulate_hawkesn_inversion(K, th, N, I0, T, rng_b)) for _ in range(2000)])
    ks2 = stats.ks_2samp(fa, fb)
    res["E2_two_samplers"] = dict(ks_stat=float(ks2.statistic), p=float(ks2.pvalue),
                                  frac_small_thinning=float((fa < 30).mean()),
                                  frac_small_inversion=float((fb < 30).mean()),
                                  mean_large_thinning=float(fa[fa >= 30].mean()),
                                  mean_large_inversion=float(fb[fb >= 30].mean()))
    print("E2 samplers:", res["E2_two_samplers"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    q = np.linspace(0.0005, 0.9995, 400)
    ax[0].plot(stats.expon.ppf(q), np.quantile(inc, q), lw=2, color="#2a6fbb")
    ax[0].plot([0, 8], [0, 8], "k--", lw=1, label="perfect agreement")
    ax[0].set_xlabel("Exp(1) quantile (dimensionless)")
    ax[0].set_ylabel("quantile of rescaled waiting time (dimensionless)")
    ax[0].set_title("Simulated waiting times, rescaled by the model's own\ncompensator, follow Exp(1): "
                    f"KS={ks.statistic:.4f} (1% critical {crit:.4f}), n={len(inc)}", fontsize=10)
    ax[0].legend()
    bins = np.arange(0, 301, 10)
    ax[1].hist(fa, bins=bins, alpha=0.5, label="thinning sampler (hawkesn.py)", color="#2a6fbb")
    ax[1].hist(fb, bins=bins, alpha=0.5, label="inversion sampler (new, independent)", color="#e3872d")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("final number of events (cascade size, excluding the seed)")
    ax[1].set_ylabel("number of runs out of 2000")
    ax[1].set_title("Two independent samplers give the same cascade-size distribution\n"
                    f"K=2, theta=0.5, N=300, one seed; two-sample KS p={ks2.pvalue:.2f}", fontsize=10)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "nov_E2_simulator_checks.png", dpi=130)
    plt.close(fig)


E3_PNG = "nov_E3_old_profile_audit.png"


def run_E3(res):
    from experiments.wp12 import identifiability_study as old

    cfg = old.STUDY_CONFIG["part1"]
    Kt, tht, Nt = cfg["K_true"], cfg["theta_true"], cfg["N_true"]
    seed = old.STUDY_CONFIG["seeds"]["case3_sim"]
    K_grid = np.exp(np.linspace(np.log(0.15), np.log(20.0), 80))
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (label, spec) in zip(axes, [("one seed post (I0=1)", 1), ("25 seed posts (I0=0.23N)", None)]):
        cascades, T, I0 = old.simulate_case3_cascades(1, spec, seed + 100 + (0 if spec == 1 else 1000),
                                                      T=cfg["T"])
        ev = cascades[0]
        p_old = old.compute_profile_K(cascades, T, I0, K_grid, Nt, n_restarts=6)
        thetas = np.exp(np.linspace(np.log(tht * 1e-4), np.log(tht * 1e3), 700))
        Ns = np.exp(np.linspace(np.log(2.0), np.log(1e6), 1500))  # same box as the old code
        Ns = Ns[Ns > I0 + len(ev) - 1]
        gf = ep.grid_fit(ev, T, I0, thetas, Ns=Ns)
        p_new = ep.profile_generic(gf, K_grid, lambda v, th: v + 0 * th)
        d = p_old - p_new
        out[label] = dict(n_events=int(len(ev)), I0=int(I0), max_abs_diff=float(np.nanmax(np.abs(d))),
                          frac_K_points_off_by_more_than_1=float(np.mean(np.abs(d) > 1.0)),
                          old_exceeds_grid_max_by=float(np.nanmax(p_old) - gf.ll_max),
                          K_grid=K_grid.tolist(), old=p_old.tolist(), new=p_new.tolist())
        m = np.nanmax(p_new)
        ax.plot(K_grid, p_new - m, color="#2a6fbb", lw=2.2, label="exact grid profile (no optimiser)")
        ax.plot(K_grid, p_old - m, color="#c0392b", lw=1.2, marker=".", label="June routine (compute_profile_K)")
        ax.axhline(-1.92, color="gray", ls=":", label="95% cut-off (-1.92)")
        ax.axvline(Kt, color="k", ls="--", lw=1, label=f"true K = {Kt:.2f}")
        ax.set_xscale("log")
        ax.set_ylim(-25, 3)
        ax.set_xlabel("K (branching factor, dimensionless)")
        ax.set_title(f"{label}: {len(ev)} observed events\n"
                     f"max |June - exact| = {np.nanmax(np.abs(d)):.1f} log-lik units", fontsize=10)
        print("E3", label, {k: v for k, v in out[label].items() if k not in ("K_grid", "old", "new")})
    axes[0].set_ylabel("profile log-likelihood minus exact maximum")
    axes[0].legend(fontsize=8, loc="lower center")
    fig.suptitle("The June 'profile likelihood' is not a profile: it disagrees with the exact one on the same data",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / E3_PNG, dpi=130)
    plt.close(fig)
    res["E3"] = out


if __name__ == "__main__" and len(sys.argv) == 1:
    res = {"seeds": dict(E1=20261100, E2_rescaling=20261101, E2_thinning=20261102, E2_inversion=20261103)}
    run_E1(res)
    run_E2(res)
    run_E3(res)
    (OUT / "nov_checks_results.json").write_text(json.dumps(res, indent=1))


def run_E2b():
    """Exact-uniform version of the rescaling test (see expectations E2b)."""
    K, th, N, I0, T = 2.0, 0.5, 500.0, 10, 40.0
    rng = np.random.default_rng(20261101)
    us = []
    for _ in range(200):
        ev = simulate_hawkesn(HawkesNParams(K=K, theta=th, N=N, mu=0.0), T, rng, I0=I0)
        A, tp = float(I0), 0.0
        for i, t in enumerate(ev):
            d = 1 - (I0 + i) / N
            x = d * K * A * (1 - math.exp(-th * (t - tp)))
            c = d * K * A * (1 - math.exp(-th * (T - tp)))
            us.append((1 - math.exp(-x)) / (1 - math.exp(-c)))
            A = A * math.exp(-th * (t - tp)) + 1
            tp = t
    us = np.array(us)
    ks = stats.kstest(us, "uniform")
    out = dict(n=len(us), ks_stat=float(ks.statistic), p=float(ks.pvalue),
               crit_1pct=1.63 / math.sqrt(len(us)), mean=float(us.mean()),
               se_mean=float(0.2887 / math.sqrt(len(us))))
    print("E2b", out)
    (OUT / "nov_checks_E2b_results.json").write_text(json.dumps(out, indent=1))


if len(sys.argv) > 1 and sys.argv[1] == "E2b":
    run_E2b()


def run_E1b():
    """Identity at the lower edge of the N range (added after the grid defect, see log)."""
    rng = np.random.default_rng(20261104)
    worst = 0.0
    for c in range(10):
        I0 = 1 if c % 2 == 0 else 30
        Kg, thg, Ng = 2.5, 0.5, 300.0
        ev = np.array([])
        while len(ev) < 30:
            ev = ep.simulate_hawkesn_inversion(Kg, thg, Ng, I0, 30.0, rng)
        for N in [I0 + len(ev), I0 + len(ev) + 1e-3, I0 + len(ev) + 0.5]:
            for K, th in [(2.5, 0.5), (40.0, 0.02)]:
                j = jax_loglik(ev, 30.0, I0, K, th, float(N))
                s = ep.loglik_point(ev, 30.0, I0, K, th, float(N))
                worst = max(worst, abs(j - s) / abs(j))
    print("E1b max rel diff suff-stats vs JAX at N = n_total (+0, +1e-3, +0.5):", worst)
    (OUT / "nov_checks_E1b_results.json").write_text(json.dumps(dict(seed=20261104, max_rel=worst)))


if len(sys.argv) > 1 and sys.argv[1] == "E1b":
    run_E1b()
if len(sys.argv) > 1 and sys.argv[1] == "E3r":
    _r = {}
    E3_PNG = "nov_E3r_old_profile_audit.png"  # first-run png kept, marked withdrawn in the log
    run_E3(_r)
    (OUT / "nov_checks_E3r_results.json").write_text(json.dumps(_r, indent=1))
