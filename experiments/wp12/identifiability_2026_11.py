"""E4 / E5 of experiments/WP-12/expectations-2026-11.md.

Single-cascade identifiability of HawkesN (mu = 0, I0 known) by exact, optimiser-free
grid profiles, and a comparison with the existing L-BFGS fitter `hawkesn.fit_hawkesn`.

  examples   one cascade per scenario: profile surface over (K, theta) + 1-D profiles
  window     the same cascade observed for longer and longer
  replicate  200 cascades per scenario: point estimates, 95% profile-interval coverage,
             fitter-vs-grid log-likelihood gap, Wald-interval coverage

Run (about 4 worker processes):
  JAX_PLATFORMS=cpu uv run python experiments/wp12/identifiability_2026_11.py examples
  JAX_PLATFORMS=cpu uv run python experiments/wp12/identifiability_2026_11.py window
  JAX_PLATFORMS=cpu uv run python experiments/wp12/identifiability_2026_11.py replicate
Outputs: experiments/WP-12/nov_ident_*.{png,json}
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "WP-12"

from experiments.wp12 import exact_profile as ep  # noqa: E402

BASE_SEED = 20261110
SCENARIOS = {
    # name: K, theta, N, I0, T, min_events, description
    "P": dict(K=5.0, theta=0.2, N=1300.0, I0=300, T=11.0, min_events=0,
              desc="paper tutorial setting: 300 seed posts, N=1300, K=5"),
    "M1": dict(K=1.7, theta=0.4, N=110.0, I0=1, T=75.0, min_events=20,
               desc="meme-like, small: one seed post, N=110, K=1.7"),
    "M2": dict(K=1.7, theta=0.4, N=1000.0, I0=1, T=75.0, min_events=20,
               desc="meme-like, medium: one seed post, N=1000, K=1.7"),
    "M3": dict(K=3.0, theta=0.4, N=1000.0, I0=1, T=75.0, min_events=20,
               desc="meme-like, strong: one seed post, N=1000, K=3"),
}
SCN_INDEX = {k: i for i, k in enumerate(SCENARIOS)}
QUANT = ["K", "theta", "N", "beta", "r"]
QLABEL = {"K": "K  (offspring per post)", "theta": "theta  (decay rate, 1/time)",
          "N": "N  (population size)", "beta": "K*theta  (SIR beta)",
          "r": "theta*(K-1)  (early growth rate)"}
COL = {"K": "#257de0", "theta": "#f47230", "N": "#00b894", "beta": "#6a4ce2", "r": "#ec4647"}
LS = {"K": "-", "theta": "--", "N": "-.", "beta": ":", "r": (0, (5, 1, 1, 1))}


def truth(s):
    return dict(K=s["K"], theta=s["theta"], N=s["N"], beta=s["K"] * s["theta"],
                r=s["theta"] * (s["K"] - 1))


def draw_cascade(name: str, rep: int):
    s = SCENARIOS[name]
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, SCN_INDEX[name], rep]))
    n_tries = 0
    while True:
        n_tries += 1
        ev = ep.simulate_hawkesn_inversion(s["K"], s["theta"], s["N"], s["I0"], s["T"], rng)
        if len(ev) >= s["min_events"]:
            return ev, n_tries


def theta_grid(theta_true, n=400):
    return np.exp(np.linspace(math.log(theta_true * 1e-3), math.log(theta_true * 1e3), n))


def all_profiles(gf, tr, nv=160):
    """1-D profiles (relative to the grid maximum) and 95% intervals for the five quantities."""
    th = gf.ss.thetas
    out = {}
    Kv = np.exp(np.linspace(math.log(0.2), math.log(2000.0), nv + 60))
    out["K"] = (Kv, ep.profile_generic(gf, Kv, lambda v, t: v + 0 * t))
    bv = tr["beta"] * np.exp(np.linspace(math.log(0.05), math.log(20), nv))
    out["beta"] = (bv, ep.profile_generic(gf, bv, lambda v, t: v / t))
    rv = tr["r"] * np.exp(np.linspace(math.log(0.05), math.log(20), nv))
    out["r"] = (rv, ep.profile_generic(gf, rv, lambda v, t: 1 + v / t))
    out["theta"] = (th, ep.profile_theta(gf))
    out["N"] = (gf.Ns, ep.profile_N(gf))
    iv = {q: ep.interval_from_profile(x, p, ll_max=gf.ll_max) for q, (x, p) in out.items()}
    return out, iv


def old_fitter(ev, T, I0, **kw):
    from experiments.wp12.hawkesn import fit_hawkesn

    p, info = fit_hawkesn(ev, T=T, mu_fixed=0.0, I0=I0, **kw)
    return p, info


# ---------------------------------------------------------------------------
# examples (E4)
# ---------------------------------------------------------------------------


def run_examples():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    for name, s in SCENARIOS.items():
        tr = truth(s)
        ev, _ = draw_cascade(name, 0)
        T, I0 = s["T"], s["I0"]
        gf = ep.grid_fit(ev, T, I0, theta_grid(s["theta"], 500))
        profs, iv = all_profiles(gf, tr)
        p_old, info = old_fitter(ev, T, I0)
        gap = gf.ll_max - (-info["neg_log_likelihood"])
        Ks = np.exp(np.linspace(math.log(0.3), math.log(300.0), 220))
        surf = ep.surface_K_theta(gf, Ks) - gf.ll_max
        results[name] = dict(
            scenario=s, n_events=int(len(ev)), grid_mle=gf.mle, ll_max=gf.ll_max, intervals=iv,
            old_fitter=dict(K=p_old.K, theta=p_old.theta, N=p_old.N,
                            loglik=-info["neg_log_likelihood"], gap_below_grid_max=gap,
                            success=info["success"]),
        )
        print(name, "n=%d" % len(ev), "grid MLE", {k: round(v, 4) for k, v in gf.mle.items() if k in ("K", "theta", "N")},
              "| old fitter K=%.3f th=%.4f N=%.1f gap=%.3f" % (p_old.K, p_old.theta, p_old.N, gap))
        for q in QUANT:
            print("   %-5s true %.4g  95%% [%.4g%s, %.4g%s]" % (
                q, tr[q], iv[q]["lo"], "(open)" if iv[q]["lo_open"] else "",
                iv[q]["hi"], "(open)" if iv[q]["hi_open"] else ""))

        fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
        # (a) data
        tt = np.concatenate([[0], ev, [T]])
        cc = I0 + np.concatenate([[0], np.arange(1, len(ev) + 1), [len(ev)]])
        ax[0].step(tt, cc, where="post", color="#257de0", lw=2)
        ax[0].axhline(s["N"], color="#666666", ls="--", lw=1)
        ax[0].text(T * 0.98, s["N"], "population N = %d" % s["N"], ha="right", va="bottom", fontsize=9, color="#444")
        ax[0].set_xlabel("time since first post (model time units; 1/theta = %.1f)" % (1 / s["theta"]))
        ax[0].set_ylabel("cumulative posts (including %d seed post%s)" % (I0, "" if I0 == 1 else "s"))
        ax[0].set_title("The data: one simulated cascade, %d events\n%s" % (len(ev), s["desc"]), fontsize=10)
        ax[0].grid(alpha=0.25)
        # (b) surface
        th = gf.ss.thetas
        sel = (th > s["theta"] * 0.01) & (th < s["theta"] * 30)
        X, Y = np.meshgrid(Ks, th[sel], indexing="ij")
        pcm = ax[1].pcolormesh(X, Y, np.clip(surf[:, sel], -12, 0), cmap="Blues", shading="auto", vmin=-12, vmax=0)
        cs = ax[1].contour(X, Y, surf[:, sel], levels=[-3.0], colors=["#191919"], linewidths=1.6)
        ax[1].clabel(cs, fmt={-3.0: "95% joint region"}, fontsize=8)
        kk = np.exp(np.linspace(math.log(0.3), math.log(300), 200))
        ax[1].plot(kk, tr["beta"] / kk, color="#6a4ce2", ls=":", lw=1.8, label="K*theta = true value")
        kk2 = kk[kk > 1.02]
        ax[1].plot(kk2, tr["r"] / (kk2 - 1), color="#ec4647", ls=(0, (5, 1, 1, 1)), lw=1.5,
                   label="theta*(K-1) = true value")
        ax[1].plot(s["K"], s["theta"], marker="*", ms=16, color="#f2a600", mec="k", ls="none", label="truth")
        ax[1].plot(gf.mle["K"], gf.mle["theta"], marker="o", ms=8, color="white", mec="k", ls="none",
                   label="exact maximum (grid)")
        ax[1].plot(p_old.K, p_old.theta, marker="x", ms=10, mew=2.5, color="#ec4647", ls="none",
                   label="existing fitter's answer")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlim(0.3, 300); ax[1].set_ylim(th[sel][0], th[sel][-1])
        ax[1].set_xlabel("K  (offspring per post, dimensionless)")
        ax[1].set_ylabel("theta  (decay rate, 1/time)")
        fig.colorbar(pcm, ax=ax[1], label="log-likelihood below maximum (N optimised out), clipped at -12")
        ax[1].legend(fontsize=7.5, loc="lower left")
        ax[1].set_title("Likelihood surface over (K, theta)\ndark = supported by the data", fontsize=10)
        # (c) 1-D profiles
        for q in QUANT:
            x, p = profs[q]
            ax[2].plot(x / tr[q], p - gf.ll_max, color=COL[q], ls=LS[q], lw=2, label=QLABEL[q])
        ax[2].axhline(-ep.CHI2_1_95, color="#666", lw=1)
        ax[2].text(0.052, -ep.CHI2_1_95 + 0.08, "95% cut-off: values whose curve is above this line are in the interval",
                   fontsize=7.5, color="#444")
        ax[2].axvline(1, color="k", lw=0.8)
        ax[2].set_xscale("log"); ax[2].set_xlim(0.05, 200); ax[2].set_ylim(-8, 0.3)
        ax[2].set_xlabel("candidate value / true value  (1 = truth)")
        ax[2].set_ylabel("profile log-likelihood below maximum")
        ax[2].legend(fontsize=8, loc="lower right")
        ax[2].grid(alpha=0.25)
        fig.suptitle("PLACEHOLDER", fontsize=12)
        fig._wp12_axes = ax
        results[name]["_fig"] = fig
    return results


def finish_examples(results, titles):
    import matplotlib.pyplot as plt

    for name, r in results.items():
        fig = r.pop("_fig")
        fig._wp12_axes[2].set_title("One-parameter profiles: a narrow curve = well determined", fontsize=10)
        fig.suptitle(titles[name], fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f"nov_ident_example_{name}.png", dpi=125)
        plt.close(fig)
    (OUT / "nov_ident_examples.json").write_text(json.dumps(results, indent=1, default=float))


def auto_title(name, r):
    iv = r["intervals"]
    tr = truth(SCENARIOS[name])

    def w(q):
        if iv[q]["lo_open"] or iv[q]["hi_open"]:
            return "%s unbounded" % q
        return "%s x%.1f" % (q, iv[q]["hi"] / iv[q]["lo"])

    return ("Scenario %s (%d events). 95%% interval width as a ratio upper/lower: " % (name, r["n_events"])
            + ", ".join(w(q) for q in QUANT)
            + ".  Existing fitter is %.2f log-lik units below the true maximum." % r["old_fitter"]["gap_below_grid_max"])


# ---------------------------------------------------------------------------
# window series
# ---------------------------------------------------------------------------


def run_window():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fracs = [0.10, 0.25, 0.50, 0.75, 1.00]
    res = {}
    for name in ["M2", "M3"]:
        s = SCENARIOS[name]
        tr = truth(s)
        ev, _ = draw_cascade(name, 0)
        rows = []
        cuts = [(f, float(ev[int(round(f * len(ev))) - 1])) for f in fracs] + [("3x", 3 * float(ev[-1]))]
        # for the 3x window the cascade must be simulated that far: re-draw is not allowed
        # (different data), so only use it if it lies inside the simulated horizon.
        for f, T in cuts:
            if T > s["T"]:
                # the simulation stopped at s["T"]; events beyond were never generated
                T = s["T"]
                f = "to T=%g" % s["T"]
            sub = ev[ev <= T]
            gf = ep.grid_fit(sub, T, s["I0"], theta_grid(s["theta"], 500))
            _, iv = all_profiles(gf, tr)
            p_old, info = old_fitter(sub, T, s["I0"])
            rows.append(dict(frac=f, T=T, n=int(len(sub)), intervals=iv, mle=gf.mle,
                             old=dict(K=p_old.K, theta=p_old.theta, N=p_old.N,
                                      gap=gf.ll_max + info["neg_log_likelihood"])))
            print(name, f, "T=%.2f n=%d" % (T, len(sub)),
                  {q: ("%.3g%s-%.3g%s" % (iv[q]["lo"], "*" if iv[q]["lo_open"] else "", iv[q]["hi"],
                                           "*" if iv[q]["hi_open"] else "")) for q in QUANT},
                  "old gap %.3f" % rows[-1]["old"]["gap"])
        res[name] = dict(scenario=s, n_total=int(len(ev)), rows=rows)

        fig, axes = plt.subplots(1, 5, figsize=(19, 4.6), sharey=True)
        for ax, q in zip(axes, QUANT):
            for i, row in enumerate(rows):
                iv = row["intervals"][q]
                lo, hi = iv["lo"] / tr[q], iv["hi"] / tr[q]
                ax.plot([i, i], [max(lo, 1e-2), min(hi, 1e2)], color=COL[q], lw=5, solid_capstyle="butt")
                if iv["hi_open"] or hi > 1e2:
                    ax.annotate("", xy=(i, 1.6e2), xytext=(i, 0.9e2), arrowprops=dict(arrowstyle="->", color=COL[q], lw=2))
                if (iv["lo_open"] and q != "N") or lo < 1e-2:  # N has a hard lower bound (the observed count), not an open end
                    ax.annotate("", xy=(i, 0.6e-2), xytext=(i, 1.1e-2), arrowprops=dict(arrowstyle="->", color=COL[q], lw=2))
                mode = iv["mode"] / tr[q]
                ax.plot(i, min(max(mode, 1e-2), 1e2), "o", ms=6, color="white", mec="#191919")
            ax.axhline(1, color="k", lw=0.8)
            ax.set_yscale("log"); ax.set_ylim(0.5e-2, 2e2)
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels(["%s\n(%d ev.)" % (("%d%%" % (100 * r["frac"])) if not isinstance(r["frac"], str) else r["frac"], r["n"])
                                for r in rows], fontsize=8)
            ax.set_title(QLABEL[q], fontsize=10)
            ax.grid(alpha=0.25, axis="y")
            ax.set_xlabel("share of the cascade's events inside the window")
        axes[0].set_ylabel("95% interval / true value  (log scale; 1 = truth)\narrow = no bound inside the search box; dot = best estimate")
        res[name]["_fig"] = fig
    return res


# ---------------------------------------------------------------------------
# replicates (E5)
# ---------------------------------------------------------------------------


def one_replicate(args):
    name, rep = args
    s = SCENARIOS[name]
    tr = truth(s)
    ev, n_tries = draw_cascade(name, rep)
    T, I0 = s["T"], s["I0"]
    gf = ep.grid_fit(ev, T, I0, theta_grid(s["theta"], 400))
    _, iv = all_profiles(gf, tr, nv=120)
    cov = ep.covers(gf, s["K"], s["theta"], s["N"])
    ll_true = ep.loglik_point(ev, T, I0, s["K"], s["theta"], s["N"])
    row = dict(scenario=name, rep=rep, n=int(len(ev)), n_tries=n_tries, ll_max_grid=gf.ll_max,
               ll_true=ll_true,
               K=gf.mle["K"], theta=gf.mle["theta"], N=gf.mle["N"],
               theta_at_edge=gf.mle["theta_at_edge"], N_at_edge=gf.mle["N_at_edge"],
               cover=cov, intervals=iv,
               prof_at_truth=ep.profile_at_truth(gf, s["K"], s["theta"], s["N"]),
               box20=ep.box_max(gf, 20.0), box10=ep.box_max(gf, 10.0))
    # existing fitter, default settings
    try:
        p, info = old_fitter(ev, T, I0)
        ll_old = -info["neg_log_likelihood"]
        c = np.asarray(info["covariance"])
        est = np.log([p.K, p.theta, p.N])
        se = np.sqrt(np.clip(np.diag(c), 0, None))
        se_beta = math.sqrt(max(c[0, 0] + c[1, 1] + 2 * c[0, 1], 0))
        tv = np.log([s["K"], s["theta"], s["N"]])
        wald = {q: bool(abs(est[i] - tv[i]) <= 1.96 * se[i]) for i, q in enumerate(["K", "theta", "N"])}
        wald["beta"] = bool(abs(est[0] + est[1] - math.log(tr["beta"])) <= 1.96 * se_beta)
        row["old"] = dict(K=p.K, theta=p.theta, N=p.N, ll=ll_old, success=bool(info["success"]),
                          se_logK=float(se[0]), se_logtheta=float(se[1]), se_logN=float(se[2]),
                          wald_cover=wald)
        # same optimiser started at the grid maximum: separates "bad start" from "bad optimiser"
        from experiments.wp12.hawkesn import HawkesNParams

        Kc = min(max(gf.mle["K"], 0.011), 19.9)
        p2, info2 = old_fitter(ev, T, I0, n_restarts=1,
                               initial_params=HawkesNParams(K=Kc, theta=gf.mle["theta"], N=gf.mle["N"], mu=0.0))
        row["old_warm"] = dict(K=p2.K, theta=p2.theta, N=p2.N, ll=-info2["neg_log_likelihood"])
    except Exception as e:  # recorded, not swallowed
        row["old_error"] = repr(e)
    return row


def run_replicates(n_rep=200, n_proc=4):
    """Checkpointed: each finished replicate is appended to a .jsonl file, and a restart
    skips what is already there (the machine rebooted twice during this session)."""
    import multiprocessing as mp

    ck = OUT / "nov_ident_replicates.jsonl"
    done = set()
    if ck.exists():
        for line in ck.read_text().splitlines():
            r = json.loads(line)
            done.add((r["scenario"], r["rep"]))
    tasks = [(name, rep) for name in SCENARIOS for rep in range(1, n_rep + 1)  # rep 0 = the example
             if (name, rep) not in done]
    print("resuming: %d done, %d to do" % (len(done), len(tasks)), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(n_proc) as pool, open(ck, "a") as fh:
        for i, row in enumerate(pool.imap_unordered(one_replicate, tasks, chunksize=2)):
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            if (i + 1) % 50 == 0:
                print("  %d/%d  %.0fs" % (i + 1, len(tasks), time.time() - t0), flush=True)
    rows = [json.loads(line) for line in ck.read_text().splitlines()]
    rows.sort(key=lambda r: (r["scenario"], r["rep"]))
    (OUT / "nov_ident_replicates.json").write_text(json.dumps(
        dict(base_seed=BASE_SEED, seed_scheme="SeedSequence([base_seed, scenario_index, rep])",
             scenarios=SCENARIOS, rows=rows), default=float))
    print("done in %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "examples":
        r = run_examples()
        finish_examples(r, {k: auto_title(k, v) for k, v in r.items()})
    elif mode == "window":
        import matplotlib.pyplot as plt

        r = run_window()
        for name, v in r.items():
            fig = v.pop("_fig")
            fig.suptitle("Scenario %s, one cascade of %d events observed for longer and longer: which quantities are pinned down when"
                         % (name, v["n_total"]), fontsize=12)
            fig.tight_layout()
            fig.savefig(OUT / f"nov_ident_window_{name}.png", dpi=125)
            plt.close(fig)
        (OUT / "nov_ident_window.json").write_text(json.dumps(r, indent=1, default=float))
    elif mode == "replicate":
        run_replicates(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
