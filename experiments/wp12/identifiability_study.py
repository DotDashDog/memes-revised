"""WP-12 identifiability study — Parts 1, 2, 3.

PRE-REGISTERED HYPOTHESES (written before running):

Part 1 — Profile-likelihood diagnostic:
  Hypothesis:  The (K,θ) likelihood surface for Phase-A case 3 (K=1.73, θ=0.41,
               N=109, I₀=1, near-critical) has a ridge along Kθ=const making K
               not individually identifiable from a single cascade. The boundary-
               pinning observed in Phase A is a symptom of structural degeneracy,
               not an optimizer pathology.
  Prediction:  Profile ℓ(K) will be flat or gently sloped on a single cascade;
               profile ℓ(β=Kθ) will be sharp.  Pooling 5/10/20 cascades will
               sharpen the K profile proportional to √(n_cascades).  At
               I₀/N=0.23 (paper conditions) profiles will be sharper than at
               I₀=1 for the same number of cascades.
  Falsifier:   A sharp interior K-peak on a SINGLE cascade → optimizer pathology
               (not data). K-profile flat even at 20 pooled full cascades at
               I₀/N=0.23 → structural degeneracy (K and β not separable from
               exponential kernel + finite-population depletion).

Part 2 — Paper-matched recovery:
  Hypothesis:  Fitting 20 cascades separately under paper params (N=1300,
               I₀=300, K=5, θ=0.2, T=11) recovers K and N well but NOT θ
               individually (matching upstream θ-sd=8.52). β=Kθ is recovered.
  Prediction:  Median K ≈ 5 ± 2 (upstream reported 5.5 ± 2.1), N ≈ 1292 ± 11,
               θ wide sd; β well-recovered.
  Falsifier:   Median K error >60% OR median N error >10% → implementation
               divergence; audit before proceeding.

Part 3 — Information scaling (only if Part 1 gate passes):
  Hypothesis:  K profile-interval width scales as 1/√(n_cascades); I₀/N=0.23
               gives substantially narrower intervals than I₀=1.
  Prediction:  log-log slope of profile-K interval width vs n_cascades ≈ −0.5.
               At N=1300, 20 pooled cascades, full obs, interval width at
               I₀/N=0.23 ≤ half that at I₀=1/N.
  Falsifier:   K profile-width does NOT decrease with I₀/N → extinction not the
               driver; diagnose optimizer/bounds.

Gate: Part 3 runs ONLY if Part 1 finds the single-cascade K-profile is genuinely
flat (not a sharp interior peak missed by optimizer).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar, minimize

jax.config.update("jax_enable_x64", True)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.wp12.hawkesn import (
    HawkesNParams,
    hawkesn_log_likelihood,
    simulate_hawkesn,
    fit_hawkesn,
)

WP12_OUTPUT = _PROJECT_ROOT / "experiments" / "WP-12"

# ---------------------------------------------------------------------------
# Seeds and config (reproducibility)
# ---------------------------------------------------------------------------

STUDY_CONFIG: dict[str, Any] = {
    "run_id": "wp12-identifiability-001",
    "date": "2026-06-11",
    "seeds": {
        "case3_sim": 7777,          # seed for re-simulating case-3 cascades
        "paper_match_sim": 8888,    # seed for paper-matched 20 cascades
        "part3_sim": 9999,          # seed for Part 3 scaling grid
    },
    "part1": {
        # Phase-A case 3 true params (from synthetic_recovery_results.json)
        "K_true": 1.7269310030189216,
        "theta_true": 0.4108074885956497,
        "N_true": 109.0,
        "I0_values": [1, None],   # None means I0 = int(0.23*N)
        "n_cascade_values": [1, 5, 10, 20],
        "T": 11.0 / 0.4108074885956497,  # approx T = 10/theta (≈ 26.8)
        "n_K_grid": 60,
        "n_theta_grid": 60,
        "n_N_grid": 40,
        "K_grid": [0.2, 25.0],
        "theta_grid": [0.01, 5.0],
        "N_grid_factor": [0.3, 3.0],   # relative to N_true
        "n_restarts_profile": 10,
    },
    "part2": {
        # Paper params (Rizoiu et al. 2018)
        "K_true": 5.0,
        "theta_true": 0.2,
        "N_true": 1300.0,
        "I0": 300,
        "T": 11.0,
        "n_cascades": 20,
        "n_restarts": 10,
    },
    "part3": {
        "n_cascade_values": [1, 3, 5, 10, 20],
        "I0_fracs": [None, 0.23],    # None = I0=1; 0.23 = paper fraction
        "N_values": [109, 500, 1300],
        "T_multiples": [2.0, 10.0, 55.0],  # T = multiple / theta
        "K_true": 1.7269310030189216,  # same as case-3
        "theta_true": 0.4108074885956497,
        "n_restarts": 10,
        "n_profile_K_points": 40,
        "K_profile_range": [0.3, 20.0],
    },
}


# ---------------------------------------------------------------------------
# Helper: pooled log-likelihood over multiple cascades
# ---------------------------------------------------------------------------

def pooled_log_likelihood(
    log_K: jax.Array,
    log_theta: jax.Array,
    log_N: jax.Array,
    cascades: list[np.ndarray],
    T: float,
    I0: int,
) -> jax.Array:
    """Sum of HawkesN log-likelihoods over multiple cascades (μ=0 fixed)."""
    log_mu = jnp.array(np.log(1e-12), dtype=jnp.float64)
    total = jnp.zeros((), dtype=jnp.float64)
    for evts in cascades:
        times_jax = jnp.asarray(evts, dtype=jnp.float64)
        ll = hawkesn_log_likelihood(log_K, log_theta, log_N, log_mu, times_jax, T, I0=I0)
        total = total + ll
    return total


def pooled_nll_fn(cascades, T, I0, fix_N=None):
    """Return a scipy-compatible (value, grad) function over (log_K, log_theta[, log_N])."""
    fix_n = fix_N is not None

    def _nll_jax(v):
        log_K = v[0]
        log_theta = v[1]
        log_N = jnp.log(jnp.array(fix_N, dtype=jnp.float64)) if fix_n else v[2]
        return -pooled_log_likelihood(log_K, log_theta, log_N, cascades, T, I0)

    _nll_and_grad = jax.jit(jax.value_and_grad(_nll_jax))

    def scipy_obj(v_np):
        v_jax = jnp.asarray(v_np, dtype=jnp.float64)
        val, g = _nll_and_grad(v_jax)
        return float(val), np.asarray(g, dtype=np.float64)

    return scipy_obj


def fit_pooled(cascades, T, I0, n_restarts=10, fix_N=None):
    """MLE for (K, theta, N) jointly over all cascades; μ fixed at 0."""
    fix_n = fix_N is not None
    scipy_obj = pooled_nll_fn(cascades, T, I0, fix_N=fix_N)

    # Bounds
    bounds = [
        (np.log(0.01), np.log(100.0)),   # log K
        (np.log(1e-4), np.log(1e4)),     # log theta
    ]
    if not fix_n:
        bounds.append((np.log(2.0), np.log(1e6)))  # log N

    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None

    for i in range(n_restarts):
        if i == 0:
            # Heuristic start
            all_evts = np.concatenate(cascades) if cascades else np.array([])
            theta_init = 1.0 / max(float(np.median(np.diff(all_evts))) if len(all_evts) > 1 else 1.0, 1e-6)
            theta_init = np.clip(theta_init, 1e-4, 1e4)
            N_est = max(len(all_evts) * 2 / len(cascades), 10.0)
            x0 = np.array(
                [np.log(2.0), np.log(theta_init)]
                + ([] if fix_n else [np.log(N_est)]),
                dtype=np.float64,
            )
        else:
            x0 = np.array(
                [np.log(rng.uniform(0.5, 10.0)),
                 np.log(rng.uniform(0.05, 2.0))]
                + ([] if fix_n else [np.log(rng.uniform(50.0, 2000.0))]),
                dtype=np.float64,
            )
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

        try:
            res = minimize(
                scipy_obj, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-9},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    if best_x is None:
        best_x = np.zeros(2 if fix_n else 3)
    K_fit = float(np.exp(best_x[0]))
    theta_fit = float(np.exp(best_x[1]))
    N_fit = float(fix_N) if fix_n else float(np.exp(best_x[2]))
    return K_fit, theta_fit, N_fit, float(best_val)


# ---------------------------------------------------------------------------
# Part 1: Profile-likelihood
# ---------------------------------------------------------------------------

def compute_profile_K(cascades, T, I0, K_grid, N_true, n_restarts=8):
    """Profile log-likelihood over K, with theta and N optimised out."""
    profile_ll = []
    scipy_obj_full = pooled_nll_fn(cascades, T, I0)
    bounds_sub = [
        (np.log(1e-4), np.log(1e4)),    # log theta
        (np.log(2.0), np.log(1e6)),     # log N
    ]
    rng = np.random.default_rng(42)

    for K_val in K_grid:
        log_K_fixed = np.log(K_val)

        def nll_sub(v_sub):
            v_full = np.array([log_K_fixed, v_sub[0], v_sub[1]])
            return scipy_obj_full(v_full)

        best_val = np.inf
        for i in range(n_restarts):
            if i == 0:
                x0 = np.array([np.log(0.3), np.log(N_true)])
            else:
                x0 = np.array([
                    np.log(rng.uniform(0.05, 3.0)),
                    np.log(rng.uniform(N_true * 0.2, N_true * 5.0)),
                ])
                x0 = np.clip(x0, [b[0] for b in bounds_sub], [b[1] for b in bounds_sub])
            try:
                res = minimize(
                    nll_sub, x0, method="L-BFGS-B", jac=True, bounds=bounds_sub,
                    options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
                )
                if res.fun < best_val:
                    best_val = res.fun
            except Exception:
                pass
        profile_ll.append(-best_val)

    return np.array(profile_ll)


def compute_profile_beta(cascades, T, I0, beta_grid, N_true, n_restarts=8):
    """Profile log-likelihood over β=Kθ, with theta and N optimised (K=β/θ)."""
    profile_ll = []
    rng = np.random.default_rng(42)

    for beta_val in beta_grid:
        # For each (theta, N) compute K = beta/theta, evaluate pooled LL
        def nll_beta(v):
            log_theta, log_N = v[0], v[1]
            theta_v = np.exp(log_theta)
            K_v = beta_val / theta_v
            if K_v <= 0.001 or K_v > 500:
                return (np.inf, np.zeros(2))
            log_K_v = np.log(K_v)
            v_full = jnp.array([log_K_v, log_theta, log_N], dtype=jnp.float64)
            log_mu = jnp.array(np.log(1e-12), dtype=jnp.float64)
            total_ll = jnp.zeros((), dtype=jnp.float64)
            for evts in cascades:
                times_jax = jnp.asarray(evts, dtype=jnp.float64)
                total_ll = total_ll + hawkesn_log_likelihood(
                    log_K_v, log_theta, log_N, log_mu, times_jax, T, I0=I0)
            # Compute gradient w.r.t (log_theta, log_N) only
            return float(-total_ll), np.zeros(2)

        def nll_beta_scalar(v):
            log_theta, log_N = v[0], v[1]
            K_v = beta_val / np.exp(log_theta)
            if K_v <= 0.001 or K_v > 500:
                return np.inf
            log_K_v = jnp.array(np.log(K_v), dtype=jnp.float64)
            ltheta = jnp.array(log_theta, dtype=jnp.float64)
            lN = jnp.array(log_N, dtype=jnp.float64)
            log_mu = jnp.array(np.log(1e-12), dtype=jnp.float64)
            total_ll = jnp.zeros((), dtype=jnp.float64)
            for evts in cascades:
                times_jax = jnp.asarray(evts, dtype=jnp.float64)
                total_ll = total_ll + hawkesn_log_likelihood(
                    log_K_v, ltheta, lN, log_mu, times_jax, T, I0=I0)
            return float(-total_ll)

        bounds_sub = [
            (np.log(1e-4), np.log(1e4)),
            (np.log(2.0), np.log(1e6)),
        ]
        best_val = np.inf
        for i in range(n_restarts):
            if i == 0:
                x0 = np.array([np.log(0.3), np.log(N_true)])
            else:
                x0 = np.array([
                    np.log(rng.uniform(0.05, 3.0)),
                    np.log(rng.uniform(N_true * 0.2, N_true * 5.0)),
                ])
                x0 = np.clip(x0, [b[0] for b in bounds_sub], [b[1] for b in bounds_sub])
            try:
                res = minimize(
                    nll_beta_scalar, x0, method="Nelder-Mead",
                    options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9},
                )
                if res.fun < best_val:
                    best_val = res.fun
            except Exception:
                pass
        profile_ll.append(-best_val)

    return np.array(profile_ll)


def compute_2d_surface_KN(cascades, T, I0, K_grid, N_grid, theta_fixed):
    """2D log-likelihood surface over (K, N) with theta fixed."""
    nK, nN = len(K_grid), len(N_grid)
    ll_surface = np.full((nK, nN), np.nan)
    log_mu = np.log(1e-12)
    log_theta = np.log(theta_fixed)
    for i, K_v in enumerate(K_grid):
        log_K = np.log(K_v)
        for j, N_v in enumerate(N_grid):
            log_N = np.log(N_v)
            total = 0.0
            for evts in cascades:
                times_jax = jnp.asarray(evts, dtype=jnp.float64)
                ll = float(hawkesn_log_likelihood(
                    jnp.array(log_K), jnp.array(log_theta),
                    jnp.array(log_N), jnp.array(log_mu),
                    times_jax, T, I0=I0,
                ))
                total += ll
            ll_surface[i, j] = total
    return ll_surface


def simulate_case3_cascades(n_cascades, I0_val, seed, T=None):
    """Simulate n_cascades using case-3 true params."""
    cfg = STUDY_CONFIG["part1"]
    K_true = cfg["K_true"]
    theta_true = cfg["theta_true"]
    N_true = cfg["N_true"]
    if T is None:
        T = cfg["T"]
    if I0_val is None:
        I0_val = max(1, int(0.23 * N_true))
    params = HawkesNParams(K=K_true, theta=theta_true, N=N_true, mu=0.0)
    rng = np.random.default_rng(seed)
    cascades = []
    for i in range(n_cascades):
        evts = simulate_hawkesn(params, T, rng, I0=I0_val)
        # Ensure non-empty
        for attempt in range(20):
            if len(evts) >= 3:
                break
            evts = simulate_hawkesn(params, T, rng, I0=I0_val)
        cascades.append(evts)
    return cascades, T, I0_val


def run_part1():
    """Part 1: Profile-likelihood diagnostic."""
    print("\n" + "=" * 70)
    print("PART 1 — Profile-likelihood diagnostic")
    print("=" * 70)

    cfg = STUDY_CONFIG["part1"]
    K_true = cfg["K_true"]
    theta_true = cfg["theta_true"]
    N_true = cfg["N_true"]
    seed = STUDY_CONFIG["seeds"]["case3_sim"]
    T_window = cfg["T"]

    print(f"\nCase 3 params: K={K_true:.4f}, θ={theta_true:.4f}, N={N_true:.0f}")
    print(f"β=Kθ = {K_true*theta_true:.4f}, T≈{T_window:.1f}")
    print(f"\nPRE-REGISTERED FALSIFIER:")
    print("  Sharp interior K-peak on single cascade → optimizer pathology")
    print("  Flat K-profile even at 20 pooled cascades → structural degeneracy")
    print("  K-profile sharpens with pooled data → low-information (resolvable)")

    # K grid for profiles
    K_profile_grid = np.exp(np.linspace(np.log(0.15), np.log(20.0), 80))
    beta_profile_grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 60))

    results_part1 = {
        "K_true": K_true, "theta_true": theta_true, "N_true": N_true,
        "K_profile_grid": K_profile_grid.tolist(),
        "beta_profile_grid": beta_profile_grid.tolist(),
        "profiles": {},
        "mle_fits": {},
    }

    fig_profiles, axes_profiles = plt.subplots(2, 4, figsize=(18, 9))
    fig_profiles.suptitle(
        f"Part 1: Profile LL — Case 3 (K={K_true:.2f}, θ={theta_true:.2f}, N={N_true:.0f})\n"
        f"β=Kθ={K_true*theta_true:.3f}, T≈{T_window:.1f}",
        fontsize=12
    )

    col = 0
    for I0_label, I0_val_spec in [("I0=1", 1), ("I0=0.23N", None)]:
        I0_val = I0_val_spec if I0_val_spec is not None else max(1, int(0.23 * N_true))

        profiles_K = {}
        profiles_beta = {}
        mle_list = []

        for n_cas in [1, 5, 10, 20]:
            label = f"n={n_cas}"
            print(f"\n  [{I0_label}, {label}] simulating and computing profiles...")
            t0 = time.time()

            cascades, T_val, I0_used = simulate_case3_cascades(
                n_cas, I0_val_spec, seed + n_cas * 100 + (0 if I0_val_spec == 1 else 1000),
                T=T_window
            )
            n_events_total = sum(len(c) for c in cascades)
            print(f"    Total events: {n_events_total} across {n_cas} cascade(s)")

            # Profile over K (N profiled out by optimizing over theta, N)
            pK = compute_profile_K(cascades, T_val, I0_used, K_profile_grid, N_true, n_restarts=6)
            profiles_K[label] = pK.tolist()

            # Also compute MLE
            K_fit, theta_fit, N_fit, neg_ll = fit_pooled(
                cascades, T_val, I0_used, n_restarts=10
            )
            mle_list.append({
                "n_cascades": n_cas,
                "K_fit": K_fit, "theta_fit": theta_fit, "N_fit": N_fit,
                "beta_fit": K_fit * theta_fit,
                "neg_ll": neg_ll,
                "n_events": n_events_total,
            })
            print(f"    MLE: K={K_fit:.3f} (true={K_true:.3f}), "
                  f"θ={theta_fit:.3f} (true={theta_true:.3f}), "
                  f"N={N_fit:.1f} (true={N_true:.1f}), "
                  f"β={K_fit*theta_fit:.3f} (true={K_true*theta_true:.3f}), "
                  f"time={time.time()-t0:.1f}s")

            profiles_K[label] = pK.tolist()

        # Store
        results_part1["profiles"][I0_label] = {"K": profiles_K, "beta": {}}
        results_part1["mle_fits"][I0_label] = mle_list

        # Plot profile K
        ax = axes_profiles[0, col]
        for lbl, pK in profiles_K.items():
            pK_arr = np.array(pK)
            # Normalise by subtracting max
            pK_norm = pK_arr - np.nanmax(pK_arr)
            ax.plot(K_profile_grid, pK_norm, label=lbl, alpha=0.8)
        ax.axvline(K_true, color="k", ls="--", label=f"K_true={K_true:.2f}")
        ax.axhline(-1.92, color="gray", ls=":", alpha=0.7, label="95% CI (-1.92)")
        ax.set_xlabel("K")
        ax.set_ylabel("Profile ℓ(K) − max")
        ax.set_title(f"Profile ℓ(K): {I0_label}")
        ax.set_xlim([0, 22])
        ax.set_ylim([-20, 2])
        ax.legend(fontsize=7)

        # 2D surface for single cascade at true theta
        ax2 = axes_profiles[1, col]
        cascades_1, T_val, I0_used_1 = simulate_case3_cascades(
            1, I0_val_spec, seed + 50 + (0 if I0_val_spec == 1 else 1000),
            T=T_window
        )
        K_2d_grid = np.exp(np.linspace(np.log(0.2), np.log(20.0), 40))
        N_2d_grid = np.exp(np.linspace(np.log(max(N_true * 0.3, 5)), np.log(N_true * 4), 40))
        print(f"\n  [{I0_label}] Computing 2D (K,N) surface at θ_true...")
        ll_surf = compute_2d_surface_KN(cascades_1, T_val, I0_used_1, K_2d_grid, N_2d_grid, theta_true)
        ll_surf_norm = ll_surf - np.nanmax(ll_surf)
        KK, NN = np.meshgrid(K_2d_grid, N_2d_grid, indexing="ij")
        pcm = ax2.pcolormesh(KK, NN, np.clip(ll_surf_norm, -30, 0), cmap="viridis")
        ax2.axvline(K_true, color="red", ls="--", lw=1.5, label=f"K_true")
        ax2.axhline(N_true, color="orange", ls="--", lw=1.5, label=f"N_true")
        plt.colorbar(pcm, ax=ax2, label="ΔlogL")
        ax2.set_xlabel("K")
        ax2.set_ylabel("N")
        ax2.set_title(f"2D (K,N) surface\n{I0_label}, 1 cascade, θ=θ_true")
        ax2.legend(fontsize=7)

        col += 1

    # Also add profile K comparison across I0 for n=1 and n=20
    for plot_col, n_cas in [(2, 1), (3, 20)]:
        ax = axes_profiles[0, plot_col]
        for I0_label, I0_val_spec in [("I0=1", 1), ("I0=0.23N", None)]:
            pK_list = results_part1["profiles"][I0_label]["K"].get(f"n={n_cas}", [])
            if pK_list:
                pK_arr = np.array(pK_list)
                pK_norm = pK_arr - np.nanmax(pK_arr)
                ax.plot(K_profile_grid, pK_norm, label=I0_label, alpha=0.8)
        ax.axvline(K_true, color="k", ls="--", lw=2, label=f"K_true")
        ax.axhline(-1.92, color="gray", ls=":", alpha=0.7)
        ax.set_xlabel("K")
        ax.set_ylabel("Profile ℓ(K) − max")
        ax.set_title(f"I₀ comparison: n={n_cas} cascade(s)")
        ax.set_xlim([0, 22])
        ax.set_ylim([-20, 2])
        ax.legend(fontsize=7)

        ax2 = axes_profiles[1, plot_col]
        # Plot MLE K estimates vs n_cascades
        for I0_label in ["I0=1", "I0=0.23N"]:
            mles = results_part1["mle_fits"][I0_label]
            ns = [m["n_cascades"] for m in mles]
            Ks = [m["K_fit"] for m in mles]
            ax2.plot(ns, Ks, "o-", label=I0_label)
        ax2.axhline(K_true, color="k", ls="--", label=f"K_true={K_true:.2f}")
        ax2.set_xlabel("# cascades (pooled)")
        ax2.set_ylabel("MLE K")
        ax2.set_title("MLE K vs # pooled cascades")
        ax2.legend(fontsize=7)
        break  # only do this once for the last 2 cols

    plt.tight_layout()
    prof_plot_path = WP12_OUTPUT / "part1_profiles.png"
    fig_profiles.savefig(prof_plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig_profiles)
    print(f"\n  Part 1 profiles plot saved: {prof_plot_path}")

    # Save results
    results_part1_json_path = WP12_OUTPUT / "part1_results.json"
    results_part1_json_path.write_text(json.dumps(results_part1, indent=2))
    print(f"  Part 1 results saved: {results_part1_json_path}")

    # Verdict: is K-profile flat on single cascade?
    # Check if the 95% CI interval is wide (>5x K_true range)
    pK_single_I0_1 = np.array(results_part1["profiles"]["I0=1"]["K"].get("n=1", []))
    verdict_flat = True
    if len(pK_single_I0_1) > 0:
        pK_norm = pK_single_I0_1 - np.nanmax(pK_single_I0_1)
        above_95 = K_profile_grid[pK_norm > -1.92]
        if len(above_95) > 1:
            ci_width = above_95[-1] - above_95[0]
            print(f"\n  Single-cascade 95% K-CI width: {ci_width:.2f} (K_true={K_true:.2f})")
            # A 'sharp' peak would be narrower than K_true/2
            if ci_width < K_true / 2:
                verdict_flat = False
                print("  GATE RESULT: SHARP PEAK on single cascade → optimizer pathology suspected")
            else:
                print("  GATE RESULT: FLAT/BROAD profile on single cascade → structural degeneracy")
        else:
            print("  GATE RESULT: Profile entirely above threshold → flat ridge (degenerate)")

    return results_part1, verdict_flat


# ---------------------------------------------------------------------------
# Part 2: Paper-matched recovery
# ---------------------------------------------------------------------------

def run_part2():
    """Part 2: Paper-matched recovery (20 cascades, fit separately)."""
    print("\n" + "=" * 70)
    print("PART 2 — Paper-matched recovery")
    print("=" * 70)

    cfg = STUDY_CONFIG["part2"]
    K_true = cfg["K_true"]
    theta_true = cfg["theta_true"]
    N_true = cfg["N_true"]
    I0 = cfg["I0"]
    T = cfg["T"]
    n_cascades = cfg["n_cascades"]
    n_restarts = cfg["n_restarts"]
    seed = STUDY_CONFIG["seeds"]["paper_match_sim"]

    print(f"\nPaper params: K={K_true}, θ={theta_true}, N={N_true}, I₀={I0}, T={T}")
    print(f"β=Kθ={K_true*theta_true:.3f}, R₀=K={K_true}")
    print(f"Upstream reported: median K=5.5 (sd 2.1), N=1292 (sd 11), θ sd=8.52")
    print(f"\nPRE-REGISTERED FALSIFIER:")
    print("  Median K error >60% OR N error >10% → implementation divergence")

    params = HawkesNParams(K=K_true, theta=theta_true, N=N_true, mu=0.0)
    rng = np.random.default_rng(seed)

    results_per_cascade = []
    for i in range(n_cascades):
        # Simulate cascade
        evts = simulate_hawkesn(params, T, rng, I0=I0)
        for attempt in range(20):
            if len(evts) >= 5:
                break
            evts = simulate_hawkesn(params, T, rng, I0=I0)

        # Fit separately (μ=0 per design)
        fit_params, fit_info = fit_hawkesn(
            evts, T=T, mu_fixed=0.0, n_restarts=n_restarts, I0=I0
        )

        n_events = len(evts)
        final_size_frac = n_events / N_true

        results_per_cascade.append({
            "cascade": i,
            "n_events": n_events,
            "final_size_frac": final_size_frac,
            "K_fit": fit_params.K,
            "theta_fit": fit_params.theta,
            "N_fit": fit_params.N,
            "beta_fit": fit_params.K * fit_params.theta,
            "K_err_pct": abs(fit_params.K - K_true) / K_true * 100,
            "theta_err_pct": abs(fit_params.theta - theta_true) / theta_true * 100,
            "N_err_pct": abs(fit_params.N - N_true) / N_true * 100,
            "beta_err_pct": abs(fit_params.K * fit_params.theta - K_true * theta_true) / (K_true * theta_true) * 100,
            "fit_success": fit_info["success"],
        })
        print(f"  Cascade {i:2d}: n={n_events:4d}, "
              f"K={fit_params.K:.2f} (err={results_per_cascade[-1]['K_err_pct']:.0f}%), "
              f"N={fit_params.N:.0f} (err={results_per_cascade[-1]['N_err_pct']:.0f}%), "
              f"β={fit_params.K*fit_params.theta:.3f} (err={results_per_cascade[-1]['beta_err_pct']:.0f}%), "
              f"θ={fit_params.theta:.3f}")

    # Summary stats
    K_fits = np.array([r["K_fit"] for r in results_per_cascade])
    theta_fits = np.array([r["theta_fit"] for r in results_per_cascade])
    N_fits = np.array([r["N_fit"] for r in results_per_cascade])
    beta_fits = np.array([r["beta_fit"] for r in results_per_cascade])
    fs_fracs = np.array([r["final_size_frac"] for r in results_per_cascade])

    K_errs = np.array([r["K_err_pct"] for r in results_per_cascade])
    N_errs = np.array([r["N_err_pct"] for r in results_per_cascade])
    beta_errs = np.array([r["beta_err_pct"] for r in results_per_cascade])

    print(f"\n  Summary (n={n_cascades} cascades, fit separately):")
    print(f"  K:   median={np.median(K_fits):.2f}, sd={np.std(K_fits):.2f}, "
          f"median_err={np.median(K_errs):.1f}%")
    print(f"  θ:   median={np.median(theta_fits):.3f}, sd={np.std(theta_fits):.3f}")
    print(f"  N:   median={np.median(N_fits):.0f}, sd={np.std(N_fits):.0f}, "
          f"median_err={np.median(N_errs):.1f}%")
    print(f"  β=Kθ: median={np.median(beta_fits):.3f} (true={K_true*theta_true:.3f}), "
          f"sd={np.std(beta_fits):.3f}, median_err={np.median(beta_errs):.1f}%")
    print(f"  Final size/N: median={np.median(fs_fracs):.3f}, sd={np.std(fs_fracs):.3f}")
    print(f"\n  Upstream comparison:")
    print(f"  K:    ours median={np.median(K_fits):.2f} ± {np.std(K_fits):.2f}  "
          f"vs  upstream median=5.5 ± 2.1")
    print(f"  N:    ours median={np.median(N_fits):.0f} ± {np.std(N_fits):.0f}  "
          f"vs  upstream median=1292 ± 11")
    print(f"  θ:    ours sd={np.std(theta_fits):.3f}  vs  upstream sd=8.52")
    print(f"  β:    ours median={np.median(beta_fits):.3f} ± {np.std(beta_fits):.3f} "
          f"(true={K_true*theta_true:.3f})")

    # Gate check
    k_err_pct = abs(np.median(K_fits) - K_true) / K_true * 100
    n_err_pct = abs(np.median(N_fits) - N_true) / N_true * 100
    print(f"\n  GATE CHECK:")
    print(f"  Median K error: {k_err_pct:.1f}% (falsifier: >60%)")
    print(f"  Median N error: {n_err_pct:.1f}% (falsifier: >10%)")
    if k_err_pct > 60:
        print("  *** FALSIFIER TRIGGERED: K error >60% → implementation divergence! ***")
    elif n_err_pct > 10:
        print("  *** FALSIFIER TRIGGERED: N error >10% → implementation divergence! ***")
    else:
        print("  Gate PASSED — proceeding to Part 3")

    # Plot: scatter and histograms of fitted params
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        f"Part 2: Paper-matched recovery\n"
        f"N=1300, I₀=300, K=5, θ=0.2, T=11, {n_cascades} cascades, μ=0",
        fontsize=12
    )

    params_list = [
        ("K", K_fits, K_true, f"Upstream: 5.5 ± 2.1"),
        ("θ", theta_fits, theta_true, f"Upstream: sd=8.52 (θ not id.)"),
        ("N", N_fits, N_true, f"Upstream: 1292 ± 11"),
        ("β=Kθ", beta_fits, K_true * theta_true, f"true={K_true*theta_true:.3f}"),
    ]

    for col, (name, vals, true_val, note) in enumerate(params_list):
        ax_hist = axes[0, col]
        ax_hist.hist(vals, bins=15, alpha=0.7, color="steelblue", edgecolor="white")
        ax_hist.axvline(true_val, color="red", ls="--", lw=2, label=f"true={true_val:.3f}")
        ax_hist.axvline(np.median(vals), color="orange", ls="-", lw=2,
                        label=f"median={np.median(vals):.3f}")
        ax_hist.set_title(f"{name} distribution\n{note}")
        ax_hist.set_xlabel(name)
        ax_hist.legend(fontsize=7)

        ax_scat = axes[1, col]
        ax_scat.scatter(range(n_cascades), vals, alpha=0.6, s=30)
        ax_scat.axhline(true_val, color="red", ls="--", lw=2, label=f"true")
        ax_scat.axhline(np.median(vals), color="orange", ls="-", lw=1.5, label="median")
        ax_scat.set_xlabel("Cascade index")
        ax_scat.set_ylabel(name)
        ax_scat.legend(fontsize=7)

    plt.tight_layout()
    p2_plot_path = WP12_OUTPUT / "part2_paper_recovery.png"
    fig.savefig(p2_plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Part 2 plot saved: {p2_plot_path}")

    # Save JSON
    summary = {
        "params_true": {"K": K_true, "theta": theta_true, "N": N_true,
                        "I0": I0, "T": T, "beta": K_true * theta_true},
        "n_cascades": n_cascades,
        "per_cascade": results_per_cascade,
        "summary": {
            "K_median": float(np.median(K_fits)), "K_sd": float(np.std(K_fits)),
            "K_median_err_pct": float(k_err_pct),
            "theta_median": float(np.median(theta_fits)), "theta_sd": float(np.std(theta_fits)),
            "N_median": float(np.median(N_fits)), "N_sd": float(np.std(N_fits)),
            "N_median_err_pct": float(n_err_pct),
            "beta_median": float(np.median(beta_fits)), "beta_sd": float(np.std(beta_fits)),
            "beta_median_err_pct": float(np.median(beta_errs)),
            "final_size_median": float(np.median(fs_fracs)), "final_size_sd": float(np.std(fs_fracs)),
        },
        "upstream_comparison": {
            "K_upstream_median": 5.5, "K_upstream_sd": 2.1,
            "N_upstream_median": 1292, "N_upstream_sd": 11,
            "theta_upstream_sd": 8.52,
        },
        "gate_passed": bool(k_err_pct <= 60 and n_err_pct <= 10),
    }
    p2_json_path = WP12_OUTPUT / "part2_results.json"

    def _json_safe(obj):
        """Recursively convert numpy types to Python native for JSON."""
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        else:
            return obj

    p2_json_path.write_text(json.dumps(_json_safe(summary), indent=2))
    print(f"  Part 2 results saved: {p2_json_path}")

    return summary


# ---------------------------------------------------------------------------
# Part 3: Information scaling
# ---------------------------------------------------------------------------

def compute_profile_K_interval_width(cascades, T, I0, K_true, n_points=40,
                                     K_range=(0.3, 20.0)):
    """Return 95% profile-CI half-width for K (or None if profile is too flat)."""
    K_grid = np.exp(np.linspace(np.log(K_range[0]), np.log(K_range[1]), n_points))
    N_true_approx = max(sum(len(c) for c in cascades) * 1.5, 50.0)
    pK = compute_profile_K(cascades, T, I0, K_grid, N_true_approx, n_restarts=6)
    pK_norm = pK - np.nanmax(pK)
    above = K_grid[pK_norm > -1.92]
    if len(above) >= 2:
        return float(above[-1] - above[0])
    elif len(above) == 0:
        return None  # entirely flat / all above threshold
    else:
        return float(above[-1] - above[0])


def run_part3(cfg_p1=None):
    """Part 3: Information scaling."""
    print("\n" + "=" * 70)
    print("PART 3 — Information scaling")
    print("=" * 70)

    cfg = STUDY_CONFIG["part3"]
    K_true_c3 = STUDY_CONFIG["part1"]["K_true"]
    theta_true_c3 = STUDY_CONFIG["part1"]["theta_true"]
    seed = STUDY_CONFIG["seeds"]["part3_sim"]

    print(f"\nPRE-REGISTERED PREDICTION:")
    print("  log-log slope of profile-K CI width vs n_cascades ≈ −0.5")
    print("  At I₀/N=0.23, CI width ≤ half that at I₀=1 (N=1300, 20 cascades, full obs)")
    print(f"\nFALSIFIER: K width does NOT decrease as I₀/N increases → extinction not the driver")

    # Primary grid: n_cascades × I0
    n_cas_vals = cfg["n_cascade_values"]
    N_vals = cfg["N_values"]
    T_mults = cfg["T_multiples"]

    results_part3 = []

    # Primary: N=109 (case-3 N), full obs (T=10/theta)
    N_primary = 109
    theta_for_T = theta_true_c3
    T_primary = 10.0 / theta_for_T

    print(f"\n--- PRIMARY GRID: N={N_primary}, T=10/θ={T_primary:.1f} ---")
    print(f"{'n_cas':>6} {'I0_type':>10} {'K_fit':>8} {'K_err%':>8} {'β_err%':>8} {'N_err%':>8} {'CI_width':>10}")

    for i0_type, i0_spec in [("I0=1", 1), ("I0=0.23N", None)]:
        for n_cas in n_cas_vals:
            I0_val = i0_spec if i0_spec is not None else max(1, int(0.23 * N_primary))
            params = HawkesNParams(K=K_true_c3, theta=theta_true_c3, N=float(N_primary), mu=0.0)
            rng = np.random.default_rng(seed + n_cas * 17 + (0 if i0_spec == 1 else 5000))
            cascades = []
            for j in range(n_cas):
                evts = simulate_hawkesn(params, T_primary, rng, I0=I0_val)
                for _ in range(20):
                    if len(evts) >= 2:
                        break
                    evts = simulate_hawkesn(params, T_primary, rng, I0=I0_val)
                cascades.append(evts)

            K_fit, theta_fit, N_fit, _ = fit_pooled(cascades, T_primary, I0_val, n_restarts=10)
            K_err = abs(K_fit - K_true_c3) / K_true_c3 * 100
            beta_err = abs(K_fit * theta_fit - K_true_c3 * theta_true_c3) / (K_true_c3 * theta_true_c3) * 100
            N_err = abs(N_fit - N_primary) / N_primary * 100

            # Profile CI width
            ci_width = compute_profile_K_interval_width(
                cascades, T_primary, I0_val, K_true_c3,
                n_points=cfg["n_profile_K_points"],
                K_range=tuple(cfg["K_profile_range"])
            )

            row = {
                "n_cascades": n_cas, "I0_type": i0_type, "I0_val": I0_val,
                "N": N_primary, "T": T_primary, "T_multiple": 10.0,
                "K_fit": K_fit, "theta_fit": theta_fit, "N_fit": N_fit,
                "beta_fit": K_fit * theta_fit,
                "K_err_pct": K_err, "beta_err_pct": beta_err, "N_err_pct": N_err,
                "profile_K_ci_width": ci_width,
            }
            results_part3.append(row)
            print(f"  {n_cas:4d} {i0_type:>10} {K_fit:8.2f} {K_err:8.1f} {beta_err:8.1f} {N_err:8.1f} {str(ci_width):>10}")

    # Secondary: N grid at n=20 cascades
    print(f"\n--- SECONDARY GRID: n_cascades=20, T=10/θ, I0/N ∈ {{1/N, 0.23}} ---")
    for N_v in N_vals:
        T_v = 10.0 / theta_true_c3
        for i0_type, i0_spec in [("I0=1", 1), ("I0=0.23N", None)]:
            I0_val = i0_spec if i0_spec is not None else max(1, int(0.23 * N_v))
            params = HawkesNParams(K=K_true_c3, theta=theta_true_c3, N=float(N_v), mu=0.0)
            rng = np.random.default_rng(seed + N_v + (0 if i0_spec == 1 else 5000))
            cascades = []
            for j in range(20):
                evts = simulate_hawkesn(params, T_v, rng, I0=I0_val)
                for _ in range(20):
                    if len(evts) >= 2:
                        break
                    evts = simulate_hawkesn(params, T_v, rng, I0=I0_val)
                cascades.append(evts)

            K_fit, theta_fit, N_fit, _ = fit_pooled(cascades, T_v, I0_val, n_restarts=10)
            K_err = abs(K_fit - K_true_c3) / K_true_c3 * 100
            beta_err = abs(K_fit * theta_fit - K_true_c3 * theta_true_c3) / (K_true_c3 * theta_true_c3) * 100
            N_err = abs(N_fit - N_v) / N_v * 100

            ci_width = compute_profile_K_interval_width(
                cascades, T_v, I0_val, K_true_c3,
                n_points=cfg["n_profile_K_points"],
                K_range=tuple(cfg["K_profile_range"])
            )
            row = {
                "n_cascades": 20, "I0_type": i0_type, "I0_val": I0_val,
                "N": N_v, "T": T_v, "T_multiple": 10.0,
                "K_fit": K_fit, "theta_fit": theta_fit, "N_fit": N_fit,
                "beta_fit": K_fit * theta_fit,
                "K_err_pct": K_err, "beta_err_pct": beta_err, "N_err_pct": N_err,
                "profile_K_ci_width": ci_width,
            }
            results_part3.append(row)
            print(f"  N={N_v:5d} {i0_type:>10}: "
                  f"K_fit={K_fit:.2f} (err={K_err:.1f}%), β_err={beta_err:.1f}%, "
                  f"N_err={N_err:.1f}%, CI_width={ci_width}")

    # T-multiple secondary: N=109, n=20, I0=1
    print(f"\n--- SECONDARY: T multiple grid, N={N_primary}, n=20, I0=1 ---")
    for T_mult in T_mults:
        T_v = T_mult / theta_true_c3
        params = HawkesNParams(K=K_true_c3, theta=theta_true_c3, N=float(N_primary), mu=0.0)
        rng = np.random.default_rng(seed + int(T_mult * 100))
        cascades = []
        for j in range(20):
            evts = simulate_hawkesn(params, T_v, rng, I0=1)
            for _ in range(20):
                if len(evts) >= 2:
                    break
                evts = simulate_hawkesn(params, T_v, rng, I0=1)
            cascades.append(evts)
        K_fit, theta_fit, N_fit, _ = fit_pooled(cascades, T_v, 1, n_restarts=8)
        K_err = abs(K_fit - K_true_c3) / K_true_c3 * 100
        beta_err = abs(K_fit * theta_fit - K_true_c3 * theta_true_c3) / (K_true_c3 * theta_true_c3) * 100
        N_err = abs(N_fit - N_primary) / N_primary * 100
        ci_width = compute_profile_K_interval_width(
            cascades, T_v, 1, K_true_c3,
            n_points=cfg["n_profile_K_points"],
            K_range=tuple(cfg["K_profile_range"])
        )
        row = {
            "n_cascades": 20, "I0_type": "I0=1", "I0_val": 1,
            "N": N_primary, "T": T_v, "T_multiple": T_mult,
            "K_fit": K_fit, "theta_fit": theta_fit, "N_fit": N_fit,
            "beta_fit": K_fit * theta_fit,
            "K_err_pct": K_err, "beta_err_pct": beta_err, "N_err_pct": N_err,
            "profile_K_ci_width": ci_width,
        }
        results_part3.append(row)
        print(f"  T={T_mult:.1f}/θ={T_v:.1f}: K_fit={K_fit:.2f} (err={K_err:.1f}%), "
              f"β_err={beta_err:.1f}%, CI_width={ci_width}")

    # Plotting Part 3
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Part 3: Information Scaling (K-profile CI width vs #cascades & I₀/N)", fontsize=12)

    # 1. CI width vs n_cascades for primary grid
    ax = axes[0, 0]
    primary = [r for r in results_part3 if r["N"] == N_primary and r["T_multiple"] == 10.0]
    for i0_type in ["I0=1", "I0=0.23N"]:
        sub = [r for r in primary if r["I0_type"] == i0_type]
        sub.sort(key=lambda r: r["n_cascades"])
        ns = [r["n_cascades"] for r in sub]
        ws = [r["profile_K_ci_width"] if r["profile_K_ci_width"] is not None else np.nan for r in sub]
        ax.plot(ns, ws, "o-", label=i0_type)
    ax.set_xlabel("# pooled cascades")
    ax.set_ylabel("95% Profile-K CI width")
    ax.set_title(f"N={N_primary}, T=10/θ")
    ax.legend()
    ax.set_yscale("log")

    # 2. log-log: CI width vs n_cascades
    ax2 = axes[0, 1]
    for i0_type in ["I0=1", "I0=0.23N"]:
        sub = [r for r in primary if r["I0_type"] == i0_type]
        sub.sort(key=lambda r: r["n_cascades"])
        ns = np.array([r["n_cascades"] for r in sub], dtype=float)
        ws = np.array([r["profile_K_ci_width"] if r["profile_K_ci_width"] is not None else np.nan for r in sub])
        valid = ~np.isnan(ws) & (ws > 0)
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(np.log(ns[valid]), np.log(ws[valid]), 1)
            ax2.loglog(ns[valid], ws[valid], "o", label=f"{i0_type} (slope={slope:.2f})")
            ax2.loglog(ns[valid], np.exp(intercept) * ns[valid] ** slope, "--", alpha=0.5)
        else:
            ax2.loglog(ns, ws, "o-", label=i0_type)
    ax2.set_xlabel("# pooled cascades (log)")
    ax2.set_ylabel("CI width (log)")
    ax2.set_title("log-log: CI width vs n_cascades\n(pred slope ≈ −0.5)")
    ax2.legend()

    # 3. K error vs n_cascades
    ax3 = axes[0, 2]
    for i0_type in ["I0=1", "I0=0.23N"]:
        sub = [r for r in primary if r["I0_type"] == i0_type]
        sub.sort(key=lambda r: r["n_cascades"])
        ns = [r["n_cascades"] for r in sub]
        es = [r["K_err_pct"] for r in sub]
        ax3.plot(ns, es, "o-", label=i0_type)
    ax3.set_xlabel("# pooled cascades")
    ax3.set_ylabel("|K error| %")
    ax3.set_title("K error vs n_cascades")
    ax3.legend()

    # 4. CI width vs N (n=20, T=10/θ)
    ax4 = axes[1, 0]
    n20_rows = [r for r in results_part3 if r["n_cascades"] == 20 and r["T_multiple"] == 10.0]
    for i0_type in ["I0=1", "I0=0.23N"]:
        sub = [r for r in n20_rows if r["I0_type"] == i0_type]
        sub.sort(key=lambda r: r["N"])
        Ns = [r["N"] for r in sub]
        ws = [r["profile_K_ci_width"] if r["profile_K_ci_width"] is not None else np.nan for r in sub]
        ax4.plot(Ns, ws, "o-", label=i0_type)
    ax4.set_xlabel("N")
    ax4.set_ylabel("95% K CI width")
    ax4.set_title("N=109→1300, n=20, T=10/θ\n(extinction hypothesis test)")
    ax4.legend()

    # 5. CI width vs T multiple (N=primary, n=20, I0=1)
    ax5 = axes[1, 1]
    t_rows = [r for r in results_part3 if r["N"] == N_primary and r["n_cascades"] == 20 and r["I0_type"] == "I0=1"]
    t_rows.sort(key=lambda r: r["T_multiple"])
    ts = [r["T_multiple"] for r in t_rows]
    ws_t = [r["profile_K_ci_width"] if r["profile_K_ci_width"] is not None else np.nan for r in t_rows]
    ax5.plot(ts, ws_t, "o-")
    ax5.set_xlabel("T × θ (obs window as multiple of 1/θ)")
    ax5.set_ylabel("95% K CI width")
    ax5.set_title(f"Observation window effect\nN={N_primary}, n=20, I0=1")

    # 6. β error vs I0 type at various N, n=20
    ax6 = axes[1, 2]
    for i0_type in ["I0=1", "I0=0.23N"]:
        sub = [r for r in n20_rows if r["I0_type"] == i0_type]
        sub.sort(key=lambda r: r["N"])
        Ns = [r["N"] for r in sub]
        es = [r["beta_err_pct"] for r in sub]
        ax6.plot(Ns, es, "o-", label=i0_type)
    ax6.set_xlabel("N")
    ax6.set_ylabel("|β=Kθ error| %")
    ax6.set_title("β error vs N (n=20 pooled)")
    ax6.legend()

    plt.tight_layout()
    p3_plot_path = WP12_OUTPUT / "part3_scaling.png"
    fig.savefig(p3_plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Part 3 plot saved: {p3_plot_path}")

    # Save JSON
    p3_json_path = WP12_OUTPUT / "part3_results.json"
    # Convert None values to null-friendly
    results_part3_clean = [
        {k: (v if v is not None else None) for k, v in r.items()}
        for r in results_part3
    ]
    p3_json_path.write_text(json.dumps(results_part3_clean, indent=2))
    print(f"  Part 3 results saved: {p3_json_path}")

    # Extinction hypothesis check
    print(f"\n  EXTINCTION HYPOTHESIS CHECK:")
    print("  Does K recovery improve as I₀/N rises? (N=1300, n=20, T=10/θ)")
    n1300 = [r for r in n20_rows if r["N"] == 1300]
    for r in n1300:
        print(f"    {r['I0_type']:>10}: K_err={r['K_err_pct']:.1f}%, CI_width={r['profile_K_ci_width']}")

    return results_part3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("WP-12 Identifiability Study — Parts 1, 2, 3")
    print("PRE-REGISTERED — PROPOSED/UNCERTIFIED")
    print("=" * 70)

    t_start = time.time()

    # Save config
    cfg_path = WP12_OUTPUT / "identifiability_config.json"
    import jax
    import numpy as np
    import scipy
    import matplotlib
    cfg_with_versions = {
        **STUDY_CONFIG,
        "library_versions": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    cfg_path.write_text(json.dumps(cfg_with_versions, indent=2))
    print(f"\nConfig saved: {cfg_path}")

    # Part 1
    results_p1, gate_passes = run_part1()

    print(f"\n{'='*70}")
    print(f"GATE RESULT: {'PASS — running Part 3' if gate_passes else 'FAIL — stopping after Part 2'}")
    print(f"{'='*70}")

    # Part 2 (always run)
    results_p2 = run_part2()

    # Part 3 (only if gate passes)
    results_p3 = None
    if gate_passes:
        results_p3 = run_part3()
    else:
        print("\nPart 3 NOT run — single-cascade K-profile showed sharp peak, "
              "suggesting optimizer pathology (not structural degeneracy).")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Study complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Outputs in: {WP12_OUTPUT}")
    print("NOTE: Results are PROPOSED/UNCERTIFIED — review required before acceptance.")
    print("=" * 70)

    return results_p1, results_p2, results_p3, gate_passes


if __name__ == "__main__":
    main()
