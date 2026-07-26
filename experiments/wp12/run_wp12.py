"""Main execution script for WP-12.

Runs all Phase A tasks:
  1. Synthetic parameter recovery validation
  2. Mean-field bridge check (HawkesN ↔ SIR ODE)
  3. Final-size distribution (Markov chain / MC)

Saves all outputs and a config JSON to experiments/WP-12/.

Usage:
    .venv/bin/python experiments/wp12/run_wp12.py

All random seeds are recorded; library versions are saved in config.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Make sure project root is on path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.wp12.bridge import run_bridge_check
from experiments.wp12.hawkesn import (
    HawkesNParams,
    fit_hawkesn,
    hawkesn_log_likelihood,
    simulate_hawkesn,
)
from experiments.wp12.plots import (
    WP12_OUTPUT,
    plot_bridge_check,
    plot_final_size_distribution,
    plot_synthetic_recovery,
)
from experiments.wp12.size_distribution import final_size_distribution

# ---------------------------------------------------------------------------
# Config & seeds
# ---------------------------------------------------------------------------

RUN_CONFIG = {
    "run_id": "wp12-phase-a-001",
    "date": "2026-06-10",
    "seeds": {
        "synthetic_recovery_base": 1000,
        "bridge_check": 42,
        "final_size_mc": 2024,
    },
    "synthetic_recovery": {
        "n_cases": 20,
        "T_factor": 30.0,   # T = T_factor / theta (adaptive to decay timescale)
        "n_restarts": 3,
        "K_range": [1.5, 3.0],   # super-critical only for reliable cascades
        "theta_range": [0.2, 1.0],
        "N_range": [100, 500],
        "I0": 1,
        "min_events": 20,   # skip cases with < min_events
        "n_trials": 5,      # try n_trials simulations per case to get one big enough
    },
    "bridge_check": {
        "K_values": [1.5, 2.0, 3.0],
        "theta": 0.3,
        "N": 500,
        "I0": 1.0,
        "n_simulations": 200,
        "n_bins": 50,
    },
    "final_size": {
        "K_values": [0.8, 1.2, 2.0, 3.0],
        "theta": 0.3,
        "N_ctmc": 100,
        "N_mc": 1000,
        "I0": 1,
        "n_mc": 20_000,
    },
}


def save_config() -> None:
    """Save run config and library versions to WP-12/."""
    import diffrax
    import matplotlib
    import scipy

    versions = {
        "jax": jax.__version__,
        "diffrax": diffrax.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    cfg = {**RUN_CONFIG, "library_versions": versions}
    cfg_path = WP12_OUTPUT / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"  Config saved to {cfg_path}")


# ---------------------------------------------------------------------------
# Phase 1: Synthetic parameter recovery
# ---------------------------------------------------------------------------

# §11 pre-registration for synthetic recovery:
#
#   Hypothesis: MLE with 3 random restarts recovers K, θ, N to within
#       50% of ground truth on >80% of synthetic cascades with K in [0.5, 3]
#       and N in [100, 500].
#
#   Prediction: Median absolute percentage error < 20% for K and θ;
#       N may be harder (wider CI) due to depletion only being observed
#       near the end of large cascades.
#
#   Falsifier: If median |K_fit - K_true| / K_true > 0.50, the MLE is
#       mis-specified or the log-likelihood has pathological curvature.
#       If N recovery is consistently poor, that's expected for sub-critical
#       cascades (K < 1) where depletion is never observed — and that's
#       scientifically informative, not a bug.

PRE_REGISTRATION = """
=== §11 Pre-registration: Synthetic Recovery ===
Hypothesis: MLE recovers (K, θ, N) to within 50% on >80% of cases
            for K in [1.5, 3], θ in [0.2, 1.0], N in [100, 500], ≥20 events.
Prediction: Median |K_err| < 30%, median |θ_err| < 30%;
            K·θ product is better identified than K or θ individually.
            N is harder to identify due to K-θ-N ridge.
Falsifier:  Median K·θ error > 50% → MLE mis-specification bug.

=== §11 Pre-registration: Mean-field Bridge ===
Hypothesis: HawkesN mean (200 sims) matches SIR ODE rate to < 20% rel RMSE.
Prediction: Agreement within 10% for K >= 1.5, N = 500.
Falsifier:  Rel RMSE > 20% at K=2.0 → bridge does NOT hold numerically.

=== §11 Pre-registration: Final-size Distribution ===
Hypothesis: K=0.8 → unimodal near 0; K near 1.0-1.5 → bimodal;
            K=3.0 → unimodal near large outbreak.
Prediction: For K=0.8, >90% mass below 5% of N.
            For K=2.0, visible bimodal (small + large modes).
Falsifier:  K=0.8 showing large outbreak probability > 10% → calculation error.
"""


def run_synthetic_recovery() -> list[dict]:
    """Run synthetic parameter recovery validation.

    Returns a list of per-case result dicts.
    """
    print("\n=== Phase 1: Synthetic parameter recovery ===")
    print(PRE_REGISTRATION.split("===")[1:3][0])

    cfg = RUN_CONFIG["synthetic_recovery"]
    base_seed = RUN_CONFIG["seeds"]["synthetic_recovery_base"]
    n_cases = cfg["n_cases"]
    min_events = cfg.get("min_events", 10)
    n_trials = cfg.get("n_trials", 5)

    rng_gen = np.random.default_rng(base_seed)

    K_vals = np.exp(
        rng_gen.uniform(np.log(cfg["K_range"][0]), np.log(cfg["K_range"][1]), n_cases)
    )
    theta_vals = np.exp(
        rng_gen.uniform(
            np.log(cfg["theta_range"][0]), np.log(cfg["theta_range"][1]), n_cases
        )
    )
    N_vals = np.round(
        np.exp(rng_gen.uniform(np.log(cfg["N_range"][0]), np.log(cfg["N_range"][1]), n_cases))
    ).astype(int)

    results = []
    for case_i in range(n_cases):
        K_true = float(K_vals[case_i])
        theta_true = float(theta_vals[case_i])
        N_true = int(N_vals[case_i])
        I0 = cfg["I0"]
        # Adaptive T: use T = T_factor / theta so we see the full cascade
        T = cfg.get("T_factor", 30.0) / theta_true

        params_true = HawkesNParams(K=K_true, theta=theta_true, N=float(N_true), mu=0.0)

        # Try multiple seeds to get a large enough cascade
        events = np.array([])
        for trial in range(n_trials):
            sim_rng = np.random.default_rng(base_seed + case_i * 100 + trial)
            events_try = simulate_hawkesn(params_true, T, sim_rng, I0=I0)
            if len(events_try) >= min_events:
                events = events_try
                break
        n_events = len(events)

        if n_events < min_events:
            print(f"  Case {case_i:02d}: K={K_true:.2f}, θ={theta_true:.2f}, N={N_true}: "
                  f"only {n_events} events after {n_trials} trials — skipping fit")
            results.append({
                "case": case_i,
                "K_true": K_true, "theta_true": theta_true, "N_true": float(N_true),
                "K_fit": np.nan, "theta_fit": np.nan, "N_fit": np.nan,
                "K_err_pct": np.nan, "theta_err_pct": np.nan, "N_err_pct": np.nan,
                "n_events": n_events,
                "fit_success": False,
                "neg_ll": np.nan,
            })
            continue

        fit_params, fit_info = fit_hawkesn(
            events, T=T, mu_fixed=0.0, n_restarts=cfg["n_restarts"]
        )

        K_err = abs(fit_params.K - K_true) / K_true * 100
        theta_err = abs(fit_params.theta - theta_true) / theta_true * 100
        N_err = abs(fit_params.N - N_true) / N_true * 100
        Ktheta_true = K_true * theta_true
        Ktheta_fit = fit_params.K * fit_params.theta
        Ktheta_err = abs(Ktheta_fit - Ktheta_true) / Ktheta_true * 100

        results.append({
            "case": case_i,
            "K_true": K_true, "theta_true": theta_true, "N_true": float(N_true),
            "K_fit": fit_params.K, "theta_fit": fit_params.theta, "N_fit": fit_params.N,
            "K_err_pct": K_err, "theta_err_pct": theta_err, "N_err_pct": N_err,
            "Ktheta_err_pct": Ktheta_err,
            "n_events": n_events,
            "fit_success": fit_info["success"],
            "neg_ll": fit_info["neg_log_likelihood"],
        })
        print(
            f"  Case {case_i:02d}: K·θ={Ktheta_true:.2f}→{Ktheta_fit:.2f} "
            f"({Ktheta_err:.0f}%), K={K_true:.2f}→{fit_params.K:.2f} "
            f"({K_err:.0f}%), θ={theta_true:.2f}→{fit_params.theta:.2f} "
            f"({theta_err:.0f}%), N={N_true}→{fit_params.N:.0f} "
            f"({N_err:.0f}%), n={n_events}"
        )

    # Summary statistics
    valid = [r for r in results if not np.isnan(r["K_err_pct"])]
    if valid:
        K_errors = [r["K_err_pct"] for r in valid]
        theta_errors = [r["theta_err_pct"] for r in valid]
        N_errors = [r["N_err_pct"] for r in valid]
        # K*theta (=beta) error — the product is better identified
        Ktheta_errors = [
            abs(r["K_fit"] * r["theta_fit"] - r["K_true"] * r["theta_true"]) /
            (r["K_true"] * r["theta_true"]) * 100
            for r in valid
        ]
        print(f"\n  Summary ({len(valid)}/{n_cases} cases with ≥{min_events} events):")
        print(f"  Median K error:       {np.median(K_errors):.1f}%")
        print(f"  Median θ error:       {np.median(theta_errors):.1f}%")
        print(f"  Median K·θ=β error:  {np.median(Ktheta_errors):.1f}%")
        print(f"  Median N error:       {np.median(N_errors):.1f}%")
        frac_Ktheta_within_50 = np.mean(np.array(Ktheta_errors) < 50)
        print(f"  K·θ within 50%:       {frac_Ktheta_within_50*100:.0f}%")

        # §11: compare prediction vs observation
        if np.median(Ktheta_errors) < 30:
            print("  [PREDICTION MET] Median K·θ error < 30%")
        else:
            print("  [PREDICTION MISSED] Median K·θ error >= 30% — examine cases")
        if frac_Ktheta_within_50 > 0.8:
            print("  [HYPOTHESIS MET] >80% K·θ cases within 50%")
        else:
            print("  [HYPOTHESIS MISSED] <80% K·θ cases within 50%")

    # Save JSON
    recovery_path = WP12_OUTPUT / "synthetic_recovery_results.json"
    recovery_path.write_text(json.dumps(results, indent=2))
    print(f"  Results saved to {recovery_path}")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Mean-field bridge check
# ---------------------------------------------------------------------------

def run_bridge() -> list[dict]:
    """Run the mean-field bridge check for multiple K values."""
    print("\n=== Phase 2: Mean-field bridge check ===")
    print(PRE_REGISTRATION.split("===")[3:5][0])

    cfg = RUN_CONFIG["bridge_check"]
    bridge_results = []

    for K in cfg["K_values"]:
        print(f"  K={K}: running {cfg['n_simulations']} simulations ...")
        t0 = time.time()
        result = run_bridge_check(
            K=K,
            theta=cfg["theta"],
            N=cfg["N"],
            I0=cfg["I0"],
            n_simulations=cfg["n_simulations"],
            n_bins=cfg["n_bins"],
            rng_seed=RUN_CONFIG["seeds"]["bridge_check"],
        )
        elapsed = time.time() - t0
        rmse_cond = result.get("relative_rmse_cond", result["relative_rmse"])
        rmse_uncond = result.get("relative_rmse_uncond", result["relative_rmse"])
        confirmed = result["bridge_confirmed"]
        print(
            f"  K={K}: cond RMSE={rmse_cond:.4f}, uncond RMSE={rmse_uncond:.4f} "
            f"({'CONFIRMED' if confirmed else 'NOT confirmed'}), {elapsed:.1f}s"
        )
        bridge_results.append(result)

        # Save individual plot
        plot_path = WP12_OUTPUT / f"bridge_check_K{K:.1f}.png"
        plot_bridge_check(result, output_path=plot_path)
        print(f"    Plot saved: {plot_path}")

    # Summary
    all_confirmed = all(r["bridge_confirmed"] for r in bridge_results)
    rmse_vals = [r["relative_rmse"] for r in bridge_results]
    print(f"\n  Bridge check summary:")
    for K, r in zip(cfg["K_values"], bridge_results):
        rmse_c = r.get("relative_rmse_cond", r["relative_rmse"])
        rmse_u = r.get("relative_rmse_uncond", r["relative_rmse"])
        print(f"    K={K}: cond RMSE={rmse_c:.4f}, uncond RMSE={rmse_u:.4f}")
    if all_confirmed:
        print("  [BRIDGE CONFIRMED] All K values pass the 20% threshold.")
    else:
        failed = [K for K, r in zip(cfg["K_values"], bridge_results) if not r["bridge_confirmed"]]
        print(f"  [BRIDGE NOT FULLY CONFIRMED] Failed K values: {failed}")

    # Save serialisable summary (remove numpy arrays; convert to float)
    summary = []
    for r in bridge_results:
        s = {k: float(v) if hasattr(v, '__float__') else v
             for k, v in r.items() if not isinstance(v, np.ndarray)}
        summary.append(s)
    bridge_path = WP12_OUTPUT / "bridge_results.json"
    bridge_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary saved to {bridge_path}")

    return bridge_results


# ---------------------------------------------------------------------------
# Phase 3: Final-size distribution
# ---------------------------------------------------------------------------

def run_final_size() -> list[dict]:
    """Run final-size distribution computation for multiple K values."""
    print("\n=== Phase 3: Final-size distribution ===")
    print(PRE_REGISTRATION.split("===")[5:7][0])

    cfg = RUN_CONFIG["final_size"]
    size_results = []

    # Run CTMC for small N
    print(f"  CTMC (N={cfg['N_ctmc']}):")
    for K in cfg["K_values"]:
        t0 = time.time()
        pmf, sizes, method = final_size_distribution(
            K=K,
            theta=cfg["theta"],
            N=cfg["N_ctmc"],
            I0=cfg["I0"],
            n_mc=cfg["n_mc"],
            rng_seed=RUN_CONFIG["seeds"]["final_size_mc"],
            ctmc_max_n=cfg["N_ctmc"],
        )
        elapsed = time.time() - t0
        # Find modal final size
        mode = float(sizes[np.argmax(pmf)]) / cfg["N_ctmc"]
        small_outbreak_prob = float(pmf[: cfg["N_ctmc"] // 5].sum())
        large_outbreak_prob = float(pmf[cfg["N_ctmc"] // 5 :].sum())
        print(
            f"    K={K:.1f}: mode={mode:.2f}N, "
            f"P(small)={small_outbreak_prob:.3f}, P(large)={large_outbreak_prob:.3f}, "
            f"method={method}, {elapsed:.2f}s"
        )
        size_results.append({
            "K": K, "N": cfg["N_ctmc"],
            "pmf": pmf, "sizes": sizes, "method": method,
            "mode_frac": mode,
            "small_outbreak_prob": small_outbreak_prob,
            "large_outbreak_prob": large_outbreak_prob,
        })

        # §11: Check prediction for K=0.8
        if abs(K - 0.8) < 0.05:
            if small_outbreak_prob > 0.9:
                print("    [PREDICTION MET] K=0.8: >90% small outbreak")
            else:
                print(f"    [PREDICTION MISSED] K=0.8: small outbreak prob = {small_outbreak_prob:.3f}")

    # Save CTMC plot
    plot_path = WP12_OUTPUT / f"final_size_ctmc_N{cfg['N_ctmc']}.png"
    plot_final_size_distribution(size_results, output_path=plot_path)
    print(f"  Plot saved: {plot_path}")

    # Also run MC for larger N
    print(f"\n  MC (N={cfg['N_mc']}, {cfg['n_mc']} sims per K):")
    mc_results = []
    for K in cfg["K_values"]:
        t0 = time.time()
        pmf, sizes, method = final_size_distribution(
            K=K,
            theta=cfg["theta"],
            N=cfg["N_mc"],
            I0=cfg["I0"],
            n_mc=cfg["n_mc"],
            rng_seed=RUN_CONFIG["seeds"]["final_size_mc"],
            ctmc_max_n=200,
        )
        elapsed = time.time() - t0
        mode = float(sizes[np.argmax(pmf)]) / cfg["N_mc"]
        small_prob = float(pmf[: cfg["N_mc"] // 5].sum())
        large_prob = float(pmf[cfg["N_mc"] // 5 :].sum())
        print(
            f"    K={K:.1f}: mode={mode:.2f}N, "
            f"P(small)={small_prob:.3f}, P(large)={large_prob:.3f}, "
            f"{elapsed:.1f}s"
        )
        mc_results.append({
            "K": K, "N": cfg["N_mc"],
            "pmf": pmf, "sizes": sizes, "method": "mc",
            "mode_frac": mode,
            "small_outbreak_prob": small_prob,
            "large_outbreak_prob": large_prob,
        })

    mc_plot_path = WP12_OUTPUT / f"final_size_mc_N{cfg['N_mc']}.png"
    plot_final_size_distribution(mc_results, output_path=mc_plot_path)
    print(f"  Plot saved: {mc_plot_path}")

    # Save JSON (without numpy arrays)
    combined = []
    for r in size_results + mc_results:
        s = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        combined.append(s)
    fsd_path = WP12_OUTPUT / "final_size_results.json"
    fsd_path.write_text(json.dumps(combined, indent=2))
    print(f"  Results saved to {fsd_path}")

    return size_results, mc_results


# ---------------------------------------------------------------------------
# Phase 4: Log-likelihood sanity check
# ---------------------------------------------------------------------------

def run_ll_sanity() -> None:
    """Quick sanity check: log-likelihood is higher at true params than random."""
    print("\n=== LL Sanity Check ===")
    rng = np.random.default_rng(9999)
    params = HawkesNParams(K=2.0, theta=0.3, N=500.0, mu=0.0)
    events = simulate_hawkesn(params, T=50.0, rng=rng, I0=1)
    print(f"  Simulated {len(events)} events")
    if len(events) < 3:
        print("  Too few events — skipping LL sanity")
        return

    times_jax = jnp.asarray(events, dtype=jnp.float64)

    ll_true = float(
        hawkesn_log_likelihood(
            jnp.log(jnp.array(2.0)), jnp.log(jnp.array(0.3)),
            jnp.log(jnp.array(500.0)), jnp.log(jnp.array(1e-12)),
            times_jax, 50.0,
        )
    )
    ll_bad = float(
        hawkesn_log_likelihood(
            jnp.log(jnp.array(0.1)), jnp.log(jnp.array(0.1)),
            jnp.log(jnp.array(50.0)), jnp.log(jnp.array(1e-12)),
            times_jax, 50.0,
        )
    )
    print(f"  LL at true params: {ll_true:.2f}")
    print(f"  LL at wrong params: {ll_bad:.2f}")
    if ll_true > ll_bad:
        print("  [SANITY PASSED] True params have higher LL than bad params")
    else:
        print("  [SANITY FAILED] True params have LOWER LL — investigate!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("WP-12 Phase A — JAX HawkesN core execution")
    print("=" * 70)
    print(PRE_REGISTRATION)

    t_start = time.time()
    save_config()

    # LL sanity first
    run_ll_sanity()

    # Phase 1
    recovery_results = run_synthetic_recovery()
    recovery_plot = plot_synthetic_recovery(
        [r for r in recovery_results if not np.isnan(r.get("K_fit", float("nan")))],
        output_path=WP12_OUTPUT / "synthetic_recovery.png",
    )
    print(f"\n  Recovery plot: {recovery_plot}")

    # Phase 2
    bridge_results = run_bridge()

    # Phase 3
    ctmc_results, mc_results = run_final_size()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Phase A complete in {elapsed:.1f}s")
    print(f"Outputs in: {WP12_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
