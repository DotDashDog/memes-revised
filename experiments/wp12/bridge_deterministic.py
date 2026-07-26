"""Deterministic mean-field bridge test: HawkesN-ODE vs SIR-ODE.

PRE-REGISTRATION (written before any results are computed)
==========================================================

Claim under test
----------------
The HawkesN process (exponential kernel, μ=0, N→∞ mean-field limit) is
equivalent to the SIR ODE under the parameter map:

    γ = θ,   β = K·θ,   R₀ = K,   N_hawkes = N_sir,   I₀/N = initial fraction

(Rizoiu et al., WWW 2018, Prop. 1; cited in research-notes/hawkes-processes.md §3)

Claim type (method-discipline §2)
-----------------------------------
This is a *deterministic identity / asymptotic limit*.  Per §2, the correct
test is deterministic-to-integrator-tolerance.  Stochastic simulations and
conditioning are NOT part of this test.

HawkesN deterministic mean-field ODE — independent derivation
--------------------------------------------------------------
The HawkesN conditional intensity is:

    λ*(t) = (1 - N_t/N) · Σ_{t_i < t} K·θ·exp(-θ·(t - t_i))      [μ=0]

Define the *normalized excitation state*:

    a(t) = (1/N) · Σ_{t_i < t} K·θ·exp(-θ·(t - t_i))
         = A(t)/N  where  A(t) = Σ K·θ·exp(-θ·(t-t_i))

Between events, A decays at rate θ.  Each new event adds K·θ to A.
In the mean-field limit, event rate → λ*(t) = (1-x)·N·a(t) = s·N·a(t)
where s(t) = 1 - N_t/N is the remaining susceptible fraction.

Evolution equations (mean-field, continuous limit):

    da/dt = -θ·a  (decay term)  +  K·θ · (λ*(t)/N)  (new-event term)
          = -θ·a + K·θ · s·a
          = θ·a · (K·s - 1)

    ds/dt = -(λ*(t)/N) = -s·a

Initial conditions (I₀ seeds at t=0⁻):
    s(0) = 1 - I₀/N
    a(0) = (I₀/N) · K·θ          [each seed contributes K·θ to A at t=0]

Non-circularity argument
------------------------
The HawkesN mean-field system is expressed in variables (s, a) where
a = normalized cumulative excitation.  We integrate this 2D ODE as written.

Separately we integrate the *textbook* SIR ODE:
    dS/dt = -β·S·I
    dI/dt = β·S·I - γ·I
with β = K·θ, γ = θ, S(0) = 1-I₀/N, I(0) = I₀/N.

The two systems use different state variables (a vs I), different RHS
functions, and are integrated by the same ODE solver independently.
They should agree because one can derive algebraically that I(t) = a(t)/(K·θ),
but that derivation is a theorem, not a tautology, and a coding error in
either system would break agreement.  Therefore a match is a non-trivial
implementation check.  The circularity risk is that both systems are
ultimately derived from the same claim — that is the honest limitation
(see "metric blind spot" below).

Pre-registered metric and threshold (method-discipline §3)
----------------------------------------------------------
Primary metric: relative maximum absolute deviation (relative max-abs)

    ε = max_t |I_hawkes(t) - I_sir(t)| / max_t I_sir(t)

where I_hawkes(t) = a(t)/(K·θ) is the infected fraction inferred from the
HawkesN mean-field ODE, and I_sir(t) is the infected fraction from the
independent SIR ODE solve.

Threshold derivation:
    Both ODEs are integrated with rtol=1e-7, atol=1e-9 (Dopri5 / RK45).
    Local truncation error accumulates over the integration interval; for
    a well-conditioned system the global error is O(rtol + atol / scale).
    With rtol=1e-7, the global error on each ODE is ≲ ~10·rtol ≈ 1e-6
    (rough, for O(1) condition number).  Two independent solves compound
    as ~√2 · 1e-6 ≈ 1.4e-6.  If the claim is TRUE, ε should be at or
    below this ~1e-5 level (a conservative factor-of-10 buffer for
    conditioning and parameter variation).

    FALSIFIER: ε > 1e-3 (0.1%) falsifies the deterministic identity claim.
    The threshold is 100× the expected numerical noise, so any real
    identity error at the level of a fraction of a percent would register.
    A threshold you cannot derive is not a test (method-discipline §3) —
    this threshold is derived from solver tolerances, not chosen to pass.

    Secondary metric (for context only, not the primary falsifier):
    relative RMS deviation:
        δ = sqrt( mean_t( (I_hawkes - I_sir)² ) ) / max_t I_sir

Metric blind spot (method-discipline §4)
-----------------------------------------
A near-exact numerical match confirms that *this implementation* correctly
maps HawkesN parameters to SIR parameters and that both ODEs are coded
consistently — it does NOT prove the analytic theorem (which is
established in the literature and would require a symbolic proof).
Additionally, relative max-abs is insensitive to errors near t=0 and t=T
where I(t) ≈ 0 (denominator is small), but normalization by the peak
max_t I_sir(t) rather than I(t) avoids that; however, early/late errors
are still down-weighted by the peak-normalization.

Pre-committed expected plot appearance (method-discipline §5)
--------------------------------------------------------------
If the identity HOLDS:
  - The HawkesN mean-field I(t) and SIR I(t) curves should be visually
    indistinguishable (overlapping within line width) across the full
    time range for all tested (K, θ) combinations.
  - The residual panel I_hawkes(t) - I_sir(t) should be a near-flat line
    hugging zero, with amplitude < 1e-5 · max(I_sir).
  - No systematic trend (no phase lead/lag, no amplitude offset).

If the identity FAILS:
  - The curves would show a visible gap, phase offset, or amplitude
    mismatch.
  - The residuals would show a structured curve (e.g., one-sided bump
    near the peak, systematic phase lead/lag, or a constant offset).
  - Any structured residual at a level > 0.1% of peak is a failure.

The plot is produced BEFORE reporting the scalar metric, and the plot
verdict takes priority over the number (method-discipline §5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.wp02.sir import SirParameters, solve_sir  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------
FALSIFIER_THRESHOLD = 1e-3   # relative max-abs > this falsifies the identity
EXPECTED_NOISE_LEVEL = 1e-5  # expected agreement level (10× solver tolerance)
ODE_RTOL = 1e-7
ODE_ATOL = 1e-9

# ---------------------------------------------------------------------------
# HawkesN deterministic mean-field ODE
# ---------------------------------------------------------------------------

def _hawkesn_mf_rhs(
    t: jax.Array,
    y: jax.Array,
    args: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """RHS of the HawkesN mean-field ODE in (s, a) coordinates.

    State: y = [s, a]
      s(t) = susceptible fraction = 1 - N_t(t)/N
      a(t) = normalized excitation = (1/N) · Σ_{t_i<t} K·θ·exp(-θ(t-t_i))

    Equations (derived independently from HawkesN intensity, see module
    docstring):
      ds/dt = -s · a
      da/dt =  θ · a · (K·s - 1)
    """
    del t
    K, theta = args
    s, a = y[0], y[1]
    ds = -s * a
    da = theta * a * (K * s - 1.0)
    return jnp.array([ds, da], dtype=y.dtype)


def solve_hawkesn_mf(
    K: float,
    theta: float,
    I0_frac: float,
    times: jax.Array,
) -> jax.Array:
    """Integrate the HawkesN deterministic mean-field ODE.

    Parameters
    ----------
    K       : branching ratio (R₀ analogue)
    theta   : exponential decay rate
    I0_frac : initial infected fraction (I₀/N)
    times   : evaluation times, sorted, shape (T,); t0 = times[0]

    Returns
    -------
    ys : array shape (T, 2) — columns [s(t), a(t)]
    """
    K_j = jnp.array(K, dtype=jnp.float64)
    theta_j = jnp.array(theta, dtype=jnp.float64)

    s0 = jnp.array(1.0 - I0_frac, dtype=jnp.float64)
    # Each I0 seed contributes K·θ to A = N·a at t=0, so a(0) = I0_frac · K·θ
    a0 = jnp.array(I0_frac * K * theta, dtype=jnp.float64)
    y0 = jnp.array([s0, a0], dtype=jnp.float64)

    term = diffrax.ODETerm(_hawkesn_mf_rhs)
    solution = diffrax.diffeqsolve(
        term,
        solver=diffrax.Dopri5(),
        t0=times[0],
        t1=times[-1],
        dt0=0.01,
        y0=y0,
        args=(K_j, theta_j),
        saveat=diffrax.SaveAt(ts=times),
        stepsize_controller=diffrax.PIDController(rtol=ODE_RTOL, atol=ODE_ATOL),
        adjoint=diffrax.DirectAdjoint(),
        max_steps=16384,
    )
    return solution.ys  # shape (T, 2)


# ---------------------------------------------------------------------------
# SIR ODE (reuses wp02/sir.py — an independent implementation)
# ---------------------------------------------------------------------------

def solve_sir_bridge(
    K: float,
    theta: float,
    I0_frac: float,
    times: jax.Array,
) -> jax.Array:
    """Integrate the standard SIR ODE using wp02/sir.py.

    Parameters
    ----------
    K       : branching ratio; β = K·θ
    theta   : decay rate; γ = θ
    I0_frac : initial infected fraction
    times   : evaluation times

    Returns
    -------
    ys : array shape (T, 4) — columns [S, I, R, cumI] (all fractions)
    """
    beta = K * theta
    gamma = theta
    # theta_vec for solve_sir: [log_beta, log_gamma, log_scale, logit_i0]
    # scale=1.0 because we want fractions
    theta_vec = jnp.array(
        [
            jnp.log(beta),
            jnp.log(gamma),
            jnp.log(1.0),          # scale = 1 (fractions)
            jnp.log(I0_frac / (1.0 - I0_frac)),  # logit(i0)
        ],
        dtype=jnp.float64,
    )
    return solve_sir(theta_vec, times)  # shape (T, 4)


# ---------------------------------------------------------------------------
# Bridge comparison
# ---------------------------------------------------------------------------

def run_deterministic_bridge(
    K: float,
    theta: float,
    N: int = 1000,
    I0: int = 1,
    n_points: int = 500,
    T: float | None = None,
) -> dict:
    """Run the deterministic HawkesN mean-field vs SIR-ODE comparison.

    Pre-registered metric: relative max-abs deviation of I(t).
    Falsifier: ε > 1e-3 (0.1%).

    Parameters
    ----------
    K, theta : HawkesN parameters (branching ratio, decay rate)
    N        : population size (sets I0_frac = I0/N)
    I0       : initial infected count (absolute)
    n_points : number of time evaluation points
    T        : observation window; defaults to 12/theta (covers full epidemic)

    Returns
    -------
    dict with all pre-registered fields plus raw trajectories
    """
    I0_frac = float(I0) / float(N)
    if T is None:
        T = 12.0 / theta  # ≥12 decay times: well past epidemic peak

    times = jnp.linspace(0.0, T, n_points, dtype=jnp.float64)

    # --- HawkesN mean-field ODE (independent system in (s, a)) ---
    hawkesn_ys = solve_hawkesn_mf(K, theta, I0_frac, times)
    s_hawkesn = np.asarray(hawkesn_ys[:, 0])
    a_hawkesn = np.asarray(hawkesn_ys[:, 1])
    # Convert a → I using the derived relation I = a/(K·θ)
    I_hawkesn = a_hawkesn / (K * theta)

    # --- SIR ODE (independent implementation from wp02/sir.py) ---
    sir_ys = solve_sir_bridge(K, theta, I0_frac, times)
    S_sir = np.asarray(sir_ys[:, 0])
    I_sir = np.asarray(sir_ys[:, 1])

    # --- Pre-registered metric: relative max-abs ---
    residuals = I_hawkesn - I_sir
    peak_I_sir = float(np.max(I_sir))
    rel_maxabs = float(np.max(np.abs(residuals))) / peak_I_sir

    # Secondary (context only)
    rel_rms = float(np.sqrt(np.mean(residuals ** 2))) / peak_I_sir

    # Verdict (proposed, uncertified — per anti-self-certification rule)
    passes_falsifier = rel_maxabs < FALSIFIER_THRESHOLD

    return {
        # Configuration
        "K": K,
        "theta": theta,
        "N": N,
        "I0": I0,
        "I0_frac": I0_frac,
        "T": float(T),
        "n_points": n_points,
        # Pre-registered metric
        "rel_maxabs": rel_maxabs,
        "falsifier_threshold": FALSIFIER_THRESHOLD,
        "expected_noise_level": EXPECTED_NOISE_LEVEL,
        "passes_falsifier": passes_falsifier,  # proposed only, not certified
        # Secondary
        "rel_rms": rel_rms,
        # Trajectories for plotting
        "_times": np.asarray(times),
        "_s_hawkesn": s_hawkesn,
        "_a_hawkesn": a_hawkesn,
        "_I_hawkesn": I_hawkesn,
        "_S_sir": S_sir,
        "_I_sir": I_sir,
        "_residuals": residuals,
    }


# ---------------------------------------------------------------------------
# Plotting (pre-committed appearance described in module docstring)
# ---------------------------------------------------------------------------

def plot_bridge_deterministic(
    results: list[dict],
    out_path: Path,
) -> None:
    """Two-panel plot: I(t) overlay + residuals.

    Appearance if identity holds (pre-committed):
      Top panel: HawkesN mf and SIR I(t) curves visually identical.
      Bottom panel: residuals hugging zero, amplitude << 1% of peak.

    Appearance if identity fails:
      Visible gap, phase offset, or structured residual ≫ 1e-5.
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), sharex="col")
    if n == 1:
        axes = axes.reshape(2, 1)

    colors = {"hawkesn": "#1f77b4", "sir": "#d62728"}

    for col, res in enumerate(results):
        t = res["_times"]
        I_h = res["_I_hawkesn"]
        I_s = res["_I_sir"]
        resid = res["_residuals"]
        K = res["K"]
        theta = res["theta"]
        eps = res["rel_maxabs"]
        thresh = res["falsifier_threshold"]
        passes = res["passes_falsifier"]

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Top: overlay
        ax_top.plot(t, I_s * 100, color=colors["sir"], lw=2.0,
                    label="SIR ODE (wp02/sir.py)", zorder=3)
        ax_top.plot(t, I_h * 100, color=colors["hawkesn"], lw=1.5,
                    linestyle="--", label="HawkesN mean-field ODE", zorder=4)
        ax_top.set_ylabel("Infected fraction I(t) [%]")
        ax_top.set_title(
            f"K={K}, θ={theta}\n"
            f"ε = {eps:.2e}  |  threshold = {thresh:.0e}\n"
            f"{'PROPOSED PASS' if passes else 'PROPOSED FAIL'} (uncertified)"
        )
        ax_top.legend(fontsize=8)
        ax_top.set_xlim(t[0], t[-1])

        # Bottom: residuals (absolute, in units of % of peak)
        peak_pct = float(np.max(I_s)) * 100
        resid_pct = resid * 100
        ax_bot.axhline(0, color="gray", lw=0.7, zorder=1)
        ax_bot.axhline(thresh * peak_pct, color="orange", lw=1.0,
                       linestyle=":", label=f"falsifier @ {thresh*100:.1f}% of peak")
        ax_bot.axhline(-thresh * peak_pct, color="orange", lw=1.0, linestyle=":")
        ax_bot.plot(t, resid_pct, color="black", lw=1.2,
                    label=f"I_hawkesn - I_sir [%]")
        ax_bot.set_ylabel("Residual I_hawkes − I_sir [%]")
        ax_bot.set_xlabel("Time t")
        ax_bot.legend(fontsize=8)

    fig.suptitle(
        "Deterministic HawkesN Mean-Field vs SIR ODE — Bridge Test\n"
        "Pre-registered: ε = max|I_hawkesn − I_sir| / max(I_sir) < 1e-3 to pass\n"
        "Pre-committed expected appearance: curves visually identical, residuals ≈ 0",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(output_dir: Path | None = None) -> dict:
    """Run deterministic bridge test for multiple (K, theta) combinations.

    No random seeds needed — pure ODE integration.

    Returns
    -------
    summary : dict with all results and proposed (uncertified) verdict
    """
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "experiments" / "WP-12"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test cases: cover sub- and super-critical, different time scales
    test_cases = [
        {"K": 1.5, "theta": 0.3, "N": 1000, "I0": 1},
        {"K": 2.0, "theta": 0.3, "N": 1000, "I0": 1},
        {"K": 3.0, "theta": 0.3, "N": 1000, "I0": 1},
        {"K": 2.0, "theta": 1.0, "N": 500,  "I0": 1},
        {"K": 2.0, "theta": 0.1, "N": 2000, "I0": 1},
    ]

    results = []
    for tc in test_cases:
        print(f"Running K={tc['K']}, θ={tc['theta']}, N={tc['N']} ...")
        res = run_deterministic_bridge(**tc, n_points=500)
        # Remove trajectory arrays for JSON serialization (save separately)
        results.append(res)
        eps = res["rel_maxabs"]
        status = "PROPOSED PASS" if res["passes_falsifier"] else "PROPOSED FAIL"
        print(f"  ε (rel max-abs) = {eps:.3e}  →  {status} (uncertified)")
        print(f"  δ (rel rms)     = {res['rel_rms']:.3e}")

    # --- Plot: produce before reporting aggregate verdict ---
    # Use first 3 cases (K=1.5, 2.0, 3.0 at θ=0.3) for main plot
    plot_path = output_dir / "bridge_deterministic.png"
    plot_bridge_deterministic(results[:3], plot_path)
    print(f"Plot saved: {plot_path}")

    # Additional plot: θ variation
    plot_path_theta = output_dir / "bridge_deterministic_theta_var.png"
    plot_bridge_deterministic(
        [results[1], results[3], results[4]],  # K=2.0, θ=0.3,1.0,0.1
        plot_path_theta,
    )
    print(f"Plot saved: {plot_path_theta}")

    # --- Serialize results (without trajectory arrays) ---
    serializable = []
    for res in results:
        d = {k: v for k, v in res.items() if not k.startswith("_")}
        serializable.append(d)

    results_path = output_dir / "bridge_deterministic_results.json"
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results: {results_path}")

    # --- Config / reproducibility record (01-scientific-rigor §6) ---
    import importlib.metadata

    def _pkg_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except Exception:
            return "unknown"

    config_record = {
        "run_id": "wp12-bridge-det-001",
        "date": "2026-06-11",
        "no_random_seeds_needed": True,
        "ode_solver": "diffrax.Dopri5",
        "ode_rtol": ODE_RTOL,
        "ode_atol": ODE_ATOL,
        "falsifier_threshold": FALSIFIER_THRESHOLD,
        "expected_noise_level": EXPECTED_NOISE_LEVEL,
        "test_cases": test_cases,
        "library_versions": {
            "jax": _pkg_version("jax"),
            "diffrax": _pkg_version("diffrax"),
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
            "matplotlib": _pkg_version("matplotlib"),
        },
        "parameter_map": {
            "gamma_sir": "theta_hawkesn",
            "beta_sir": "K * theta_hawkesn",
            "R0": "K",
            "N_sir": "N_hawkesn",
            "I0_frac": "I0 / N",
        },
        "hawkesn_mf_ode": {
            "state": ["s (susceptible fraction)", "a (normalized excitation)"],
            "equations": [
                "ds/dt = -s * a",
                "da/dt = theta * a * (K*s - 1)",
            ],
            "initial_conditions": [
                "s(0) = 1 - I0/N",
                "a(0) = (I0/N) * K * theta",
            ],
            "converted_to_I_via": "I(t) = a(t) / (K * theta)",
        },
        "sir_ode": {
            "implementation": "experiments/wp02/sir.py: solve_sir()",
            "equations": [
                "dS/dt = -beta * S * I",
                "dI/dt = beta * S * I - gamma * I",
            ],
        },
        "non_circularity_note": (
            "HawkesN MF ODE is in (s,a) variables derived from the HawkesN "
            "intensity definition. SIR ODE is in (S,I) variables from wp02/sir.py. "
            "They share no code. Agreement tests the parameter map and both "
            "implementations simultaneously. Circularity risk: both systems are "
            "derived from the same analytic claim — a numerical match confirms "
            "implementation correctness, not the underlying theorem."
        ),
    }

    config_path = output_dir / "bridge_deterministic_config.json"
    with open(config_path, "w") as f:
        json.dump(config_record, f, indent=2)
    print(f"Config: {config_path}")

    # --- Overall proposed verdict ---
    all_pass = all(r["passes_falsifier"] for r in results)
    max_eps = max(r["rel_maxabs"] for r in results)
    worst_case = max(results, key=lambda r: r["rel_maxabs"])

    summary = {
        "proposed_verdict_uncertified": "PASS" if all_pass else "FAIL",
        "certified": False,
        "certifying_agent": None,
        "note": (
            "Proposed verdict only. Per AGENTS.md anti-self-certification rule, "
            "this result must be certified by the orchestrating agent or user "
            "after inspecting the plot and raw numbers. DO NOT record as certified "
            "in docs/02-state.md or WP-12 Outcome until reviewed."
        ),
        "max_rel_maxabs_across_cases": max_eps,
        "falsifier_threshold": FALSIFIER_THRESHOLD,
        "worst_case": {"K": worst_case["K"], "theta": worst_case["theta"]},
        "all_cases_below_threshold": all_pass,
        "per_case_results": serializable,
        "plots": [str(plot_path), str(plot_path_theta)],
        "config": str(config_path),
    }

    summary_path = output_dir / "bridge_deterministic_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    return summary


if __name__ == "__main__":
    result = main()
    print("\n=== PROPOSED VERDICT (UNCERTIFIED) ===")
    print(f"  Verdict:       {result['proposed_verdict_uncertified']}")
    print(f"  Max ε (rel):   {result['max_rel_maxabs_across_cases']:.3e}")
    print(f"  Threshold:     {result['falsifier_threshold']:.0e}")
    print(f"  All pass:      {result['all_cases_below_threshold']}")
    print(f"\n  NOTE: {result['note']}")
