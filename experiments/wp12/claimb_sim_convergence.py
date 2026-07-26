"""WP-12 Claim B — Stochastic HawkesN simulator convergence to mean-field.

PRE-REGISTRATION (written before any results are computed)
==========================================================

Claim under test (Claim B)
---------------------------
The stochastic HawkesN simulator (Ogata thinning, finite N) converges in the
mean-field sense to the certified mean-field I_mf(t) = a(t)/(Kθ) as N → ∞,
in the ε₀-fixed initial-fraction regime (I₀/N = ε₀ = 0.01).

This is a stochastic-process convergence claim, not the analytic identity of
Claim A.  Per method-discipline §2 (link-naming), this is a SEPARATE link from
the HawkesN-ODE ≡ SIR-ODE identity already certified as Claim A.

Claim type (method-discipline §2)
-----------------------------------
*Stochastic-estimator behaviour* — specifically:
  LLN link:  E[I_N(t)] → I_mf(t) as N → ∞ (bias of the sample-mean → 0)
  CLT link:  std[I_N(t)] ~ N^(−1/2) (fluctuations scale as Kurtz CLT predicts)

These two links are tested separately, as required by the approved corrections.

Pre-committed expected plot appearance (method-discipline §5) — BEFORE RESULTS
-------------------------------------------------------------------------------
Written here, pre-committed, before any numbers are seen:

PLOT 1 — Mean-bias scaling:
  If Claim B LLN link holds:
    - ε(N) = max_t |Ī_N(t) − I_mf(t)| / max_t I_mf(t) should DECREASE as N
      grows, with a log-log slope of approximately −1 (Kurtz system-size
      expansion: the bias of the sample mean is O(1/N)).
    - On the log-log plot, the points (log N, log ε) should fall on a
      downward-sloping line with slope ≈ −1.
    - Bootstrap CI for the slope should be entirely negative (upper bound < 0).
  If it fails:
    - ε(N) would plateau (flat line on log-log plot), increase, or show a
      non-monotone pattern; CI for the slope would include 0 or be positive.

PLOT 2 — Single-run std scaling:
  If Claim B CLT link holds (Kurtz fluctuation theorem):
    - σ_indiv(N) = std across R runs of the peak-time I value should scale as
      N^(−1/2) — log-log slope ≈ −0.5.
    - On the log-log plot, points should fall on a downward line, slope ≈ −0.5.
    - Bootstrap CI for the slope should include −0.5 and be entirely negative.
  If it fails:
    - Slope significantly different from −0.5 (e.g., 0 = no shrinkage = not
      a standard Kurtz fluctuation; or −1 = shrinking too fast).

PLOT 3 — Raw-time overlay (one case, multiple N):
  If converging:
    - All ensemble mean curves Ī_N(t) should cluster tightly around I_mf(t).
    - At large N (e.g. 4000, 8000), the mean curve should be nearly invisible
      from the mean-field.  No systematic phase lead or lag.
    - At small N (250, 500), a small phase lead is expected (finite-N effect;
      this is the failure mode flagged in the withdrawn test).
    - Width of the ensemble (±1 std band) should visibly narrow with N.
  If failing:
    - All N would show a systematic phase lead/lag that does NOT close with N;
      the mean curves would not converge to I_mf(t).

PLOT 4 — Milestone-based mean trajectory:
  Monotone milestone parameterization (cumulative-infected fraction) removes
  the rising-vs-falling ambiguity.  If converging, the milestone mean curves
  should overlay I_mf(t) at the same cumulative-infected levels, with the gap
  shrinking with N.

Pre-registered falsifier (correction 3 — mandatory)
----------------------------------------------------
Convergence of the mean is SUPPORTED iff:
  1. ε(N) is decreasing with N (visually on the log-log plot), AND
  2. The fitted log-log slope for ε(N) vs N has a 95% bootstrap CI with
     upper bound < 0 (significantly negative slope).

Convergence is FALSIFIED if:
  - ε(N) plateaus (CI includes 0) or increases (CI upper bound > 0).
  - A systematic phase lead does NOT close with N in Plot 3.

(Dropped: the old "ratio > 0.5" hard threshold tied to a 0.25 expectation.)

Pre-registered hypotheses (corrections 1 & 2)
----------------------------------------------
  H1 (LLN):  fitted mean-bias slope ≈ −1  (Kurtz O(1/N) bias)
  H2 (CLT):  fitted std slope ≈ −0.5      (Kurtz O(1/√N) fluctuations)

The metric blind spot (method-discipline §4)
--------------------------------------------
The max-time relative error metric ε(N) is peak-normalised and therefore
down-weights early and late time errors.  More specifically: it is a single
scalar summary of a trajectory.  A process that converges at the peak but has
a systematic phase lead or trailing offset would partially evade this metric.
The raw-time overlay plot (Plot 3) is the primary guard against this blind spot
— if the plot shows a phase lead, that is an open finding regardless of ε(N).

Scope caveat (correction 6 — mandatory)
-----------------------------------------
This validates the **ε₀-fixed mean-field family** only:
  - I₀/N = ε₀ = 0.01 fixed across all N.
  - This eliminates large take-off variance and ensures negligible extinctions
    at N ≥ 500.
Single-seed (I₀ = 1) cascades — the actual real-meme regime — are a DIFFERENT,
HARDER regime: large take-off jitter, real extinction probability, and
conditioning-on-survival bias.  That regime is NOT covered here.

Seeds (method-discipline §6, reproducibility)
----------------------------------------------
seed = case_index * 1_000_000 + N * 10_000 + r    (r = 0-indexed replication)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import importlib.metadata
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.wp12.hawkesn import HawkesNParams, simulate_hawkesn  # noqa: E402
from experiments.wp12.bridge_deterministic import solve_hawkesn_mf     # noqa: E402

import jax.numpy as jnp  # noqa: E402
import jax               # noqa: E402

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------

EPSILON_0 = 0.01         # fixed initial fraction I₀/N
R_REPS = 1000            # replications per (N, case)
N_SEQUENCE = [250, 500, 1000, 2000, 4000, 8000]
PARAMETER_CASES = [
    {"K": 2.0, "theta": 0.3, "label": "K=2.0,θ=0.3 (supercritical)"},
    {"K": 1.5, "theta": 0.3, "label": "K=1.5,θ=0.3 (mild supercritical)"},
    {"K": 3.0, "theta": 0.3, "label": "K=3.0,θ=0.3 (strong supercritical)"},
]
T_FACTOR = 12.0 / 0.3    # T = 12/θ = 40.0 for θ=0.3
N_TIME_POINTS = 400       # evaluation grid for mean-field ODE and raw overlay
N_MILESTONES = 10         # number of cumulative-infected milestones


# ---------------------------------------------------------------------------
# Helper: get mean-field curve and cumulative-infected milestone times
# ---------------------------------------------------------------------------

def _get_mf(K: float, theta: float, T: float, n_pts: int = N_TIME_POINTS):
    """Return (times, I_mf, cumI_mf) where cumI_mf = 1 − S(t).
    Milestones on the cumI axis are monotone — no rising/falling ambiguity.
    """
    times = jnp.linspace(0.0, T, n_pts, dtype=jnp.float64)
    ys = solve_hawkesn_mf(K, theta, EPSILON_0, times)  # (n_pts, 2): [s, a]
    s_mf = np.asarray(ys[:, 0])
    a_mf = np.asarray(ys[:, 1])
    I_mf = a_mf / (K * theta)                          # I_mf(t) = a(t)/(Kθ)
    cumI_mf = 1.0 - s_mf                               # cumulative infected = 1-S
    times_np = np.asarray(times)
    return times_np, I_mf, cumI_mf


def _milestone_levels(cumI_mf: np.ndarray, n_milestones: int) -> np.ndarray:
    """Return n_milestones evenly-spaced levels of cumI in (0, max(cumI)).

    Parameterized by cumulative-infected fraction → monotone, no rising/falling
    ambiguity (correction 4).  Levels span from 5% to 95% of the final level.
    """
    max_cumI = cumI_mf[-1]
    levels = np.linspace(0.05 * max_cumI, 0.95 * max_cumI, n_milestones)
    return levels


def _milestone_times(cumI_mf: np.ndarray, times: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Find the times at which cumI_mf first reaches each level (monotone — unique)."""
    milestone_ts = []
    for lv in levels:
        idx = np.searchsorted(cumI_mf, lv)
        idx = np.clip(idx, 0, len(times) - 1)
        milestone_ts.append(times[idx])
    return np.array(milestone_ts)


# ---------------------------------------------------------------------------
# Helper: convert event times to I(t) on a fixed grid
# ---------------------------------------------------------------------------

def _events_to_I_grid(
    event_times: np.ndarray,
    times: np.ndarray,
    K: float,
    theta: float,
    N: int,
    I0: int,
) -> np.ndarray:
    """Convert a simulated event sequence to I(t) on a fixed time grid.

    I(t) = (excitation sum at t) / (K * theta) = a(t),
    where a(t) = (1/N) * sum_{t_i <= t} K*theta * exp(-theta*(t-t_i)).

    This matches the mean-field ODE state a(t) = normalized excitation.
    Seed events (I0 at t=0) are included in the sum.

    Parameters
    ----------
    event_times : observed event times (NOT including seeds)
    times       : evaluation grid
    K, theta    : model parameters
    N           : population size (for normalization)
    I0          : initial infected (seeds at t=0)

    Returns
    -------
    I_grid : array shape (len(times),)
    """
    # All event times including seeds at t=0
    seeds = np.zeros(I0, dtype=np.float64)
    all_events = np.concatenate([seeds, event_times]) if len(event_times) > 0 else seeds

    I_grid = np.zeros(len(times), dtype=np.float64)
    for j, t in enumerate(times):
        mask = all_events <= t
        if not np.any(mask):
            I_grid[j] = 0.0
        else:
            dt = t - all_events[mask]
            contrib = np.sum(np.exp(-theta * dt))
            # a(t) = (1/N) * K*theta * sum exp(-theta*(t-ti))
            # I(t) = a(t)/(K*theta) = (1/N) * sum exp(...)
            I_grid[j] = contrib / N
    return I_grid


# ---------------------------------------------------------------------------
# Run simulations for one (N, case) block
# ---------------------------------------------------------------------------

def run_block(
    case_idx: int,
    K: float,
    theta: float,
    N: int,
    T: float,
    times: np.ndarray,
    I_mf: np.ndarray,
    mf_peak_idx: int,
) -> dict:
    """Run R_REPS replications for one (case, N) block and return statistics.

    Parameters
    ----------
    case_idx    : case index (for seed formula)
    K, theta    : model parameters
    N           : population size
    T           : end time
    times       : fixed time grid
    I_mf        : certified mean-field I(t) on the time grid
    mf_peak_idx : index of the peak of I_mf (for σ_indiv measurement)

    Returns
    -------
    dict with:
      - mean_I : mean trajectory over R_REPS runs, shape (len(times),)
      - std_I  : std trajectory over R_REPS runs, shape (len(times),)
      - sigma_indiv : std of peak-time I across runs (scalar)
      - n_runs : R_REPS
    """
    I0 = max(1, round(EPSILON_0 * N))
    params = HawkesNParams(K=K, theta=theta, N=float(N), mu=0.0)

    trajectories = np.zeros((R_REPS, len(times)), dtype=np.float64)

    for r in range(R_REPS):
        seed = case_idx * 1_000_000 + N * 10_000 + r
        rng = np.random.default_rng(seed)
        ev = simulate_hawkesn(params, T, rng, I0=I0)
        traj = _events_to_I_grid(ev, times, K, theta, N, I0)
        trajectories[r] = traj

    mean_I = trajectories.mean(axis=0)
    std_I = trajectories.std(axis=0, ddof=1)

    # σ_indiv: std of I at the mean-field peak time (correction 2)
    sigma_indiv = float(std_I[mf_peak_idx])

    return {
        "mean_I": mean_I,
        "std_I": std_I,
        "sigma_indiv": sigma_indiv,
        "n_runs": R_REPS,
        "I0": I0,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI for log-log slope
# ---------------------------------------------------------------------------

def fit_loglog_slope_with_ci(
    xs: np.ndarray,
    ys: np.ndarray,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    rng_seed: int = 99,
) -> dict:
    """Fit log(y) = slope * log(x) + intercept; return slope with bootstrap CI.

    Parameters
    ----------
    xs, ys     : 1-D arrays of positive values
    n_boot     : number of bootstrap replicates
    ci_level   : confidence level (0.95 → 95% CI)
    rng_seed   : seed for reproducibility

    Returns
    -------
    dict: slope, intercept, ci_lo, ci_hi, ci_level, n_boot
    """
    log_x = np.log(xs)
    log_y = np.log(ys)

    # OLS fit
    A = np.column_stack([log_x, np.ones_like(log_x)])
    slope, intercept = np.linalg.lstsq(A, log_y, rcond=None)[0]

    # Bootstrap CI (resample points)
    rng = np.random.default_rng(rng_seed)
    n = len(xs)
    boot_slopes = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        lx_b = log_x[idx]
        ly_b = log_y[idx]
        A_b = np.column_stack([lx_b, np.ones_like(lx_b)])
        try:
            s_b, _ = np.linalg.lstsq(A_b, ly_b, rcond=None)[0]
            boot_slopes[b] = s_b
        except Exception:
            boot_slopes[b] = slope

    alpha = 1.0 - ci_level
    ci_lo = float(np.quantile(boot_slopes, alpha / 2))
    ci_hi = float(np.quantile(boot_slopes, 1.0 - alpha / 2))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_level": ci_level,
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Compute ε(N) — mean-bias error metric
# ---------------------------------------------------------------------------

def compute_epsilon(mean_I: np.ndarray, I_mf: np.ndarray) -> float:
    """Compute relative max-abs bias of sample mean vs mean-field.

    ε = max_t |Ī(t) − I_mf(t)| / max_t I_mf(t)
    """
    peak_mf = float(np.max(I_mf))
    return float(np.max(np.abs(mean_I - I_mf))) / peak_mf


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bias_scaling(
    case_results: list[dict],
    out_path: Path,
) -> None:
    """Plot 1: ε(N) log-log scaling for all cases (mean-bias, slope ≈ −1 expected)."""
    fig, axes = plt.subplots(1, len(case_results), figsize=(6 * len(case_results), 5),
                             sharey=False)
    if len(case_results) == 1:
        axes = [axes]

    for ax, cr in zip(axes, case_results):
        Ns = np.array([r["N"] for r in cr["N_results"]])
        eps = np.array([r["epsilon"] for r in cr["N_results"]])
        slope_info = cr["bias_slope"]

        ax.loglog(Ns, eps, "o-", color="#1f77b4", ms=8, lw=2, label=f"ε(N) measured")
        # Reference N^{-1} line
        ref = eps[0] * (Ns[0] / Ns)
        ax.loglog(Ns, ref, "--", color="gray", lw=1.5, label="N^{-1} reference")
        # Reference N^{-0.5} line
        ref05 = eps[0] * (Ns[0] / Ns) ** 0.5
        ax.loglog(Ns, ref05, ":", color="brown", lw=1.2, label="N^{-0.5} reference")

        slope = slope_info["slope"]
        ci_lo = slope_info["ci_lo"]
        ci_hi = slope_info["ci_hi"]
        ax.set_xlabel("N")
        ax.set_ylabel("ε(N) = max|Ī(t) − I_mf(t)| / max(I_mf)")
        ax.set_title(
            f"{cr['label']}\n"
            f"fitted slope = {slope:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]\n"
            f"H1: slope ≈ −1;  CI upper < 0: {'YES' if ci_hi < 0 else 'NO'}"
        )
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Claim B — Mean-bias convergence ε(N) (pre-registered: slope ≈ −1)\n"
        "PRE-COMMITTED: expect downward slope ≈ −1 on log-log; CI upper bound < 0\n"
        "PROPOSED / UNCERTIFIED",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_std_scaling(
    case_results: list[dict],
    out_path: Path,
) -> None:
    """Plot 2: σ_indiv(N) log-log scaling (CLT link, slope ≈ −1/2 expected)."""
    fig, axes = plt.subplots(1, len(case_results), figsize=(6 * len(case_results), 5),
                             sharey=False)
    if len(case_results) == 1:
        axes = [axes]

    for ax, cr in zip(axes, case_results):
        Ns = np.array([r["N"] for r in cr["N_results"]])
        sigmas = np.array([r["sigma_indiv"] for r in cr["N_results"]])
        slope_info = cr["std_slope"]

        ax.loglog(Ns, sigmas, "s-", color="#d62728", ms=8, lw=2, label="σ_indiv(N) measured")
        ref05 = sigmas[0] * (Ns[0] / Ns) ** 0.5
        ax.loglog(Ns, ref05, "--", color="gray", lw=1.5, label="N^{-0.5} reference")
        ref1 = sigmas[0] * (Ns[0] / Ns)
        ax.loglog(Ns, ref1, ":", color="brown", lw=1.2, label="N^{-1} reference")

        slope = slope_info["slope"]
        ci_lo = slope_info["ci_lo"]
        ci_hi = slope_info["ci_hi"]
        ax.set_xlabel("N")
        ax.set_ylabel("σ_indiv(N) = std[I(t_peak)] across R runs")
        ax.set_title(
            f"{cr['label']}\n"
            f"fitted slope = {slope:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]\n"
            f"H2: slope ≈ −0.5;  CI upper < 0: {'YES' if ci_hi < 0 else 'NO'}"
        )
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Claim B — Single-run std σ_indiv(N) (pre-registered: slope ≈ −0.5, Kurtz CLT)\n"
        "PRE-COMMITTED: expect downward slope ≈ −0.5 on log-log\n"
        "PROPOSED / UNCERTIFIED",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_raw_time_overlay(
    case_label: str,
    N_results_for_case: list[dict],
    I_mf: np.ndarray,
    times: np.ndarray,
    K: float,
    theta: float,
    out_path: Path,
) -> None:
    """Plot 3: Raw-time overlay of mean trajectory ± 1 std for multiple N.

    Pre-committed: if converging, mean curves cluster around I_mf; no systematic
    phase lead at large N.  Shaded band narrows with N.
    """
    n_N = len(N_results_for_case)
    fig, axes = plt.subplots(1, n_N, figsize=(5 * n_N, 4), sharex=True, sharey=True)
    if n_N == 1:
        axes = [axes]

    cmap = plt.cm.Blues
    for ax, nr in zip(axes, N_results_for_case):
        N_val = nr["N"]
        mean_I = nr["mean_I"]
        std_I = nr["std_I"]

        ax.fill_between(times, mean_I - std_I, mean_I + std_I,
                        alpha=0.3, color="#1f77b4", label="±1 std")
        ax.plot(times, mean_I, color="#1f77b4", lw=1.8, label=f"Ī_N(t) (N={N_val})")
        ax.plot(times, I_mf, color="#d62728", lw=2.0, ls="--", label="I_mf(t)")

        eps_val = nr["epsilon"]
        ax.set_title(f"N={N_val}\nε={eps_val:.4f}", fontsize=9)
        ax.set_xlabel("time t")
        ax.set_ylabel("I(t)")
        ax.legend(fontsize=7)
        ax.set_xlim(times[0], times[-1])

    fig.suptitle(
        f"Claim B — Raw-time overlay: {case_label}\n"
        f"PRE-COMMITTED: mean curves converge to I_mf(t); no phase lead at large N\n"
        f"K={K}, θ={theta} | PROPOSED / UNCERTIFIED",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def plot_milestone_overlay(
    case_label: str,
    N_results_for_case: list[dict],
    I_mf: np.ndarray,
    times: np.ndarray,
    milestone_ts: np.ndarray,
    K: float,
    theta: float,
    out_path: Path,
) -> None:
    """Plot 4: milestone-parameterized mean trajectory overlay.

    Milestone times are where cumulative infected fraction reaches evenly spaced
    levels — monotone, no rising/falling ambiguity (correction 4).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.Blues
    colors = [cmap(0.3 + 0.7 * i / (len(N_results_for_case) - 1))
              for i in range(len(N_results_for_case))]

    ax.plot(times, I_mf, color="#d62728", lw=2.5, ls="--", zorder=10, label="I_mf(t)")

    for color, nr in zip(colors, N_results_for_case):
        N_val = nr["N"]
        mean_I = nr["mean_I"]
        ax.plot(times, mean_I, color=color, lw=1.5, label=f"N={N_val}", alpha=0.85)
        # Mark milestones
        for mt in milestone_ts:
            idx = np.searchsorted(times, mt)
            idx = np.clip(idx, 0, len(times) - 1)
            ax.axvline(mt, color="gray", lw=0.5, alpha=0.4)

    ax.set_xlabel("time t")
    ax.set_ylabel("I(t) = prevalence fraction")
    ax.set_title(
        f"Claim B — Milestone overlay: {case_label}\n"
        f"K={K}, θ={theta}  |  Milestones: cumulative-infected fraction (monotone)"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(times[0], times[-1])
    fig.suptitle("PROPOSED / UNCERTIFIED", fontsize=9, color="red")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Residual bias diagnostic (correction 5)
# ---------------------------------------------------------------------------

def diagnose_residual_bias(case_results: list[dict]) -> dict:
    """Correction 5: diagnose whether residual bias is genuine O(1/N) or a bug.

    If the fitted log-log slope for ε(N) is consistently near −1 and continues
    to decrease at the largest N values, that is the genuine finite-N correction.
    If ε plateaus at a nonzero offset at large N, that points to a systematic bug
    (e.g. I₀-seed contribution to a(0) / depletion init).

    We assess this by:
    1. Checking whether the last two ε values continue to decrease.
    2. Comparing the extrapolated (slope * log N + intercept) to the observed
       large-N values.
    """
    diagnoses = []
    for cr in case_results:
        Ns = np.array([r["N"] for r in cr["N_results"]])
        eps = np.array([r["epsilon"] for r in cr["N_results"]])
        slope_info = cr["bias_slope"]
        slope = slope_info["slope"]

        # Ratio of last to second-last ε (should be < 1 if decreasing)
        last_ratio = float(eps[-1] / eps[-2]) if eps[-2] > 0 else np.nan
        # Expected ratio at the two largest N from the OLS fit
        expected_ratio = (Ns[-1] / Ns[-2]) ** slope

        # Is it still decreasing at largest N?
        still_decreasing = bool(eps[-1] < eps[-2])

        # Does the OLS line extrapolate correctly (ratio close to expected)?
        ratio_matches_fit = abs(last_ratio - expected_ratio) < 0.3 * abs(expected_ratio - 1.0) + 0.05

        if still_decreasing and slope < -0.5:
            diagnosis = "GENUINE_O(1/N): ε continues to decrease at large N; consistent with finite-N Kurtz correction."
        elif not still_decreasing:
            diagnosis = "POSSIBLE_BUG_OR_FLOOR: ε does not decrease from N={} to N={}; may be noise floor or systematic offset.".format(
                int(Ns[-2]), int(Ns[-1]))
        else:
            diagnosis = "UNCLEAR: slope is weak; inspect plots to determine if plateau or slow decrease."

        diagnoses.append({
            "label": cr["label"],
            "slope": slope,
            "eps_last_two": [float(eps[-2]), float(eps[-1])],
            "N_last_two": [int(Ns[-2]), int(Ns[-1])],
            "last_ratio": last_ratio,
            "expected_ratio_from_fit": float(expected_ratio),
            "still_decreasing_at_large_N": still_decreasing,
            "diagnosis": diagnosis,
        })
    return {"diagnoses": diagnoses}


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(output_dir: Path | None = None) -> dict:
    """Run Claim B stochastic convergence test."""
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "experiments" / "WP-12"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print("=" * 70)
    print("WP-12 Claim B — Stochastic HawkesN mean-field convergence test")
    print("PRE-REGISTERED FALSIFIER (before any results):")
    print("  Convergence SUPPORTED iff:")
    print("    1. ε(N) is decreasing with N (log-log plot), AND")
    print("    2. fitted slope CI upper bound < 0 (significantly negative)")
    print("  Convergence FALSIFIED if ε plateaus or CI includes 0.")
    print()
    print("PRE-COMMITTED EXPECTED APPEARANCE:")
    print("  Plot 1 (ε(N)): downward slope ≈ −1 on log-log")
    print("  Plot 2 (σ_indiv): downward slope ≈ −0.5 on log-log")
    print("  Plot 3 (raw overlay): mean curves converge to I_mf; no phase lead at large N")
    print("  Plot 4 (milestone overlay): same convergence, without rising/falling ambiguity")
    print("=" * 70)
    print()

    T = T_FACTOR  # 40.0

    # Pre-compute mean-field curves for each case
    case_mf = []
    for case_idx, case in enumerate(PARAMETER_CASES):
        K, theta = case["K"], case["theta"]
        times, I_mf, cumI_mf = _get_mf(K, theta, T, n_pts=N_TIME_POINTS)
        mf_peak_idx = int(np.argmax(I_mf))
        levels = _milestone_levels(cumI_mf, N_MILESTONES)
        milestone_ts = _milestone_times(cumI_mf, times, levels)
        case_mf.append({
            "K": K,
            "theta": theta,
            "label": case["label"],
            "times": times,
            "I_mf": I_mf,
            "cumI_mf": cumI_mf,
            "mf_peak_idx": mf_peak_idx,
            "milestone_levels": levels,
            "milestone_ts": milestone_ts,
        })

    # Run all (case, N) blocks
    all_case_results = []
    for case_idx, cm in enumerate(case_mf):
        K, theta = cm["K"], cm["theta"]
        times = cm["times"]
        I_mf = cm["I_mf"]
        mf_peak_idx = cm["mf_peak_idx"]
        print(f"\nCase {case_idx}: {cm['label']}")

        N_results = []
        for N in N_SEQUENCE:
            I0 = max(1, round(EPSILON_0 * N))
            print(f"  N={N:5d}  I0={I0:3d}  (R={R_REPS} reps)  ...", end="", flush=True)
            t0 = time.time()

            block = run_block(case_idx, K, theta, N, T, times, I_mf, mf_peak_idx)
            epsilon = compute_epsilon(block["mean_I"], I_mf)

            print(f"  ε={epsilon:.5f}  σ_indiv={block['sigma_indiv']:.5f}  "
                  f"[{time.time()-t0:.1f}s]")

            N_results.append({
                "N": N,
                "I0": block["I0"],
                "n_runs": block["n_runs"],
                "epsilon": epsilon,
                "sigma_indiv": block["sigma_indiv"],
                # Store arrays for plotting (not JSON-serialized later)
                "_mean_I": block["mean_I"],
                "_std_I": block["std_I"],
            })

        # Fit log-log slopes
        Ns_arr = np.array([r["N"] for r in N_results], dtype=float)
        eps_arr = np.array([r["epsilon"] for r in N_results])
        sigma_arr = np.array([r["sigma_indiv"] for r in N_results])

        bias_slope = fit_loglog_slope_with_ci(Ns_arr, eps_arr, n_boot=2000,
                                              rng_seed=case_idx * 100 + 1)
        std_slope = fit_loglog_slope_with_ci(Ns_arr, sigma_arr, n_boot=2000,
                                             rng_seed=case_idx * 100 + 2)

        print(f"  Bias slope: {bias_slope['slope']:.3f}  95% CI [{bias_slope['ci_lo']:.3f}, {bias_slope['ci_hi']:.3f}]")
        print(f"  Std  slope: {std_slope['slope']:.3f}  95% CI [{std_slope['ci_lo']:.3f}, {std_slope['ci_hi']:.3f}]")

        # Determine falsifier outcome
        falsifier_supported = (
            bool(eps_arr[-1] < eps_arr[0]) and
            bool(bias_slope["ci_hi"] < 0)
        )
        falsifier_outcome = "SUPPORTED" if falsifier_supported else "FALSIFIED"
        print(f"  Falsifier outcome: {falsifier_outcome}")

        all_case_results.append({
            "case_idx": case_idx,
            "K": K,
            "theta": theta,
            "label": cm["label"],
            "N_results": N_results,
            "bias_slope": bias_slope,
            "std_slope": std_slope,
            "falsifier_outcome": falsifier_outcome,
            "falsifier_supported": falsifier_supported,
        })

    # -----------------------------------------------------------------------
    # Plots (produced before final verdict — §5)
    # -----------------------------------------------------------------------

    # Plot 1: ε(N) scaling (bias, slope ≈ −1)
    # Attach mean_I arrays for plotting
    for cr in all_case_results:
        for nr in cr["N_results"]:
            nr["mean_I"] = nr["_mean_I"]
            nr["std_I"] = nr["_std_I"]

    plot1_path = output_dir / "claimb_bias_scaling.png"
    plot_bias_scaling(all_case_results, plot1_path)

    # Plot 2: σ_indiv(N) scaling (CLT, slope ≈ −0.5)
    plot2_path = output_dir / "claimb_std_scaling.png"
    plot_std_scaling(all_case_results, plot2_path)

    # Plot 3 & 4: Per-case overlays (use case 0 = K=2.0 for primary)
    plot_paths_overlay = []
    plot_paths_milestone = []
    for case_idx, cr in enumerate(all_case_results):
        cm = case_mf[case_idx]
        times = cm["times"]
        I_mf = cm["I_mf"]

        p3 = output_dir / f"claimb_raw_overlay_case{case_idx}.png"
        plot_raw_time_overlay(
            cr["label"], cr["N_results"], I_mf, times,
            cr["K"], cr["theta"], p3
        )
        plot_paths_overlay.append(str(p3))

        p4 = output_dir / f"claimb_milestone_overlay_case{case_idx}.png"
        plot_milestone_overlay(
            cr["label"], cr["N_results"], I_mf, times,
            cm["milestone_ts"], cr["K"], cr["theta"], p4
        )
        plot_paths_milestone.append(str(p4))

    # -----------------------------------------------------------------------
    # Residual bias diagnosis (correction 5)
    # -----------------------------------------------------------------------
    diagnosis_result = diagnose_residual_bias(all_case_results)
    print("\n--- Residual bias diagnosis ---")
    for d in diagnosis_result["diagnoses"]:
        print(f"  {d['label']}:")
        print(f"    {d['diagnosis']}")

    # -----------------------------------------------------------------------
    # Summary and verdict
    # -----------------------------------------------------------------------
    overall_supported = all(cr["falsifier_supported"] for cr in all_case_results)
    proposed_verdict = "CONVERGENCE SUPPORTED" if overall_supported else "CONVERGENCE FALSIFIED OR UNCERTAIN"

    t_elapsed = time.time() - t_start
    print(f"\nTotal elapsed time: {t_elapsed:.1f}s")
    print(f"\nPROPOSED (UNCERTIFIED) VERDICT: {proposed_verdict}")

    # -----------------------------------------------------------------------
    # Serialize results (strip numpy arrays)
    # -----------------------------------------------------------------------
    def _pkg_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except Exception:
            return "unknown"

    library_versions = {
        "numpy": _pkg_version("numpy"),
        "jax": _pkg_version("jax"),
        "diffrax": _pkg_version("diffrax"),
        "scipy": _pkg_version("scipy"),
        "matplotlib": _pkg_version("matplotlib"),
    }

    # Build JSON-safe results
    json_cases = []
    for cr in all_case_results:
        json_N_results = []
        for nr in cr["N_results"]:
            json_N_results.append({
                "N": nr["N"],
                "I0": nr["I0"],
                "n_runs": nr["n_runs"],
                "epsilon": nr["epsilon"],
                "sigma_indiv": nr["sigma_indiv"],
            })
        json_cases.append({
            "case_idx": cr["case_idx"],
            "K": cr["K"],
            "theta": cr["theta"],
            "label": cr["label"],
            "N_results": json_N_results,
            "bias_slope": cr["bias_slope"],
            "std_slope": cr["std_slope"],
            "falsifier_outcome": cr["falsifier_outcome"],
            "falsifier_supported": cr["falsifier_supported"],
        })

    config_and_results = {
        "run_id": "wp12-claimb-001",
        "date": "2026-06-11",
        "claim": "B — stochastic HawkesN simulator converges to mean-field as N→∞",
        "regime": "epsilon0-fixed (I0/N = 0.01 fixed)",
        "scope_caveat": (
            "This validates the ε₀-fixed mean-field family ONLY. "
            "Single-seed (I₀=1) cascades — the actual real-meme regime — are "
            "a DIFFERENT, HARDER regime (large take-off jitter, real extinction) "
            "and are NOT covered here."
        ),
        "pre_registered_falsifier": {
            "supported_condition": "ε(N) decreasing AND bias slope CI upper bound < 0",
            "falsified_condition": "ε plateaus (CI includes 0) or increases",
            "dropped_rule": "old 'ratio > 0.5' hard threshold tied to 0.25 expectation — DROPPED per correction 3",
        },
        "pre_registered_hypotheses": {
            "H1_LLN": "fitted mean-bias slope ≈ −1  (Kurtz O(1/N) bias)",
            "H2_CLT": "fitted std slope ≈ −0.5  (Kurtz O(1/√N) fluctuations)",
        },
        "metric_blind_spot": (
            "Max-time relative error ε(N) is peak-normalised and summarises "
            "a whole trajectory in one scalar. A process with a systematic "
            "phase lead or trailing offset would partially evade this metric. "
            "The raw-time overlay plot is the primary guard against this blind spot."
        ),
        "milestone_choice": (
            "Milestones parameterized by cumulative-infected fraction (monotone) — "
            "no rising/falling ambiguity. 10 levels from 5% to 95% of final cumI."
        ),
        "parameters": {
            "epsilon_0": EPSILON_0,
            "R_reps": R_REPS,
            "N_sequence": N_SEQUENCE,
            "T": T,
            "parameter_cases": PARAMETER_CASES,
        },
        "seed_formula": "seed = case_idx * 1_000_000 + N * 10_000 + r  (r = 0-indexed)",
        "library_versions": library_versions,
        "case_results": json_cases,
        "residual_bias_diagnosis": diagnosis_result,
        "proposed_verdict_uncertified": proposed_verdict,
        "overall_supported": overall_supported,
        "elapsed_seconds": t_elapsed,
        "certification_note": (
            "PROPOSED / UNCERTIFIED. Per AGENTS.md anti-self-certification rule, "
            "this verdict must be reviewed by the orchestrating agent or user "
            "after inspecting plots and raw numbers. DO NOT record as certified."
        ),
        "plots": {
            "bias_scaling": str(plot1_path),
            "std_scaling": str(plot2_path),
            "raw_overlay": plot_paths_overlay,
            "milestone_overlay": plot_paths_milestone,
        },
    }

    config_path = output_dir / "claimb_config_results.json"
    with open(config_path, "w") as f:
        json.dump(config_and_results, f, indent=2)
    print(f"\nConfig+results: {config_path}")

    return config_and_results


if __name__ == "__main__":
    result = main()
    print("\n" + "=" * 70)
    print("PROPOSED VERDICT (UNCERTIFIED) — must be graded by orchestrator/user")
    print("=" * 70)
    for cr in result["case_results"]:
        print(f"\n  {cr['label']}")
        print(f"    Bias slope: {cr['bias_slope']['slope']:.3f}  "
              f"CI [{cr['bias_slope']['ci_lo']:.3f}, {cr['bias_slope']['ci_hi']:.3f}]")
        print(f"    Std  slope: {cr['std_slope']['slope']:.3f}  "
              f"CI [{cr['std_slope']['ci_lo']:.3f}, {cr['std_slope']['ci_hi']:.3f}]")
        print(f"    Falsifier: {cr['falsifier_outcome']}")
    print(f"\n  Overall: {result['proposed_verdict_uncertified']}")
    print(f"\n  {result['certification_note']}")
