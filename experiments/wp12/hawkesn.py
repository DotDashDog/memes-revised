"""JAX HawkesN simulator, log-likelihood, and MLE/MAP fitter.

HawkesN (Rizoiu et al. WWW 2018) is a finite-population Hawkes process whose
conditional intensity depletes as events accumulate:

    λ*(t) = μ + (1 - N_t/N) · Σ_{t_i < t} K·θ·exp(-θ·(t - t_i))

where
    N       = finite population size
    N_t     = number of events in [0, t)
    K       = branching ratio  (≡ R₀ in the SIR bridge)
    θ       = exponential-decay rate of influence (≡ γ in SIR bridge; β = K·θ)
    μ       = background / immigration rate

Parameter map to WP-02 SIR parameterisation
--------------------------------------------
    γ = θ           (recovery rate ↔ decay rate)
    β = K·θ         (infection rate ↔ K × decay rate)
    R₀ = β/γ = K    (basic reproduction number = branching ratio)
    N_hawkes = N_sir (finite population size in both models)
    I₀/N ≈ initial fraction infected

The mean-field bridge holds only when μ = 0 (no external immigration).

Interface contract (Branch-A reusable interface, per ADR 0014)
--------------------------------------------------------------
    fit_hawkesn(event_times, N=None, ...)
        -> (params: HawkesNParams, uncertainty: dict)

Works on a SINGLE event stream (one cluster's timestamps) with no dependence
on holistic/field machinery.

Notes
-----
- Ogata thinning simulation: numpy-based (random state passed explicitly).
- Log-likelihood: pure JAX (differentiable, supports jit/grad).
- MLE: JAX value+grad → scipy L-BFGS-B.
- All x64 arithmetic.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HawkesNParams:
    """Natural HawkesN parameters.

    Attributes
    ----------
    K  : branching ratio (R₀ analogue); must be in (0, 1) for sub-critical
    theta : exponential decay rate θ > 0
    N  : finite population size > 0
    mu : background rate μ ≥ 0 (set to 0 for pure SIR bridge)
    """

    K: float
    theta: float
    N: float
    mu: float = 0.0


# ---------------------------------------------------------------------------
# Simulator (Ogata / Lewis-Shedler thinning)
# ---------------------------------------------------------------------------


def simulate_hawkesn(
    params: HawkesNParams,
    T: float,
    rng: np.random.Generator,
    *,
    I0: int = 1,
    max_events: int = 100_000,
) -> np.ndarray:
    """Simulate a HawkesN process on [0, T] via Ogata thinning.

    Uses the depleting kernel:
        λ*(t) = (1 - n/N) · [μ + Σ_{t_i < t} K·θ·exp(-θ·(t-tᵢ))]
    where n = number of events so far (counting both seed and generated events).

    The SIR bridge requires I0 > 0 initial infected.  These are modelled as
    I0 seed events at t = 0⁻ (just before observation) that contribute to the
    excitation kernel at t=0 but are NOT included in the returned event times
    (they correspond to initial conditions, not observed cascade events).

    Parameters
    ----------
    params  : HawkesNParams
    T       : end time (float)
    rng     : numpy random Generator (seed externally for reproducibility)
    I0      : number of initially infected (seed events at t=0⁻)
    max_events : safety cap on number of events

    Returns
    -------
    event_times : np.ndarray of shape (n_events,), sorted, in (0, T]
                  Does NOT include the I0 seed events.
    """
    K, theta, N, mu = params.K, params.theta, params.N, params.mu

    events: list[float] = []
    t = 0.0

    # Initialise kernel_sum to account for I0 seed events at t=0⁻.
    # Each seed event contributes K·θ·exp(-θ·(t - 0)) = K·θ at t=0.
    # Since they're at t=0, kernel_sum_0 = I0 * K * theta.
    kernel_sum = float(I0) * K * theta  # at t=0

    # Track total consumed (seed events count toward depletion)
    n_consumed = int(I0)  # already consumed by seed events

    while t < T and len(events) < max_events:
        n_so_far = n_consumed + len(events)
        depletion = max(1.0 - n_so_far / N, 0.0)

        # Upper bound: λ*(t) is non-increasing between events.
        # Use current value as the upper bound.
        lambda_bar = depletion * (mu + kernel_sum)
        # add a small floor to avoid infinite wait when intensity hits 0
        lambda_bar = max(lambda_bar, 1e-15)

        # Propose next arrival
        u1 = rng.uniform()
        dt = -np.log(u1) / lambda_bar
        t_candidate = t + dt

        if t_candidate >= T:
            break

        # Decay kernel_sum to t_candidate
        kernel_sum_at_candidate = kernel_sum * np.exp(-theta * dt)

        # Evaluate true intensity at t_candidate (depletion unchanged)
        lambda_true = depletion * (mu + kernel_sum_at_candidate)

        # Thinning acceptance
        u2 = rng.uniform()
        if u2 <= lambda_true / max(lambda_bar, 1e-300):
            events.append(t_candidate)
            # Add contribution of the new event to kernel_sum
            kernel_sum = kernel_sum_at_candidate + K * theta
        else:
            kernel_sum = kernel_sum_at_candidate

        t = t_candidate

    return np.array(events, dtype=np.float64)


# ---------------------------------------------------------------------------
# Log-likelihood (pure JAX — differentiable)
# ---------------------------------------------------------------------------


def hawkesn_log_likelihood(
    log_K: jax.Array,
    log_theta: jax.Array,
    log_N: jax.Array,
    log_mu: jax.Array,
    event_times: jax.Array,
    T: float,
    I0: int = 1,
) -> jax.Array:
    """Compute log P(event_times | params, I0) under the HawkesN model.

    The log-likelihood of a temporal point process is:

        log L = Σ_{i} log λ*(tᵢ) - ∫_0^T λ*(t) dt

    where λ*(t) = (1 - N_t/N) · [μ + K·θ · Σ_{t_j < t} exp(-θ·(t - t_j))]

    and N_t counts ALL events including the I0 seeds.

    The I0 initial infected are treated as seed events at t=0⁻, contributing
    to the excitation kernel but NOT included in `event_times`.

    Parameters
    ----------
    log_K, log_theta, log_N, log_mu : unconstrained real scalars
    event_times : sorted observed event times in (0, T], shape (n,)
    T           : observation window end (scalar float)
    I0          : number of seed events at t=0⁻ (initial infected)

    Returns
    -------
    log_likelihood : scalar (negative = worse fit)

    Notes
    -----
    - O(n) time and memory via forward+backward scans.
    - Compensator is EXACT for the depleting exponential model via a backward
      scan that accounts for all depletion changes.
    """
    K = jnp.exp(log_K)
    theta = jnp.exp(log_theta)
    N = jnp.exp(log_N)
    mu = jnp.exp(log_mu)

    n = event_times.shape[0]
    I0_f = jnp.array(float(I0), dtype=jnp.float64)

    # Full timeline: I0 seeds at t=0 (indices 0..I0-1), then observed events
    # (indices I0..I0+n-1), observation end at T.
    # Depletion AFTER event j (0-indexed in the full sequence) is: 1 - (j+1)/N.
    # (j+1 events have been consumed including event j itself.)

    # For computations below, create the full event array:
    # seeds at t=0 (I0 copies) + observed events
    # shape (I0+n,)
    seeds = jnp.zeros(I0, dtype=jnp.float64)  # all seeds at t=0
    all_times = jnp.concatenate([seeds, event_times])  # shape (I0+n,)
    n_all = I0 + n

    # Extended times: [seeds..., event_times..., T]
    # = all_times concatenated with [T]
    times_full = jnp.concatenate([all_times, jnp.array([T], dtype=jnp.float64)])
    # shape (I0+n+1,)

    # -------------------------------------------------------------------------
    # Part 1: Sum of log-intensities at OBSERVED event times (indices I0..I0+n-1)
    #
    # Forward scan: carry = (A, t_prev) where A = Σ_{past events} exp(-θ(t-t_past))
    # Initial state: just before index I0 (first observed event),
    #   A = I0 * exp(-θ * t_first) + ... = seed contributions decayed to t_first
    # We'll scan from the first observed event onward.
    # -------------------------------------------------------------------------

    def fwd_intensity_scan(
        carry: tuple[jax.Array, jax.Array],
        obs_i: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        """Carry: (A_prev, t_prev).  obs_i: 0-indexed in observed events.

        A_prev is the kernel sum Σ exp(-θ(t_prev - t_past)) just after event
        at t_prev fired.  At time t_i, before event obs_i fires:
          A_at_ti = A_prev * exp(-θ * (t_i - t_prev))
        After event obs_i fires, A increases by 1.
        """
        A_prev, t_prev = carry
        t_i = event_times[obs_i]
        dt = t_i - t_prev
        A_at_ti = jnp.exp(-theta * dt) * A_prev
        # Number of events consumed before obs_i: I0 seeds + obs_i observed
        n_before = I0_f + obs_i
        dep_i = jnp.maximum(1.0 - n_before / N, 0.0)
        lambda_i = dep_i * (mu + K * theta * A_at_ti)
        lambda_i = jnp.maximum(lambda_i, 1e-300)
        log_lambda_i = jnp.log(lambda_i)
        A_after = A_at_ti + 1.0  # event fires
        return (A_after, t_i), log_lambda_i

    # Initial carry:
    # Just after the last seed fires at t=0, A = I0 (all seeds at t=0 → exp(0)=1 each).
    A_init = I0_f
    t_prev_init = jnp.zeros((), dtype=jnp.float64)
    obs_indices = jnp.arange(n, dtype=jnp.int32)
    _, log_lambdas = jax.lax.scan(
        fwd_intensity_scan,
        (A_init, t_prev_init),
        obs_indices,
    )
    sum_log_lambdas = jnp.sum(log_lambdas)

    # -------------------------------------------------------------------------
    # Part 2: Exact compensator ∫_0^T λ*(t) dt
    #
    # Split into background (μ) and excitation (K·θ) parts.
    #
    # Background:
    #   Λ_bg = μ · ∫_0^T (1 - N_t/N) dt
    #        = μ · Σ_{j=0}^{n_all} dep_j · Δ_j
    # where interval j = [times_full[j], times_full[j+1]],
    # dep_j = 1 - j/N  (after j events have fired, before interval j+1),
    # Δ_j = times_full[j+1] - times_full[j].
    # Note: before any event (j=0), dep_0 = 1.
    #
    # Excitation:
    # Each source event s (0..n_all-1) at time t_s contributes:
    #   C_s = K · Σ_{j>=s} dep_j · (exp(-θ(t_j - t_s)) - exp(-θ(t_{j+1} - t_s)))
    #       = K · Σ_{j>=s} dep_j · exp(-θ(t_j - t_s)) · (1 - exp(-θ Δ_j))
    # where dep_j = 1 - (j+1)/N (depletion AFTER event j fires).
    #
    # Total Λ_exc = Σ_s C_s
    #
    # Efficient O(n) computation via backward scan:
    # Define B_j = Σ_{k=j}^{n_all-1} dep_k · (1 - exp(-θ Δ_k)) · exp(-θ (t_k - t_j))
    # Recurrence: B_j = dep_j · (1-exp(-θ Δ_j)) + exp(-θ Δ_j) · B_{j+1}
    # Then C_s = K · B_s.
    # -------------------------------------------------------------------------

    # Interval lengths: Δ_j = times_full[j+1] - times_full[j], j=0..n_all-1
    delta_full = jnp.diff(times_full)  # shape (n_all,)  (n_all = I0+n intervals)
    # Note: the last interval is (last_event, T), so this covers 0..T.
    # Depletion in interval j (after event j fires):
    # Interval 0 is (t_0, t_1) where t_0 = seeds[0] = 0, t_1 = seeds[1] or event[0].
    # After event j fires (0-indexed), (j+1) events consumed.
    # So dep_j = 1 - (j+1)/N
    j_indices_f = jnp.arange(n_all, dtype=jnp.float64)  # 0..n_all-1
    dep_after_j = jnp.maximum(1.0 - (j_indices_f + 1.0) / N, 0.0)  # shape (n_all,)

    # Background compensator:
    # dep in interval j (BEFORE event j fires) = 1 - j/N
    dep_before_j = jnp.maximum(1.0 - j_indices_f / N, 0.0)
    background_compensator = mu * jnp.sum(dep_before_j * delta_full)

    # Excitation compensator via backward scan:
    # B_{n_all} = 0 (no intervals after last event)
    # For j = n_all-1 down to 0:
    #   B_j = dep_after_j[j] · (1 - exp(-θ·Δ_j)) + exp(-θ·Δ_j) · B_{j+1}
    # C_s = K · B_s  (for each source event s)
    # Total Λ_exc = Σ_s K · B_s

    def bwd_compensator_scan(
        B_next: jax.Array,
        j_rev: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """j_rev: reversed index (n_all-1 down to 0)."""
        j = j_rev
        dj = delta_full[j]
        decay_j = jnp.exp(-theta * dj)
        dep_j = dep_after_j[j]
        contrib = dep_j * (1.0 - decay_j)
        B_j = contrib + decay_j * B_next
        return B_j, B_j  # carry and output

    # Scan backwards: j goes from n_all-1 down to 0
    j_reversed = jnp.arange(n_all - 1, -1, -1, dtype=jnp.int32)  # shape (n_all,)
    _, B_reversed = jax.lax.scan(
        bwd_compensator_scan,
        jnp.zeros((), dtype=jnp.float64),
        j_reversed,
    )
    # B_reversed[0] = B_{n_all-1}, ..., B_reversed[n_all-1] = B_0
    B_values = jnp.flip(B_reversed)  # B_values[j] = B_j, shape (n_all,)
    # Total excitation compensator = K · Σ_{s=0}^{n_all-1} B_s
    excitation_compensator = K * jnp.sum(B_values)

    compensator = background_compensator + excitation_compensator

    return sum_log_lambdas - compensator


# ---------------------------------------------------------------------------
# Unconstrained parameterisation helpers
# ---------------------------------------------------------------------------


def params_to_unconstrained(params: HawkesNParams) -> np.ndarray:
    """Map natural parameters to unconstrained (log-space) vector."""
    return np.array(
        [
            np.log(params.K),
            np.log(params.theta),
            np.log(params.N),
            np.log(params.mu + 1e-12),  # guard log(0)
        ],
        dtype=np.float64,
    )


def unconstrained_to_params(v: np.ndarray) -> HawkesNParams:
    """Map unconstrained vector back to natural parameters."""
    return HawkesNParams(
        K=float(np.exp(v[0])),
        theta=float(np.exp(v[1])),
        N=float(np.exp(v[2])),
        mu=float(np.exp(v[3])),
    )


# ---------------------------------------------------------------------------
# MLE / MAP fitter (Branch-A interface)
# ---------------------------------------------------------------------------

_BOUNDS_LOG = (
    (np.log(0.01), np.log(20.0)),   # log K: branching ratio 0.01–20
    (np.log(1e-4), np.log(1e4)),    # log theta: decay rate
    (np.log(2.0), np.log(1e7)),     # log N: population size ≥ 2
    (np.log(1e-12), np.log(1e3)),   # log mu: background rate
)


def fit_hawkesn(
    event_times: np.ndarray,
    T: float | None = None,
    N_fixed: float | None = None,
    mu_fixed: float | None = None,
    initial_params: HawkesNParams | None = None,
    n_restarts: int = 3,
    I0: int = 1,
) -> tuple[HawkesNParams, dict]:
    """Fit a HawkesN model to a single event stream.

    Branch-A interface: (event_times[, T, ...]) → (params, uncertainty).

    Parameters
    ----------
    event_times  : sorted array of event timestamps
    T            : observation window end; defaults to max(event_times)
    N_fixed      : if given, fix N and do not optimise it
    mu_fixed     : if given, fix μ and do not optimise it (use 0 for pure SIR)
    initial_params : starting point; if None, heuristics are used
    n_restarts   : number of random restarts

    Returns
    -------
    best_params  : HawkesNParams at MLE
    info         : dict with keys 'neg_log_likelihood', 'hessian', 'covariance',
                   'success', 'message', 'n_events', 'T', 'n_restarts_tried'
    """
    event_times = np.asarray(event_times, dtype=np.float64)
    if event_times.ndim != 1:
        raise ValueError("event_times must be 1-D")
    n = len(event_times)
    if T is None:
        T = float(event_times[-1]) if n > 0 else 1.0

    times_jax = jnp.asarray(event_times, dtype=jnp.float64)

    # Build objective
    # We may fix N and/or mu; reduce the dimension accordingly.
    fix_N = N_fixed is not None
    fix_mu = mu_fixed is not None

    # Build a JAX neg-log-likelihood over the free parameters only
    # We always optimize log_K, log_theta; optionally log_N, log_mu.
    def _nll(free_vec: jax.Array) -> jax.Array:
        idx = 0
        log_K = free_vec[idx]; idx += 1
        log_theta = free_vec[idx]; idx += 1
        log_N = jnp.log(jnp.array(N_fixed, dtype=jnp.float64)) if fix_N else free_vec[idx]
        if not fix_N:
            idx += 1
        log_mu_val = jnp.log(jnp.array(mu_fixed + 1e-12, dtype=jnp.float64)) if fix_mu else free_vec[idx]
        return -hawkesn_log_likelihood(log_K, log_theta, log_N, log_mu_val, times_jax, T, I0=I0)

    nll_and_grad = jax.jit(jax.value_and_grad(_nll))

    def scipy_obj(v: np.ndarray) -> tuple[float, np.ndarray]:
        val, g = nll_and_grad(jnp.asarray(v, dtype=jnp.float64))
        return float(val), np.asarray(g, dtype=np.float64)

    # Build bounds for free params
    bounds_list = [_BOUNDS_LOG[0], _BOUNDS_LOG[1]]  # log_K, log_theta
    if not fix_N:
        bounds_list.append(_BOUNDS_LOG[2])
    if not fix_mu:
        bounds_list.append(_BOUNDS_LOG[3])

    # Heuristic initial point
    if initial_params is not None:
        x0_base = np.array(
            [np.log(initial_params.K), np.log(initial_params.theta)]
            + ([] if fix_N else [np.log(initial_params.N)])
            + ([] if fix_mu else [np.log(initial_params.mu + 1e-12)]),
            dtype=np.float64,
        )
    else:
        # Heuristics: K~0.8, theta ~ 1/(typical inter-event time)
        if n > 1:
            median_iet = float(np.median(np.diff(event_times)))
            theta_init = 1.0 / max(median_iet, 1e-6)
        else:
            theta_init = 1.0
        theta_init = np.clip(theta_init, np.exp(_BOUNDS_LOG[1][0]), np.exp(_BOUNDS_LOG[1][1]))
        K_init = 0.8
        N_init = max(float(n) * 2, 10.0)
        x0_base = np.array(
            [np.log(K_init), np.log(theta_init)]
            + ([] if fix_N else [np.log(N_init)])
            + ([] if fix_mu else [np.log(1e-6)]),
            dtype=np.float64,
        )

    rng_restarts = np.random.default_rng(42)
    best_val = np.inf
    best_result = None

    for restart_i in range(n_restarts):
        if restart_i == 0:
            x0 = x0_base.copy()
        else:
            # Perturb initial point
            x0 = x0_base + rng_restarts.normal(0, 0.5, size=x0_base.shape)
            # Clip to bounds
            for dim_i, (lo, hi) in enumerate(bounds_list):
                x0[dim_i] = np.clip(x0[dim_i], lo, hi)

        try:
            result = minimize(
                scipy_obj,
                x0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds_list,
                options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if result.fun < best_val:
                best_val = result.fun
                best_result = result
        except Exception:
            pass

    if best_result is None:
        # Fallback: use initial point
        class _FallbackResult:
            x = x0_base
            fun = np.inf
            success = False
            message = "all restarts failed"

        best_result = _FallbackResult()

    # Reconstruct params
    v = best_result.x
    idx = 0
    K_fit = float(np.exp(v[idx])); idx += 1
    theta_fit = float(np.exp(v[idx])); idx += 1
    N_fit = float(N_fixed) if fix_N else float(np.exp(v[idx]))
    if not fix_N:
        idx += 1
    mu_fit = float(mu_fixed) if fix_mu else float(np.exp(v[idx]))
    best_params = HawkesNParams(K=K_fit, theta=theta_fit, N=N_fit, mu=mu_fit)

    # Compute Hessian of NLL at best params for uncertainty
    free_jax = jnp.asarray(best_result.x, dtype=jnp.float64)
    hessian_nll = np.asarray(jax.hessian(_nll)(free_jax), dtype=np.float64)
    # Pseudo-invert for covariance
    sym_h = 0.5 * (hessian_nll + hessian_nll.T)
    cov = np.linalg.pinv(sym_h + 1e-8 * np.eye(sym_h.shape[0]))

    info = {
        "neg_log_likelihood": float(best_val),
        "hessian": hessian_nll,
        "covariance": cov,
        "success": bool(best_result.success),
        "message": str(best_result.message),
        "n_events": n,
        "T": T,
        "n_restarts_tried": n_restarts,
    }
    return best_params, info
