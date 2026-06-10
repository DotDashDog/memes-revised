"""Likelihood, MLE, and local posterior geometry for WP-02."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from scipy.optimize import minimize

from experiments.wp02.sir import bin_means

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FitResult:
    """Maximum-likelihood fit and Laplace approximation for one curve."""

    theta_mle: np.ndarray
    natural_params: dict[str, float]
    neg_log_likelihood: float
    hessian: np.ndarray
    covariance: np.ndarray
    success: bool
    message: str


def poisson_neg_log_likelihood(
    theta: jax.Array,
    counts: jax.Array,
    bin_edges: jax.Array,
) -> jax.Array:
    """Poisson negative log-likelihood for binned SIR counts."""

    means = bin_means(theta, bin_edges)
    return jnp.sum(means - counts * jnp.log(means) + gammaln(counts + 1.0))


def observed_information(
    theta: jax.Array,
    counts: jax.Array,
    bin_edges: jax.Array,
) -> jax.Array:
    """Observed information matrix: Hessian of the negative log-likelihood."""

    return jax.hessian(poisson_neg_log_likelihood)(theta, counts, bin_edges)


def covariance_from_hessian(hessian: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Pseudo-invert a Hessian into a local covariance estimate."""

    sym_hessian = 0.5 * (hessian + hessian.T)
    return np.linalg.pinv(sym_hessian + jitter * np.eye(sym_hessian.shape[0]))


def _natural_params_dict(theta: np.ndarray) -> dict[str, float]:
    log_beta, log_gamma, log_scale, logit_i0 = theta
    i0 = 1.0 / (1.0 + np.exp(-logit_i0))
    return {
        "beta": float(np.exp(log_beta)),
        "gamma": float(np.exp(log_gamma)),
        "scale": float(np.exp(log_scale)),
        "i0": float(i0),
    }


def fit_mle(
    initial_theta: np.ndarray,
    counts: np.ndarray,
    bin_edges: np.ndarray,
    bounds: tuple[tuple[float, float], ...] | None = None,
) -> FitResult:
    """Fit ``theta`` by Poisson MLE using JAX values/gradients and SciPy."""

    if bounds is None:
        bounds = (
            (np.log(0.03), np.log(30.0)),
            (np.log(0.01), np.log(10.0)),
            (np.log(1e-3), np.log(1e8)),
            (-14.0, 0.0),
        )

    counts_jax = jnp.asarray(counts, dtype=jnp.float64)
    bin_edges_jax = jnp.asarray(bin_edges, dtype=jnp.float64)
    value_and_grad = jax.value_and_grad(poisson_neg_log_likelihood)

    def objective(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        theta = jnp.asarray(theta_np, dtype=jnp.float64)
        value, grad = value_and_grad(theta, counts_jax, bin_edges_jax)
        return float(value), np.asarray(grad, dtype=float)

    result = minimize(
        objective,
        np.asarray(initial_theta, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
    )
    hessian = np.asarray(
        observed_information(
            jnp.asarray(result.x, dtype=jnp.float64),
            counts_jax,
            bin_edges_jax,
        ),
        dtype=float,
    )
    covariance = covariance_from_hessian(hessian)
    return FitResult(
        theta_mle=np.asarray(result.x, dtype=float),
        natural_params=_natural_params_dict(np.asarray(result.x, dtype=float)),
        neg_log_likelihood=float(result.fun),
        hessian=hessian,
        covariance=covariance,
        success=bool(result.success),
        message=str(result.message),
    )
