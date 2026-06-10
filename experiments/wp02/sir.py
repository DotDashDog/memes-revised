"""Differentiable SIR forward model for binned count observations.

The natural scientific parameters are ``(beta, gamma, scale, i0)``. For
optimization and MCMC we use an unconstrained vector
``theta = (log_beta, log_gamma, log_scale, logit_i0)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import config

config.update("jax_enable_x64", True)

import diffrax
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SirParameters:
    """Natural SIR parameters for a single synthetic meme curve."""

    beta: float
    gamma: float
    scale: float
    i0: float


def theta_from_params(params: SirParameters) -> jax.Array:
    """Convert natural positive parameters to the unconstrained fit vector."""

    if not (0.0 < params.i0 < 1.0):
        raise ValueError("i0 must be in (0, 1)")
    return jnp.array(
        [
            jnp.log(params.beta),
            jnp.log(params.gamma),
            jnp.log(params.scale),
            jnp.log(params.i0 / (1.0 - params.i0)),
        ],
        dtype=jnp.float64,
    )


def params_from_theta(theta: jax.Array) -> SirParameters:
    """Convert an unconstrained fit vector to natural SIR parameters."""

    log_beta, log_gamma, log_scale, logit_i0 = theta
    return SirParameters(
        beta=float(jnp.exp(log_beta)),
        gamma=float(jnp.exp(log_gamma)),
        scale=float(jnp.exp(log_scale)),
        i0=float(jax.nn.sigmoid(logit_i0)),
    )


def _unpack_theta(theta: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    log_beta, log_gamma, log_scale, logit_i0 = theta
    return (
        jnp.exp(log_beta),
        jnp.exp(log_gamma),
        jnp.exp(log_scale),
        jax.nn.sigmoid(logit_i0),
    )


def _sir_rhs(
    t: jax.Array,
    y: jax.Array,
    args: tuple[jax.Array, jax.Array],
) -> jax.Array:
    del t
    beta, gamma = args
    susceptible, infected, recovered, cumulative_infected_time = y
    del recovered, cumulative_infected_time

    d_susceptible = -beta * susceptible * infected
    d_infected = beta * susceptible * infected - gamma * infected
    d_recovered = gamma * infected
    d_cumulative_infected_time = infected
    return jnp.array(
        [d_susceptible, d_infected, d_recovered, d_cumulative_infected_time],
        dtype=y.dtype,
    )


def solve_sir(theta: jax.Array, times: jax.Array) -> jax.Array:
    """Solve SIR and save ``(S, I, R, integral I dt)`` at requested times."""

    beta, gamma, _, i0 = _unpack_theta(theta)
    y0 = jnp.array([1.0 - i0, i0, 0.0, 0.0], dtype=jnp.float64)
    term = diffrax.ODETerm(_sir_rhs)
    solution = diffrax.diffeqsolve(
        term,
        solver=diffrax.Dopri5(),
        t0=times[0],
        t1=times[-1],
        dt0=0.05,
        y0=y0,
        args=(beta, gamma),
        saveat=diffrax.SaveAt(ts=times),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
        adjoint=diffrax.DirectAdjoint(),
        max_steps=4096,
    )
    return solution.ys


def bin_means(theta: jax.Array, bin_edges: jax.Array) -> jax.Array:
    """Expected Poisson counts per bin from ``scale * integral_bin I(t) dt``."""

    _, _, scale, _ = _unpack_theta(theta)
    cumulative_infected_time = solve_sir(theta, bin_edges)[:, 3]
    means = scale * jnp.diff(cumulative_infected_time)
    return jnp.clip(means, min=1e-12)


def simulate_counts(
    key: jax.Array,
    params: SirParameters,
    bin_edges: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Draw Poisson binned counts for a synthetic SIR curve."""

    theta = theta_from_params(params)
    means = bin_means(theta, bin_edges)
    return jax.random.poisson(key, means), means
