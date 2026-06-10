"""NUTS posterior sampling for WP-02 synthetic SIR fits."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from experiments.wp02.sir import bin_means

jax.config.update("jax_enable_x64", True)

THETA_BOUNDS = {
    "log_beta": (np.log(0.03), np.log(30.0)),
    "log_gamma": (np.log(0.01), np.log(10.0)),
    "log_scale": (np.log(1e-3), np.log(1e8)),
    "logit_i0": (-14.0, 0.0),
}


@dataclass(frozen=True)
class NutsResult:
    """Posterior samples and diagnostics from one NumPyro NUTS run."""

    samples: dict[str, np.ndarray]
    samples_by_chain: dict[str, np.ndarray]
    diagnostics: dict[str, dict[str, float]]
    extra_fields: dict[str, np.ndarray]


def sir_poisson_model(counts: np.ndarray, bin_edges: np.ndarray) -> None:
    """Broad-prior Poisson SIR model in unconstrained coordinates."""

    log_beta = numpyro.sample("log_beta", dist.Uniform(*THETA_BOUNDS["log_beta"]))
    log_gamma = numpyro.sample("log_gamma", dist.Uniform(*THETA_BOUNDS["log_gamma"]))
    log_scale = numpyro.sample("log_scale", dist.Uniform(*THETA_BOUNDS["log_scale"]))
    logit_i0 = numpyro.sample("logit_i0", dist.Uniform(*THETA_BOUNDS["logit_i0"]))
    theta = jnp.array([log_beta, log_gamma, log_scale, logit_i0], dtype=jnp.float64)
    means = bin_means(theta, jnp.asarray(bin_edges, dtype=jnp.float64))
    numpyro.sample("counts", dist.Poisson(means), obs=jnp.asarray(counts))


def _clip_to_prior_interior(name: str, value: float, epsilon: float = 1e-6) -> float:
    lower, upper = THETA_BOUNDS[name]
    return float(np.clip(value, lower + epsilon, upper - epsilon))


def run_nuts(
    key: jax.Array,
    counts: np.ndarray,
    bin_edges: np.ndarray,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    chain_method: str = "sequential",
    init_theta: np.ndarray | None = None,
) -> NutsResult:
    """Run NUTS for one synthetic curve and return posterior samples."""

    nuts_kwargs = {"dense_mass": True}
    if init_theta is not None:
        clipped_theta = np.array(
            [
                _clip_to_prior_interior("log_beta", init_theta[0]),
                _clip_to_prior_interior("log_gamma", init_theta[1]),
                _clip_to_prior_interior("log_scale", init_theta[2]),
                _clip_to_prior_interior("logit_i0", init_theta[3]),
            ]
        )
        nuts_kwargs["init_strategy"] = init_to_value(
            values={
                "log_beta": float(clipped_theta[0]),
                "log_gamma": float(clipped_theta[1]),
                "log_scale": float(clipped_theta[2]),
                "logit_i0": float(clipped_theta[3]),
            }
        )
    kernel = NUTS(sir_poisson_model, **nuts_kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )
    mcmc.run(key, counts=np.asarray(counts), bin_edges=np.asarray(bin_edges))
    samples = {name: np.asarray(value) for name, value in mcmc.get_samples().items()}
    samples_by_chain = {
        name: np.asarray(value) for name, value in mcmc.get_samples(group_by_chain=True).items()
    }
    diagnostics = summary(samples_by_chain, group_by_chain=True)
    extra_fields = {name: np.asarray(value) for name, value in mcmc.get_extra_fields().items()}
    return NutsResult(
        samples=samples,
        samples_by_chain=samples_by_chain,
        diagnostics=diagnostics,
        extra_fields=extra_fields,
    )


def theta_samples_from_nuts(samples: dict[str, np.ndarray]) -> np.ndarray:
    """Stack NUTS samples into ``(n_samples, 4)`` unconstrained theta rows."""

    return np.column_stack(
        [
            samples["log_beta"],
            samples["log_gamma"],
            samples["log_scale"],
            samples["logit_i0"],
        ]
    )
