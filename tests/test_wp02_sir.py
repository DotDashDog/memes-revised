import jax
import jax.numpy as jnp
import numpy as np

from experiments.wp02.fit import poisson_neg_log_likelihood
from experiments.wp02.sir import SirParameters, bin_means, simulate_counts, theta_from_params


def test_bin_means_are_positive_and_match_bins() -> None:
    params = SirParameters(beta=1.0, gamma=0.3, scale=1000.0, i0=0.01)
    theta = theta_from_params(params)
    bin_edges = jnp.arange(0.0, 15.0, 1.0)

    means = bin_means(theta, bin_edges)

    assert means.shape == (bin_edges.shape[0] - 1,)
    assert jnp.all(jnp.isfinite(means))
    assert jnp.all(means > 0.0)


def test_poisson_likelihood_has_finite_gradient() -> None:
    params = SirParameters(beta=1.1, gamma=0.4, scale=750.0, i0=0.005)
    theta = theta_from_params(params)
    bin_edges = jnp.arange(0.0, 12.0, 1.0)
    counts, _ = simulate_counts(jax.random.PRNGKey(123), params, bin_edges)

    value, grad = jax.value_and_grad(poisson_neg_log_likelihood)(theta, counts, bin_edges)

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))
