"""Run one WP-02 synthetic SIR identifiability case.

This is intentionally small: it verifies the JAX/diffrax forward model,
Poisson likelihood, MLE/Hessian path, and optional NUTS path before scaling to
the full parameter grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.wp02.fit import fit_mle
from experiments.wp02.plots import save_corner_plot
from experiments.wp02.posterior import run_nuts, theta_samples_from_nuts
from experiments.wp02.sir import SirParameters, simulate_counts, theta_from_params


def _positive_semidefinite_covariance(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    clipped = np.clip(eigvals, 1e-12, None)
    return eigvecs @ np.diag(clipped) @ eigvecs.T


def _theta_to_natural_samples(theta_samples: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.exp(theta_samples[:, 0]),
            np.exp(theta_samples[:, 1]),
            np.exp(theta_samples[:, 2]),
            1.0 / (1.0 + np.exp(-theta_samples[:, 3])),
        ]
    )


def _summarize_samples(samples: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    quantiles = np.quantile(samples, [0.05, 0.16, 0.5, 0.84, 0.95], axis=0)
    summary: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        summary[name] = {
            "mean": float(np.mean(samples[:, index])),
            "std": float(np.std(samples[:, index], ddof=1)),
            "q05": float(quantiles[0, index]),
            "q16": float(quantiles[1, index]),
            "median": float(quantiles[2, index]),
            "q84": float(quantiles[3, index]),
            "q95": float(quantiles[4, index]),
        }
    return summary


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _summarize_extra_fields(extra_fields: dict[str, np.ndarray]) -> dict[str, object]:
    summary: dict[str, object] = {}
    if "diverging" in extra_fields:
        diverging = np.asarray(extra_fields["diverging"], dtype=bool)
        summary["divergences"] = int(np.sum(diverging))
        summary["draws"] = int(diverging.size)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--days", type=float, default=60.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--scale", type=float, default=5000.0)
    parser.add_argument("--i0", type=float, default=0.003)
    parser.add_argument("--nuts", action="store_true")
    parser.add_argument("--nuts-warmup", type=int, default=500)
    parser.add_argument("--nuts-samples", type=int, default=500)
    parser.add_argument("--nuts-chains", type=int, default=4)
    parser.add_argument("--nuts-chain-method", choices=["parallel", "sequential", "vectorized"], default="sequential")
    parser.add_argument(
        "--nuts-init",
        choices=["map", "default"],
        default="map",
        help="Initialize NUTS near the MAP or use NumPyro's default prior-based initializer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/WP-02/smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    params = SirParameters(
        beta=args.beta,
        gamma=args.gamma,
        scale=args.scale,
        i0=args.i0,
    )
    true_theta = np.asarray(theta_from_params(params), dtype=float)
    bin_edges = np.asarray(jnp.arange(0.0, args.days + args.bin_width, args.bin_width))

    key = jax.random.PRNGKey(args.seed)
    counts, means = simulate_counts(key, params, jnp.asarray(bin_edges))
    counts_np = np.asarray(counts, dtype=float)
    means_np = np.asarray(means, dtype=float)

    fit = fit_mle(true_theta, counts_np, bin_edges)
    rng = np.random.default_rng(args.seed)
    hessian_covariance = _positive_semidefinite_covariance(fit.covariance)
    hessian_samples = rng.multivariate_normal(
        fit.theta_mle,
        hessian_covariance,
        size=4000,
        method="svd",
    )
    save_corner_plot(
        hessian_samples,
        args.output_dir / "hessian_corner.png",
        truths=true_theta,
        title="Laplace approximation from observed Hessian",
    )
    hessian_natural_samples = _theta_to_natural_samples(hessian_samples)
    natural_names = ["beta", "gamma", "scale", "i0"]
    theta_names = ["log_beta", "log_gamma", "log_scale", "logit_i0"]

    result = {
        "seed": args.seed,
        "true_params": params.__dict__,
        "true_theta": true_theta.tolist(),
        "fit_success": fit.success,
        "fit_message": fit.message,
        "mle_theta": fit.theta_mle.tolist(),
        "mle_natural_params": fit.natural_params,
        "neg_log_likelihood": fit.neg_log_likelihood,
        "counts_total": float(counts_np.sum()),
        "expected_counts_total": float(means_np.sum()),
        "hessian_eigenvalues": np.linalg.eigvalsh(0.5 * (fit.hessian + fit.hessian.T)).tolist(),
        "hessian_theta_summary": _summarize_samples(hessian_samples, theta_names),
        "hessian_natural_summary": _summarize_samples(hessian_natural_samples, natural_names),
    }

    if args.nuts:
        nuts_key = jax.random.fold_in(key, 1)
        nuts_result = run_nuts(
            nuts_key,
            counts_np,
            bin_edges,
            num_warmup=args.nuts_warmup,
            num_samples=args.nuts_samples,
            num_chains=args.nuts_chains,
            chain_method=args.nuts_chain_method,
            init_theta=fit.theta_mle if args.nuts_init == "map" else None,
        )
        theta_samples = theta_samples_from_nuts(nuts_result.samples)
        save_corner_plot(
            theta_samples,
            args.output_dir / "nuts_corner.png",
            truths=true_theta,
            title="NUTS posterior samples",
        )
        nuts_natural_samples = _theta_to_natural_samples(theta_samples)
        result["nuts_config"] = {
            "warmup": args.nuts_warmup,
            "samples_per_chain": args.nuts_samples,
            "chains": args.nuts_chains,
            "chain_method": args.nuts_chain_method,
            "init": args.nuts_init,
        }
        result["nuts_samples"] = {
            name: value.shape[0] for name, value in nuts_result.samples.items()
        }
        result["nuts_diagnostics"] = _json_safe(nuts_result.diagnostics)
        result["nuts_extra_fields"] = _summarize_extra_fields(nuts_result.extra_fields)
        result["nuts_theta_summary"] = _summarize_samples(theta_samples, theta_names)
        result["nuts_natural_summary"] = _summarize_samples(nuts_natural_samples, natural_names)
        result["posterior_mean_delta_hessian_minus_nuts"] = {
            name: float(
                result["hessian_theta_summary"][name]["mean"]
                - result["nuts_theta_summary"][name]["mean"]
            )
            for name in theta_names
        }

    with (args.output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
