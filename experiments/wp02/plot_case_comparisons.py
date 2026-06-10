"""Create combined time-series and posterior comparison plots for WP-02 cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import corner
import jax
import jax.numpy as jnp
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.wp02.fit import fit_mle
from experiments.wp02.plots import THETA_LABELS
from experiments.wp02.posterior import THETA_BOUNDS, run_nuts, theta_samples_from_nuts
from experiments.wp02.sir import SirParameters, bin_means, simulate_counts, theta_from_params

THETA_NAMES = ("log_beta", "log_gamma", "log_scale", "logit_i0")
DEFAULT_CASE_DIRS = (
    Path("experiments/WP-02/nuts-check-case-008"),
    Path("experiments/WP-02/nuts-check-case-067"),
    Path("experiments/WP-02/nuts-check-case-019"),
    Path("experiments/WP-02/nuts-check-case-052"),
)


def _positive_semidefinite_covariance(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    clipped = np.clip(eigvals, 1e-12, None)
    return eigvecs @ np.diag(clipped) @ eigvecs.T


def _load_case(case_dir: Path) -> dict[str, object]:
    with (case_dir / "result.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _case_id(case_dir: Path) -> str:
    return case_dir.name.replace("nuts-check-", "")


def _simulate_from_result(result: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, SirParameters]:
    params_dict = result["true_params"]
    params = SirParameters(
        beta=float(params_dict["beta"]),
        gamma=float(params_dict["gamma"]),
        scale=float(params_dict["scale"]),
        i0=float(params_dict["i0"]),
    )
    bin_edges = np.asarray(jnp.arange(0.0, 60.0 + 1.0, 1.0))
    key = jax.random.PRNGKey(int(result["seed"]))
    counts, true_means = simulate_counts(key, params, jnp.asarray(bin_edges))
    return np.asarray(counts, dtype=float), np.asarray(true_means, dtype=float), bin_edges, params


def _load_or_run_samples(
    *,
    cache_path: Path,
    counts: np.ndarray,
    bin_edges: np.ndarray,
    params: SirParameters,
    seed: int,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    chain_method: str,
    force: bool,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and not force:
        with np.load(cache_path) as cache:
            return {name: np.asarray(cache[name]) for name in cache.files}

    true_theta = np.asarray(theta_from_params(params), dtype=float)
    fit = fit_mle(true_theta, counts, bin_edges)
    rng = np.random.default_rng(seed)
    hessian_samples = rng.multivariate_normal(
        fit.theta_mle,
        _positive_semidefinite_covariance(fit.covariance),
        size=4000,
        method="svd",
    )
    nuts_result = run_nuts(
        jax.random.fold_in(jax.random.PRNGKey(seed), 1),
        counts,
        bin_edges,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        init_theta=fit.theta_mle,
    )
    nuts_samples = theta_samples_from_nuts(nuts_result.samples)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        hessian_samples=hessian_samples,
        nuts_samples=nuts_samples,
        map_theta=fit.theta_mle,
        nuts_median_theta=np.median(nuts_samples, axis=0),
        true_theta=true_theta,
    )
    return {
        "hessian_samples": hessian_samples,
        "nuts_samples": nuts_samples,
        "map_theta": fit.theta_mle,
        "nuts_median_theta": np.median(nuts_samples, axis=0),
        "true_theta": true_theta,
    }


def _clip_to_plot_ranges(samples: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    clipped = np.asarray(samples, dtype=float).copy()
    for index, (lower, upper) in enumerate(ranges):
        epsilon = 1e-9 * (upper - lower)
        clipped[:, index] = np.clip(clipped[:, index], lower + epsilon, upper - epsilon)
    return clipped


def _plot_case(
    *,
    case_id: str,
    result: dict[str, object],
    counts: np.ndarray,
    true_means: np.ndarray,
    bin_edges: np.ndarray,
    sample_data: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    hessian_samples = sample_data["hessian_samples"]
    nuts_samples = sample_data["nuts_samples"]
    map_theta = sample_data["map_theta"]
    nuts_median_theta = sample_data["nuts_median_theta"]
    true_theta = sample_data["true_theta"]

    plot_ranges = [THETA_BOUNDS[name] for name in THETA_NAMES]
    hessian_plot_samples = _clip_to_plot_ranges(hessian_samples, plot_ranges)
    nuts_plot_samples = _clip_to_plot_ranges(nuts_samples, plot_ranges)
    figure = corner.corner(
        hessian_plot_samples,
        labels=THETA_LABELS,
        truths=true_theta,
        range=plot_ranges,
        color="tab:blue",
        bins=36,
        plot_datapoints=False,
        fill_contours=False,
        hist_kwargs={"density": True, "alpha": 0.35},
        contour_kwargs={"linewidths": 1.2},
        max_n_ticks=3,
    )
    corner.corner(
        nuts_plot_samples,
        fig=figure,
        labels=THETA_LABELS,
        truths=true_theta,
        range=plot_ranges,
        color="tab:orange",
        bins=36,
        plot_datapoints=False,
        fill_contours=False,
        hist_kwargs={"density": True, "alpha": 0.45},
        contour_kwargs={"linewidths": 1.2},
        max_n_ticks=3,
    )

    axis = figure.add_axes([0.54, 0.61, 0.39, 0.30])
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    map_means = np.asarray(bin_means(jnp.asarray(map_theta), jnp.asarray(bin_edges)), dtype=float)
    nuts_median_means = np.asarray(
        bin_means(jnp.asarray(nuts_median_theta), jnp.asarray(bin_edges)),
        dtype=float,
    )
    axis.scatter(centers, counts, color="black", s=14, alpha=0.75, label="observed counts")
    axis.plot(centers, map_means, color="tab:blue", lw=1.8, label="MAP expected counts")
    axis.plot(
        centers,
        nuts_median_means,
        color="tab:orange",
        lw=1.8,
        label="NUTS median expected counts",
    )
    axis.plot(centers, true_means, color="black", lw=1.0, ls="--", alpha=0.5, label="truth")
    axis.set_title("Binned time series", fontsize=10)
    axis.set_xlabel("day", fontsize=9)
    axis.set_ylabel("count / bin", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.legend(fontsize=7, frameon=False)

    diagnostics = result.get("nuts_diagnostics", {})
    max_rhat = max(float(value["r_hat"]) for value in diagnostics.values()) if diagnostics else float("nan")
    divergences = int(result.get("nuts_extra_fields", {}).get("divergences", 0))
    figure.suptitle(
        (
            f"{case_id}: NUTS vs Hessian/Laplace in fit coordinates\n"
            f"counts={result['counts_total']:.0f}, max R-hat={max_rhat:.3g}, divergences={divergences}"
        ),
        fontsize=14,
    )
    figure.legend(
        handles=[
            mlines.Line2D([], [], color="tab:blue", label="Hessian/Laplace samples"),
            mlines.Line2D([], [], color="tab:orange", label="NUTS samples"),
            mlines.Line2D([], [], color="black", label="truth"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/WP-02/comparison-plots-001"))
    parser.add_argument("--nuts-warmup", type=int, default=500)
    parser.add_argument("--nuts-samples", type=int, default=500)
    parser.add_argument("--nuts-chains", type=int, default=4)
    parser.add_argument("--nuts-chain-method", choices=["parallel", "sequential", "vectorized"], default="sequential")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dirs = args.case_dir or list(DEFAULT_CASE_DIRS)
    for case_dir in case_dirs:
        result = _load_case(case_dir)
        counts, true_means, bin_edges, params = _simulate_from_result(result)
        case_name = _case_id(case_dir)
        sample_data = _load_or_run_samples(
            cache_path=args.output_dir / "cache" / f"{case_name}.npz",
            counts=counts,
            bin_edges=bin_edges,
            params=params,
            seed=int(result["seed"]),
            num_warmup=args.nuts_warmup,
            num_samples=args.nuts_samples,
            num_chains=args.nuts_chains,
            chain_method=args.nuts_chain_method,
            force=args.force_rerun,
        )
        output_path = args.output_dir / f"{case_name}-comparison.png"
        _plot_case(
            case_id=case_name,
            result=result,
            counts=counts,
            true_means=true_means,
            bin_edges=bin_edges,
            sample_data=sample_data,
            output_path=output_path,
        )
        print(output_path)


if __name__ == "__main__":
    main()
