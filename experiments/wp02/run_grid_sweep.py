"""Run a Hessian/Laplace identifiability sweep over synthetic SIR parameters."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.wp02.fit import fit_mle
from experiments.wp02.sir import SirParameters, simulate_counts, theta_from_params

PARAMETER_NAMES = ("beta", "gamma", "scale", "i0")
THETA_NAMES = ("log_beta", "log_gamma", "log_scale", "logit_i0")
DEFAULT_BOUNDS = (
    (np.log(0.03), np.log(30.0)),
    (np.log(0.01), np.log(10.0)),
    (np.log(1e-3), np.log(1e8)),
    (-14.0, 0.0),
)


def _parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item]


def _natural_jacobian(theta: np.ndarray) -> np.ndarray:
    beta, gamma, scale = np.exp(theta[:3])
    i0 = 1.0 / (1.0 + np.exp(-theta[3]))
    return np.diag([beta, gamma, scale, i0 * (1.0 - i0)])


def _relative_errors(fitted: dict[str, float], true: SirParameters) -> dict[str, float]:
    true_values = true.__dict__
    return {
        f"{name}_rel_error": float((fitted[name] - true_values[name]) / true_values[name])
        for name in PARAMETER_NAMES
    }


def _boundary_hits(theta: np.ndarray, tolerance: float = 1e-4) -> dict[str, bool]:
    return {
        f"{name}_boundary_hit": bool(
            theta[index] <= DEFAULT_BOUNDS[index][0] + tolerance
            or theta[index] >= DEFAULT_BOUNDS[index][1] - tolerance
        )
        for index, name in enumerate(THETA_NAMES)
    }


def _fit_case(
    *,
    case_id: str,
    seed: int,
    params: SirParameters,
    bin_edges: np.ndarray,
) -> dict[str, object]:
    key = jax.random.PRNGKey(seed)
    theta_true = np.asarray(theta_from_params(params), dtype=float)
    counts, means = simulate_counts(key, params, jnp.asarray(bin_edges))
    counts_np = np.asarray(counts, dtype=float)
    means_np = np.asarray(means, dtype=float)

    fit = fit_mle(theta_true, counts_np, bin_edges)
    theta_covariance = 0.5 * (fit.covariance + fit.covariance.T)
    theta_std = np.sqrt(np.clip(np.diag(theta_covariance), 0.0, np.inf))
    natural_covariance = _natural_jacobian(fit.theta_mle) @ theta_covariance @ _natural_jacobian(
        fit.theta_mle
    ).T
    natural_std = np.sqrt(np.clip(np.diag(natural_covariance), 0.0, np.inf))
    hessian_eigenvalues = np.linalg.eigvalsh(0.5 * (fit.hessian + fit.hessian.T))
    positive_eigenvalues = hessian_eigenvalues[hessian_eigenvalues > 0.0]
    hessian_positive_definite = bool(len(positive_eigenvalues) == len(hessian_eigenvalues))
    condition_number = (
        float(positive_eigenvalues[-1] / positive_eigenvalues[0])
        if hessian_positive_definite
        else float("inf")
    )
    covariance_valid = bool(
        hessian_positive_definite
        and np.all(np.isfinite(theta_std))
        and np.all(np.isfinite(natural_std))
        and np.all(theta_std > 0.0)
        and np.all(natural_std > 0.0)
    )

    row: dict[str, object] = {
        "case_id": case_id,
        "seed": seed,
        "fit_success": fit.success,
        "fit_message": fit.message,
        "expected_counts_total": float(means_np.sum()),
        "observed_counts_total": float(counts_np.sum()),
        "neg_log_likelihood": fit.neg_log_likelihood,
        "hessian_min_eigenvalue": float(hessian_eigenvalues[0]),
        "hessian_max_eigenvalue": float(hessian_eigenvalues[-1]),
        "hessian_positive_definite": hessian_positive_definite,
        "covariance_valid": covariance_valid,
        "hessian_condition_number": condition_number,
        "max_abs_relative_error": float(
            max(abs(value) for value in _relative_errors(fit.natural_params, params).values())
        ),
    }
    row.update({f"true_{name}": getattr(params, name) for name in PARAMETER_NAMES})
    row.update({f"mle_{name}": fit.natural_params[name] for name in PARAMETER_NAMES})
    row.update(_relative_errors(fit.natural_params, params))
    row.update({f"theta_std_{name}": float(theta_std[index]) for index, name in enumerate(THETA_NAMES)})
    row.update({f"natural_std_{name}": float(natural_std[index]) for index, name in enumerate(PARAMETER_NAMES)})
    row.update(
        {
            f"natural_rel_std_{name}": float(natural_std[index] / fit.natural_params[name])
            for index, name in enumerate(PARAMETER_NAMES)
        }
    )
    row["max_natural_rel_std"] = float(
        max(row[f"natural_rel_std_{name}"] for name in PARAMETER_NAMES)
    )
    row.update(_boundary_hits(fit.theta_mle))
    row["any_boundary_hit"] = bool(any(row[f"{name}_boundary_hit"] for name in THETA_NAMES))
    return row


def _select_cases(rows: list[dict[str, object]]) -> dict[str, str]:
    successful = [row for row in rows if row["fit_success"]]
    valid = [
        row
        for row in successful
        if row["covariance_valid"] and not row["any_boundary_hit"]
    ]
    if not successful or not valid:
        return {}

    by_condition = sorted(valid, key=lambda row: float(row["hessian_condition_number"]))
    by_uncertainty = sorted(valid, key=lambda row: float(row["max_natural_rel_std"]))
    by_counts = sorted(successful, key=lambda row: float(row["observed_counts_total"]))
    boundary = [row for row in successful if row["any_boundary_hit"]]
    invalid_hessian = [row for row in successful if not row["hessian_positive_definite"]]
    median_index = len(by_condition) // 2

    selected = {
        "well_identified": str(by_uncertainty[0]["case_id"]),
        "median_condition": str(by_condition[median_index]["case_id"]),
        "worst_condition": str(by_condition[-1]["case_id"]),
        "worst_uncertainty": str(by_uncertainty[-1]["case_id"]),
        "lowest_count": str(by_counts[0]["case_id"]),
    }
    if boundary:
        selected["boundary_case"] = str(
            max(boundary, key=lambda row: float(row["max_natural_rel_std"]))["case_id"]
        )
    if invalid_hessian:
        selected["non_positive_hessian"] = str(
            min(invalid_hessian, key=lambda row: float(row["hessian_min_eigenvalue"]))["case_id"]
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--days", type=float, default=60.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    parser.add_argument("--betas", default="0.3,1.0,3.0")
    parser.add_argument("--gammas", default="0.1,0.35,1.0")
    parser.add_argument("--scales", default="300,1500,7500")
    parser.add_argument("--i0s", default="0.001,0.003,0.01")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/WP-02/grid-sweep-001"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = np.asarray(jnp.arange(0.0, args.days + args.bin_width, args.bin_width))
    rows: list[dict[str, object]] = []

    grid = list(
        product(
            _parse_float_list(args.betas),
            _parse_float_list(args.gammas),
            _parse_float_list(args.scales),
            _parse_float_list(args.i0s),
        )
    )
    for index, (beta, gamma, scale, i0) in enumerate(grid):
        case_id = f"case-{index:03d}"
        case_seed = int(jax.random.fold_in(jax.random.PRNGKey(args.seed), index)[0])
        row = _fit_case(
            case_id=case_id,
            seed=case_seed,
            params=SirParameters(beta=beta, gamma=gamma, scale=scale, i0=i0),
            bin_edges=bin_edges,
        )
        rows.append(row)
        print(
            f"{case_id}: beta={beta:g} gamma={gamma:g} scale={scale:g} i0={i0:g} "
            f"success={row['fit_success']} counts={row['observed_counts_total']:.0f} "
            f"max_rel_std={row['max_natural_rel_std']:.3g} "
            f"cond={row['hessian_condition_number']:.3g}"
        )

    csv_path = args.output_dir / "hessian_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    selected_cases = _select_cases(rows)
    summary = {
        "seed": args.seed,
        "days": args.days,
        "bin_width": args.bin_width,
        "grid_size": len(rows),
        "parameter_grid": {
            "betas": _parse_float_list(args.betas),
            "gammas": _parse_float_list(args.gammas),
            "scales": _parse_float_list(args.scales),
            "i0s": _parse_float_list(args.i0s),
        },
        "fit_successes": sum(bool(row["fit_success"]) for row in rows),
        "boundary_hits": sum(bool(row["any_boundary_hit"]) for row in rows),
        "positive_definite_hessians": sum(bool(row["hessian_positive_definite"]) for row in rows),
        "valid_covariances": sum(bool(row["covariance_valid"]) for row in rows),
        "selected_cases": selected_cases,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
