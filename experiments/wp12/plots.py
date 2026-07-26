"""Plotting helpers for WP-12.

All figures saved to experiments/WP-12/ alongside their configs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Output directory
WP12_OUTPUT = Path(__file__).parent.parent / "WP-12"
WP12_OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_synthetic_recovery(
    results: list[dict],
    output_path: Path | None = None,
) -> Path:
    """Plot recovered vs true (K, theta, N) from synthetic validation.

    Parameters
    ----------
    results : list of dicts, each with keys 'K_true', 'theta_true', 'N_true',
              'K_fit', 'theta_fit', 'N_fit', 'K_err_pct', 'theta_err_pct', 'N_err_pct'
    output_path : override default path
    """
    if output_path is None:
        output_path = WP12_OUTPUT / "synthetic_recovery.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = ["K", "theta", "N"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, p in zip(axes, params):
        true_vals = np.array([r[f"{p}_true"] for r in results])
        fit_vals = np.array([r[f"{p}_fit"] for r in results])
        ax.scatter(true_vals, fit_vals, alpha=0.7, s=40)
        lo, hi = min(true_vals.min(), fit_vals.min()), max(true_vals.max(), fit_vals.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
        ax.set_xlabel(f"{p} true")
        ax.set_ylabel(f"{p} recovered")
        ax.set_title(p)
        ax.legend(fontsize=9)

    fig.suptitle("Synthetic parameter recovery (HawkesN MLE)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_bridge_check(
    result: dict,
    output_path: Path | None = None,
) -> Path:
    """Plot HawkesN mean-field rate vs SIR ODE rate.

    Parameters
    ----------
    result : output of bridge.run_bridge_check()
    """
    if output_path is None:
        output_path = WP12_OUTPUT / "bridge_check.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    times = result["times"]
    sir_rate = result["sir_rate"]
    mean_rate = result["hawkesn_mean_rate"]
    std_rate = result["hawkesn_std_rate"]
    mean_rate_cond = result.get("hawkesn_mean_rate_cond", mean_rate)
    rmse_uncond = result.get("relative_rmse_uncond", result["relative_rmse"])
    rmse_cond = result.get("relative_rmse_cond", result["relative_rmse"])
    K = result["K"]
    theta = result["theta"]
    N = result["N"]
    n_sims = result["n_sims"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, (mean_r, lbl, rmse_val, tag) in zip(
        axes,
        [
            (mean_rate, "Unconditional mean", rmse_uncond, "uncond"),
            (mean_rate_cond, "Conditional mean (large cascades)", rmse_cond, "cond"),
        ],
    ):
        ax.plot(times, sir_rate, "b-", lw=2, label=f"SIR ODE (K={K})")
        ax.plot(times, mean_r, "r-", lw=1.5, label=lbl)
        ax.set_xlabel("Time")
        ax.set_ylabel("Event rate")
        threshold = result["pre_registered_threshold"]
        confirmed = rmse_val < threshold
        ax.set_title(
            f"K={K}, {tag}: rel. RMSE={rmse_val:.3f} "
            f"({'CONFIRMED' if confirmed else 'NOT conf.'})"
        )
        ax.legend(fontsize=9)

    fig.suptitle(f"Mean-field bridge: K={K}, θ={theta}, N={N}, {n_sims} sims")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_final_size_distribution(
    results_list: list[dict],
    output_path: Path | None = None,
) -> Path:
    """Plot final-size distributions for multiple K values.

    Parameters
    ----------
    results_list : list of dicts, each with keys 'K', 'pmf', 'sizes', 'method', 'N'
    """
    if output_path is None:
        output_path = WP12_OUTPUT / "final_size_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_plots = len(results_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        K = res["K"]
        pmf = res["pmf"]
        sizes = res["sizes"]
        N = res["N"]
        method = res.get("method", "mc")

        # Plot as bar / smoothed
        ax.bar(sizes / N, pmf, width=1.0 / N, color="steelblue", alpha=0.7)
        ax.set_xlabel("Final size fraction (R/N)")
        ax.set_ylabel("Probability")
        ax.set_title(f"K = {K:.2f}, N={N} ({method})")
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle("Final-size distribution P(R(∞) = r) for HawkesN / stochastic SIR")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
