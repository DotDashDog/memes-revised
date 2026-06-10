"""Plotting helpers for WP-02."""

from __future__ import annotations

from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

THETA_LABELS = [r"$\log \beta$", r"$\log \gamma$", r"$\log s$", r"$\mathrm{logit}(I_0)$"]


def save_corner_plot(
    samples: np.ndarray,
    output_path: Path,
    *,
    truths: np.ndarray | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Save a corner plot for Hessian-drawn or NUTS posterior samples."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = corner.corner(
        samples,
        labels=labels or THETA_LABELS,
        truths=truths,
        show_titles=True,
        title_fmt=".2g",
    )
    if title:
        figure.suptitle(title)
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)
