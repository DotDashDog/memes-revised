"""WP-18 measurement 1 plots. Reads m1_authors_monthly.csv / m1_summary.json.

    uv run python -m experiments.wp18.m1_plot
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.wp18.common import BACKBONE, COLORS, OUT, SUBS

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
START = pd.Timestamp("2016-01-01")
EVENT = pd.Timestamp("2023-06-01")  # API-pricing protest blackout month


def panels(title, subtitle):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED, linespacing=1.35)
    axes[1, 3].axis("off")
    return fig, [axes[0, i] for i in range(4)] + [axes[1, i] for i in range(3)], axes[1, 3]


def main() -> None:
    m = pd.read_csv(OUT / "m1_authors_monthly.csv", dtype={"subreddit": str})
    m["t"] = pd.to_datetime(m.month + "-01")
    summ = {c["subreddit"]: c for c in json.loads((OUT / "m1_summary.json").read_text())["communities"]}
    titles = json.loads((OUT / "m1_plot_titles.json").read_text())

    # ---- figure 1: unique named authors per month
    fig, axs, note = panels(titles["authors_title"], titles["authors_subtitle"])
    for ax, sub in zip(axs, SUBS):
        d = m[(m.subreddit == sub) & (m.t >= START)]
        s = summ[sub]
        act = pd.Timestamp(s["active_from"] + "-01")
        ax.axvspan(START, max(act, START), color="#f3f3f3", lw=0)
        ax.plot(d.t, d.posts.clip(lower=0.5), color="#b0b0b0", lw=1.2, label="all posts")
        ax.plot(d.t, d.n_named.clip(lower=0.5), color=COLORS[sub], lw=2, label="unique named authors")
        ax.axhline(100, color="#c0392b", lw=1, ls=(0, (4, 3)))
        ax.axvline(EVENT, color=MUTED, lw=0.8, ls=":")
        for st in s["steps_ge_2x"]:
            t = pd.Timestamp(st["month"] + "-01")
            if t >= START:
                ax.plot([t], [max(st["n"], 0.5)], marker="v", ms=8, color="#c0392b", mec="white", mew=1.2, ls="")
        # post-hoc addition (not pre-registered): steps of 1.5x-2x, which the 2x flag is blind to
        da = d[d.t >= act].copy()
        f = da.n_named / da.n_named.shift(1)
        for _, r in da[((f >= 1.5) & (f < 2)) | ((f <= 1 / 1.5) & (f > 0.5))].iterrows():
            ax.plot([r.t], [r.n_named], marker="v", ms=8, color="white", mec="#c0392b", mew=1.5, ls="")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 1e6)
        ax.set_xlim(START, pd.Timestamp("2026-01-01"))
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        role = "long-running set" if sub in BACKBONE else "newer community"
        ax.set_title(f"r/{sub}  ({role})\nmin {s['min_n_named']:,} in {s['min_n_named_month']}; "
                     f"Dec 2025: {s['n_named_2025_12']:,}", fontsize=10, loc="left", color=INK)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in (axs[0], axs[4]):
        ax.set_ylabel("count per calendar month (log scale)")
    for ax in axs[4:]:
        ax.set_xlabel("month (UTC)")
    h = [plt.Line2D([], [], color=INK, lw=2), plt.Line2D([], [], color="#b0b0b0", lw=1.2),
         plt.Line2D([], [], color="#c0392b", lw=1, ls=(0, (4, 3))),
         plt.Line2D([], [], color="#c0392b", marker="v", ms=8, ls=""),
         plt.Line2D([], [], color="white", mec="#c0392b", mew=1.5, marker="v", ms=8, ls=""),
         plt.Line2D([], [], color=MUTED, lw=0.8, ls=":"),
         plt.Rectangle((0, 0), 1, 1, color="#f3f3f3")]
    note.legend(h, ["unique named authors that month (coloured line)", "all submissions that month",
                    "100 authors: pre-registered 'near-zero' line",
                    "month-on-month change of 2x or more (pre-registered flag)",
                    "change of 1.5x to 2x (added after seeing the data)",
                    "June 2023 (API-pricing protest blackout)",
                    "before the community's first 1,000-post month (not judged)"],
                loc="center left", frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "m1_unique_authors.png", dpi=130)
    plt.close(fig)

    # ---- figure 2: share of posts whose author is [deleted], and share removed
    fig, axs, note = panels(titles["deleted_title"], titles["deleted_subtitle"])
    for ax, sub in zip(axs, SUBS):
        s = summ[sub]
        d = m[(m.subreddit == sub) & (m.t >= pd.Timestamp(s["active_from"] + "-01")) & (m.t >= START)]
        ax.plot(d.t, 100 * d.removed_share, color="#b0b0b0", lw=1.2)
        ax.plot(d.t, 100 * d.deleted_share, color=COLORS[sub], lw=2)
        ax.axvline(EVENT, color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(0, 100)
        ax.set_xlim(START, pd.Timestamp("2026-01-01"))
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"r/{sub}\n'[deleted]' share: {s['deleted_share_by_year_pct'].get('2022', float('nan')):.0f}% in 2022, "
                     f"{s['deleted_share_by_year_pct']['2025']:.0f}% in 2025", fontsize=10, loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in (axs[0], axs[4]):
        ax.set_ylabel("share of that month's posts (%)")
    for ax in axs[4:]:
        ax.set_xlabel("month (UTC)")
    note.legend([plt.Line2D([], [], color=INK, lw=2), plt.Line2D([], [], color="#b0b0b0", lw=1.2),
                 plt.Line2D([], [], color=MUTED, lw=0.8, ls=":")],
                ["posts whose author field is '[deleted]' (coloured line):\nthese authors cannot be counted",
                 "posts flagged removed (by moderators, spam filter,\nReddit or the author) in the archive",
                 "June 2023 (API-pricing protest blackout)"],
                loc="center left", frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "m1_deleted_author_share.png", dpi=130)
    plt.close(fig)
    print("wrote plots")


if __name__ == "__main__":
    main()
