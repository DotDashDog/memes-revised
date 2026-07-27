"""WP-16 gate artifact: is the r/memes decline real, or a collection artifact?

Plots the dump's monthly submission curve with independent Arctic Shift day-counts
overlaid, rescaled to a monthly equivalent so they sit on the same axis.

Regenerate the inputs with:
    experiments/wp16/count_monthly.py   -> monthly.json   (from the .zst dumps)
    experiments/wp16/arctic_shift.py    -> exact day counts (see WP-16's table)

Counts below are EXACT 24-hour scans. An earlier version of this plot used four
sampled hours scaled x6, which over-estimated by ~34% and produced a spurious
"the decline has flattened" reading. Do not reintroduce sampling here.
"""
import calendar
import datetime as dt
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
ARCHIVE, INDEP, GRID, RULE = "#2a78d6", "#eb6834", "#e5e4e0", "#9a9994"

MONTHLY = sys.argv[1] if len(sys.argv) > 1 else "/home/linus/.claude/jobs/b2e9a557/tmp/monthly.json"
OUT = "/home/linus/memes-revised/experiments/WP-16/decline_crosscheck.png"

# Exact 24-hour Arctic Shift counts. `indep` marks days on/after 2023-04, when Arctic
# Shift switched to its own Reddit-API collection; earlier days share pushshift lineage.
ARCTIC = {
    "2020-08-15": (12158, False),
    "2022-08-15": (1690, False),
    "2024-08-15": (714, True),
    "2025-08-15": (378, True),
    "2026-07-20": (287, True),
}

mon = json.load(open(MONTHLY))["2025cut"]
pts = sorted((dt.date(int(k[:4]), int(k[5:]), 1), v) for k, v in mon.items() if k >= "2017-01")
xs, ys = [p[0] for p in pts], [p[1] for p in pts]

fig, ax = plt.subplots(figsize=(11, 5.6), dpi=170)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_yscale("log")
ax.grid(True, which="major", axis="y", color=GRID, lw=0.8, zorder=0)
ax.grid(True, which="minor", axis="y", color=GRID, lw=0.4, alpha=.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)

split = dt.date(2023, 4, 1)
ax.axvline(split, color=RULE, lw=1.2, ls=(0, (5, 3)), zorder=1)
ax.annotate("2023-04 — Arctic Shift switches to\nits own Reddit-API collection.\n"
            "Points right of this line are\nindependent of the dumps.",
            xy=(split, 1.7e5), xytext=(10, 0), textcoords="offset points",
            fontsize=8.5, color=INK2, va="center", ha="left", linespacing=1.45)

ax.plot(xs, ys, color=ARCHIVE, lw=2.0, zorder=3, label="Pushshift dump",
        solid_capstyle="round")

for day, (n, indep) in ARCTIC.items():
    d = dt.datetime.strptime(day, "%Y-%m-%d").date()
    monthly_equiv = n * calendar.monthrange(d.year, d.month)[1]
    ax.plot([d], [monthly_equiv], "o" if indep else "s", ms=10 if indep else 8,
            color=INDEP if indep else SURFACE, mec=INDEP, mew=2.0, zorder=5,
            label="_nolegend_")

ax.plot([], [], "o", ms=10, color=INDEP, mec=INDEP, mew=2, label="Arctic Shift — independent")
ax.plot([], [], "s", ms=8, color=SURFACE, mec=INDEP, mew=2, label="Arctic Shift — same lineage")

last_n = ARCTIC["2026-07-20"][0]
ax.annotate(f"{last_n}/day in Jul 2026 —\nstill falling, not flattening",
            xy=(dt.date(2026, 7, 20), last_n * 31),
            xytext=(-16, -30), textcoords="offset points", fontsize=8.5, color=INK,
            ha="right", linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color=RULE, lw=1))

ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2,
          handletextpad=.7, bbox_to_anchor=(0.005, 0.99))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
ax.set_ylabel("submissions per month  (log scale)", color=INK2, fontsize=9.5, labelpad=8)
ax.set_title("Cross-check: is the r/memes decline real, or a collection artifact?",
             color=INK, fontsize=13.5, fontweight="600", loc="left", pad=16)
ax.text(0, 1.015,
        "Exact 24-hour counts. All four overlapping days match the dump exactly, "
        "so the dump is not under-collecting recent years.",
        transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")

fig.tight_layout()
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
print("wrote", OUT)
