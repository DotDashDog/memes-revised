import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import datetime as dt

SURFACE, INK, INK2, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
GRID, RULE = "#e5e4e0", "#9a9994"

d = json.load(open("/home/linus/.claude/jobs/b2e9a557/tmp/monthly.json"))["2025cut"]
pts = sorted((dt.date(int(k[:4]), int(k[5:]), 1), v) for k, v in d.items() if k >= "2017-01")
xs, ys = [p[0] for p in pts], [p[1] for p in pts]

fig, ax = plt.subplots(figsize=(11, 5.4), dpi=170)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

ax.set_yscale("log")
ax.grid(True, which="major", axis="y", color=GRID, lw=0.8, zorder=0)
ax.grid(True, which="minor", axis="y", color=GRID, lw=0.4, alpha=.6, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)

# event markers
lock = dt.date(2023, 5, 1)
ax.axvline(lock, color=RULE, lw=1.4, ls=(0, (5, 3)), zorder=1)
ax.annotate("May 2023 — Reddit\nrevokes Pushshift API", xy=(lock, 1.15e4),
            xytext=(-8, 0), textcoords="offset points", fontsize=8.5,
            color=INK2, va="center", ha="right", linespacing=1.4)
cut = dt.date(2023, 12, 31)
ax.axvline(cut, color=RULE, lw=1.0, ls=(0, (2, 3)), alpha=.75, zorder=1)
ax.annotate("old 2023-12\ncut ends here", xy=(cut, 1.15e4), xytext=(8, 0),
            textcoords="offset points", fontsize=8.5, color=INK2, ha="left", va="center", linespacing=1.4)

ax.plot(xs, ys, color=SERIES, lw=2.0, zorder=3, solid_capstyle="round")

# direct labels: peak and final
pk = max(pts, key=lambda p: p[1])
ax.plot([pk[0]], [pk[1]], "o", ms=8, color=SERIES, mec=SURFACE, mew=2, zorder=4)
ax.annotate(f"peak {pk[1]:,}  ({pk[0]:%b %Y})", xy=pk, xytext=(10, -4),
            textcoords="offset points", fontsize=9, color=INK, ha="left", fontweight="600")
last = pts[-1]
ax.plot([last[0]], [last[1]], "o", ms=8, color=SERIES, mec=SURFACE, mew=2, zorder=4)
ax.annotate(f"{last[1]:,}\n{last[0]:%b %Y}", xy=last, xytext=(8, -2),
            textcoords="offset points", fontsize=9, color=INK, ha="left")

ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
ax.set_ylabel("submissions per month  (log scale)", color=INK2, fontsize=9.5, labelpad=8)
ax.set_title("r/memes submissions per month, 2017–2025",
             color=INK, fontsize=13.5, fontweight="600", loc="left", pad=16)
ax.text(0, 1.015, "From the 2025-12 Pushshift dump. Log axis: a constant percentage decline is a straight line.",
        transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")

fig.tight_layout()
out = "/home/linus/memes-revised/experiments/WP-16/submissions_per_month.png"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)

# monthly detail across the boundary
print("\nmonthly, 2023-01 .. 2024-06:")
for k in sorted(d):
    if "2023-01" <= k <= "2024-06": print(f"  {k}  {d[k]:>8,}")
