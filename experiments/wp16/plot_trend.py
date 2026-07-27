import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
# palette slots 1-7; every series is direct-labeled, so identity is never colour-alone
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

DATA = [                       # 2020-08-15, 2023-08-15, 2026-07-20  (exact 24h counts)
    ("r/memes",        [12158, 1531, 287]),
    ("r/dankmemes",    [ 3608,  192,  77]),
    ("r/comedyheaven", [  662,   13,  65]),
    ("r/meirl",        [  144,  231, 147]),
    ("r/shitposting",  [   69,  789, 169]),
    ("r/196",          [    0,  616, 167]),
    ("r/whenthe",      [    0,   57, 312]),
]
X = [0, 1, 2]
LBL = ["Aug 2020", "Aug 2023", "Jul 2026"]

fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=170)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.set_yscale("log")
ax.grid(True, which="major", axis="y", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5, length=3, color=GRID)

import math
ends = []
for i, (name, ys) in enumerate(DATA):
    xs = [x for x, y in zip(X, ys) if y > 0]
    vs = [y for y in ys if y > 0]
    ax.plot(xs, vs, color=C[i], lw=2.0, zorder=3, solid_capstyle="round")
    ax.plot([xs[-1]], [vs[-1]], "o", ms=8, color=C[i], mec=SURFACE, mew=2, zorder=4)
    ends.append([math.log10(vs[-1]), vs[-1], name, C[i]])
    if len(xs) < 3:
        ax.annotate("did not exist", xy=(xs[0], vs[0]), xytext=(-8, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="center", ha="right", style="italic")

# de-collide right-edge labels in log space, then draw a leader to the true point
ends.sort(key=lambda e: e[0])
GAP = 0.085
for j in range(1, len(ends)):
    if ends[j][0] - ends[j-1][0] < GAP:
        ends[j][0] = ends[j-1][0] + GAP
for lab_y, true_y, name, col in ends:
    ax.annotate(f"{name}  {true_y:,}", xy=(2, true_y), xytext=(2.10, 10**lab_y),
                textcoords="data", fontsize=9.5, color=INK, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=col, lw=1.0,
                                shrinkA=4, shrinkB=0, alpha=.55))

ax.set_xticks(X); ax.set_xticklabels(LBL, fontsize=10, color=INK2)
ax.set_xlim(-0.28, 2.95)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
ax.set_ylabel("submissions that day  (log scale)", color=INK2, fontsize=9.5, labelpad=8)
ax.set_title("Reddit meme activity migrated — it did not simply disappear",
             color=INK, fontsize=13.5, fontweight="600", loc="left", pad=26)
ax.text(0, 1.015, "Exact 24-hour post counts via Arctic Shift. The classic big subs collapsed ~98%; newer ones grew into the gap.",
        transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
fig.tight_layout()
out = "/home/linus/memes-revised/experiments/WP-16/subreddit_migration.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight"); print("wrote", out)
