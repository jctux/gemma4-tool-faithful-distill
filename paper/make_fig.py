import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# (label, model, score or None for timeout)
rows = [
    ("E4B base, 40 calls / 8 min", "E4B", 0.05),
    ("E4B + prompt2", "E4B", 0.03),
    ("E4B + public-leader prompt, 28 calls", "E4B", 0.05),
    ("E4B + Budget-Fit prompt, thinking 4096", "E4B", 0.03),
    ("31B base, 40 calls / 8 min, thinking 1024", "31B", 0.13),
    ("31B, thinking 4096", "31B", None),
    ("31B, 60 calls / 12 min", "31B", None),
    ("31B, public-leader prompt, thinking 4096", "31B", None),
    ("31B, public-leader prompt, thinking 1024", "31B", 0.12),
    ("31B, Budget-Fit prompt, thinking 4096", "31B", 0.13),
    ("31B, Budget-Fit prompt, 40 calls / 12 min", "31B", None),
]
fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
n = len(rows)
ax.axvspan(0.13 - 0.027, 0.13 + 0.027, color=GRID, alpha=0.8, zorder=0)
ax.text(0.13, -0.95, "±0.027\nsingle-run noise", ha="center", va="top", fontsize=7.5, color=MUTED)
for i, (lab, m, s) in enumerate(rows):
    y = n - 1 - i
    c = BLUE if m == "E4B" else ORANGE
    ax.hlines(y, 0, 0.30, color=GRID, lw=0.8, zorder=1)
    if s is None:
        ax.plot(0.0, y, marker="X", ms=9, color=SURF, mec=c, mew=1.8, zorder=3, clip_on=False)
        ax.text(0.012, y, "timeout, no score", va="center", fontsize=8, color=MUTED)
    else:
        ax.plot(s, y, "o", ms=9, color=c, mec=SURF, mew=2, zorder=3)
        ax.text(s + 0.012, y, f"{s:.2f}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5, color=INK)
ax.set_xlim(-0.01, 0.30); ax.set_ylim(-1.9, n - 0.4)
ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Hidden-task score (public leaderboard split)", fontsize=9, color=MUTED)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_title("Every 31B run that finished scored 0.12-0.13; four timed out", loc="left", fontsize=10.5, color=INK, pad=10)
h = [Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="E4B"),
     Line2D([], [], marker="o", ls="", color=ORANGE, ms=8, label="31B"),
     Line2D([], [], marker="X", ls="", mfc=SURF, mec=MUTED, mew=1.6, ms=8, label="Notebook Timeout")]
ax.legend(handles=h, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
fig.tight_layout()
fig.savefig("paper/fig_hidden_scores.png", facecolor=SURF)
print("ok")
