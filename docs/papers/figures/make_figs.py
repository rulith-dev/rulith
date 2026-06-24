import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/sessions/serene-magical-carson/mnt/galaxy-core/docs/papers/figures")
OUT.mkdir(parents=True, exist_ok=True)

BOARD = "#1b6ca8"   # board = blue
BASE  = "#e07b39"   # baseline = orange
GRID  = "#dddddd"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

# ---------- Fig B: pre/post harness (qwen3.6-35b-a3b) ----------
fig, ax = plt.subplots(figsize=(6, 4))
fams = ["arith", "audit", "coding"]
pre, post = [72, 42, 86], [100, 100, 100]
x = np.arange(len(fams)); w = 0.38
b1 = ax.bar(x - w/2, pre,  w, label="pre-harness", color=BASE)
b2 = ax.bar(x + w/2, post, w, label="post-harness", color=BOARD)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{int(b.get_height())}",
            ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(fams)
ax.set_ylim(0, 108); ax.set_ylabel("board score (%)")
ax.set_title("qwen3.6-35b-a3b (3B-active): the harness was hiding the capability")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
save(fig, "fig_prepost_harness")

# ---------- Fig C: belief revision dual-axis ----------
fig, ax = plt.subplots(figsize=(6.6, 4.2))
sizes = [8, 13, 19, 27]
base_acc = [96.5, 94.7, 93.9, 94.0]
base_err = [9, 24, 39, 55]
ax.plot(sizes, [100]*4, "-o", color=BOARD, label="board accuracy (100%)")
ax.plot(sizes, base_acc, "-s", color=BASE, label="baseline accuracy")
ax.set_xlabel("belief-network size (avg conclusions / case)")
ax.set_ylabel("per-conclusion accuracy (%)")
ax.set_ylim(90, 101); ax.set_xticks(sizes)
ax2 = ax.twinx()
ax2.plot(sizes, [0]*4, "--^", color=BOARD, alpha=0.55, label="board over-retractions (0)")
ax2.plot(sizes, base_err, "--D", color=BASE, alpha=0.9, label="baseline over-retractions")
ax2.set_ylabel("model over-retraction errors (count)")
ax2.set_ylim(0, 60)
ax.set_title("Belief revision: board exact + scale-invariant; model erodes with scale")
l1, la = ax.get_legend_handles_labels(); l2, lb = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la + lb, frameon=False, fontsize=8, loc="center left")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
save(fig, "fig_belief_revision")

# ---------- Fig A: thinking x driving 2x2 ----------
fig, ax = plt.subplots(figsize=(6.6, 5.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.add_patch(plt.Rectangle((0, 0.5), 0.5, 0.5, color=BOARD, alpha=0.08))
ax.axvline(0.5, color="#aaaaaa", lw=1); ax.axhline(0.5, color="#aaaaaa", lw=1)
ax.text(0.25, 0.96, "SWEET SPOT\nboard recovers most", ha="center", va="top",
        fontsize=9.5, color=BOARD, weight="bold")
ax.text(0.75, 0.96, "board can't help\n(driving collapses)", ha="center", va="top", fontsize=9, color="#888")
ax.text(0.25, 0.04, "board draws even\n+ structural guarantee", ha="center", va="bottom", fontsize=9, color="#666")
ax.text(0.75, 0.04, "board can't help", ha="center", va="bottom", fontsize=9, color="#888")
pts = [
    (0.16, 0.86, "qwen3.6-35b-a3b ★ (3B-act)", BOARD, 9),
    (0.30, 0.80, "qwen3.6-27b", BOARD, 8),
    (0.20, 0.74, "gemma-4-31b", BOARD, 8),
    (0.34, 0.64, "deepseek (audit)", BOARD, 8),
    (0.58, 0.58, "gemma-4-26b-a4b (4B-act)", "#c47f3d", 8),
    (0.26, 0.22, "strong models\n(easy arith, coding-trust)", "#666", 8),
    (0.76, 0.52, "gemma-4-12b", BASE, 8),
    (0.87, 0.44, "qwen3.5-9b", BASE, 8),
]
for px, py, t, c, fs in pts:
    ax.plot(px, py, "o", color=c, ms=9)
    ax.annotate(t, (px, py), textcoords="offset points", xytext=(9, 2), fontsize=fs, color=c)
ax.set_xlabel("Driving   (clean  →  collapsed)")
ax.set_ylabel("Free-reasoning error   (low  →  high)")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Thinking × Driving: where the board helps")
save(fig, "fig_2x2")

# ---------- Fig D (bonus): board vs baseline across models (arith + audit) ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
models = ["qwen3.6-35b-a3b", "qwen3.6-27b", "gemma-4-31b", "deepseek", "gpt-oss-120b", "qwen3.5-9b †", "gemma-4-12b †"]
arith_board = [100, 100, 100, 100, 96, 22, 50]
arith_base  = [6, 4, 4, 98, 88, 6, 0]
aud_board   = [100, 100, 92, 100, 86, 27, 17]
aud_base    = [24, 100, 90, 52, 56, 33, 42]
def panel(ax, board, base, title):
    x = np.arange(len(models)); w = 0.4
    ax.bar(x - w/2, base, w, label="baseline", color=BASE)
    ax.bar(x + w/2, board, w, label="board", color=BOARD)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 108); ax.set_title(title); ax.set_ylabel("score (%)")
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
panel(axes[0], arith_board, arith_base, "arith")
panel(axes[1], aud_board, aud_base, "audit")
axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
fig.suptitle("Board vs baseline  († = boundary probe, small n)", y=1.01)
save(fig, "fig_main_results")

# ---------- Fig 1: rulith board architecture (schematic) ----------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(11.5, 4.4))
ax.set_xlim(0, 14.5); ax.set_ylim(0, 6); ax.axis("off")
def rbox(x, y, w, h, fc, ec, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", fc=fc, ec=ec, lw=lw))
rbox(0.3, 2.1, 2.7, 1.9, "#f5efe6", BASE)
ax.text(1.65, 3.55, "MODEL", ha="center", fontsize=11, weight="bold", color="#222")
ax.text(1.65, 3.08, "(proposes)", ha="center", fontsize=9, color="#555")
ax.text(1.65, 2.55, "latent · text", ha="center", fontsize=9, color="#555", style="italic")
ax.plot([3.95, 3.95], [0.6, 5.2], ls=(0, (5, 3)), color="#999", lw=1.5)
ax.text(3.95, 5.42, "commitment membrane", ha="center", va="bottom", fontsize=8, color="#777")
rbox(5.1, 0.7, 6.7, 4.5, "#edf3f8", BOARD, 1.5)
ax.text(8.45, 4.78, "BOARD — working memory + stratified-negation closure",
        ha="center", fontsize=9, weight="bold", color=BOARD)
for i, t in enumerate([
        "derivation gate — only derived facts count (no self-cert)",
        "truth maintenance — retract cascades along evidenceRefs",
        "exact-or-fail arithmetic — exact within ±2^53, else fail",
        "grounding floor — derived > attested > asserted"]):
    ax.text(5.35, 4.05 - i * 0.72, "•  " + t, ha="left", va="center", fontsize=8.5, color="#333")
ar = dict(arrowstyle="-|>", color="#555", lw=1.5, mutation_scale=15)
ax.add_patch(FancyArrowPatch((3.05, 3.0), (5.1, 3.0), **ar))
ax.text(4.075, 3.45, "facts · rules · actions · claims", ha="center", va="bottom", fontsize=6.5, color="#666")
ax.add_patch(FancyArrowPatch((11.8, 3.0), (12.9, 3.0), **ar))
ax.text(13.0, 3.5, "derived ✓", ha="left", fontsize=8.5, color=BOARD, weight="bold")
ax.text(13.0, 3.0, "effect", ha="left", fontsize=8, color="#555")
ax.text(13.0, 2.5, "rejected", ha="left", fontsize=8, color=BASE)
ax.set_title("rulith: the model proposes, the board adjudicates", fontsize=11.5)
save(fig, "fig_architecture")

print("OK:", sorted(p.name for p in OUT.glob("*.png")))
