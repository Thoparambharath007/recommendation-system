from pathlib import Path
# Renders the use-case diagram (Fig 3.1) for the SRS.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch
import numpy as np

TEAL = "#31859C"
fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=200)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

ax.add_patch(FancyBboxPatch((3.1, 0.6), 5.2, 8.8, boxstyle="square,pad=0.05",
                             fc="white", ec=TEAL, lw=1.4))
ax.text(5.7, 9.05, "SmartShirts Query Application", ha="center", fontsize=10,
        color=TEAL, fontweight="bold")

cases = [
    (8.05, "Ask in plain English"),
    (6.75, "Filter by exact colour /\npattern / sleeve / price"),
    (5.45, "Sort: cheapest /\nhighest-rated / most-reviewed"),
    (4.15, "Include / exclude uncertain\nattributes"),
    (2.85, "Browse ranked product cards"),
    (1.45, "Update verified catalog\n(backed up, audited)"),
]
for y, t in cases:
    ax.add_patch(Ellipse((5.7, y), 3.7, 1.0, fc="#DDEEF3", ec=TEAL, lw=1.1))
    ax.text(5.7, y, t, ha="center", va="center", fontsize=8, linespacing=1.2)

def actor(x, y, name):
    ax.plot([x, x], [y, y + 0.9], color="black", lw=1.2)          # body
    ax.add_patch(Ellipse((x, y + 1.15), 0.42, 0.42, fc="white", ec="black", lw=1.2))
    ax.plot([x - 0.32, x + 0.32], [y + 0.62, y + 0.62], color="black", lw=1.2)  # arms
    ax.plot([x, x - 0.28], [y, y - 0.45], color="black", lw=1.2)
    ax.plot([x, x + 0.28], [y, y - 0.45], color="black", lw=1.2)
    ax.text(x, y - 0.85, name, ha="center", fontsize=9, fontweight="bold")

actor(1.1, 6.4, "Shopper")
for y, _ in cases[:5]:
    ax.add_patch(FancyArrowPatch((1.42, 6.85), (3.85, y), arrowstyle="-", color="black", lw=0.9))
actor(9.35, 1.2, "Maintainer")
ax.add_patch(FancyArrowPatch((9.0, 1.45), (7.55, 1.45), arrowstyle="-", color="black", lw=0.9))

plt.tight_layout(pad=0.3)
out = Path(__file__).resolve().parent / "usecase.png"
plt.savefig(out, facecolor="white")
print("saved", out)
