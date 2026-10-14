from pathlib import Path
# Renders the end-to-end system architecture figure used in the SRS (Chapter 4).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow

TEAL = "#31859C"; LIGHT = "#DDEEF3"; TXT = "#123642"

def box(ax, x, y, w, h, text, fill=LIGHT, fs=8.6, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                 fc=fill, ec=TEAL, lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(ax, x1, y, x2):
    ax.add_patch(FancyArrow(x1, y, x2 - x1, 0, width=0.012, head_width=0.035,
                            head_length=0.02, fc=TEAL, ec=TEAL, length_includes_head=True))

fig, ax = plt.subplots(figsize=(10.6, 4.6), dpi=200)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

W, H, GAP = 0.178, 0.15, 0.024
xs = [0.012 + i * (W + GAP) for i in range(5)]

# Row 1: query runtime
ax.text(0.012, 0.93, "QUERY RUNTIME  (every user request)", fontsize=10.5, color=TEAL, fontweight="bold")
y1 = 0.70
r1 = ["Plain-English question\nShirt search web UI\n(127.0.0.1:8010)",
      "LLM generates SQL\nNVIDIA Nemotron\n3.5 Lightning API",
      "SQLGlot AST validation\nSELECT parent_asin only\nunsafe SQL rejected",
      "DuckDB scans Parquet\n50,224 verified shirts\nno DB server",
      "Ranked product cards\nprice tier + Bayesian\nrating + reviews"]
for x, t in zip(xs, r1):
    hl = t.startswith("DuckDB")
    box(ax, x, y1, W, H, t, fill=TEAL if hl else LIGHT, tc="white" if hl else "black", bold=hl)
for i in range(4):
    arrow(ax, xs[i] + W, y1 + H/2, xs[i+1])

# connector: build row feeds the DuckDB box
ax.add_patch(FancyArrow(xs[3] + W/2 - 0.005, 0.505, 0, 0.17, width=0.012,
                        head_width=0.035, head_length=0.025, fc=TEAL, ec=TEAL,
                        length_includes_head=True))
ax.text(xs[3] + W/2 + 0.025, 0.585, "shirts_master.parquet", fontsize=8.5, color=TEAL, style="italic")

# Row 2: catalogue build
ax.text(0.012, 0.445, "CATALOGUE BUILD  (offline, one time)", fontsize=10.5, color=TEAL, fontweight="bold")
y2 = 0.215
r2 = ["7.2M Amazon\nmetadata records",
      "Rule segregation\n63,190 candidates",
      "CLIP ViT-B/32 verify\n50,224 shirts confirmed",
      "Price estimation, review\nmining, fabric/fit,\nKaggle v3.2 colour+pattern",
      "*_original audit columns,\nbackups, exact\nID/URL checks"]
for x, t in zip(xs, r2):
    box(ax, x, y2, W, H, t, fs=8.2)
for i in range(4):
    arrow(ax, xs[i] + W, y2 + H/2, xs[i+1])

ax.text(0.012, 0.075, "Future (Milestone 2): reviews_v2 verified-purchase join  +  sentiment ranking  +  Recall@10 / NDCG@10 / MRR evaluation",
        fontsize=8, color=TXT, style="italic")

plt.tight_layout(pad=0.4)
out = Path(__file__).resolve().parent / "architecture.png"
plt.savefig(out, facecolor="white")
print("saved", out)
