from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUTPUT_DIR / "utsd_selection_schematic.png"
SVG_PATH = OUTPUT_DIR / "utsd_selection_schematic.svg"

SEED = 42
DOMAINS = ["Energy", "Health", "IoT", "Nature", "Transport"]
ORIGINAL_COUNTS = [10, 8, 7, 9, 8]
SELECTED_COUNTS = [3, 2, 2, 3, 2]
REDUCTION_TEXT = "10% Representative Core Set"

rng = np.random.default_rng(SEED)

def make_series(domain_index, sample_index, n=90):
    t = np.linspace(0, 1, n)
    if domain_index == 0:
        y = 0.45 * np.sin(2 * np.pi * (2.0 + 0.08 * sample_index) * t) + 0.20 * t
    elif domain_index == 1:
        y = 0.35 * np.sin(2 * np.pi * (1.0 + 0.05 * sample_index) * t + 0.4)
        y += 0.20 * np.exp(-((t - 0.65) / 0.12) ** 2)
    elif domain_index == 2:
        y = 0.28 * np.sin(2 * np.pi * (5.0 + 0.12 * sample_index) * t)
        y += 0.12 * np.sin(2 * np.pi * 11 * t)
    elif domain_index == 3:
        y = 0.25 * np.sin(2 * np.pi * (1.5 + 0.06 * sample_index) * t)
        y += 0.35 * (t - 0.5) ** 2
    else:
        y = 0.30 * np.sin(2 * np.pi * 3 * t)
        y += 0.18 * np.sin(2 * np.pi * 7 * t + 0.5)
    y += 0.05 * rng.normal(size=n)
    y += 0.025 * sample_index
    return y

def add_panel(ax, x, y, w, h, title):
    panel = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.03",
                           linewidth=1.4)
    ax.add_patch(panel)
    ax.text(x + w / 2, y + h - 0.055, title,
            ha="center", va="top", fontsize=17, fontweight="bold")

def draw_domain_group(ax, x0, y0, width, height, domain_name,
                      domain_index, count, selected=False):
    ax.text(x0, y0 + height + 0.018, domain_name,
            ha="left", va="bottom", fontsize=12, fontweight="bold")
    gap = height / max(count, 1)
    for i in range(count):
        series = make_series(domain_index, i)
        series = (series - series.mean()) / (series.std() + 1e-8)
        xs = np.linspace(x0, x0 + width, len(series))
        center_y = y0 + height - (i + 0.5) * gap
        scale = gap * (0.22 if not selected else 0.28)
        ys = center_y + scale * series
        ax.plot(xs, ys, linewidth=0.8 if not selected else 1.3)

fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

add_panel(ax, 0.03, 0.08, 0.38, 0.84, "Original UTSD")
# ax.text(0.78, 0.13, "Original UTSD",
        # ha="center", va="center", fontsize=13, fontweight="bold")

arrow = FancyArrowPatch((0.44, 0.50), (0.56, 0.50),
                        arrowstyle="-|>", mutation_scale=24, linewidth=1.8)
ax.add_patch(arrow)
ax.text(0.50, 0.56, "Select", ha="center", va="center",
        fontsize=16, fontweight="bold")
ax.text(0.50, 0.44, "Representative sampling",
        ha="center", va="center", fontsize=10)

add_panel(ax, 0.59, 0.08, 0.38, 0.84, "Representative Core Set")
ax.text(0.78, 0.13, REDUCTION_TEXT,
        ha="center", va="center", fontsize=13, fontweight="bold")

top = 0.79
group_h = 0.115
group_gap = 0.025

for idx, domain in enumerate(DOMAINS):
    y = top - idx * (group_h + group_gap)
    draw_domain_group(ax, 0.065, y, 0.31, group_h * 0.78,
                      domain, idx, ORIGINAL_COUNTS[idx], selected=False)
    draw_domain_group(ax, 0.625, y, 0.31, group_h * 0.78,
                      domain, idx, SELECTED_COUNTS[idx], selected=True)

fig.tight_layout()
fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
fig.savefig(SVG_PATH, bbox_inches="tight")
print(PNG_PATH)