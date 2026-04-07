"""Generate a study area map for the 4GDH paper.

Building footprints coloured by type over street network.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from config import (
    PAPER_FIGURE_DIR,
    PLOTS_DIR,
    GRID_CALCULATION_DIR,
    buildingstock_results_path,
)

# ── Paths ────────────────────────────────────────────────────────────────
STREETS_PATH = GRID_CALCULATION_DIR / "streets_only_clean_25832.gpkg"

# ── Data ─────────────────────────────────────────────────────────────────
buildings = gpd.read_parquet(buildingstock_results_path("unrenovated"))
buildings = buildings[buildings["NFA"] >= 30].copy()
buildings = buildings.set_crs(epsg=25832)

streets = gpd.read_file(STREETS_PATH)

# ── Building type classification ─────────────────────────────────────────
label_map = {
    "mfh": "MFH",
    "ab": "AB",
    "sfh": "SFH",
    "th": "TH",
}

buildings["type_label"] = buildings["building_usage"].map(
    lambda x: label_map.get(x, "NR")
)

# ── Colours (colourblind-friendly, categorical) ──────────────────────────
# Wong (2011) palette adapted for map legibility
type_colours = {
    "MFH": "#0072B2",     # blue
    "AB": "#D55E00",      # vermillion
    "SFH": "#009E73",     # bluish green
    "TH": "#CC79A7",      # reddish purple
    "NR": "#999999",      # grey for non-residential
}

type_order = ["MFH", "AB", "SFH", "TH", "NR"]

# ── Figure ───────────────────────────────────────────────────────────────
# District is elongated east-west (~2.4 km x ~0.6 km), so use wide aspect
fig, ax_main = plt.subplots(figsize=(12, 4.5))

# Streets
streets.plot(ax=ax_main, color="#cccccc", linewidth=0.5, zorder=1)

# Buildings by type (NR first so residential overlays)
for ttype in reversed(type_order):
    subset = buildings[buildings["type_label"] == ttype]
    if len(subset) == 0:
        continue
    subset.plot(
        ax=ax_main,
        color=type_colours[ttype],
        edgecolor="black",
        linewidth=0.15,
        zorder=2 if ttype == "NR" else 3,
        label=ttype,
    )

# ── Axis extent ──────────────────────────────────────────────────────────
bx = buildings.total_bounds  # minx, miny, maxx, maxy
buf = 50  # metres
ax_main.set_xlim(bx[0] - buf, bx[2] + buf)
ax_main.set_ylim(bx[1] - buf, bx[3] + buf)
ax_main.set_aspect("equal")
ax_main.set_axis_off()

# ── Scale bar (placed in empty space, upper-right) ───────────────────────
bar_length = 200  # metres
bar_x = bx[2] - bar_length - 30
bar_y = bx[3] - 20
ax_main.plot(
    [bar_x, bar_x + bar_length], [bar_y, bar_y],
    color="black", linewidth=2, zorder=10,
)
# End ticks
for x in [bar_x, bar_x + bar_length]:
    ax_main.plot(
        [x, x], [bar_y - 8, bar_y + 8],
        color="black", linewidth=1.5, zorder=10,
    )
ax_main.text(
    bar_x + bar_length / 2, bar_y - 15, f"{bar_length} m",
    ha="center", va="top", fontsize=9, fontweight="bold", zorder=10,
)

# ── North arrow (axes fraction, upper-left, clear spacing) ───────────────
ax_main.annotate(
    "", xy=(0.02, 0.95), xytext=(0.02, 0.78),
    xycoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"),
)
ax_main.text(
    0.02, 0.97, "N",
    transform=ax_main.transAxes,
    fontsize=10, fontweight="bold", ha="center", va="bottom",
)

# ── Legend ────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(
        facecolor=type_colours[t], edgecolor="black", linewidth=0.3, label=t
    )
    for t in type_order
]
legend_handles.append(
    Line2D([0], [0], color="#cccccc", linewidth=0.8, label="Streets")
)

ax_main.legend(
    handles=legend_handles,
    loc="lower left",
    fontsize=8,
    framealpha=0.85,
    edgecolor="grey",
    title="Building type",
    title_fontsize=9,
)

fig.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────
out_main = PLOTS_DIR / "study_area_map.png"

fig.savefig(out_main, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_main.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print(f"Saved to {out_main}")

if PAPER_FIGURE_DIR and PAPER_FIGURE_DIR.exists():
    fig.savefig(PAPER_FIGURE_DIR / "study_area_map.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PAPER_FIGURE_DIR / "study_area_map.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved to {PAPER_FIGURE_DIR / 'study_area_map.png'}")

plt.close()
