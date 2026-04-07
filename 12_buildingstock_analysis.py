import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from config import PAPER_FIGURE_DIR, PLOTS_DIR, buildingstock_results_path
from costs.renovation_costs import renovation_costs_iwu

unrenovated_buildingstock = gpd.read_parquet(buildingstock_results_path("unrenovated"))
unrenovated_buildingstock = unrenovated_buildingstock[
    unrenovated_buildingstock["NFA"] >= 30
]

renovated_buildingstock = gpd.read_parquet(buildingstock_results_path("renovated"))
renovated_buildingstock = renovated_buildingstock[renovated_buildingstock["NFA"] >= 30]
renovated_buildingstock.fillna(0, inplace=True)


building_types = unrenovated_buildingstock["building_usage"].unique()

average_specific_ued_unrenovated = unrenovated_buildingstock.groupby("building_usage")[
    "specific_ued"
].mean()
average_specific_ued_renovated = renovated_buildingstock.groupby("building_usage")[
    "specific_ued"
].mean()


_label_map = {
    "mfh": "MFH", "sfh": "SFH", "ab": "AB", "th": "TH",
    "other": "Other", "trade": "Trade", "education": "Education",
    "health": "Health", "office": "Office",
}
_type_order = ["mfh", "ab", "sfh", "th", "other", "trade", "education", "health", "office"]
_raw_labels = [t for t in _type_order if t in average_specific_ued_unrenovated.index]
labels = [_label_map[l] for l in _raw_labels]
values_unrenovated = average_specific_ued_unrenovated.reindex(_raw_labels).values
values_renovated = average_specific_ued_renovated.reindex(_raw_labels).values

width = 0.35  # Bar width
x = np.arange(len(labels))  # Label locations
fig, ax = plt.subplots(figsize=(12, 7))  # Increased size to accommodate labels better

# Define monochromatic colorblind-friendly colors using different shades of blue
monochromatic_colors = {
    "unrenovated": "#2C5282",  # Dark blue - for unrenovated (higher values)
    "renovated": "#90CDF4",  # Light blue - for renovated (lower values)
}

# Plot unrenovated bars with hatching for additional distinction
rects1 = ax.bar(
    x - width / 2,
    values_unrenovated,
    width,
    label="Unrenovated",
    color=monochromatic_colors["unrenovated"],
    edgecolor="black",
    linewidth=0.5,
    hatch="///",  # Add diagonal hatching pattern
)

# Plot renovated bars
rects2 = ax.bar(
    x + width / 2,
    values_renovated,
    width,
    label="Renovated",
    color=monochromatic_colors["renovated"],
    edgecolor="black",
    linewidth=0.5,
)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Average Specific UED [kWh/(m\u00b2\u00b7yr)]", fontsize=18)


# ax.set_title("Average Specific UED by Building Type and Renovation Status")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.yticks(fontsize=16)

plt.xticks(rotation=45, ha="right", fontsize=18)  # Rotate labels to prevent overlap
fig.tight_layout()

# Save with bbox_inches='tight' to prevent label cutoff
plt.savefig(
    PLOTS_DIR / "buildingstock_demand_pre_post_renovation.png", dpi=300, bbox_inches="tight"
)
if PAPER_FIGURE_DIR and PAPER_FIGURE_DIR.exists():
    plt.savefig(
        PAPER_FIGURE_DIR / "buildingstock_demand_pre_post_renovation.png",
        dpi=300,
        bbox_inches="tight",
    )


number_not_renovated = renovated_buildingstock.groupby("building_usage")[
    "insulation_thickness"
].apply(lambda x: x.isna().sum())

number_renovations = renovated_buildingstock.groupby("building_usage")[
    "insulation_thickness"
].apply(lambda x: (x > 0).sum())

total_buildingstock = unrenovated_buildingstock.groupby("building_usage").size()

convert2020_2023 = 188.40 / 133.90
renovation_costs = renovation_costs_iwu(renovated_buildingstock, convert2020_2023)

average_costs_renovations = renovation_costs.groupby("building_usage")[
    "total_cost"
].mean()
average_costs_renovations = average_costs_renovations.fillna(0)
renovated_buildingstock["renovation_cost_m2"] = (
    renovation_costs["total_cost"] / renovation_costs["NFA"]
)
average_costs_renovations_m2 = (
    renovated_buildingstock.groupby("building_usage")["renovation_cost_m2"]
    .mean()
    .fillna(0)
    .apply(lambda x: int(round(x, 0)))
)
