import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from costs.renovation_costs import renovation_costs_iwu

unrenovated_buildingstock_path = Path(
    "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results_unrenovated.parquet"
)
unrenovated_buildingstock = gpd.read_parquet(unrenovated_buildingstock_path)
unrenovated_buildingstock = unrenovated_buildingstock[
    unrenovated_buildingstock["NFA"] >= 30
]

renovated_buildingstock_path = Path(
    "building_analysis/results/renovated_whole_buildingstock/buildingstock_results_renovated.parquet"
)
renovated_buildingstock = gpd.read_parquet(renovated_buildingstock_path)
renovated_buildingstock = renovated_buildingstock[renovated_buildingstock["NFA"] >= 30]


building_types = unrenovated_buildingstock["building_usage"].unique()

average_specific_ued_unrenovated = unrenovated_buildingstock.groupby("building_usage")[
    "specific_ued"
].mean()
average_specific_ued_renovated = renovated_buildingstock.groupby("building_usage")[
    "specific_ued"
].mean()


labels = building_types
values_unrenovated = average_specific_ued_unrenovated.values
values_renovated = average_specific_ued_renovated.values

width = 0.35  # Bar width
x = np.arange(len(labels))  # Label locations
fig, ax = plt.subplots(figsize=(10, 6))

# Plot unrenovated bars
rects1 = ax.bar(x - width / 2, values_unrenovated, width, label="Unrenovated")

# Plot renovated bars
rects2 = ax.bar(x + width / 2, values_renovated, width, label="Renovated")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Average Specific UED (kWh/m²a)")
ax.set_title("Average Specific UED by Building Type and Renovation Status")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.xticks(rotation=45, ha="right")  # Rotate labels to prevent overlap
plt.savefig("plots/buildingstock_demand_pre_post_renovation.png", dpi=300)


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
renovated_buildingstock["renovation_cost_m2"] = (
    renovation_costs["total_cost"] / renovation_costs["NFA"]
)
average_costs_renovations_m2 = (
    renovated_buildingstock.groupby("building_usage")["renovation_cost_m2"]
    .mean()
    .fillna(0)
    .apply(lambda x: int(round(x, 0)))
)
