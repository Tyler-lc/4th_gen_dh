import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from config import PLOTS_DIR, buildingstock_results_path
from costs.renovation_costs import renovation_costs_iwu

unrenovated_buildingstock = gpd.read_parquet(buildingstock_results_path("unrenovated"))
unrenovated_buildingstock = unrenovated_buildingstock[
    unrenovated_buildingstock["NFA"] >= 30
]

types = [
    "ab",
    "sfh",
    "mfh",
    "th",
]

unrenovated_residential = unrenovated_buildingstock[
    unrenovated_buildingstock["building_usage"].isin(types)
]
total_energy_demand_res = (
    unrenovated_residential["yearly_space_heating"].sum()
    + unrenovated_residential["yearly_dhw_energy"].sum()
)
total_NFA_res = unrenovated_residential["NFA"].sum()

total_NFA = unrenovated_buildingstock["NFA"].sum()
total_energy_demand = (
    unrenovated_buildingstock["yearly_space_heating"].sum()
    + unrenovated_buildingstock["yearly_dhw_energy"].sum()
) / 1000000  # GWh

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Load the buildingstock results with geometry
gdf = gpd.read_parquet(buildingstock_results_path("unrenovated"))

# Filter out buildings with very small NFA (noise)
gdf = gdf[gdf["NFA"] >= 30]

# Calculate total heat demand (space heating + DHW)
gdf["total_heat_demand"] = gdf["yearly_space_heating"] + gdf["yearly_dhw_energy"]

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 12))

# Plot buildings colored by total heat demand
gdf.plot(
    column="total_heat_demand",
    ax=ax,
    legend=True,
    legend_kwds={
        "label": "Total Heat Demand (kWh/year)",
        "orientation": "vertical",
        "shrink": 0.7,
    },
    cmap="YlOrRd",  # Yellow to Orange to Red colormap
    edgecolor="black",
    linewidth=0.3,
)

ax.set_title("Building Heat Demand Distribution", fontsize=18)
ax.set_xlabel("Easting (m)", fontsize=12)
ax.set_ylabel("Northing (m)", fontsize=12)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("building_map_heat_demand.png", dpi=300, bbox_inches="tight")
plt.show()
