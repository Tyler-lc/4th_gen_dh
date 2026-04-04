"""
Calculate total installed heat pump capacity per scenario for the paper.

Used for R1-2.6 + R5-SC6: Electrical grid implications paragraph in the
Limitations section.

Central HP capacities are taken from Table 2 of the paper (design parameters).
Booster capacities are computed from the buildingstock parquet.
"""

import pandas as pd
from pathlib import Path

# ── Central HP capacities (from Table 2 / scenario scripts) ──────────────
scenarios = {
    "HT": {"n_central_units": 3, "capacity_per_unit_MW": 22.0},
    "LT+Reno": {"n_central_units": 2, "capacity_per_unit_MW": 16.5},
    "Booster": {"n_central_units": 2, "capacity_per_unit_MW": 24.5},
}

for name, params in scenarios.items():
    params["total_central_MW"] = params["n_central_units"] * params["capacity_per_unit_MW"]

# ── Booster building-level capacities ────────────────────────────────────
# NFA >= 30 m² filter matches 08_Booster_Scenario.py line 119.
# Buildings below this threshold are excluded from the TEO analysis.
booster_parquet = Path(
    "building_analysis/results/booster_whole_buildingstock/"
    "buildingstock_booster_whole_buildingstock_results.parquet"
)
booster_df = pd.read_parquet(booster_parquet)
n_total = len(booster_df)
booster_df = booster_df[booster_df["NFA"] >= 30]

hp_col = "heat_pump_size [kW]"
n_boosters = len(booster_df)
total_booster_kW = booster_df[hp_col].sum()
total_booster_MW = total_booster_kW / 1000
mean_booster_kW = booster_df[hp_col].mean()
median_booster_kW = booster_df[hp_col].median()

# Booster maintenance cost
fixed_costs_boosters = 250  # €/booster per year (from Vivian et al. 2018)
total_annual_maintenance = n_boosters * fixed_costs_boosters

# ── Summary ──────────────────────────────────────────────────────────────
print("=" * 65)
print("INSTALLED HEAT PUMP CAPACITY PER SCENARIO")
print("=" * 65)

print(f"\n{'Scenario':<12} {'Central':<15} {'Distributed':<15} {'Total':<10}")
print("-" * 52)

for name, params in scenarios.items():
    central = f"{params['total_central_MW']} MW"
    if name == "Booster":
        distributed = f"{total_booster_MW:.1f} MW"
        total = f"{params['total_central_MW'] + total_booster_MW:.1f} MW"
    else:
        distributed = "—"
        total = f"{params['total_central_MW']} MW"
    print(f"{name:<12} {central:<15} {distributed:<15} {total:<10}")

print(f"\n{'BOOSTER DETAILS':}")
print(f"  Buildings in parquet:              {n_total}")
print(f"  After NFA >= 30 filter:            {n_boosters}")
print(f"  Total distributed capacity:        {total_booster_MW:.1f} MW")
print(f"  Mean capacity per booster:         {mean_booster_kW:.1f} kW")
print(f"  Median capacity per booster:       {median_booster_kW:.1f} kW")

print(f"\n{'BOOSTER MAINTENANCE':}")
print(f"  Fixed cost per booster:            {fixed_costs_boosters} €/year")
print(f"  Total annual maintenance:          {total_annual_maintenance:,.0f} €/year")

# ── Electricity demand (booster-side) ────────────────────────────────────
elec_col = "total_demand_electricity [kWh]"
total_booster_elec_MWh = booster_df[elec_col].sum() / 1000
print(f"\n{'BOOSTER ELECTRICITY DEMAND':}")
print(f"  Total annual electricity (boosters): {total_booster_elec_MWh:,.0f} MWh")
