"""Compute the carbon price needed to fund renovation subsidies from carbon tax revenue.

Policy question: if a government wants to subsidise X% of renovation costs using
carbon tax revenue collected from the district's gas consumption, what carbon price
is needed over a given payback period?

This is a district-level calculation using actual building stock data.
Reads from: building stock parquets (renovation costs, gas demand)
Writes nothing — output is printed to console.
"""

import sys
from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import buildingstock_results_path
from costs.renovation_costs import renovation_costs_iwu

# ──────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────
CARBON_INTENSITY = 0.2  # tCO2/MWh of natural gas
BOILER_EFFICIENCY = 0.9
CONVERT_2020_2023 = 188.40 / 133.90  # construction cost index update
SUBSIDY_FRACTIONS = [0.2, 0.5, 0.8, 1.0]
PAYBACK_YEARS = [10, 15, 25]

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────
print("Loading building stock data...")

# Renovated building stock — for renovation costs
bs_reno = gpd.read_parquet(buildingstock_results_path("renovated"))
bs_reno = bs_reno[bs_reno["NFA"] >= 30]
reno_costs = renovation_costs_iwu(bs_reno, CONVERT_2020_2023)
reno_costs["total_cost"] = reno_costs["total_cost"].fillna(0)

total_renovation_cost = reno_costs["total_cost"].sum()
n_renovated = (reno_costs["total_cost"] > 0).sum()
n_total = len(reno_costs)

# Unrenovated building stock — for gas consumption (baseline)
bs_unren = gpd.read_parquet(buildingstock_results_path("unrenovated"))
bs_unren = bs_unren[bs_unren["NFA"] >= 30]
useful_demand_kwh = (
    bs_unren["yearly_dhw_energy"] + bs_unren["yearly_space_heating"]
).sum()
gas_consumption_mwh = useful_demand_kwh / BOILER_EFFICIENCY / 1000
annual_co2 = gas_consumption_mwh * CARBON_INTENSITY  # tCO2/yr

# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("DISTRICT DATA")
print("=" * 70)
print(f"  Buildings: {n_total} total, {n_renovated} requiring renovation")
print(f"  Total renovation cost: {total_renovation_cost / 1e6:.1f} M EUR")
print(f"  Average per renovated building: {total_renovation_cost / n_renovated / 1e3:.0f} k EUR")
print(f"  Annual gas consumption (baseline): {gas_consumption_mwh:,.0f} MWh")
print(f"  Annual CO2 emissions (baseline): {annual_co2:,.0f} tCO2")

print()
print("=" * 70)
print("CARBON PRICE TO FUND RENOVATION SUBSIDIES")
print("=" * 70)
print()
print("Assumes: carbon tax revenue from district gas consumption funds the subsidy.")
print("Simple undiscounted payback (tax collected over N years = subsidy cost).")
print()

header = f"{'Subsidy':>10s} {'Cost (M€)':>10s}"
for y in PAYBACK_YEARS:
    header += f" {f'{y}yr':>10s}"
print(header)
print("-" * len(header))

for frac in SUBSIDY_FRACTIONS:
    subsidy_cost = total_renovation_cost * frac
    row = f"{frac * 100:>9.0f}% {subsidy_cost / 1e6:>9.1f}"
    for years in PAYBACK_YEARS:
        annual_needed = subsidy_cost / years
        carbon_price = annual_needed / annual_co2
        row += f" {carbon_price:>8.0f} €/t"
    print(row)

print()
print("Note: this is a simplified calculation assuming the full district gas")
print("consumption is subject to the carbon tax. In practice, the tax base")
print("would be larger (city/national level), yielding lower per-tCO2 prices.")
print("The values above represent the district-level upper bound.")
