"""Test LCOH/NPV consistency using actual HT scenario parameters.

Two cases:
  Case A: LCOH_HP (25yr) + LCOH_dhg (50yr), sell at LCOH, OUC residual → NPV should be ≈ 0
  Case B: LCOH_HP (25yr) + LCOH_dhg (25yr), sell at LCOH, no residual  → NPV should be ≈ 0

This script reads input data (building stock, grid costs) but writes nothing.
All calculations are self-contained and do not modify any existing output.
"""

import sys
from pathlib import Path

import numpy as np
import numpy_financial as npf
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    area_results_path,
    grid_results_parquet,
    weather_data_path,
)
from costs.heat_supply import (
    calculate_future_values,
    calculate_lcoh,
    calculate_revenues,
    capital_costs_hp,
    fixed_oem_hp,
    var_oem_hp,
)
from costs.renovation_costs import npv_2
from heat_supply.carnot_efficiency import carnot_cop

# ──────────────────────────────────────────────────────────────────────
# Parameters (identical to 05b_HT_Scenario.py)
# ──────────────────────────────────────────────────────────────────────
supply_temperature = 90
approach_temperature = 5
r = 0.05                    # discount rate (same for HP, DHG, and operator NPV)
n_years_hp = 25
dhg_lifetime = 50
heat_pump_lifetime = 25     # NPV horizon
margin = 0
reduction_factor = 1
safety_factor = 1.2
n_heat_pumps = 3
initial_electricity_cost = 0.1776

DK_to_DE = 109.1 / 148.5
update2022_2023 = 126.6 / 111.2

# ──────────────────────────────────────────────────────────────────────
# Read input data (read-only, same sources as 05b)
# ──────────────────────────────────────────────────────────────────────
print("Reading input data...")
ember_results = pd.read_parquet(grid_results_parquet("unrenovated"))
investment_costs_dhg = ember_results["cost_total"].sum() / 1e6  # M€

areas_demand = pd.read_csv(area_results_path("unrenovated"), index_col=0)
areas_demand.index = pd.to_datetime(areas_demand.index)
areas_demand["total_useful_demand"] = areas_demand["dhw_energy"] + areas_demand["space_heating"]
areas_demand["delivered_energy"] = areas_demand["total_useful_demand"] / 0.8
total_power_losses = ember_results["Losses [W]"].sum()
areas_demand["hourly grid losses [kWh]"] = total_power_losses / 1000
areas_demand["hourly heat generated in Large HP [kWh]"] = (
    areas_demand["hourly grid losses [kWh]"] + areas_demand["delivered_energy"]
)

estimated_capacity = areas_demand["hourly heat generated in Large HP [kWh]"].max()
heat_pump_load = areas_demand["hourly heat generated in Large HP [kWh]"] / n_heat_pumps / 1000
capacity_single_hp = estimated_capacity / n_heat_pumps * safety_factor / 1000

outside_temp = pd.read_csv(weather_data_path(), usecols=["T2m"])
outside_temp.index = areas_demand.index
supply_temp = pd.DataFrame(supply_temperature, index=outside_temp.index, columns=["supply_temp"])
cop_hourly = carnot_cop(supply_temp, outside_temp, approach_temperature)
P_el = areas_demand["hourly heat generated in Large HP [kWh]"] / cop_hourly

# ──────────────────────────────────────────────────────────────────────
# Derived costs (same as 05b)
# ──────────────────────────────────────────────────────────────────────
installation_cost_HP = capital_costs_hp(capacity_single_hp, "air") * DK_to_DE * update2022_2023
total_installation_costs = installation_cost_HP * n_heat_pumps * capacity_single_hp  # M€
single_var_oem = var_oem_hp(capacity_single_hp, "air", heat_pump_load.sum()) * DK_to_DE * update2022_2023
single_fix_oem = fixed_oem_hp(capacity_single_hp, "air") * DK_to_DE * update2022_2023
total_var_oem = single_var_oem * n_heat_pumps
total_fixed_oem = single_fix_oem * n_heat_pumps * capacity_single_hp
yearly_heat_supplied = areas_demand["delivered_energy"].sum() / 1000  # MWh

future_el_prices = calculate_future_values({"electricity": initial_electricity_cost}, n_years_hp)
total_electricity_cost = P_el.sum() * future_el_prices / 1e6  # M€
total_electricity_cost_df = pd.DataFrame(total_electricity_cost)

# Common data for LCOH calculations
fixed_oem_hp_df = calculate_future_values({"Fixed O&M": total_fixed_oem}, n_years_hp)
var_oem_hp_df = calculate_future_values({"Variable O&M": total_var_oem}, n_years_hp)
heat_supplied_hp_df = pd.DataFrame({"Heat Supplied (MW)": [yearly_heat_supplied] * n_years_hp})

dhg_zeros = pd.DataFrame(np.zeros(dhg_lifetime))
dhg_zeros_25 = pd.DataFrame(np.zeros(n_years_hp))
heat_supplied_dhg_50 = pd.DataFrame({"Heat Supplied (MW)": [yearly_heat_supplied] * dhg_lifetime})
heat_supplied_dhg_25 = pd.DataFrame({"Heat Supplied (MW)": [yearly_heat_supplied] * n_years_hp})

# Operator annual costs (same for all cases)
total_yearly_costs_hps = (
    total_var_oem + total_fixed_oem + total_electricity_cost.iloc[0, 0] * 1e6
)  # EUR/year
overnight_costs = (total_installation_costs + investment_costs_dhg) * 1e6  # EUR

print(f"\n{'='*70}")
print(f"HT Scenario Parameters")
print(f"{'='*70}")
print(f"  HP investment:  {total_installation_costs:.2f} M€")
print(f"  DHG investment: {investment_costs_dhg:.2f} M€")
print(f"  Total overnight: {overnight_costs/1e6:.2f} M€")
print(f"  Annual heat delivered: {yearly_heat_supplied:,.0f} MWh")
print(f"  Annual operator costs: {total_yearly_costs_hps/1e6:.2f} M€")
print(f"  Discount rate: {r*100:.0f}%")


def run_npv(lcoh_total, residual_pct, label):
    """Run operator NPV with given LCOH and residual value."""
    # Revenue: sell all heat at LCOH (margin=0, RF=1)
    annual_revenue = yearly_heat_supplied * 1000 * lcoh_total  # kWh * €/kWh = €
    future_revenues = calculate_future_values({"revenues": annual_revenue}, heat_pump_lifetime)
    future_revenues.iloc[-1] += investment_costs_dhg * 1e6 * residual_pct

    future_expenses = calculate_future_values({"costs": total_yearly_costs_hps}, heat_pump_lifetime)
    npv_val, _ = npv_2(-overnight_costs, future_expenses, future_revenues, r)

    print(f"\n--- {label} ---")
    print(f"  LCOH total:    {lcoh_total*1000:.2f} €/MWh")
    print(f"  Annual revenue: {annual_revenue/1e6:.2f} M€")
    print(f"  Residual value: {residual_pct*100:.1f}% = {investment_costs_dhg * residual_pct:.2f} M€ (undiscounted)")
    print(f"  Residual PV:    {investment_costs_dhg * residual_pct / (1+r)**25:.2f} M€")
    print(f"  Operator NPV:  {npv_val/1e6:+.4f} M€")
    return npv_val


def compute_ouc(dhg_inv_eur, lcoh_dhg, annual_heat_kwh, r, npv_years, lcoh_years):
    """Compute Outstanding Unrecovered Capital at the end of NPV horizon.

    The OUC is the portion of the DHG investment not yet recovered by
    LCOH_dhg-based revenue over npv_years, expressed as an undiscounted
    value at year npv_years.
    """
    # PV of heat output over LCOH horizon (same discounting as calculate_lcoh)
    pv_heat_lcoh = sum(annual_heat_kwh / (1 + r) ** (t + 1) for t in range(lcoh_years))
    # PV of heat output over NPV horizon
    pv_heat_npv = sum(annual_heat_kwh / (1 + r) ** (t + 1) for t in range(npv_years))

    # Fraction of DHG investment recovered through npv_years of LCOH revenue
    recovery_fraction = pv_heat_npv / pv_heat_lcoh
    # Outstanding unrecovered capital in PV terms (year 0)
    ouc_pv = dhg_inv_eur * (1 - recovery_fraction)
    # Project forward to year npv_years (undiscounted at that point)
    ouc_at_year_n = ouc_pv * (1 + r) ** npv_years

    pct = ouc_at_year_n / dhg_inv_eur
    print(f"\n  OUC calculation:")
    print(f"    PV heat (LCOH {lcoh_years}yr): {pv_heat_lcoh/1e6:.2f}M kWh-equiv")
    print(f"    PV heat (NPV  {npv_years}yr): {pv_heat_npv/1e6:.2f}M kWh-equiv")
    print(f"    Recovery fraction: {recovery_fraction*100:.2f}%")
    print(f"    OUC (PV, year 0): {ouc_pv/1e6:.2f} M€")
    print(f"    OUC (undiscounted, year {npv_years}): {ouc_at_year_n/1e6:.2f} M€")
    print(f"    As % of DHG investment: {pct*100:.1f}%")
    return pct


# ══════════════════════════════════════════════════════════════════════
# CASE A: LCOH_HP (25yr) + LCOH_dhg (50yr), OUC residual
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"CASE A: LCOH_HP (25yr) + LCOH_dhg (50yr) + OUC residual")
print(f"{'='*70}")

LCOH_HP_25 = calculate_lcoh(
    total_installation_costs * 1e6,
    fixed_oem_hp_df, var_oem_hp_df,
    total_electricity_cost_df * 1e6,
    heat_supplied_hp_df * 1000,
    r,
)
LCOH_dhg_50 = calculate_lcoh(
    investment_costs_dhg * 1e6,
    dhg_zeros, dhg_zeros, dhg_zeros,
    heat_supplied_dhg_50 * 1000,
    r,
)
LCOH_A = LCOH_HP_25 + LCOH_dhg_50
print(f"  LCOH_HP (25yr):  {LCOH_HP_25*1000:.2f} €/MWh")
print(f"  LCOH_dhg (50yr): {LCOH_dhg_50*1000:.2f} €/MWh")
print(f"  LCOH total:      {LCOH_A*1000:.2f} €/MWh")

ouc_pct = compute_ouc(
    investment_costs_dhg * 1e6,
    LCOH_dhg_50,
    yearly_heat_supplied * 1000,  # kWh
    r,
    npv_years=25,
    lcoh_years=50,
)
npv_A = run_npv(LCOH_A, ouc_pct, "Case A: 50yr DHG LCOH + OUC residual")

# Also show what happens with 40% and 50% residual for reference
run_npv(LCOH_A, 0.40, "Case A reference: 50yr DHG LCOH + 40% residual")
run_npv(LCOH_A, 0.50, "Case A reference: 50yr DHG LCOH + 50% residual")


# ══════════════════════════════════════════════════════════════════════
# CASE B: LCOH_HP (25yr) + LCOH_dhg (25yr), no residual needed
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"CASE B: LCOH_HP (25yr) + LCOH_dhg (25yr), no residual")
print(f"{'='*70}")

LCOH_dhg_25 = calculate_lcoh(
    investment_costs_dhg * 1e6,
    dhg_zeros_25, dhg_zeros_25, dhg_zeros_25,
    heat_supplied_dhg_25 * 1000,
    r,
)
LCOH_B = LCOH_HP_25 + LCOH_dhg_25
print(f"  LCOH_HP (25yr):  {LCOH_HP_25*1000:.2f} €/MWh")
print(f"  LCOH_dhg (25yr): {LCOH_dhg_25*1000:.2f} €/MWh")
print(f"  LCOH total:      {LCOH_B*1000:.2f} €/MWh")

npv_B = run_npv(LCOH_B, 0.0, "Case B: 25yr DHG LCOH + 0% residual")

# With 50% residual as a bonus (grid still has value after 25yr)
run_npv(LCOH_B, 0.50, "Case B bonus: 25yr DHG LCOH + 50% residual (extra value)")


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Case A (50yr DHG + OUC):     NPV = {npv_A/1e6:+.4f} M€  (expect ≈ 0)")
print(f"  Case B (25yr DHG + no res):  NPV = {npv_B/1e6:+.4f} M€  (expect ≈ 0)")
print()
print(f"  LCOH_A = {LCOH_A*1000:.2f} €/MWh  (50yr DHG amortisation, lower price)")
print(f"  LCOH_B = {LCOH_B*1000:.2f} €/MWh  (25yr DHG amortisation, higher price)")
print(f"  Difference: {(LCOH_B - LCOH_A)*1000:.2f} €/MWh")
print()
if abs(npv_A) < 1e4 and abs(npv_B) < 1e4:
    print("  ✓ Both cases return NPV ≈ 0 — LCOH/NPV consistency verified.")
else:
    print("  ✗ Mismatch detected — investigate further.")
    if abs(npv_A) > 1e4:
        print(f"    Case A off by {npv_A/1e6:.4f} M€")
    if abs(npv_B) > 1e4:
        print(f"    Case B off by {npv_B/1e6:.4f} M€")
