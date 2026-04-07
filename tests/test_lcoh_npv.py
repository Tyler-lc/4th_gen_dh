"""LCOH/NPV consistency tests.

Validates that selling heat at exactly LCOH (margin=0, reduction_factor=1)
produces NPV ≈ 0 for the DH operator, using actual HT scenario data.

Two cases:
  Case A: LCOH_HP (25yr) + LCOH_dhg (50yr) + OUC residual → NPV ≈ 0
  Case B: LCOH_HP (25yr) + LCOH_dhg (25yr) + no residual  → NPV ≈ 0

These tests read input data (building stock, grid costs) but write nothing.
"""

import numpy as np
import pandas as pd
import pytest

from config import area_results_path, grid_results_parquet, weather_data_path
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


# ── HT scenario parameters (identical to 05b_HT_Scenario.py) ───────

SUPPLY_TEMPERATURE = 90
APPROACH_TEMPERATURE = 5
DISCOUNT_RATE = 0.05
N_YEARS_HP = 25
DHG_LIFETIME = 50
MARGIN = 0
SAFETY_FACTOR = 1.2
N_HEAT_PUMPS = 3
INITIAL_ELECTRICITY_COST = 0.1776

DK_TO_DE = 109.1 / 148.5
UPDATE_2022_2023 = 126.6 / 111.2


@pytest.fixture(scope="module")
def ht_scenario_data():
    """Load and compute all intermediate values needed for LCOH/NPV tests."""
    # Grid costs
    ember_results = pd.read_parquet(grid_results_parquet("unrenovated"))
    investment_costs_dhg = ember_results["cost_total"].sum() / 1e6  # M€

    # Area demand
    areas_demand = pd.read_csv(area_results_path("unrenovated"), index_col=0)
    areas_demand.index = pd.to_datetime(areas_demand.index)
    areas_demand["total_useful_demand"] = (
        areas_demand["dhw_energy"] + areas_demand["space_heating"]
    )
    areas_demand["delivered_energy"] = areas_demand["total_useful_demand"] / 0.8

    total_power_losses = ember_results["Losses [W]"].sum()
    areas_demand["hourly grid losses [kWh]"] = total_power_losses / 1000
    areas_demand["hourly heat generated in Large HP [kWh]"] = (
        areas_demand["hourly grid losses [kWh]"] + areas_demand["delivered_energy"]
    )

    estimated_capacity = areas_demand["hourly heat generated in Large HP [kWh]"].max()
    heat_pump_load = (
        areas_demand["hourly heat generated in Large HP [kWh]"] / N_HEAT_PUMPS / 1000
    )
    capacity_single_hp = estimated_capacity / N_HEAT_PUMPS * SAFETY_FACTOR / 1000

    # COP
    outside_temp = pd.read_csv(weather_data_path(), usecols=["T2m"])
    outside_temp.index = areas_demand.index
    supply_temp = pd.DataFrame(
        SUPPLY_TEMPERATURE, index=outside_temp.index, columns=["supply_temp"]
    )
    cop_hourly = carnot_cop(supply_temp, outside_temp, APPROACH_TEMPERATURE)
    P_el = areas_demand["hourly heat generated in Large HP [kWh]"] / cop_hourly

    # Costs
    installation_cost_HP = (
        capital_costs_hp(capacity_single_hp, "air") * DK_TO_DE * UPDATE_2022_2023
    )
    total_installation_costs = installation_cost_HP * N_HEAT_PUMPS * capacity_single_hp
    single_var_oem = (
        var_oem_hp(capacity_single_hp, "air", heat_pump_load.sum())
        * DK_TO_DE * UPDATE_2022_2023
    )
    single_fix_oem = (
        fixed_oem_hp(capacity_single_hp, "air") * DK_TO_DE * UPDATE_2022_2023
    )
    total_var_oem = single_var_oem * N_HEAT_PUMPS
    total_fixed_oem = single_fix_oem * N_HEAT_PUMPS * capacity_single_hp
    yearly_heat_supplied = areas_demand["delivered_energy"].sum() / 1000  # MWh

    future_el_prices = calculate_future_values(
        {"electricity": INITIAL_ELECTRICITY_COST}, N_YEARS_HP
    )
    total_electricity_cost = P_el.sum() * future_el_prices / 1e6

    # Precomputed DataFrames for LCOH
    fixed_oem_hp_df = calculate_future_values(
        {"Fixed O&M": total_fixed_oem}, N_YEARS_HP
    )
    var_oem_hp_df = calculate_future_values(
        {"Variable O&M": total_var_oem}, N_YEARS_HP
    )
    heat_supplied_hp_df = pd.DataFrame(
        {"Heat Supplied (MW)": [yearly_heat_supplied] * N_YEARS_HP}
    )
    total_electricity_cost_df = pd.DataFrame(total_electricity_cost)

    dhg_zeros = pd.DataFrame(np.zeros(DHG_LIFETIME))
    dhg_zeros_25 = pd.DataFrame(np.zeros(N_YEARS_HP))
    heat_supplied_dhg_50 = pd.DataFrame(
        {"Heat Supplied (MW)": [yearly_heat_supplied] * DHG_LIFETIME}
    )
    heat_supplied_dhg_25 = pd.DataFrame(
        {"Heat Supplied (MW)": [yearly_heat_supplied] * N_YEARS_HP}
    )

    total_yearly_costs_hps = (
        total_var_oem + total_fixed_oem + total_electricity_cost.iloc[0, 0] * 1e6
    )
    overnight_costs = (total_installation_costs + investment_costs_dhg) * 1e6

    return {
        "investment_costs_dhg": investment_costs_dhg,
        "total_installation_costs": total_installation_costs,
        "yearly_heat_supplied": yearly_heat_supplied,
        "total_yearly_costs_hps": total_yearly_costs_hps,
        "overnight_costs": overnight_costs,
        "fixed_oem_hp_df": fixed_oem_hp_df,
        "var_oem_hp_df": var_oem_hp_df,
        "heat_supplied_hp_df": heat_supplied_hp_df,
        "total_electricity_cost_df": total_electricity_cost_df,
        "dhg_zeros": dhg_zeros,
        "dhg_zeros_25": dhg_zeros_25,
        "heat_supplied_dhg_50": heat_supplied_dhg_50,
        "heat_supplied_dhg_25": heat_supplied_dhg_25,
    }


def _compute_ouc_pct(dhg_inv_eur, annual_heat_kwh, r, npv_years, lcoh_years):
    """Compute Outstanding Unrecovered Capital as fraction of DHG investment."""
    pv_heat_lcoh = sum(
        annual_heat_kwh / (1 + r) ** (t + 1) for t in range(lcoh_years)
    )
    pv_heat_npv = sum(
        annual_heat_kwh / (1 + r) ** (t + 1) for t in range(npv_years)
    )
    recovery_fraction = pv_heat_npv / pv_heat_lcoh
    ouc_pv = dhg_inv_eur * (1 - recovery_fraction)
    ouc_at_year_n = ouc_pv * (1 + r) ** npv_years
    return ouc_at_year_n / dhg_inv_eur


def _run_npv(data, lcoh_total, residual_pct):
    """Compute operator NPV at given LCOH and residual value."""
    annual_revenue = data["yearly_heat_supplied"] * 1000 * lcoh_total
    future_revenues = calculate_future_values(
        {"revenues": annual_revenue}, N_YEARS_HP
    )
    future_revenues.iloc[-1] += data["investment_costs_dhg"] * 1e6 * residual_pct

    future_expenses = calculate_future_values(
        {"costs": data["total_yearly_costs_hps"]}, N_YEARS_HP
    )
    npv_val, _ = npv_2(
        -data["overnight_costs"], future_expenses, future_revenues, DISCOUNT_RATE
    )
    return npv_val


@pytest.mark.regression
@pytest.mark.slow
def test_case_a_lcoh_50yr_dhg_ouc_residual_npv_zero(ht_scenario_data):
    """LCOH_HP(25yr) + LCOH_dhg(50yr) with OUC residual → NPV ≈ 0."""
    d = ht_scenario_data

    LCOH_HP_25 = calculate_lcoh(
        d["total_installation_costs"] * 1e6,
        d["fixed_oem_hp_df"], d["var_oem_hp_df"],
        d["total_electricity_cost_df"] * 1e6,
        d["heat_supplied_hp_df"] * 1000,
        DISCOUNT_RATE,
    )
    LCOH_dhg_50 = calculate_lcoh(
        d["investment_costs_dhg"] * 1e6,
        d["dhg_zeros"], d["dhg_zeros"], d["dhg_zeros"],
        d["heat_supplied_dhg_50"] * 1000,
        DISCOUNT_RATE,
    )
    lcoh_total = LCOH_HP_25 + LCOH_dhg_50

    ouc_pct = _compute_ouc_pct(
        d["investment_costs_dhg"] * 1e6,
        d["yearly_heat_supplied"] * 1000,
        DISCOUNT_RATE,
        npv_years=25,
        lcoh_years=50,
    )

    npv_val = _run_npv(d, lcoh_total, ouc_pct)
    assert abs(npv_val) < 1e4, (
        f"Case A NPV should be ≈ 0, got {npv_val/1e6:+.4f} M€"
    )


@pytest.mark.regression
@pytest.mark.slow
def test_case_b_lcoh_25yr_dhg_no_residual_npv_zero(ht_scenario_data):
    """LCOH_HP(25yr) + LCOH_dhg(25yr) with no residual → NPV ≈ 0."""
    d = ht_scenario_data

    LCOH_HP_25 = calculate_lcoh(
        d["total_installation_costs"] * 1e6,
        d["fixed_oem_hp_df"], d["var_oem_hp_df"],
        d["total_electricity_cost_df"] * 1e6,
        d["heat_supplied_hp_df"] * 1000,
        DISCOUNT_RATE,
    )
    LCOH_dhg_25 = calculate_lcoh(
        d["investment_costs_dhg"] * 1e6,
        d["dhg_zeros_25"], d["dhg_zeros_25"], d["dhg_zeros_25"],
        d["heat_supplied_dhg_25"] * 1000,
        DISCOUNT_RATE,
    )
    lcoh_total = LCOH_HP_25 + LCOH_dhg_25

    npv_val = _run_npv(d, lcoh_total, 0.0)
    assert abs(npv_val) < 1e4, (
        f"Case B NPV should be ≈ 0, got {npv_val/1e6:+.4f} M€"
    )
