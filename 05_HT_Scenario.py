import pandas as pd
import numpy as np
from pathlib import Path
from costs.heat_supply import capital_costs_hp, var_oem_hp, fixed_oem_hp, calculate_lcoh
from heat_supply.carnot_efficiency import carnot_cop
from costs.renovation_costs import apply_inflation

###################################################################################
###################################################################################
######################## Import data about area's demand  #########################
###################################################################################
###################################################################################

path_unrenovated_area = Path(
    "building_analysis/results/unrenovated_whole_buildingstock/area_results.csv"
)
areas_demand = pd.read_csv(path_unrenovated_area, index_col=0)
areas_demand["total_useful_demand"] = (
    areas_demand["dhw_energy"] + areas_demand["space_heating"]
)
efficiency_he = (
    0.8  # efficiency of the heat exchanger to be used to calculate the delivered energy
)
areas_demand["delivered_energy"] = areas_demand["total_useful_demand"] / efficiency_he

estimated_grid_losses = 0.1  # estimated grid losses are 10% of the delivered energy

areas_demand["final_energy"] = areas_demand["delivered_energy"] * (
    1 + estimated_grid_losses
)

estimated_capacity = areas_demand["final_energy"].max()


# the estimated capacity is around 60 MW. For this reason we decide to use 3 heat pumps of 20 MW each.
# we assume that the load is equally distributed among the 3 heat pumps.
n_heat_pumps = 3
heat_pump_load = areas_demand["final_energy"] / n_heat_pumps / 1000
capacity_single_hp = estimated_capacity / n_heat_pumps * 1.2 / 1000  # MW

# let's calculate the efficiency of the heat pumps at a hourly level.
# we assume that the heat pumps are air source heat pumps.
# the COP of the heat pump is calculated as a function of the outside temperature using the Carnot formula
# the source will be the outside air. Let's import the outside air temperature data

area_name = "Frankfurt_Griesheim_Mitte"
year_start = 2019
year_end = 2019

path_outside_air = Path(
    f"irradiation_data/{area_name}_{year_start}_{year_end}/{area_name}_irradiation_data_{year_start}_{year_end}.csv"
)
outside_temp = pd.read_csv(path_outside_air, usecols=["T2m"])
outside_temp.index = areas_demand.index

# set up the outlet and inlet of the heat pump to calculate the COP
supply_temp = pd.DataFrame(100, index=outside_temp.index, columns=["supply_temp"])

cop_hourly = carnot_cop(supply_temp, outside_temp, 5)

P_el = areas_demand["final_energy"] / cop_hourly

installation_cost_HP = capital_costs_hp(capacity_single_hp, "air")  # million euros
total_installation_costs = installation_cost_HP * n_heat_pumps  # Million euros

single_var_oem_hp = var_oem_hp(
    capacity_single_hp, "air", heat_pump_load
)  # Million Euros

single_fix_oem = fixed_oem_hp(capacity_single_hp, "air")  # Million Euros

print(
    f"total costs installation: {total_installation_costs}, single HP Variable OEM: {single_var_oem_hp}, single HP Fixed OEM: {single_fix_oem}"
)


## we need to calculate also the electricity cost of running the heat pump:
initial_electricity_cost = 0.1776  # EUROSTAT 2023- semester 2
inflation_rate = 0.02
n_years = 25
future_electricity_prices = apply_inflation(
    initial_electricity_cost, n_years, inflation_rate
)

# this is the electricity cost for ALL heat pumps in the area for all n_years years
total_electricity_cost = (
    P_el.sum() * future_electricity_prices / 1000000
)  # Million euros
total_var_oem_hp = single_var_oem_hp * n_heat_pumps
total_fixed_oem_hp = single_fix_oem * n_heat_pumps
yearly_heat_supplied = areas_demand["final_energy"].sum() / 1000  # MW
heat_supplied_df = pd.DataFrame(  ## In this case we are using the heat supplied in the Grid, not the delivered heat
    {"Heat Supplied (MW)": [yearly_heat_supplied] * n_years}
)

inflated_fixed_oem_hp = apply_inflation(total_fixed_oem_hp, n_years, inflation_rate)
inflated_fixed_oem_hp_df = pd.DataFrame(inflated_fixed_oem_hp)

inflated_var_oem_hp = apply_inflation(total_var_oem_hp, n_years, inflation_rate)
inflated_var_oem_hp_df = pd.DataFrame(inflated_var_oem_hp)
total_electricity_cost_df = pd.DataFrame(total_electricity_cost)

LCOH = calculate_lcoh(
    total_installation_costs,
    inflated_fixed_oem_hp_df,
    inflated_var_oem_hp_df,
    total_electricity_cost_df,
    heat_supplied_df,
    0.05,
)  # in this case we are getting Million Euros per MWh produced. We need  to convert to Euros per kWh produced

LCOH_eurokwh = LCOH * 1000000 / 1000  # 1000000 million / 1000 kWh
print(f"LCOH: {LCOH_eurokwh}")


dhg_lifetime = 60  # years
dhg_other_costs = np.zeros(dhg_lifetime)
dhg_other_costs_df = pd.DataFrame(dhg_other_costs)
heat_supplied_dhg = pd.DataFrame(  ## In this case we are using the heat supplied in the Grid, not the delivered heat
    {"Heat Supplied (MW)": [yearly_heat_supplied] * dhg_lifetime}
)

LCOD_dhg = calculate_lcoh(
    16.28,  # comes from THERMOS
    dhg_other_costs_df,
    dhg_other_costs_df,
    dhg_other_costs_df,
    heat_supplied_dhg,
    0.05,
)

LCOH_dhg_eurokwh = LCOD_dhg * 1000000 / 1000  # 1000000 million / 1000 kWh
