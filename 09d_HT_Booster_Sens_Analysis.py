import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm
import os
from pathlib import Path
import sys
from typing import Union

from costs.heat_supply import capital_costs_hp, var_oem_hp, fixed_oem_hp, calculate_lcoh
from heat_supply.carnot_efficiency import carnot_cop
from costs.heat_supply import calculate_revenues, calculate_future_values
from costs.renovation_costs import (
    apply_inflation,
    calculate_energy_prices_future,
    consumer_size,
    calculate_expenses,
    calculate_npv_savings,
    npv,
    renovation_costs_iwu,
)
from utils.misc import get_electricity_cost

grid_temperatures = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]


# TODO: the booster's COP is in the buildingstock calculation. We need to find a better way to
# do this.
def sensitivity_analysis_booster(
    simulation_type: str,
    supply_temperature: Union[float, int] = 90,
    approach_temperature: Union[float, int] = 5,
    margin: float = 0,
    taxation: float = 0.07,
    reduction_factor: float = 1,
    oversizing_factor: float = 1.2,
    n_heat_pumps: int = 3,
    inv_cost_multiplier: Union[float, int] = 1,
    electricity_cost_multiplier: float = 1,
    gas_cost_multiplier: Union[float, int] = 1,
    max_COP: Union[float, int] = 4,
    carnot_efficiency: Union[float, int] = 0.524,
    ir: float = 0.05,
):
    """
    This is a helper function to run the sensitivity analysis for the HT and LT scenario
    Parameters:
    - simulation_type: can be "renovated" or "unrenovated"
    - supply_temperature: the supply temperature to the buildings
    - approach_temperature: the approach temperature of the heat exchanger (grid to heat pump)
    - margin: the margin to be applied to the price of the heat supplied to the customers
    - taxation: taxation rate applied to the heat supplied to the residential customers
    - reduction_factor: the reduction factor to be applied to the price of the heat supplied to the customers
    - safety_factor: the oversizing factor for the large scale heat pumps
    - n_heat_pumps: the number of large scale heat pumps installed
    - electricity_cost_multiplier: the multiplier for the electricity cost
    - max_COP: the maximum COP for the heat pumps
    - ir: the interest rate for the whole simulation

    """
    if simulation_type not in ["renovated", "unrenovated", "booster"]:
        raise ValueError(
            "simulation_type must be 'renovated', 'unrenovated' or 'booster'"
        )
    #############################################################################################
    # In this scenario we compare the NPV of the customer when they do not renovate and use gas
    # against the case when they renovate and use DH which uses a air Heat Pump.
    #############################################################################################

    #############################################################################################

    ### import EMBERS data to assess grid losses and total investment costs
    path_embers = f"grid_calculation/sensitivity_analysis/{simulation_type}/{supply_temperature}/booster_result_df_{supply_temperature}.parquet"
    embers_data = pd.read_parquet(path_embers)

    # margin, taxation and reduction_factor are now arguments of the function
    # margin = 0
    # taxation = 0.07
    # reduction_factor = 1

    n_heat_pumps = 2  # might need changing!!!!

    initial_electricity_cost = {
        "IA": 0.3279,  # these are the electricity costs based on consumption
        "IB": 0.2480,
        "IC": 0.2175,
        "ID": 0.2017,
        "IE": 0.1776,
        "IF": 0.172,
        "IG": 0.1527,
    }

    n_years_hp = 25  # for LCOH calculation
    heat_pump_lifetime = 25  # setting years until replacement

    dhg_lifetime = 25  # years
    investment_costs_dhg = (
        embers_data["cost_total"].sum() / 1000000
    )  # from EMBERS [Mil €]

    years_buildingstock = 25
    building_interest_rate = 0.05

    ### Let's find out what is the size of the customer now for GAS
    small_consumer_threshold = 20  # GJ per year
    gj_to_kwh = 1 / 3600 * 1000000  # 1 GJ = 1/3600 * 1000000 kwh - conversion factor
    medium_consumer_threshold = 200  # GJ per year
    res_types = ["mfh", "sfh", "ab", "th"]

    fixed_costs_boosters = 250  # €/booster per year
    ###################################################################################
    ###################################################################################
    ######################## Import data about area's demand  #########################
    ###################################################################################
    ###################################################################################

    ## We need to import both the unrenovated and renovated buildingstock

    path_unrenovated_area = Path(
        f"building_analysis/results//sensitivity_analysis/{simulation_type}/{simulation_type}_whole_buildingstock_{supply_temperature}/area_results_{supply_temperature}/area_results_{simulation_type}_whole_buildingstock_{supply_temperature}.csv"
    )
    areas_demand = pd.read_csv(path_unrenovated_area, index_col=0)
    areas_demand.index = pd.to_datetime(areas_demand.index)

    areas_demand["total useful demand thermal demand [kWh]"] = (
        areas_demand["area space heating demand [kWh]"]
        + areas_demand["area dhw energy demand [kWh]"]
    )

    # we need the buildingstock data to calculate the investment costs of the booster heat pumps
    # TODO: check whether the pathing is correct or not
    path_booster_buildingstock = f"building_analysis/results/sensitivity_analysis/{simulation_type}/{simulation_type}_whole_buildingstock_{supply_temperature}/buildingstock_{simulation_type}_whole_buildingstock_{supply_temperature}_results.parquet"
    booster_buildingstock = gpd.read_parquet(path_booster_buildingstock)

    efficiency_he = 0.8  # efficiency of the heat exchanger to be used to calculate the delivered energy

    areas_demand["delivered_energy_dh [kWh]"] = (
        areas_demand["area grid demand [kWh]"] / efficiency_he
    )  # in our case we are delivering now the thermal demand from the grid.

    # now we can import the data from EMBERS to assess the grid losses:

    total_power_losses = embers_data["Losses [W]"].sum()

    ### this whole part is mostly to calculate the percentage of the energy losses on the grid
    # and check whether they are in line with what we expect ~10 to 20%
    total_energy_losses = total_power_losses * 8760 / 1000  # kWh/year
    total_boosters_demand = areas_demand["area grid demand [kWh]"].sum()
    total_delivered_energy = total_boosters_demand / efficiency_he

    total_hp_heat_generated = total_delivered_energy + total_energy_losses
    percent_losses = total_energy_losses / total_hp_heat_generated

    ### now we need to calculate the capacity of the heat pumps. First of all we take the
    # power losses from embers.
    areas_demand["hourly grid losses [kWh]"] = embers_data["Losses [W]"].sum() / 1000
    areas_demand["hourly heat generated in Large HP [kWh]"] = (
        areas_demand["hourly grid losses [kWh]"]
        + areas_demand["delivered_energy_dh [kWh]"]
    )

    total_heat_pump_max_load = areas_demand[
        "hourly heat generated in Large HP [kWh]"
    ].max()

    # the estimated capacity is around 44 MW. For this reason we decide to use 3 heat pumps of 20 MW each.
    # we assume that the load is equally distributed among the 3 heat pumps.

    heat_pump_load = (  # this is only used to calculate the O&M Var for the large scale heat pump
        areas_demand["hourly heat generated in Large HP [kWh]"] / n_heat_pumps / 1000
    )  # MWh this is the load for each heat pump
    capacity_single_hp = (
        total_heat_pump_max_load / n_heat_pumps * oversizing_factor / 1000
    )  # MW

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
    supply_temp = pd.DataFrame(
        supply_temperature, index=outside_temp.index, columns=["supply_temp"]
    )

    cop_hourly = carnot_cop(
        supply_temp,
        outside_temp,
        approach_temperature,
        carnot_efficiency,
        max_COP,
    )

    P_el = (
        areas_demand["hourly heat generated in Large HP [kWh]"] / cop_hourly
    )  # this is the Electric power input for the large scale heat pump
    # the electricity demand for the booster is areas_demand["area total boosters demand [kWh]"]

    DK_to_DE = (
        109.1 / 148.5
    )  # this is the ratio of the installation costs of a heat pump in Denmark to Germany
    update2022_2023 = (
        126.6 / 111.2
    )  # this is the ratio of the installation costs of a heat pump in 2022 to 2023
    installation_cost_HP = (
        capital_costs_hp(capacity_single_hp, "air") * DK_to_DE * update2022_2023
    )  # million euros/MW_thermal installed per heat pump
    total_installation_costs = (
        installation_cost_HP * n_heat_pumps * capacity_single_hp
    ) * inv_cost_multiplier  # Million euros

    single_var_oem_hp = (
        var_oem_hp(capacity_single_hp, "air", heat_pump_load.sum())
        * DK_to_DE
        * update2022_2023
    )  # Euros/MWh_thermal produced

    single_fix_oem = (
        fixed_oem_hp(capacity_single_hp, "air") * DK_to_DE * update2022_2023
    )  # Million Euros/MW_thermal installed

    print(
        f"Data in Million Euros. Total costs installation: {total_installation_costs}\nSingle HP Variable OEM: {single_var_oem_hp}\nSingle HP Fixed OEM: {single_fix_oem}"
    )

    ###################################################################################
    ###################################################################################
    ############################ Heat Pump LCOE Data  #################################
    ###################################################################################
    ###################################################################################

    ## we need to calculate also the electricity cost of running the heat pump:
    ## we need to calculate first the electricity cost for the system in €/kWh

    # TODO: This doesn't change the result by change. But I should still split the boosters and the
    # large scale heat pumps elecricity prices. (It doesn't change because whether their,
    # electricity demand is pooled or separated, they end up both in the same category as when they
    # are indeed pooled)

    initial_electricity_cost_system = (
        get_electricity_cost(
            P_el.sum() / 1000
            + areas_demand["area total boosters demand [kWh]"].sum() / 1000,
            "non_residential",  # total consumption (Booster + central HP in MWh)
        )
        * electricity_cost_multiplier
    )  # €/kWh

    future_electricity_prices = calculate_future_values(
        {"electricity": initial_electricity_cost_system}, n_years_hp
    )

    # We need to split the large scale heat pumps and the boosters
    large_hp_total_electricity_cost = (
        P_el.sum()
        * future_electricity_prices
        / 1000000  # kWh electricity * price in €/kWh / 1000000 to convert to Million euros
    )  # Million euros
    total_var_oem_large_hp = single_var_oem_hp * n_heat_pumps
    total_fixed_oem_large_hp = single_fix_oem * n_heat_pumps * capacity_single_hp
    yearly_heat_supplied_large_hp = (
        areas_demand["hourly heat generated in Large HP [kWh]"].sum() / 1000
    )  # MW
    heat_supplied_df_large_hp = pd.DataFrame(  ## In this case we are using the heat supplied in the Grid, not the delivered heat
        {"Heat Supplied (MWh)": [yearly_heat_supplied_large_hp] * n_years_hp}
    )

    fixed_oem_hp_df = calculate_future_values(
        {"Fixed O&M": total_fixed_oem_large_hp}, n_years_hp
    )

    var_oem_hp_df = calculate_future_values(
        {"Variable O&M": total_var_oem_large_hp}, n_years_hp
    )

    total_electricity_cost_df = pd.DataFrame(large_hp_total_electricity_cost)

    LCOH_HP = calculate_lcoh(
        total_installation_costs * 1000000,  # convert to euros
        fixed_oem_hp_df,
        var_oem_hp_df,
        total_electricity_cost_df * 1000000,  # convert to euros
        heat_supplied_df_large_hp * 1000,  # # convert to kWh
        ir,
    )  # in this case we are getting Euros per kWh produced.
    print(f"LCOH of the Heat Pumps: {LCOH_HP}")

    dhg_other_costs = np.zeros(dhg_lifetime)
    dhg_other_costs_df = pd.DataFrame(dhg_other_costs)
    heat_supplied_dhg = pd.DataFrame(  ## In this case we are using the heat supplied in the Grid, not the delivered heat
        {"Heat Supplied (MW)": [yearly_heat_supplied_large_hp] * dhg_lifetime}
    )

    LCOH_dhg = calculate_lcoh(
        investment_costs_dhg * 1000000,  # comes from EMBERS
        dhg_other_costs_df,
        dhg_other_costs_df,
        dhg_other_costs_df,
        heat_supplied_dhg * 1000,
        ir,
    )

    # LCOH_dhg_eurokwh = LCOD_dhg * 1000000 / 1000  # 1000000 million / 1000 kWh
    print(f"LCOH_dhg_eurokwh: {LCOH_dhg}")

    #### Now we need to calculate the LCOH for the booster heat pumps as well.
    # We have the list of all buildings with the "installed" capacity of each booster.
    # Now the issue I see here, is that I do not know how to allocate the costs of all boosters.
    # we need to crete one pricing for all buildings (according to types and consumer size of course)
    # but each building has a differently sized booster. That also will use a different amount of energy.
    # It should not be a problem because they will all share the same "coefficient" for investment costs.

    # price_heat_eurokwh_residential = (
    #     (LCOH_HP + LCOH_dhg) * (1 + margin) * (1 + taxation) * reduction_factor
    # )
    # print(
    #     f"Lowest Price of the residential heat supplied: {price_heat_eurokwh_residential}"
    # )
    # price_heat_eurokwh_non_residential = (
    #     (LCOH_HP + LCOH_dhg) * (1 + margin) * reduction_factor
    # )
    # print(
    #     f"Lowest Price of the non-residential heat supplied: {price_heat_eurokwh_non_residential}"
    # )
    # price_heat_eurokwh_non_residential_VAT = (
    #     (LCOH_HP + LCOH_dhg) * (1 + margin) * (1 + taxation) * reduction_factor
    # )

    price_heat_eurokwh_residential = LCOH_HP + LCOH_dhg
    print(
        f"Lowest Price of the residential heat supplied: {price_heat_eurokwh_residential}"
    )
    price_heat_eurokwh_non_residential = LCOH_HP + LCOH_dhg
    print(
        f"Lowest Price of the non-residential heat supplied: {price_heat_eurokwh_non_residential}"
    )
    price_heat_eurokwh_non_residential_VAT = LCOH_HP + LCOH_dhg

    #### So far the LCOH only included the large scale heat pumps. Now we need to add also the boosters.
    # for this I need to find the O&M for the boosters as well as the investment costs. We will
    # apply the same electricity price for the boosters as for the large scale heat pumps.

    ### insert code here ###

    ### We decide to set the price of the heat supplied to the customer depending on the energy demand of the customer.
    ### We will use the R2 and NR2 as the lowest (which should be price_heat_eurokwh_residential and price_heat_eurokwh_non_residential)ù
    ### All the other demand categories will have a price higher than those. The difference in prices will be based on the same ratio already
    ### applied to the current energy prices for gas.
    gas_energy_prices = {  # eurostat data
        "r0": 0.1405,  # residential small
        "r1": 0.1145,  # residential medium
        "r2": 0.1054,  # residential large
        "nr0": 0.1312,  # non-residential small
        "nr1": 0.1070,  # non-residential medium
        "nr2": 0.0985,  # non-residential large
    }
    for keys in gas_energy_prices.keys():
        gas_energy_prices[keys] = gas_energy_prices[keys] * gas_cost_multiplier

    ratios = {
        "r0": gas_energy_prices["r2"] / gas_energy_prices["r0"],
        "r1": gas_energy_prices["r2"] / gas_energy_prices["r1"],
        "r2": gas_energy_prices["r2"] / gas_energy_prices["r2"],
        "nr0": gas_energy_prices["nr2"] / gas_energy_prices["nr0"],
        "nr1": gas_energy_prices["nr2"] / gas_energy_prices["nr1"],
        "nr2": gas_energy_prices["nr2"] / gas_energy_prices["nr2"],
    }

    hp_energy_prices = {
        "r0": price_heat_eurokwh_residential / ratios["r0"],
        "r1": price_heat_eurokwh_residential / ratios["r1"],
        "r2": price_heat_eurokwh_residential / ratios["r2"],
        "nr0": price_heat_eurokwh_non_residential / ratios["nr0"],
        "nr1": price_heat_eurokwh_non_residential / ratios["nr1"],
        "nr2": price_heat_eurokwh_non_residential / ratios["nr2"],
    }

    # operator_selling_price = {
    #     "r0": price_heat_eurokwh_residential / ratios["r0"],
    #     "r1": price_heat_eurokwh_residential / ratios["r1"],
    #     "r2": price_heat_eurokwh_residential / ratios["r2"],
    #     "nr0": price_heat_eurokwh_non_residential_VAT / ratios["nr0"],
    #     "nr1": price_heat_eurokwh_non_residential_VAT / ratios["nr1"],
    #     "nr2": price_heat_eurokwh_non_residential_VAT / ratios["nr2"],
    # }

    booster_buildingstock["total_useful_demand [kWh]"] = (
        booster_buildingstock["yearly_dhw_energy"]
        + booster_buildingstock["yearly_space_heating"]
    )
    booster_buildingstock["consumer_size"] = consumer_size(
        booster_buildingstock,
        small_consumer_threshold,
        medium_consumer_threshold,
        res_types,
        "total_useful_demand [kWh]",
    )
    ### the various data for booster heat pumps can be found here:
    # https://doi.org/10.1016/j.energy.2018.04.081

    for idx, row in tqdm(
        booster_buildingstock.iterrows(), total=len(booster_buildingstock)
    ):
        size_hp_booster = row["heat_pump_size [kW]"]
        if size_hp_booster > 500:
            specific_cost_booster = 100
        else:
            specific_cost_booster = 190
        cost_heat_boosters = specific_cost_booster * size_hp_booster
        booster_buildingstock.loc[idx, "cost_hp_booster [€]"] = cost_heat_boosters

        # i think i actually need to calculate the LCOH for each booster now. poo.

        booster_buildingstock.loc[idx, "electricity_cost_booster [€]"] = (
            booster_buildingstock.loc[idx, "total_demand_electricity [kWh]"]
            * initial_electricity_cost_system
        )  # we are using the DH operator electricity price
        # booster_buildingstock.loc[idx, "dhg_cost_booster [€]"] = booster_buildingstock.loc[
        #     idx, "total_demand_on_grid [kWh]"
        # ]
    # lcoh = (Inv + (sum( p_el*W_booster +p_lth*Q_dhg - Cm))/(i+1)^t)/(sum(Q_booster/(i+1)^t))

    fixed_oem_boosters = calculate_future_values(
        {"Fixed O&M": fixed_costs_boosters}, n_years_hp
    )  # this is the Cm in the eqution
    annual_electricity_cost_boosters = booster_buildingstock[
        "electricity_cost_booster [€]"
    ].sum()
    electricity_cost_boosters = calculate_future_values(
        {"Variable O&M": annual_electricity_cost_boosters}, n_years_hp
    )  # this is the p_el*W_booster in the eqution

    electricity_demand_boosters = booster_buildingstock[
        "total_demand_electricity [kWh]"
    ].sum()
    total_heat_supplied_by_boosters = calculate_future_values(
        {"Heat Supplied (MWh)": electricity_demand_boosters}, n_years_hp
    )  # this is Q_booster in the eqution

    total_heat_supplied_by_dhg = booster_buildingstock[
        "total_demand_on_grid [kWh]"
    ].sum()  # this is Q_dhg in the eqution
    variable_oem_boosters = calculate_future_values({"Variable O&M": 0}, n_years_hp)

    total_investment_costs_boosters = booster_buildingstock["cost_hp_booster [€]"].sum()

    numerator = (
        total_investment_costs_boosters
        + (
            electricity_cost_boosters
            + total_heat_supplied_by_dhg * (LCOH_HP + LCOH_dhg)
            - fixed_costs_boosters
        )
    ) / (1 + ir) ** 25

    # lcoh_boosters_num = total_investment_costs_boosters +
    # lcoh_booster_den = (
    #     booster_buildingstock["yearly_space_heating"].sum()
    #     + booster_buildingstock["yearly_dhw_energy"].sum()
    # ) / (1 + ir) ** 25
    # lcoh_booster = lcoh_boosters_num / lcoh_booster_den
    # print(f"simple formula LCOH booster: {lcoh_booster}")

    # the original equation from "evaluation the cost of heat for end users":
    # lcoh = (Inv + (sum( p_el*W_booster +p_lth*Q_dhg - Cm))/(i+1)^t)/(sum(Q_booster/(i+1)^t))
    # where:
    # Inv = total_investment_costs_boosters
    # p_el = initial_electricity_cost_system
    # W_booster = electricity_demand_boosters
    # p_lth = LCOH_HP + LCOH_dhg
    # Q_dhg = total_heat_supplied_by_dhg
    # Cm = fixed_costs_boosters

    lcoh_electricity_boosters = calculate_future_values(
        {"electricity": initial_electricity_cost_system * electricity_demand_boosters},
        n_years_hp,
    )
    lcoh_heat_grid_boosters = calculate_future_values(
        {"heat from grid": (LCOH_HP + LCOH_dhg) * total_heat_supplied_by_dhg},
        n_years_hp,
    )
    lcoh_total_heat_generated_boosters = calculate_future_values(
        {
            "heat_generated": booster_buildingstock["yearly_space_heating"].sum()
            + booster_buildingstock["yearly_dhw_energy"].sum()
        },
        n_years_hp,
    )
    lcoh_fixed_oem_boosters = calculate_future_values(
        {"fixed o&m": fixed_costs_boosters * len(booster_buildingstock)}, n_years_hp
    )

    lcoh_booster = calculate_lcoh(
        total_investment_costs_boosters,
        lcoh_fixed_oem_boosters,
        lcoh_heat_grid_boosters,
        lcoh_electricity_boosters,
        lcoh_total_heat_generated_boosters,
        ir,
    )
    print(f"LCOH booster: {lcoh_booster}")

    # now i want to manually calculate the price of heat supplied to the customers.
    # first we calculate the amount spent on electricity for the heat pumps:
    el_boosters = booster_buildingstock["total_demand_electricity [kWh]"].sum()

    price_heat_eurokwh_residential = (
        (lcoh_booster) * (1 + margin) * (1 + taxation) * reduction_factor
    )
    print(
        f"Lowest Price of the residential heat supplied: {price_heat_eurokwh_residential}"
    )
    price_heat_eurokwh_non_residential = (
        (lcoh_booster) * (1 + margin) * reduction_factor
    )
    print(
        f"Lowest Price of the non-residential heat supplied: {price_heat_eurokwh_non_residential}"
    )
    price_heat_eurokwh_non_residential_VAT = (
        (lcoh_booster) * (1 + margin) * (1 + taxation) * reduction_factor
    )

    operator_selling_price = {
        "r0": price_heat_eurokwh_residential / ratios["r0"],
        "r1": price_heat_eurokwh_residential / ratios["r1"],
        "r2": price_heat_eurokwh_residential / ratios["r2"],
        "nr0": price_heat_eurokwh_non_residential_VAT / ratios["nr0"],
        "nr1": price_heat_eurokwh_non_residential_VAT / ratios["nr1"],
        "nr2": price_heat_eurokwh_non_residential_VAT / ratios["nr2"],
    }

    ###################################################################################
    ###################################################################################
    ############################### NPV Customers GAS #################################
    ###################################################################################
    ###################################################################################

    # first we calculate the NPV for the customers in the case of gas heating.
    # let's import the relevant data first. In this scenario the Buildingstock is now using booster heat pumps

    ## Let's define a couple of parameteres first

    # import the data with the renovated buildingstock
    # and now let's import the unrenovated buildingstock

    year_consumption = pd.DataFrame(
        {
            "full_id": booster_buildingstock["full_id"],
            "yearly_DHW_energy_demand": booster_buildingstock["yearly_dhw_energy"],
            "yearly_space_heating": booster_buildingstock["yearly_space_heating"],
            "unrenovated_yearly_space_heating": booster_buildingstock[
                "yearly_space_heating"
            ],
        }
    )

    year_consumption["unrenovated_total_demand"] = (
        year_consumption["unrenovated_yearly_space_heating"]
        + year_consumption["yearly_DHW_energy_demand"]
    )

    #####GOT HERE
    # first we create the monetary savings for each building. We already have the energy savings.
    # Let's assess the energy prices for each building and then we can calculate the monetary savings.
    npv_data = pd.DataFrame()
    npv_data["full_id"] = booster_buildingstock["full_id"]
    npv_data["building_usage"] = booster_buildingstock["building_usage"]
    npv_data["yearly_demand_useful_unrenovated"] = year_consumption[
        "unrenovated_total_demand"
    ]  # this is for the unrenovated buildingstock. DHW+SH

    efficiency_boiler = 0.9
    npv_data["yearly_demand_delivered_unrenovated"] = (
        year_consumption["unrenovated_total_demand"] / efficiency_boiler
    )

    npv_data["yearly_demand_delivered_unrenovated_DH"] = (
        year_consumption["unrenovated_total_demand"] / efficiency_he
    )

    npv_data["consumer_size_unrenovated"] = consumer_size(
        npv_data,
        small_consumer_threshold,
        medium_consumer_threshold,
        res_types,
        "yearly_demand_delivered_unrenovated",
    )

    #### Let's fill a dataframe with the Gas prices for each consumer size
    gas_prices_future = calculate_future_values(gas_energy_prices, n_years_hp)
    ### how much would the customoers pay when using gas?
    energy_expenditure_gas = calculate_expenses(
        npv_data,
        gas_prices_future,
        "yearly_demand_delivered_unrenovated",
        years_buildingstock,
        system_efficiency=1,  # because we are calculating the gas prices on the delivered energy already
        building_state="unrenovated",
    )

    ### and how much would they pay when using heat pumps?
    dh_prices_future = calculate_future_values(hp_energy_prices, n_years_hp)
    energy_expenditure_dh = calculate_expenses(
        npv_data,
        dh_prices_future,
        "yearly_demand_delivered_unrenovated",
        years_buildingstock,
        system_efficiency=1,
        building_state="unrenovated",
    )

    # npv_data[f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}_gas"] = np.nan
    # npv_data[f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}_dh"] = np.nan
    income_buildings = pd.DataFrame(np.zeros(years_buildingstock))

    ### Calculate the costs of renovation for each building

    # TODO: Where there was no renovation now we have a NaN. We have to change these into the right data
    # because the price of the heating has changed. So they actually have a different NPV. We have to do that in the
    # renovation_costs["total_cost"]. There we get NaN data.

    for idx, row in tqdm(npv_data.iterrows(), total=len(npv_data)):
        building_id = row["full_id"]

        # Gas NPV for buildings
        energy_costs_original = energy_expenditure_gas[building_id]
        npv_gas = npv(
            0, energy_costs_original, income_buildings, building_interest_rate
        )
        npv_data.loc[
            idx,
            f"npv_Gas_{years_buildingstock}years_ir_{building_interest_rate}",
        ] = npv_gas

        # DH NPV for buildings
        energy_costs_new = energy_expenditure_dh[building_id]
        npv_dh = npv(0, energy_costs_new, income_buildings, building_interest_rate)
        npv_data.loc[
            idx,
            f"npv_DH_{years_buildingstock}years_ir_{building_interest_rate}",
        ] = npv_dh

        # NPV of savings
        npv_savings = npv_dh - npv_gas
        npv_data.loc[
            idx, f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}"
        ] = npv_savings

    overnight_costs = (
        total_installation_costs
        + investment_costs_dhg
        + total_investment_costs_boosters / 1000000
    ) * 1000000
    total_yearly_costs_large_hps = (
        total_var_oem_large_hp
        + total_fixed_oem_large_hp
        + large_hp_total_electricity_cost.iloc[0, 0] * 1000000
    )  # in Euros per year
    total_yearly_costs_boosters = (
        lcoh_fixed_oem_boosters.values[0] + lcoh_electricity_boosters.values[0]
    )
    total_yearly_costs = total_yearly_costs_large_hps + total_yearly_costs_boosters
    npv_data["operator_selling_price"] = npv_data[f"consumer_size_unrenovated"].map(
        operator_selling_price
    )
    revenues = calculate_revenues(
        npv_data[f"yearly_demand_delivered_unrenovated_DH"],
        npv_data["operator_selling_price"],
    )

    total_revenues = revenues.sum()
    future_revenues = calculate_future_values(
        {"revenues": total_revenues}, dhg_lifetime
    )
    future_expenses = calculate_future_values(
        {"costs": total_yearly_costs}, dhg_lifetime
    )
    future_expenses["costs"] = future_expenses["costs"]
    from costs.renovation_costs import npv_2

    npv_dh, df = npv_2(-overnight_costs, future_expenses, future_revenues, ir)
    print(f"NPV of the District Heating Operator: {npv_dh}")

    # We do not have energy savings for the buildingstock in this case. We do not need to import the
    # renovated buildingstock dataset.

    # TODO: we will use a 10% profit margin for the LCOH. The resulting number will be the price at which the heat will be sold to the customers.
    # We might use a slightly higher Interest Rate in the NPV to account a little bit for inflation though. We need more research on this.

    # TODO: no inflation applied for the LCOH. calculate in real terms €2023

    return npv_data, npv_dh, LCOH_dhg, LCOH_HP, max_cop


simulation = "booster"
analysis_types = [
    "supply_temperature",  # 0
    "approach_temperature",  # 1
    "electricity_price",  # 2
    "gas_price",  # 3
    "max_cop",  # 4
    "ir",  # 5
    "inv_cost_multiplier",  # 6
]
num_analysis = 0

os.makedirs(
    f"sensitivity_analysis/{simulation}/{analysis_types[num_analysis]}", exist_ok=True
)
os.makedirs(
    f"sensitivity_analysis/{simulation}/{analysis_types[num_analysis]}/plots",
    exist_ok=True,
)
os.makedirs(
    f"sensitivity_analysis/{simulation}/{analysis_types[num_analysis]}/data",
    exist_ok=True,
)
df_npv = pd.DataFrame()


###### we will create a loop for the analysis
# To set up the loop we want to create different values for the analysis. So we will first insert the number
# of steps we want to do for the analysis. Then we use these steps to create the different values for the analysis
# and then we will loop through these values.
n_steps = 10
max_value = 70
min_value = 25
step_size = (max_value - min_value) / n_steps
values = np.linspace(min_value, max_value, n_steps).astype(int)
lcoh_dhg = []
lcoh_hp = []
max_cop = []
npv_operator = []  # Add this list to collect operator NPV values

# the max_COP simulation will require also to change the carnot_efficiency.
# The max_COP we hit is anyway 3.6 with the standard carnot_efficiency value. So we do not see
# almost any diffeerence. To change the carnot_efficiency during the max_COP simulation use this
carnot_efficiency = 0.524  # 0.524 is the standard value.

for value in tqdm(values):
    print(
        f"\n Analysis type: {analysis_types[num_analysis]}, Processing value: {value} \n"
    )
    if num_analysis == 0:
        df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop = sensitivity_analysis_booster(
            simulation, supply_temperature=value
        )
    elif num_analysis == 1:
        df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop = sensitivity_analysis_booster(
            simulation, approach_temperature=value
        )
    elif num_analysis == 2:
        df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop = sensitivity_analysis_booster(
            simulation, electricity_cost_multiplier=value
        )
    elif num_analysis == 3:
        df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop = sensitivity_analysis_booster(
            simulation, gas_cost_multiplier=value
        )
    elif num_analysis == 4:
        df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop = sensitivity_analysis_booster(
            simulation, max_COP=value, carnot_efficiency=carnot_efficiency
        )
    elif num_analysis == 5:
        df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop = sensitivity_analysis_booster(
            simulation, ir=value
        )
    elif num_analysis == 6:
        df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop = sensitivity_analysis_booster(
            simulation, inv_cost_multiplier=value
        )

    df_npv.to_csv(
        f"sensitivity_analysis/{simulation}/{analysis_types[num_analysis]}/data/supply_temperature_{value}C.csv"
    )
    lcoh_dhg.append(LCOH_dhg)
    lcoh_hp.append(LCOH_HP)
    max_cop.append(cop)
    npv_operator.append(npv_dh)  # Add the operator NPV to our list

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

from operator import add

# First subplot for LCOH
ax1.plot(values, lcoh_dhg, label="LCOH DHG")
ax1.plot(values, lcoh_hp, label="LCOH HP")
ax1.plot(values, list(map(add, lcoh_dhg, lcoh_hp)), label="Total LCOH")
ax1.set_xlabel(f"{analysis_types[num_analysis]}")
ax1.set_ylabel("LCOH (€/kWh)")
ax1.set_title(f"Sensitivity Analysis - LCOH vs {analysis_types[num_analysis]}")
ax1.legend()

# Second subplot for Operator NPV
ax2.plot(values, npv_operator, label="DH Operator NPV", color="green")
ax2.set_xlabel(f"{analysis_types[num_analysis]}")
ax2.set_ylabel("NPV (€)")
ax2.set_title(
    f"Sensitivity Analysis - DH Operator NPV vs {analysis_types[num_analysis]}"
)
ax2.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig(
    f"sensitivity_analysis/{simulation}/{analysis_types[num_analysis]}/plots/sensitivity_analysis.png"
)
# plt.close()
plt.show()
