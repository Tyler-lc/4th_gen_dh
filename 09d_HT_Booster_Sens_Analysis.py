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
    supply_temperature: Union[float, int] = 50,
    approach_temperature: Union[float, int] = 5,
    margin: float = 0,
    taxation: float = 0.07,
    reduction_factor: float = 1,
    oversizing_factor: float = 1.2,
    n_heat_pumps: int = 2,
    dhg_lifetime: int = 50,
    percent_residual_value: float = 0.4,
    inv_cost_multiplier: Union[float, int] = 1,
    electricity_cost_multiplier: float = 1,
    gas_cost_multiplier: Union[float, int] = 1,
    max_COP: Union[float, int] = 4,
    carnot_efficiency: Union[float, int] = 0.524,
    ir: float = 0.05,
):
    """
    This is a helper function to run the sensitivity analysis for the HT and LT scenario
    ### Parameters:
    - simulation_type: can be "renovated" or "unrenovated"
    - supply_temperature: the supply temperature to the buildings
    - approach_temperature: the approach temperature of the heat exchanger (grid to heat pump)
    - margin: the margin to be applied to the price of the heat supplied to the customers
    - taxation: taxation rate applied to the heat supplied to the residential customers
    - reduction_factor: the reduction factor to be applied to the price of the heat supplied to the customers
    - safety_factor: the oversizing factor for the large scale heat pumps
    - n_heat_pumps: the number of large scale heat pumps installed
    - dhg_lifetime: the lifetime of the DHG
    - percent_residual_value: the percentage residual value of the DHG
    - inv_cost_multiplier: the multiplier for the investment costs of the large scale heat pumps
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

    investment_costs_dhg = (
        embers_data["cost_total"].sum() / 1000000
    )  # from EMBERS [Mil €]

    years_buildingstock = 25

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
    booster_buildingstock = booster_buildingstock[booster_buildingstock["NFA"] >= 30]

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

    dh_prices_future = calculate_future_values(operator_selling_price, n_years_hp)
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
        npv_gas = npv(0, energy_costs_original, income_buildings, ir)
        npv_data.loc[
            idx,
            f"npv_Gas_{years_buildingstock}years_ir_{ir}",
        ] = npv_gas

        # DH NPV for buildings
        energy_costs_new = energy_expenditure_dh[building_id]
        npv_dh = npv(0, energy_costs_new, income_buildings, ir)
        npv_data.loc[
            idx,
            f"npv_DH_{years_buildingstock}years_ir_{ir}",
        ] = npv_dh

        # NPV of savings
        npv_savings = npv_dh - npv_gas
        npv_data.loc[idx, f"savings_npv_{years_buildingstock}years_ir_{ir}"] = (
            npv_savings
        )

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
        {"revenues": total_revenues}, heat_pump_lifetime
    )
    future_expenses = calculate_future_values(
        {"costs": total_yearly_costs}, heat_pump_lifetime
    )
    future_revenues.iloc[len(future_revenues) - 1] += (
        investment_costs_dhg * 1000000 * percent_residual_value
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

    return npv_data, npv_dh, LCOH_dhg, LCOH_HP, max_cop, cop_hourly


simulation = "booster"
n_heat_pumps = 2
supply_temperature = 50

df_sensitivity_parameters = pd.read_excel(
    "sensitivity_analysis/sensitivity_analysis_parameters.xlsx"
)
df_sensitivity_parameters.set_index("num_analysis", inplace=True)


for num_analysis, row in df_sensitivity_parameters.iterrows():
    print(f"num_analysis: {num_analysis}")
    os.makedirs(
        f"sensitivity_analysis/{simulation}/{row['analysis_type']}", exist_ok=True
    )
    os.makedirs(
        f"sensitivity_analysis/{simulation}/{row['analysis_type']}/plots",
        exist_ok=True,
    )
    os.makedirs(
        f"sensitivity_analysis/{simulation}/{row['analysis_type']}/data",
        exist_ok=True,
    )
    df_npv = pd.DataFrame()

    ###### we will create a loop for the analysis
    # To set up the loop we want to create different values for the analysis. So we will first insert the number
    # of steps we want to do for the analysis. Then we use these steps to create the different values for the analysis
    # and then we will loop through these values.
    analysis_type = df_sensitivity_parameters.loc[num_analysis, "analysis_type"]
    n_steps = df_sensitivity_parameters.loc[num_analysis, "n_steps"]
    max_value = df_sensitivity_parameters.loc[num_analysis, "max_val"]
    min_value = df_sensitivity_parameters.loc[num_analysis, "min_val"]
    step_size = (max_value - min_value) / n_steps
    values = np.linspace(min_value, max_value, n_steps)
    lcoh_dhg = []
    lcoh_hp = []
    max_cop = []
    npv_operator = []  # Add this list to collect operator NPV values
    all_npv_data = {}  # Dictionary to store df_npv for each value
    actual_cops = []

    # the max_COP simulation will require also to change the carnot_efficiency.
    # The max_COP we hit is anyway 3.6 with the standard carnot_efficiency value. So we do not see
    # almost any diffeerence. To change the carnot_efficiency during the max_COP simulation use this
    carnot_efficiency = 0.524  # 0.524 is the standard value.

    for value in tqdm(values):
        print(f"\n Analysis type: {row['analysis_type']}, Processing value: {value} \n")
        if num_analysis == 0:  # ir
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = (
                sensitivity_analysis_booster(
                    simulation,
                    ir=value,
                    n_heat_pumps=n_heat_pumps,
                    supply_temperature=supply_temperature,
                )
            )
        elif num_analysis == 1:  # approach temperature
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = (
                sensitivity_analysis_booster(
                    simulation,
                    approach_temperature=value,
                    n_heat_pumps=n_heat_pumps,
                    supply_temperature=supply_temperature,
                )
            )
        elif num_analysis == 2:  # electricity price
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = (
                sensitivity_analysis_booster(
                    simulation,
                    electricity_cost_multiplier=value,
                    n_heat_pumps=n_heat_pumps,
                    supply_temperature=supply_temperature,
                )
            )
        elif num_analysis == 3:  # gas price
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = (
                sensitivity_analysis_booster(
                    simulation,
                    gas_cost_multiplier=value,
                    n_heat_pumps=n_heat_pumps,
                    supply_temperature=supply_temperature,
                )
            )
        elif num_analysis == 4:  # max COP
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = (
                sensitivity_analysis_booster(
                    simulation,
                    max_COP=value,
                    carnot_efficiency=carnot_efficiency,
                    n_heat_pumps=n_heat_pumps,
                    supply_temperature=supply_temperature,
                )
            )
        elif num_analysis == 5:  # supply temperature
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = (
                sensitivity_analysis_booster(
                    simulation,
                    supply_temperature=value.astype(int),
                    n_heat_pumps=n_heat_pumps,
                )
            )
        elif num_analysis == 6:  # investment cost multiplier
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = (
                sensitivity_analysis_booster(
                    simulation,
                    inv_cost_multiplier=value,
                    n_heat_pumps=n_heat_pumps,
                    supply_temperature=supply_temperature,
                )
            )
        elif num_analysis == 7:  # oversizing factor
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = (
                sensitivity_analysis_booster(
                    simulation,
                    oversizing_factor=value,
                    n_heat_pumps=n_heat_pumps,
                    supply_temperature=supply_temperature,
                )
            )
        elif num_analysis == 8:  # percent residual value
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = (
                sensitivity_analysis_booster(
                    simulation,
                    percent_residual_value=value,
                    n_heat_pumps=n_heat_pumps,
                    supply_temperature=supply_temperature,
                )
            )

        df_npv.to_csv(
            f"sensitivity_analysis/{simulation}/{row['analysis_type']}/data/supply_temperature_{value}C.csv"
        )
        lcoh_dhg.append(LCOH_dhg)
        lcoh_hp.append(LCOH_HP)
        max_cop.append(cop)
        npv_operator.append(npv_dh)  # Add the operator NPV to our list
        all_npv_data[value] = df_npv.copy()  # Store a copy of df_npv for this value
        actual_cops.append(cop_hourly)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    from operator import add

    # First subplot for LCOH
    ax1.plot(values, lcoh_dhg, label="LCOH DHG")
    ax1.plot(values, lcoh_hp, label="LCOH HP")
    ax1.plot(values, list(map(add, lcoh_dhg, lcoh_hp)), label="Total LCOH")
    ax1.set_xlabel(f"{row['analysis_type']}")
    ax1.set_ylabel("LCOH (€/kWh)")
    ax1.set_title(f"Sensitivity Analysis - LCOH vs {row['analysis_type']}")
    ax1.legend()

    # Second subplot for Operator NPV
    ax2.plot(values, npv_operator, label="DH Operator NPV", color="green")
    ax2.set_xlabel(f"{row['analysis_type']}")
    ax2.set_ylabel("NPV (€)")
    ax2.set_title(f"Sensitivity Analysis - DH Operator NPV vs {row['analysis_type']}")
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        f"sensitivity_analysis/{simulation}/{row['analysis_type']}/plots/lcoh_vs_{analysis_type}.png"
    )
    # plt.close()
    plt.show()

    # Create a figure with multiple subplots for different analyses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # First subplot for LCOH
    ax1.plot(values, lcoh_dhg, label="LCOH DHG")
    ax1.plot(values, lcoh_hp, label="LCOH HP")
    ax1.set_xlabel(f"{analysis_type}")
    ax1.set_ylabel("LCOH (€/kWh)")
    ax1.set_title(f"Sensitivity Analysis - LCOH vs {analysis_type}")
    ax1.legend()

    # Second subplot for Operator NPV
    ax2.plot(values, npv_operator, label="DH Operator NPV", color="green")
    ax2.set_xlabel(f"{analysis_type}")
    ax2.set_ylabel("NPV (€)")
    ax2.set_title(f"Sensitivity Analysis - DH Operator NPV vs {analysis_type}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_sensitivity_analysis.png"
    )
    plt.close()

    # First get all building types from any of the DataFrames
    building_types = all_npv_data[values[0]]["building_usage"].unique()
    avg_savings_data = pd.DataFrame(columns=building_types, index=values)

    for value in values:
        if analysis_type == "ir":
            ir = value
        else:
            ir = 0.05
        # Calculate average savings for each building type at this value
        avg_savings = (
            all_npv_data[value]
            .groupby("building_usage")[f"savings_npv_25years_ir_{ir}"]
            .mean()
        )
        # Fill in the row for this value
        avg_savings_data.loc[value] = avg_savings

    # No need to transpose as we've already structured it correctly
    # avg_savings_data = avg_savings_data.T

    # Create line plot
    plt.figure(figsize=(12, 8))
    for building_type in avg_savings_data.columns:
        plt.plot(
            values, avg_savings_data[building_type], marker="o", label=building_type
        )

    plt.title(f"Average NPV Savings by Building Type vs {analysis_type}", fontsize=16)
    plt.xlabel(f"{analysis_type}", fontsize=14)
    plt.ylabel("Average NPV Savings (€)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_average_savings_sensitivity.png",
        bbox_inches="tight",
    )
    plt.close()

    ######## For the time being we don'twant this plot, which is also causing issues at the moment!!!############
    # The problem is that i need to use the all_npv_data to get the data for the histogram.
    for value in values:
        if analysis_type == "ir":
            ir = value
        else:
            ir = 0.05
        n_columns = 3
        # Plot histogram of savings distribution by building type
        plt.figure(figsize=(20, 15))
        building_types = all_npv_data[value]["building_usage"].unique()
        num_types = len(building_types)
        rows = (
            num_types + 2
        ) // n_columns  # Calculate number of rows needed for n_columns

        for i, building_type in enumerate(building_types, 1):
            plt.subplot(rows, n_columns, i)
            data = all_npv_data[value][
                all_npv_data[value]["building_usage"] == building_type
            ][f"savings_npv_25years_ir_{ir}"]
            scatter = sns.histplot(data, kde=True)
            scatter.set_title(
                f"Savings Distribution - {building_type}\n{analysis_type}: {value}",
                fontsize=14,
            )
            scatter.set_xlabel("NPV Savings (€2023)", fontsize=12)
            scatter.set_ylabel("Frequency", fontsize=12)
            scatter.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(
            f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_savings_distribution_{value}.png"
        )
        plt.close()

    # Now we can plot distributions using the stored data
    n_columns = 3
    for value in values:
        if analysis_type == "ir":
            ir = value
        else:
            ir = 0.05
        plt.figure(figsize=(20, 15))
        building_types = all_npv_data[value]["building_usage"].unique()
        num_types = len(building_types)
        rows = (num_types + 2) // n_columns

        for i, building_type in enumerate(building_types, 1):
            plt.subplot(rows, n_columns, i)
            data = all_npv_data[value][
                all_npv_data[value]["building_usage"] == building_type
            ][f"savings_npv_25years_ir_{ir}"]
            scatter = sns.histplot(data, kde=True)
            scatter.set_title(
                f"Savings Distribution - {building_type}\n{analysis_type}: {value}",
                fontsize=14,
            )
            scatter.set_xlabel("NPV Savings (€2023)", fontsize=12)
            scatter.set_ylabel("Frequency", fontsize=12)
            scatter.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(
            f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_savings_distribution_{value}.png"
        )
        plt.close()

    # In the max_COP analysis section, add detailed component plotting
    if num_analysis == 4:  # max COP analysis
        # First figure with LCOH, NPV, and COP plots
        fig1, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)

        # LCOH components
        axes[0].plot(values, lcoh_dhg, label="LCOH DHG")
        axes[0].plot(values, lcoh_hp, label="LCOH HP")
        axes[0].set_xlabel("Max COP")
        axes[0].set_ylabel("LCOH (€/kWh)")
        axes[0].set_title("LCOH Components")
        axes[0].legend()

        # Operator NPV
        axes[1].plot(values, npv_operator, label="DH Operator NPV", color="green")
        axes[1].set_xlabel("Max COP")
        axes[1].set_ylabel("NPV (€)")
        axes[1].set_title("DH Operator NPV")
        axes[1].legend()

        # Average COP achieved
        axes[2].plot(values, actual_cops, label="Actual Average COP", color="orange")
        axes[2].set_xlabel("Max COP")
        axes[2].set_ylabel("Average COP")
        axes[2].set_title("Actual Average COP Achieved")
        axes[2].legend()

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.3)

        # Save first figure
        fig1.savefig(
            f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_detailed_analysis_part1.png",
            dpi=100,
        )
        plt.close(fig1)

        # Second figure for building type NPV
        fig2 = plt.figure(figsize=(10, 6), dpi=100)
        ax = fig2.add_subplot(111)

        for building_type in avg_savings_data.columns:
            ax.plot(
                values,
                avg_savings_data[building_type],
                marker="o",
                label=building_type,
                linewidth=1,
            )

        ax.set_xlabel("Max COP")
        ax.set_ylabel("NPV Savings (€)")
        ax.set_title("NPV Savings by Building Type")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        # Save second figure
        fig2.savefig(
            f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_detailed_analysis_part2.png",
            bbox_inches="tight",
            dpi=100,
        )
        plt.close(fig2)
    # Create figure and primary axis to show the savings by building type
    # and on the secondary axis the NPV of the operator
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot average customer savings on primary axis (left)
    ax1.set_xlabel(f"{analysis_type}", fontsize=12)
    ax1.set_ylabel("Average Customer Savings (€)", color="tab:blue", fontsize=12)

    # Plot each building type's savings
    colors = plt.cm.tab20(np.linspace(0.4, 0.8, len(avg_savings_data.columns)))
    for building_type, color in zip(avg_savings_data.columns, colors):
        ax1.plot(
            values,
            avg_savings_data[building_type],
            marker="o",
            label=building_type,
            color=color,
        )
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Create secondary axis (right) for DH operator NPV
    ax2 = ax1.twinx()
    ax2.set_ylabel("DH Operator NPV (M€)", color="tab:red", fontsize=12)
    ax2.plot(
        values,
        np.array(npv_operator) / 1000000,
        "r-",
        linewidth=3,
        label="DH Operator NPV",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Add title
    plt.title(
        f"DH Operator NPV vs Building Type Savings\nSensitivity to {analysis_type}",
        fontsize=14,
    )

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        fontsize=10,
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_operator_vs_building_savings.png",
        bbox_inches="tight",
    )
    plt.close()

print("done")
