import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
import os
from typing import Union

from config import (
    SENSITIVITY_PARAMS_PATH,
    grid_results_parquet,
    buildingstock_results_path,
    weather_data_path,
    sensitivity_results_dir,
)
from utils.area_demand import compute_area_demand

from costs.heat_supply import capital_costs_hp, var_oem_hp, fixed_oem_hp, calculate_lcoh, compute_ouc_residual
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


def sensitivity_analysis(
    simulation_type: str,
    areas_demand: pd.DataFrame = None,
    ember_results: pd.DataFrame = None,
    buildingstock: gpd.GeoDataFrame = None,
    supply_temperature: Union[float, int] = 90,
    approach_temperature: Union[float, int] = 5,
    margin: float = 0,
    taxation: float = 0.07,
    reduction_factor: float = 1,
    oversizing_factor: float = 1.2,
    n_heat_pumps: int = 2,
    dhg_lifetime=50,  # years
    percent_residual_value: float = compute_ouc_residual(0.05, npv_years=25, lcoh_years=50),
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
    if simulation_type == "renovated":
        supply_temperature = 50
        n_heat_pumps = 2
    #############################################################################################
    # In this scenario we compare the NPV of the customer when they do not renovate and use gas
    # against the case when they renovate and use DH which uses a air Heat Pump.
    #############################################################################################

    n_years_hp = 25  # for LCOH calculation
    heat_pump_lifetime = 25  # setting years until replacement

    if ember_results is None:
        ember_results = pd.read_parquet(grid_results_parquet(simulation_type))
    investment_costs_dhg = ember_results["cost_total"].sum() / 1000000  # Million Euros

    # investment_costs_dhg = 24203656.03 / 1000000  # from thermos with HT option

    years_buildingstock = 25

    ### Let's find out what is the size of the customer now for GAS
    small_consumer_threshold = 20  # GJ per year
    gj_to_kwh = 1 / 3600 * 1000000  # 1 GJ = 1/3600 * 1000000 kwh - conversion factor
    medium_consumer_threshold = 200  # GJ per year
    res_types = ["mfh", "sfh", "ab", "th"]

    ###################################################################################
    ###################################################################################
    ######################## Import data about area's demand  #########################
    ###################################################################################
    ###################################################################################

    ## We need to import both the unrenovated and renovated buildingstock

    if areas_demand is None:
        areas_demand = compute_area_demand(simulation_type)
    areas_demand = areas_demand.copy()

    areas_demand["total_useful_demand"] = (
        areas_demand["dhw_energy"] + areas_demand["space_heating"]
    )

    efficiency_he = 0.8  # efficiency of the heat exchanger to be used to calculate the delivered energy
    areas_demand["delivered_energy"] = (
        areas_demand["total_useful_demand"] / efficiency_he
    )

    #### we can import the losses from the EMBERS' module calculation
    total_power_losses = ember_results["Losses [W]"].sum()
    total_energy_losses = total_power_losses * 8760 / 1000  # kWh/year

    ### calculate hourly losses on the grid:
    areas_demand["hourly grid losses [kWh]"] = ember_results["Losses [W]"].sum() / 1000
    areas_demand["hourly heat generated in Large HP [kWh]"] = (
        areas_demand["hourly grid losses [kWh]"] + areas_demand["delivered_energy"]
    )

    # areas_demand["final_energy_dh"] = areas_demand["delivered_energy_dh"] * (
    #     1 + estimated_grid_losses
    # )

    estimated_capacity = areas_demand["hourly heat generated in Large HP [kWh]"].max()

    # the estimated capacity is around 60 MW. For this reason we decide to use 3 heat pumps of 20 MW each.
    # we assume that the load is equally distributed among the 3 heat pumps.

    heat_pump_load = (
        areas_demand["hourly heat generated in Large HP [kWh]"] / n_heat_pumps / 1000
    )  # MWh this is the load for each heat pump
    capacity_single_hp = (
        estimated_capacity / n_heat_pumps * oversizing_factor / 1000
    )  # MW

    # let's calculate the efficiency of the heat pumps at a hourly level.
    # we assume that the heat pumps are air source heat pumps.
    # the COP of the heat pump is calculated as a function of the outside temperature using the Carnot formula
    # the source will be the outside air. Let's import the outside air temperature data

    path_outside_air = weather_data_path()
    outside_temp = pd.read_csv(path_outside_air, usecols=["T2m"])
    outside_temp.index = areas_demand.index

    # set up the outlet and inlet of the heat pump to calculate the COP
    supply_temp = pd.DataFrame(
        supply_temperature, index=outside_temp.index, columns=["supply_temp"]
    )

    cop_hourly = carnot_cop(
        T_hot=supply_temp,
        T_cold=outside_temp,
        approach_temperature=approach_temperature,
        carnot_efficiency=carnot_efficiency,
        COP_max=max_COP,
    )
    max_cop = cop_hourly.max()
    # print(cop_hourly.head(200))

    P_el = (
        areas_demand["hourly heat generated in Large HP [kWh]"] / cop_hourly
    )  # this is the Electric power input for ALL heat pumps

    DK_to_DE = (
        109.1 / 148.5
    )  # this is the ratio of the installation costs of a heat pump in Denmark to Germany
    update2022_2023 = (
        126.6 / 111.2
    )  # this is the ratio of the installation costs of a heat pump in 2022 to 2023
    installation_cost_HP = (
        capital_costs_hp(capacity_single_hp, "air")
        * DK_to_DE
        * update2022_2023
        * inv_cost_multiplier
    )  # million euros/MW_thermal installed per heat pump
    total_installation_costs = (
        installation_cost_HP * n_heat_pumps * capacity_single_hp
    )  # Million euros

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

    initial_electricity_cost = (
        get_electricity_cost(
            P_el.sum() / 1000,
            "non_residential",  # total consumption (Booster + central HP in MWh)
        )
        * electricity_cost_multiplier
    )  # €/kWh
    ###################################################################################
    ###################################################################################
    ############################ Heat Pump LCOE Data  #################################
    ###################################################################################
    ###################################################################################

    # TODO: I could make a little function or simply a mapping to calculate the electricity cost for the DH operator
    ## we need to calculate also the electricity cost of running the heat pump:

    future_electricity_prices = calculate_future_values(
        {"electricity": initial_electricity_cost}, n_years_hp
    )

    # this is the electricity cost for ALL heat pumps in the area for all n_years years
    total_electricity_cost = (
        P_el.sum()
        * future_electricity_prices
        / 1000000  # kWh electricity * price in €/kWh / 1000000 to convert to Million euros
    )  # Million euros
    total_var_oem_hp = single_var_oem_hp * n_heat_pumps
    total_fixed_oem_hp = single_fix_oem * n_heat_pumps * capacity_single_hp
    # Compute from filtered building stock for consistency with revenue basis
    if buildingstock is None:
        buildingstock = gpd.read_parquet(buildingstock_results_path(simulation_type))
        buildingstock = buildingstock[buildingstock["NFA"] >= 30]
    yearly_heat_supplied = (
        buildingstock["yearly_dhw_energy"] + buildingstock["yearly_space_heating"]
    ).sum() / efficiency_he / 1000  # MWh
    heat_supplied_df = pd.DataFrame(  ## In this case we are using the heat supplied in the Grid, not the delivered heat
        {"Heat Supplied (MW)": [yearly_heat_supplied] * n_years_hp}
    )

    fixed_oem_hp_df = calculate_future_values(
        {"Fixed O&M": total_fixed_oem_hp}, n_years_hp
    )

    var_oem_hp_df = calculate_future_values(
        {"Variable O&M": total_var_oem_hp}, n_years_hp
    )

    total_electricity_cost_df = pd.DataFrame(total_electricity_cost)

    LCOH_HP = calculate_lcoh(
        total_installation_costs * 1000000,  # convert to euros
        fixed_oem_hp_df,
        var_oem_hp_df,
        total_electricity_cost_df * 1000000,  # convert to euros
        heat_supplied_df * 1000,  # # convert to kWh
        ir,
    )  # in this case we are getting Euros per kWh produced.
    print(f"interest rate: {ir}")
    print(f"LCOH of the Heat Pumps: {LCOH_HP}")

    dhg_other_costs = np.zeros(dhg_lifetime)
    dhg_other_costs_df = pd.DataFrame(dhg_other_costs)
    heat_supplied_dhg = pd.DataFrame(  ## In this case we are using the heat supplied in the Grid, not the delivered heat
        {"Heat Supplied (MW)": [yearly_heat_supplied] * dhg_lifetime}
    )

    LCOH_dhg = calculate_lcoh(
        investment_costs_dhg * 1000000,  # comes from THERMOS
        dhg_other_costs_df,
        dhg_other_costs_df,
        dhg_other_costs_df,
        heat_supplied_dhg * 1000,
        ir,
    )
    print(f"interest rate: {ir}")
    # LCOH_dhg_eurokwh = LCOD_dhg * 1000000 / 1000  # 1000000 million / 1000 kWh
    print(f"LCOH_dhg_eurokwh: {LCOH_dhg}")

    price_heat_eurokwh_residential = (
        (LCOH_HP + LCOH_dhg) * (1 + margin) * (1 + taxation) * reduction_factor
    )
    print(
        f"Lowest Price of the residential heat supplied: {price_heat_eurokwh_residential}"
    )
    price_heat_eurokwh_non_residential = (
        (LCOH_HP + LCOH_dhg) * (1 + margin) * reduction_factor
    )
    print(
        f"Lowest Price of the non-residential heat supplied: {price_heat_eurokwh_non_residential}"
    )
    price_heat_eurokwh_non_residential_VAT = (
        (LCOH_HP + LCOH_dhg) * (1 + margin) * (1 + taxation) * reduction_factor
    )

    # Operator revenue price: excludes VAT (pass-through to government, not operator income)
    price_heat_ex_vat = (LCOH_HP + LCOH_dhg) * (1 + margin) * reduction_factor

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

    # Operator revenue excludes VAT — VAT is collected on behalf of government
    operator_selling_price = {
        "r0": price_heat_ex_vat / ratios["r0"],
        "r1": price_heat_ex_vat / ratios["r1"],
        "r2": price_heat_ex_vat / ratios["r2"],
        "nr0": price_heat_ex_vat / ratios["nr0"],
        "nr1": price_heat_ex_vat / ratios["nr1"],
        "nr2": price_heat_ex_vat / ratios["nr2"],
    }

    ###################################################################################
    ###################################################################################
    ############################### NPV Customers GAS #################################
    ###################################################################################
    ###################################################################################

    # first we calculate the NPV for the customers in the case of gas heating.
    # let's import the relevant data first. In this scenario the Buildingstock is renovated

    ## Let's define a couple of parameteres first

    # import the data with the renovated buildingstock
    # and now let's import the unrenovated buildingstock

    # buildingstock already loaded above (before LCOH section)
    year_consumption = pd.DataFrame(
        {
            "full_id": buildingstock["full_id"],
            "yearly_DHW_energy_demand": buildingstock["yearly_dhw_energy"],
            f"{simulation_type}_yearly_space_heating": buildingstock[
                "yearly_space_heating"
            ],
            f"{simulation_type}_yearly_space_heating": buildingstock[
                "yearly_space_heating"
            ],
        }
    )

    year_consumption[f"{simulation_type}_total_demand"] = (
        year_consumption[f"{simulation_type}_yearly_space_heating"]
        + year_consumption["yearly_DHW_energy_demand"]
    )

    # first we create the monetary savings for each building. We already have the energy savings.
    # Let's assess the energy prices for each building and then we can calculate the monetary savings.
    npv_data = pd.DataFrame()
    npv_data["full_id"] = buildingstock["full_id"]
    npv_data["NFA"] = buildingstock["NFA"]
    npv_data["building_usage"] = buildingstock["building_usage"]
    npv_data[f"yearly_demand_useful_{simulation_type}"] = year_consumption[
        f"{simulation_type}_total_demand"
    ]  # this is for the unrenovated buildingstock. DHW+SH

    efficiency_boiler = 0.9
    npv_data[f"yearly_demand_delivered_{simulation_type}"] = (
        year_consumption[f"{simulation_type}_total_demand"] / efficiency_boiler
    )

    npv_data[f"yearly_demand_delivered_{simulation_type}_DH"] = (
        year_consumption[f"{simulation_type}_total_demand"] / efficiency_he
    )

    npv_data[f"consumer_size_{simulation_type}"] = consumer_size(
        npv_data,
        small_consumer_threshold,
        medium_consumer_threshold,
        res_types,
        f"yearly_demand_delivered_{simulation_type}",
    )

    #### Let's fill a dataframe with the Gas prices for each consumer size
    gas_prices_future = calculate_future_values(gas_energy_prices, n_years_hp)
    ### how much would the customoers pay when using gas?
    energy_expenditure_gas = calculate_expenses(
        npv_data,
        gas_prices_future,
        f"yearly_demand_delivered_{simulation_type}",
        years_buildingstock,
        system_efficiency=1,  # because we are calculating the gas prices on the delivered energy already
        building_state=simulation_type,
    )

    ### and how much would they pay when using heat pumps?
    dh_prices_future = calculate_future_values(hp_energy_prices, n_years_hp)
    energy_expenditure_dh = calculate_expenses(
        npv_data,
        dh_prices_future,
        f"yearly_demand_delivered_{simulation_type}",
        years_buildingstock,
        system_efficiency=1,
        building_state=simulation_type,
    )

    # npv_data[f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}_gas"] = np.nan
    # npv_data[f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}_dh"] = np.nan
    income_buildings = pd.DataFrame(np.zeros(years_buildingstock))

    ### Calculate the costs of renovation for each building

    # TODO: Where there was no renovation now we have a NaN. We have to change these into the right data
    # because the price of the heating has changed. So they actually have a different NPV. We have to do that in the
    # renovation_costs["total_cost"]. There we get NaN data.
    convert2020_2023 = 188.40 / 133.90
    renovation_costs = renovation_costs_iwu(buildingstock, convert2020_2023)
    renovation_costs["total_cost"] = renovation_costs["total_cost"].fillna(0)

    for idx, row in npv_data.iterrows():
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
        renovations = renovation_costs.loc[
            renovation_costs["full_id"] == building_id, "total_cost"
        ]
        npv_dh = npv(
            -renovations.values[0],
            energy_costs_new,
            income_buildings,
            ir,
        )
        npv_data.loc[
            idx,
            f"npv_DH_{years_buildingstock}years_ir_{ir}",
        ] = npv_dh

        # NPV of savings
        npv_savings = npv_dh - npv_gas
        npv_data.loc[idx, f"savings_npv_{years_buildingstock}years_ir_{ir}"] = (
            npv_savings
        )

    ###################################################################################
    ###################################################################################
    ############################### NPV DH Operator ###################################
    ###################################################################################
    ###################################################################################

    # now we need to calculate the NPV for the DH Operator. The operator has spent money for the
    # installation of the grid. It spends money to upkeep the Heat Pumps and run it (electricity costs.)
    # It will also receive money from the customers from the heat delivered.
    # I am not sure about the maintenance and running costs for the District Heating Network.
    overnight_costs = (total_installation_costs + investment_costs_dhg) * 1000000

    # heat_pump_replacement = pd.DataFrame()
    # heat_pump_replacement["costs"] = np.zeros(dhg_lifetime)
    # heat_pump_replacement.iloc[heat_pump_lifetime] = total_installation_costs * 1000000
    #### We need to calculate the running costs for the heat pumps. We have this data from the LCOH calculation

    total_yearly_costs_hps = (
        total_var_oem_hp
        + total_fixed_oem_hp
        + total_electricity_cost.iloc[0, 0] * 1000000
    )  # in Euros per year

    # we have different pricing schemes according to the type and size of customer.
    npv_data["operator_selling_price"] = npv_data[
        f"consumer_size_{simulation_type}"
    ].map(operator_selling_price)
    revenues = calculate_revenues(
        npv_data[f"yearly_demand_delivered_{simulation_type}_DH"],
        npv_data["operator_selling_price"],
    )
    # revenues = calculate_revenues(
    #     npv_data["yearly_demand_delivered_unrenovated_DH"],
    #     (LCOH_HP + LCOH_dhg),
    # )
    total_revenues = revenues.sum()  # in Mio €/year

    future_revenues = calculate_future_values(
        {"revenues": total_revenues}, heat_pump_lifetime
    )
    future_expenses = calculate_future_values(
        {"costs": total_yearly_costs_hps}, heat_pump_lifetime
    )
    future_revenues.iloc[len(future_revenues) - 1] += (
        investment_costs_dhg * 1000000 * percent_residual_value
    )
    future_expenses["costs"] = future_expenses["costs"]
    from costs.renovation_costs import npv_2

    npv_dh, df = npv_2(-overnight_costs, future_expenses, future_revenues, ir)
    print(f"NPV of the District Heating Operator: {npv_dh}")

    return npv_data, npv_dh, LCOH_dhg, LCOH_HP, max_cop, cop_hourly


# simulation = "unrenovated"
simulation = "renovated"
buildingstock_years = 25


df_sensitivity_parameters = pd.read_excel(SENSITIVITY_PARAMS_PATH)
df_npv = pd.DataFrame()
if simulation == "unrenovated":
    n_heat_pumps = 3
    supply_temperature = 90
elif simulation == "renovated":
    n_heat_pumps = 2
    supply_temperature = 50

# Preload data once to avoid re-reading on every sensitivity iteration
_areas_demand = compute_area_demand(simulation)
_ember_results = pd.read_parquet(grid_results_parquet(simulation))
_buildingstock = gpd.read_parquet(buildingstock_results_path(simulation))
_buildingstock = _buildingstock[_buildingstock["NFA"] >= 30]

###### we will create a loop for the analysis
# To set up the loop we want to create different values for the analysis. So we will first insert the number
# of steps we want to do for the analysis. Then we use these steps to create the different values for the analysis
# and then we will loop through these values.
for rows, columns in df_sensitivity_parameters.iterrows():

    analysis_type = df_sensitivity_parameters.loc[rows, "analysis_type"]
    n_steps = df_sensitivity_parameters.loc[rows, "n_steps"]
    max_value = df_sensitivity_parameters.loc[rows, "max_val"]
    min_value = df_sensitivity_parameters.loc[rows, "min_val"]
    step_size = (max_value - min_value) / n_steps
    values = np.linspace(min_value, max_value, n_steps)
    num_analysis = df_sensitivity_parameters.loc[rows, "num_analysis"]
    lcoh_dhg = []
    lcoh_hp = []
    max_cop = []
    npv_operator = []  # Add this list to collect operator NPV values
    all_npv_data = {}  # Dictionary to store df_npv for each value
    actual_cops = []
    print(
        f"Analysis type: {analysis_type}, Number of steps: {n_steps}, Max value: {max_value}, Min value: {min_value}, Step size: {step_size}, Values: {values}"
    )

    # creating folders for the sensitivity analysis and their results
    sens_dir = sensitivity_results_dir(simulation, analysis_type)
    os.makedirs(sens_dir, exist_ok=True)
    os.makedirs(sens_dir / "plots", exist_ok=True)
    os.makedirs(sens_dir / "data", exist_ok=True)
    # the max_COP simulation will require also to change the carnot_efficiency.
    # The max_COP we hit is anyway 3.6 with the standard carnot_efficiency value. So we do not see
    # almost any diffeerence. To change the carnot_efficiency during the max_COP simulation use this

    for value in tqdm(values):
        print(f"\n Analysis type: {analysis_type}, Processing value: {value} \n")
        if num_analysis == 0:  # interest rate
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = sensitivity_analysis(
                simulation, _areas_demand, _ember_results, _buildingstock,
                ir=value,
                n_heat_pumps=n_heat_pumps,
                supply_temperature=supply_temperature,
            )
        elif num_analysis == 1:  # approach temperature
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = sensitivity_analysis(
                simulation, _areas_demand, _ember_results, _buildingstock,
                approach_temperature=value,
                n_heat_pumps=n_heat_pumps,
                supply_temperature=supply_temperature,
            )
        elif num_analysis == 2:  # electricity price
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = sensitivity_analysis(
                simulation, _areas_demand, _ember_results, _buildingstock,
                electricity_cost_multiplier=value,
                n_heat_pumps=n_heat_pumps,
                supply_temperature=supply_temperature,
            )
        elif num_analysis == 3:  # gas price
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = sensitivity_analysis(
                simulation, _areas_demand, _ember_results, _buildingstock,
                gas_cost_multiplier=value,
                n_heat_pumps=n_heat_pumps,
                supply_temperature=supply_temperature,
            )
        elif num_analysis == 4:  # max COP

            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = sensitivity_analysis(
                simulation, _areas_demand, _ember_results, _buildingstock,
                max_COP=value,
                n_heat_pumps=n_heat_pumps,
                supply_temperature=supply_temperature,
            )
        elif num_analysis == 5:  # supply temperature
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = sensitivity_analysis(
                simulation, _areas_demand, _ember_results, _buildingstock, supply_temperature=value, n_heat_pumps=n_heat_pumps
            )
        elif num_analysis == 6:  # investment cost multiplier
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = sensitivity_analysis(
                simulation, _areas_demand, _ember_results, _buildingstock,
                inv_cost_multiplier=value,
                n_heat_pumps=n_heat_pumps,
                supply_temperature=supply_temperature,
            )
        elif num_analysis == 7:  # reduction factor
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = sensitivity_analysis(
                simulation, _areas_demand, _ember_results, _buildingstock,
                reduction_factor=value,
                n_heat_pumps=n_heat_pumps,
                supply_temperature=supply_temperature,
            )
        elif num_analysis == 8:  # percent_residual value
            df_npv, npv_dh, LCOH_dhg, LCOH_HP, cop, cop_hourly = sensitivity_analysis(
                simulation, _areas_demand, _ember_results, _buildingstock,
                percent_residual_value=value,
                n_heat_pumps=n_heat_pumps,
                supply_temperature=supply_temperature,
            )

        # Store results
        lcoh_dhg.append(LCOH_dhg)
        lcoh_hp.append(LCOH_HP)
        max_cop.append(cop)
        npv_operator.append(npv_dh)
        all_npv_data[value] = df_npv.copy()  # Store a copy of df_npv for this value
        actual_cops.append(cop_hourly)

        # Save individual NPV data
        df_npv.to_csv(sens_dir / "data" / f"{analysis_type}_{value}.csv")

    from utils.plotting import (
        lcoh_operator_NPV,
        calculate_average_savings,
        nfa_savings_operator_comparison,
        plot_savings_operator_comparison,
    )

    # Create a figure with multiple subplots for different analyses
    simulation_title = "LT+Reno Scenario"
    lcoh_operator_NPV(
        values,
        lcoh_dhg,
        lcoh_hp,
        npv_operator,
        analysis_type,
        simulation,
        simulation_title,
    )
    # First get all building types from any of the DataFrames

    avg_savings_data_nfa, avg_savings_data = calculate_average_savings(
        all_npv_data, values, analysis_type
    )

    nfa_savings_operator_comparison(
        avg_savings_data_nfa,
        npv_operator,
        all_npv_data,
        values,
        analysis_type,
        simulation,
        simulation_title,
    )
    plot_savings_operator_comparison(
        avg_savings_data, npv_operator, all_npv_data, values, analysis_type, simulation
    )

    # let's save the data for the sensitivity analysis:
    main_path = sens_dir / "data" / "multitple_graphs"
    os.makedirs(main_path, exist_ok=True)
    avg_savings_data_nfa.to_csv(main_path / "avg_savings_data_nfa.csv")
    npv_operator_df = pd.DataFrame(npv_operator)
    npv_operator_df.to_csv(main_path / "npv_operator.csv")
    for key in all_npv_data.keys():
        all_npv_data[key].to_csv(main_path / f"all_npv_data_{key}.csv")

    values_df = pd.DataFrame(values)
    values_df.to_csv(main_path / "values.csv")

print("done")
