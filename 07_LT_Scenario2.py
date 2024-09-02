import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm

from pathlib import Path

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
)

#############################################################################################
# In this scenario we compare the NPV of the customer when they do not renovate their homes
# and when they do not switch to DH vs when they do both.
#############################################################################################

###################################################################################
###################################################################################
######################## Import data about area's demand  #########################
###################################################################################
###################################################################################


## We need to import both the unrenovated and renovated buildingstock

path_renovated_area = Path(
    "building_analysis/results/renovated_whole_buildingstock/area_results_renovated.csv"
)
areas_demand = pd.read_csv(path_renovated_area, index_col=0)
areas_demand.index = pd.to_datetime(areas_demand.index)

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
safety_factor = 1.2
n_heat_pumps = 2
heat_pump_load = (
    areas_demand["final_energy"] / n_heat_pumps / 1000
)  # MWh this is the load for each heat pump
capacity_single_hp = estimated_capacity / n_heat_pumps * safety_factor / 1000  # MW

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
supply_temp = pd.DataFrame(50, index=outside_temp.index, columns=["supply_temp"])

cop_hourly = carnot_cop(supply_temp, outside_temp, 5)

P_el = (
    areas_demand["final_energy"] / cop_hourly
)  # this is the Electric power input for ALL heat pumps

DK_to_DE = (
    109.1 / 148.5
)  # this is the ratio of the installation costs of a heat pump in Denmark to Germany
update2022_2023 = (
    126.6 / 111.2
)  # this is the ratio of the installation costs of a heat pump in 2022 to 2023
installation_cost_HP = (
    capital_costs_hp(capacity_single_hp, "air") * DK_to_DE * update2022_2023
)  # million euros
total_installation_costs = installation_cost_HP * n_heat_pumps  # Million euros

single_var_oem_hp = (
    var_oem_hp(capacity_single_hp, "air", heat_pump_load) * DK_to_DE * update2022_2023
)  # Million Euros

single_fix_oem = (
    fixed_oem_hp(capacity_single_hp, "air") * DK_to_DE * update2022_2023
)  # Million Euros

print(
    f"Data in Million Euros. Total costs installation: {total_installation_costs}\nSingle HP Variable OEM: {single_var_oem_hp}\nSingle HP Fixed OEM: {single_fix_oem}"
)


###################################################################################
###################################################################################
############################ Heat Pump LCOE Data  #################################
###################################################################################
###################################################################################

# TODO: I could make a little function or simply a mapping to calculate the electricity cost for the DH operator
## we need to calculate also the electricity cost of running the heat pump:
initial_electricity_cost = (
    0.2017  # EUROSTAT 2023- semester 2 for consumption between 2000 MWH and 19999 MWH
)
n_years_hp = 25
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
total_fixed_oem_hp = single_fix_oem * n_heat_pumps
yearly_heat_supplied = areas_demand["delivered_energy"].sum() / 1000  # MW
heat_supplied_df = pd.DataFrame(  ## In this case we are using the heat supplied in the Grid, not the delivered heat
    {"Heat Supplied (MW)": [yearly_heat_supplied] * n_years_hp}
)

fixed_oem_hp_df = calculate_future_values({"Fixed O&M": total_fixed_oem_hp}, n_years_hp)

var_oem_hp_df = calculate_future_values({"Variable O&M": total_var_oem_hp}, n_years_hp)

total_electricity_cost_df = pd.DataFrame(total_electricity_cost)


ir_hp = 0.05
LCOH_HP = calculate_lcoh(
    total_installation_costs,
    fixed_oem_hp_df,
    var_oem_hp_df,
    total_electricity_cost_df,
    heat_supplied_df,  # TODO: use heat delivered
    ir_hp,
)  # in this case we are getting Million Euros per MWh produced. We need  to convert to Euros per kWh produced

LCOH_eurokwh = LCOH_HP * 1000000 / 1000  # 1000000 million / 1000 kWh
print(f"LCOH of the Heat Pumps: {LCOH_eurokwh}")


dhg_lifetime = 60  # years
dhg_other_costs = np.zeros(dhg_lifetime)
dhg_other_costs_df = pd.DataFrame(dhg_other_costs)
heat_supplied_dhg = pd.DataFrame(  ## In this case we are using the heat supplied in the Grid, not the delivered heat
    {"Heat Supplied (MW)": [yearly_heat_supplied] * dhg_lifetime}
)


investment_costs_dhg = 16.25  # from thermos with LT option
ir_dhg = 0.03
LCOD_dhg = calculate_lcoh(
    investment_costs_dhg,  # comes from THERMOS
    dhg_other_costs_df,
    dhg_other_costs_df,
    dhg_other_costs_df,
    heat_supplied_dhg,
    ir_dhg,
)

LCOH_dhg_eurokwh = LCOD_dhg * 1000000 / 1000  # 1000000 million / 1000 kWh
print(f"LCOH_dhg_eurokwh: {LCOH_dhg_eurokwh}")

margin = 0.111
taxation = 0.07
price_heat_eurokwh_residential = (
    (LCOH_eurokwh + LCOD_dhg) * (1 + margin) * (1 + taxation)
)
print(
    f"Lowest Price of the residential heat supplied: {price_heat_eurokwh_residential}"
)
price_heat_eurokwh_non_residential = (LCOH_eurokwh + LCOD_dhg) * (1 + margin)
print(
    f"Lowest Price of the non-residential heat supplied: {price_heat_eurokwh_non_residential}"
)
price_heat_eurokwh_non_residential_VAT = (
    (LCOH_eurokwh + LCOD_dhg) * (1 + margin) * (1 + taxation)
)


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
# let's import the relevant data first. In this scenario the Buildingstock is renovated


## Let's define a couple of parameteres first
years_buildingstock = 25
building_interest_rate = 0.05


# import the data with the renovated buildingstock
renovated_buildingstock_path = Path(
    "building_analysis/results/renovated_whole_buildingstock/buildingstock_renovated_results.parquet"
)

renovated_buildingstock = gpd.read_parquet(renovated_buildingstock_path)

# and now let's import the unrenovated buildingstock

unrenovated_buildingstock_path = Path(
    "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results.parquet"
)

unrenovated_buildingstock = gpd.read_parquet(unrenovated_buildingstock_path)

year_consumption = pd.DataFrame(
    {
        "full_id": renovated_buildingstock["full_id"],
        "yearly_DHW_energy_demand": renovated_buildingstock["yearly_dhw_energy"],
        "renovated_yearly_space_heating": renovated_buildingstock[
            "yearly_space_heating"
        ],
        "unrenovated_yearly_space_heating": unrenovated_buildingstock[
            "yearly_space_heating"
        ],
    }
)

year_consumption["renovated_total_demand"] = (
    year_consumption["renovated_yearly_space_heating"]
    + year_consumption["yearly_DHW_energy_demand"]
)

year_consumption["unrenovated_total_demand"] = (
    year_consumption["unrenovated_yearly_space_heating"]
    + year_consumption["yearly_DHW_energy_demand"]
)

# first we create the monetary savings for each building. We already have the energy savings.
# Let's assess the energy prices for each building and then we can calculate the monetary savings.
npv_data = pd.DataFrame()
npv_data["full_id"] = renovated_buildingstock["full_id"]
npv_data["building_usage"] = renovated_buildingstock["building_usage"]
npv_data["yearly_demand_useful_renovated"] = year_consumption[
    "renovated_total_demand"
]  # this is for the renovated buildingstock. DHW+SH

npv_data["yearly_demand_delivered_renovated"] = (
    year_consumption["renovated_total_demand"] / efficiency_he
)
npv_data["yearly_demand_useful_unrenovated"] = year_consumption[
    "unrenovated_total_demand"
]  # this is for the unrenovated buildingstock. DHW+SH

efficiency_boiler = 0.9
npv_data["yearly_demand_delivered_unrenovated"] = (
    year_consumption["unrenovated_total_demand"] / efficiency_boiler
)


### Let's find out what is the size of the customer now for GAS
small_consumer_threshold = 20  # GJ per year
gj_to_kwh = 1 / 3600 * 1000000  # 1 GJ = 1/3600 * 1000000 kwh - conversion factor
medium_consumer_threshold = 200  # GJ per year
res_types = ["mfh", "sfh", "ab", "th"]

npv_data["consumer_size_renovated"] = consumer_size(
    npv_data,
    small_consumer_threshold,
    medium_consumer_threshold,
    res_types,
    "yearly_demand_delivered_renovated",
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
    "yearly_demand_delivered_renovated",
    years_buildingstock,
    system_efficiency=1,
    building_state="renovated",
)

# npv_data[f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}_gas"] = np.nan
# npv_data[f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}_dh"] = np.nan
income_buildings = pd.DataFrame(np.zeros(years_buildingstock))

for idx, row in tqdm(npv_data.iterrows(), total=len(npv_data)):
    building_id = row["full_id"]

    # Unrenovated NPV
    energy_costs_original = energy_expenditure_gas[building_id]
    npv_gas = npv(0, energy_costs_original, income_buildings, building_interest_rate)
    npv_data.loc[
        idx,
        f"npv_unrenovated_{years_buildingstock}years_ir_{building_interest_rate}_gas",
    ] = npv_gas

    # Renovated NPV
    energy_costs_new = energy_expenditure_dh[building_id]

    npv_dh = npv(0, energy_costs_new, income_buildings, building_interest_rate)
    npv_data.loc[
        idx,
        f"npv_renovated_{years_buildingstock}years_ir_{building_interest_rate}_dh",
    ] = npv_dh

    # NPV of savings
    npv_savings = npv_dh - npv_gas
    npv_data.loc[
        idx, f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}"
    ] = npv_savings

# We do not have energy savings for the buildingstock in this case. We do not need to import the
# renovated buildingstock dataset.


# TODO: we will use a 10% profit margin for the LCOH. The resulting number will be the price at which the heat will be sold to the customers.
# We might use a slightly higher Interest Rate in the NPV to account a little bit for inflation though. We need more research on this.


# TODO: no inflation applied for the LCOH. calculate in real terms €2023


# Calculate average savings by building type
avg_savings = (
    npv_data.groupby("building_usage")[
        f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}"
    ]
    .mean()
    .sort_values(ascending=False)
)

# Plot average savings by building type
plt.figure(figsize=(12, 6))
avg_savings.plot(kind="bar")
plt.title("Average NPV Savings by Building Type")
plt.xlabel("Building Type")
plt.ylabel("Average NPV Savings (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot histogram of savings distribution by building type
plt.figure(figsize=(15, 10))
building_types = npv_data["building_usage"].unique()
num_types = len(building_types)
rows = (num_types + 1) // 2  # Calculate number of rows needed

for i, building_type in enumerate(building_types, 1):
    plt.subplot(rows, 2, i)
    data = npv_data[npv_data["building_usage"] == building_type][
        f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}"
    ]
    sns.histplot(data, kde=True)
    plt.title(f"Savings Distribution - {building_type}")
    plt.xlabel("NPV Savings (€)")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# Merge NFA data with npv_data
merged_data = npv_data.merge(renovated_buildingstock[["full_id", "NFA"]], on="full_id")

# Calculate energy savings (assuming original demand - new demand)
# merged_data['energy_savings'] = merged_data['yearly_demand_unrenovated'] - merged_data['yearly_demand_unrenovated']  # Replace with actual new demand if available

# Create scatter plots
plt.figure(figsize=(20, 15))
building_types = merged_data["building_usage"].unique()
num_types = len(building_types)
rows = (num_types + 1) // 2  # Calculate number of rows needed

for i, building_type in enumerate(building_types, 1):
    plt.subplot(rows, 2, i)
    data = merged_data[merged_data["building_usage"] == building_type]

    sns.scatterplot(
        data=data,
        x="NFA",
        y=f"savings_npv_{years_buildingstock}years_ir_{building_interest_rate}",
    )

    plt.title(f"Energy Savings vs NFA - {building_type}")
    plt.xlabel("Net Floor Area (m²)")
    plt.ylabel("Energy Savings (kWh/year)")

    # Add a trend line
    # sns.regplot(data=data, x='NFA', y='energy_savings', scatter=False, color='red')

plt.tight_layout()
plt.show()


###################################################################################
###################################################################################
############################### NPV DH Operator ###################################
###################################################################################
###################################################################################

# now we need to calculate the NPV for the DH Operator. The operator has spent money for the
# installation of the grid. It spends money to upkeep the Heat Pumps and run it (electricity costs.)
# It will also receive money from the customers from the heat delivered.
# I am not sure about the maintenance and running costs for the District Heating Network.
overnight_costs = total_installation_costs + investment_costs_dhg
heat_pump_lifetime = 25
heat_pump_replacement = pd.DataFrame()
heat_pump_replacement["costs"] = np.zeros(dhg_lifetime)
heat_pump_replacement.iloc[heat_pump_lifetime] = total_installation_costs
#### We need to calculate the running costs for the heat pumps. We have this data from the LCOH calculation

total_yearly_costs_hps = (
    total_var_oem_hp + total_fixed_oem_hp + total_electricity_cost.iloc[0, 0]
)  # in Million Euros per year

# we have different pricing schemes according to the type and size of customer.
npv_data["operator_selling_price"] = npv_data["consumer_size_renovated"].map(
    operator_selling_price
)
revenues = calculate_revenues(
    npv_data["yearly_demand_delivered_renovated"], npv_data["operator_selling_price"]
)
total_revenues = revenues.sum() / 1000000  # in Mio €/year


future_revenues = calculate_future_values({"revenues": total_revenues}, dhg_lifetime)
future_expenses = calculate_future_values(
    {"costs": total_yearly_costs_hps}, dhg_lifetime
)
future_expenses["costs"] = future_expenses["costs"] + heat_pump_replacement["costs"]

interest_rate_dh = 0.03
npv_dh = npv(-overnight_costs, future_expenses, future_revenues, interest_rate_dh)
print(f"NPV of the District Heating Operator: {npv_dh}")
