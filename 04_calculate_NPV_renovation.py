import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from costs.renovation_costs import (
    renovation_costs_iwu,
    npv,
    calculate_expenses,
    consumer_size,
    apply_inflation,
    cash_flow,
    energy_savings,
)

###################################################################################
###################################################################################
##################### Set Hyperparameters for NPV Calculation #####################
###################################################################################
###################################################################################

# these are directly used in the NPV calculation
n_years = 30
interest_rate = 0.03

# This only affects the energy prices over the years. We assume a 2% inflation rate.
# It should be possible also to use a list of inflation rates for each year. (not tested)
inflation_rate = 0.02


# We now subdvide the buildingstock by user type. We have "small, medium, large" residential buildings
# We also have small, medium, large non-residential buildings.
# We sourced the data for the non-residential energy prices as the Residential prices, without the VAT and other levies.
starting_energy_prices = {  # eurostat data
    "r0": 0.1405,  # residential small
    "r1": 0.1145,  # residential medium
    "r2": 0.1054,  # residential large
    "nr0": 0.1312,  # non-residential small
    "nr1": 0.1070,  # non-residential medium
    "nr2": 0.0985,  # non-residential large
}


# we also need (as usual) to set up the list of building codes considered residential:
res_types = ["mfh", "sfh", "ab", "th"]


# EUROSTAT provides energy prices according to the annual demand. We need to set the energy prices for each building
# a small consumer uses 20 GJ per year according to EUROSTAT. That is 5555 kWh a year.
small_consumer_threshold = 20  # GJ per year
gj_to_kwh = 1 / 3600 * 1000000  # 1 GJ = 1/3600 * 1000000 kwh - conversion factor
medium_consumer_threshold = 200  # GJ per year
medium_consumer_threshold_kwh = medium_consumer_threshold * gj_to_kwh


##################### Calculate the Energy Savings from the renovation measures #####################
generate_savings = False
if generate_savings:
    # import the data with the renovation measures
    renovated_buildingstock_path = Path(
        "building_analysis/results/renovated_whole_buildingstock/buildingstock_renovated_results.parquet"
    )
    gdf_renovated = gpd.read_parquet(renovated_buildingstock_path)

    # import the data with the unrenovated buildingstock
    unrenovated_buildingstock_path = Path(
        "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results.parquet"
    )
    gdf_unrenovated = gpd.read_parquet(unrenovated_buildingstock_path)

    renovation_costs = renovation_costs_iwu(gdf_renovated)
    renovation_costs.to_csv("costs/renovation_costs.csv")

    savings_df = energy_savings(gdf_renovated, gdf_unrenovated, rel_path=True)
    savings_df.to_csv("costs/energy_savings_renovated.csv")

##################### Load results about Energy Savings (if already calculated)  #####################

# set the paths to the data
renovation_costs_path = Path("costs/renovation_costs.csv")
energy_savings_path = Path("costs/energy_savings_renovated.csv")

# Load the data in DataFrames for further manipulation
renovation_costs = pd.read_csv("costs/renovation_costs.csv")
savings_df = pd.read_csv("costs/energy_savings_renovated.csv", index_col=0)


# the energy prices are in euros per kWh. But they also change according to user type and annual energy demand.
# so we take the energy consumption data from the buildingstock results we have already calculated
unrenovated_buildingstock_path = Path(
    "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results.parquet"
)
renovated_buildingstock_path = Path(
    "building_analysis/results/renovated_whole_buildingstock/buildingstock_renovated_results.parquet"
)
renovated_buildingstock = pd.read_parquet(renovated_buildingstock_path)
unrenovated_buildingstock = pd.read_parquet(unrenovated_buildingstock_path)

# gather the energy consumption data in a single DF for ease of use
year_consumption = pd.DataFrame(
    {
        "full_id": unrenovated_buildingstock["full_id"],
        "yearly_DHW_energy_demand": unrenovated_buildingstock["yearly_dhw_energy"],
        "unrenovated_yearly_space_heating": unrenovated_buildingstock[
            "yearly_space_heating"
        ],
        "renovated_yearly_space_heating": renovated_buildingstock[
            "yearly_space_heating"
        ],
    }
)


# We now calculate the total useful energy demand, which is the sum of DHW energy and space heating energy
# we do this for both the renovated buildingstock and the unrenovated buildingstock.
year_consumption["unrenovated_total_demand"] = (
    year_consumption["unrenovated_yearly_space_heating"]
    + year_consumption["yearly_DHW_energy_demand"]
)
year_consumption["renovated_total_demand"] = (
    year_consumption["renovated_yearly_space_heating"]
    + year_consumption["yearly_DHW_energy_demand"]
)


# first we create the monetary savings for each building. We already have the energy savings.
# Let's assess the energy prices for each building and then we can calculate the monetary savings.
npv_data = pd.DataFrame()
npv_data["full_id"] = savings_df.columns
npv_data["building_usage"] = unrenovated_buildingstock["building_usage"]
npv_data["yearly_demand_unrenovated"] = year_consumption[
    "unrenovated_total_demand"
]  # this is for the unrenovated buildingstock. DHW+SH

npv_data["consumer_size"] = consumer_size(
    npv_data, small_consumer_threshold, medium_consumer_threshold, res_types
)

npv_data["initial_gas_price"] = npv_data["consumer_size"].map(starting_energy_prices)
npv_data["renovation_costs"] = renovation_costs["total_cost"]
npv_data["yearly_demand_renovated"] = year_consumption["renovated_total_demand"]

# we create a new DataFrame to store the energy prices for each year in the NPV calculation
energy_prices_future = pd.DataFrame(
    columns=["r0", "r1", "r2", "nr0", "nr1", "nr2"], index=np.arange(n_years)
)

# fill-in the columns of the DataFrame with the energy prices for each year for each consumper type
for starting_price in starting_energy_prices:
    energy_prices_future[starting_price] = apply_inflation(
        base_energy_price=starting_energy_prices[starting_price],
        n_years=n_years,
        inflation_rate=inflation_rate,
    )

# now we calculate the energy expenditures for each building over the 25 years
energy_costs_unrenovated = pd.DataFrame(
    columns=npv_data["full_id"], index=np.arange(n_years)
)

energy_costs_unrenovated = calculate_expenses(
    npv_data, energy_prices_future, "unrenovated", n_years
)
energy_costs_renovated = calculate_expenses(
    npv_data, energy_prices_future, "renovated", n_years
)

npv_data[f"npv_unrenovated_{n_years}years_ir_{interest_rate}"] = np.nan
npv_data[f"npv_renovated_{n_years}years_ir_{interest_rate}"] = np.nan
for idx, row in npv_data.iterrows():
    building_id = row["full_id"]
    energy_costs_original = energy_costs_unrenovated[building_id]
    npv_data.loc[idx, f"npv_unrenovated_{n_years}years_ir_{interest_rate}"] = npv(
        0, energy_costs_original, 0, interest_rate
    )

    energy_costs_new = energy_costs_renovated[building_id]
    year_0_renovation = row["renovation_costs"]
    npv_data.loc[idx, f"npv_renovated_{n_years}years_ir_{interest_rate}"] = npv(
        -year_0_renovation, energy_costs_new, 0, interest_rate
    )