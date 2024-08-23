import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import warnings
import numpy as np
from typing import Union, List
import numpy_financial as npf

# we will use data from the IWU study to calculate the costs of each building's renovation


def renovation_costs_iwu(gdf: gpd.GeoDataFrame):
    """
    Calculate the costs of a renovation based on the insulation thickness and the area of the building.

    Args:
        insulation_thickness (float): The thickness of the insulation in meters.
        area (float): The area of the building in square meters.

    Returns:
        float: The cost of the renovation.
    """

    gdf_cost = gdf.copy(deep=True)

    # calculate the costs of renovating the walls
    gdf_cost["costs_walls"] = gdf_cost["walls_area"] * (
        112.18 + 3.25 * gdf_cost["insulation_thickness"] / 10
    )  # convert mm to cm

    # there is a difference in cost between a sloped roof and a flat roof
    # we assume that anything below 25 degrees is a flat roof and anything above is a sloped roof
    mask_slope = gdf_cost["roof_slope"] >= 25
    gdf_cost.loc[mask_slope, "costs_roof"] = gdf_cost["roof_area"] * (
        178.48 + 3.27 * gdf_cost["insulation_thickness"] / 10
    )  # convert mm to cm
    gdf_cost.loc[~mask_slope, "costs_roof"] = gdf_cost["roof_area"] * (
        123.29 + 4.87 * gdf_cost["insulation_thickness"] / 10
    )  # convert mm to cm

    # calculating the cost of insulating the ground contact floor
    gdf_cost["cost_ground_contact"] = gdf_cost["ground_contact_area"] * (
        10.27 + 1.86 * gdf_cost["insulation_thickness"] / 10
    )  # convert mm to cm

    # we assume that all windows are 2 sqm each. The cost of the window varies depending on its size.
    # meaning it is not a linear relationship like the other costs
    sqm_window_cost = (658.86 * 2 ** (-0.257) * 1.116) / 2
    gdf_cost["cost_windows"] = gdf_cost["windows_area"] * sqm_window_cost

    # cost of updating the door:
    sfh_mask = gdf_cost["building_usage"] == "sfh"
    gdf_cost.loc[sfh_mask, "cost_door"] = 1612.41 * gdf_cost["door_area"]
    gdf_cost.loc[np.logical_not(sfh_mask), "cost_door"] = (
        1374.99 * gdf_cost["door_area"]
    )

    gdf_cost["total_cost"] = (
        gdf_cost["costs_walls"]
        + gdf_cost["costs_roof"]
        + gdf_cost["cost_ground_contact"]
        + gdf_cost["cost_windows"]
        + gdf_cost["cost_door"]
    )

    return gdf_cost


def energy_savings(gdf_renovated, gdf_unrenovated, rel_path: bool = False):
    """
    Calculate the energy savings after a renovation. The function

    Args:
        gdf_renovated (gpd.GeoDataFrame): The GeoDataFrame with the renovated building stock.
        gdf_unrenovated (gpd.GeoDataFrame): The GeoDataFrame with the unrenovated building stock.
        rel_path (bool): If True, the paths are relative to the current working directory.
    Returns:
        pd.Dataframe: A pandas DataFrame with the energy savings.
    """

    # we first need to retrieve the index of the space heating data
    if rel_path:
        first_renovated_energy = pd.read_csv(
            f"../{gdf_renovated.iloc[0]['space_heating_path']}", index_col=0, header=0
        )
    else:
        first_renovated_energy = pd.read_csv(
            gdf_renovated.iloc[0]["space_heating_path"], index_col=0, header=0
        )
    gdf_savings = pd.DataFrame(index=first_renovated_energy.index)

    # creating a dictionary to store the energy savings of each building.
    # have to otherwise pandas will raise a warning for a fragmented dataframe
    savings_dict = {}

    for (idx_ren, row_ren), (idx_unren, row_unren) in tqdm(
        zip(gdf_renovated.iterrows(), gdf_unrenovated.iterrows()),
        total=len(gdf_renovated),
    ):
        if rel_path:
            renovated_energy = pd.read_csv(
                f"../{row_ren['space_heating_path']}", index_col=0, header=0
            )
            unrenovated_energy = pd.read_csv(
                f"../{row_unren['space_heating_path']}", index_col=0, header=0
            )
        else:
            renovated_energy = pd.read_csv(
                row_ren["space_heating_path"], index_col=0, header=0
            )
            unrenovated_energy = pd.read_csv(
                row_unren["space_heating_path"], index_col=0, header=0
            )

        building_id = row_ren["full_id"]
        building_id_unren = row_unren["full_id"]
        if building_id != building_id_unren:
            raise ValueError(
                f"Building IDs do not match. Renovated: {building_id}, Unrenovated: {building_id_unren}"
            )

        # Calculate energy savings and store in the dictionary
        savings_dict[building_id] = (
            unrenovated_energy.squeeze() - renovated_energy.squeeze()
        )

    # Convert the dictionary to a DataFrame
    gdf_savings = pd.DataFrame(savings_dict)

    return gdf_savings


def cash_flow(i: float, n_years: int, incomes, expenses, overnight_cost: float):
    """
    Calculate the cash flow of a renovation project.

    Args:
        i (float): The interest rate.
        n_years (int): The number of years.
        incomes (np.array): The income of the project.
        expenses (np.array): The expenses of the project.

    Returns:
        np.array: The cash flow of the project.
    """

    return None


def set_energy_prices(
    base_energy_price,
    n_years,
    inflation_rate: Union[float, pd.Series],
    base_year: int = 2019,
):
    """
    Creates energy prices for the duration of the NPV analysis based on the inflation rate.
    The energy prices are yearly.


    Args:
        base_energy_price (float): The energy price at the base_year
        inflation_rate (float): The inflation rate. If input is a float then it is expanded to a numpy array to fill the timespan

    Returns:
        float: The updated energy price.
    """

    if isinstance(inflation_rate, float):
        inflation = np.full(n_years, inflation_rate)

    if isinstance(inflation_rate, pd.Series):
        if len(inflation_rate) != n_years:
            raise ValueError(
                "The inflation rate must be the same length as the number of years"
            )
        inflation = inflation_rate

    # Calculate the cumulative product of the inflation rates
    cumulative_inflation = np.cumprod(1 + inflation)
    # print("cumulative inflation", cumulative_inflation)

    # Calculate the energy prices for each year
    energy_prices = base_energy_price * cumulative_inflation

    return energy_prices


def consumer_size(
    npv_base_data, small_threshold: float, medium_threshold: float, res_types: List[str]
):
    """
    Determine the size of the consumer based on the yearly energy demand. 0 for small, 1 for medium, 2 for large.

    Args:
        yearly_energy_demand (pd.Series): The yearly energy demand of the consumer.
        small_threshold (float): The threshold for a small consumer in GJ.
        medium_threshold (float): The threshold for a medium consumer in GJ.

    Returns:
        pd.Series: A pandas Series with the consumer size.
    """
    gj_to_kwh = 1 / 3600 * 1000000  # 1 GJ = 1/3600 * 1000000 kwh - conversion factor

    threshold_small_consumer_kwh = small_threshold * gj_to_kwh  # convert to kwh
    res_mask = npv_base_data["building_usage"].isin(res_types)
    mask_small_consumer = (
        npv_base_data["yearly_demand_unrenovated"] < threshold_small_consumer_kwh
    )
    medium_consumer_threshold_kwh = medium_threshold * gj_to_kwh
    mask_medium_consumer = np.logical_and(
        (npv_base_data["yearly_demand_unrenovated"] >= threshold_small_consumer_kwh),
        (npv_base_data["yearly_demand_unrenovated"] < medium_consumer_threshold_kwh),
    )

    mask_large_consumer = (
        npv_base_data["yearly_demand_unrenovated"] >= medium_consumer_threshold_kwh
    )

    consumer_size = pd.Series(index=npv_base_data.index, dtype="object")
    consumer_size[res_mask & mask_small_consumer] = "r0"
    consumer_size[res_mask & mask_medium_consumer] = "r1"
    consumer_size[res_mask & mask_large_consumer] = "r2"
    consumer_size[np.logical_not(res_mask) & mask_small_consumer] = "nr0"
    consumer_size[np.logical_not(res_mask) & mask_medium_consumer] = "nr1"
    consumer_size[np.logical_not(res_mask) & mask_large_consumer] = "nr2"
    # print(consumer_size)
    return consumer_size


def calculate_expenses(
    npv_data: pd.DataFrame,
    future_energy_prices: pd.DataFrame,
    building_state: str,
    n_years: int = 25,
    system_efficiency: float = 0.9,
):
    """
    Calculate the energy costs for each building over the 25 years.

    Args:
        npv_data (pd.DataFrame): The DataFrame with the building data.
        future_energy_prices (pd.DataFrame): The DataFrame with the future energy prices.
        building_state (str): The state of the building. Either "unrenovated" or "renovated".
        system_efficiency (float): The efficiency of the energy system.

    Returns:
        pd.DataFrame: A DataFrame with the energy costs.
    """

    # Initialize the energy_costs DataFrame with the correct shape and columns
    energy_costs = pd.DataFrame(index=np.arange(n_years), columns=npv_data["full_id"])

    # Set the first row to the renovation costs
    # energy_costs.loc[0] = npv_data["renovation_costs"]

    # Iterate over the rows to calculate and set the yearly energy expenses
    for idx, row in npv_data.iterrows():
        building_id = row["full_id"]
        consumer_size = row["consumer_size"]
        energy_prices = future_energy_prices[consumer_size] / system_efficiency
        yearly_demand = row[f"yearly_demand_{building_state}"]

        # Set the yearly energy expenses for each year (from 1 to 24)
        energy_costs[building_id] = energy_prices.values * yearly_demand

    return energy_costs


def npv(year_0, expenses, incomes, i):
    """
    Calculate the Net Present Value of a project.
    The cash flow is calculated as cash_flow = incomes - expenses.
    NB. If you insert negative expenses they become positive.
    NB.2 If on year 0 you only have expenses, you have to set it negative yourself, before you pass it to the method.


    Args:
        year_0 (float): The initial investment.
        expenses (np.array): The expenses of the project.
        incomes (np.array): The incomes of the project.
        i (float): The interest rate.

    Returns:
        float: The Net Present Value of the project.
    """

    # Calculate the cash flow
    cash_flow = incomes - expenses

    cash_flow = np.insert(cash_flow, 0, year_0)

    npv = npf.npv(i, cash_flow)

    # calculate the NPV:

    return npv


def manual_npv(year_0, expenses, incomes, i):
    """
    Calculate the Net Present Value of a project.

    Args:
        year_0 (float): The initial investment.
        expenses (np.array): The expenses of the project.
        incomes (np.array): The incomes of the project.
        i (float): The interest rate.

    Returns:
        float: The Net Present Value of the project.
    """

    # Calculate the cash flow
    cash_flow = incomes - expenses

    # Include the initial investment (year 0)
    cash_flow = np.insert(cash_flow, 0, year_0)

    # Calculate the NPV
    npv = sum(cash_flow[t] / (1 + i) ** t for t in range(len(cash_flow)))

    return npv


# TODO: we calculate first the energy prices over the 25 years, then we can apply these to the energy expenditures over the 25 years.
# after this we can also calulate the cash flows for the cases of renovation and unrenovated in the case of the customers.

# TODO: we still need to set up the building-level boosters by the way. In this case they won't be used on the renovated buildingstock.
# so the useful_energy_demand should stay the esame across these two cases.

if __name__ == "__main__":
    import os
    import sys
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    from pathlib import Path

    # # import the data with the renovation measures
    # renovated_buildingstock_path = Path(
    #     "../building_analysis/results/renovated_whole_buildingstock/buildingstock_renovated_results.parquet"
    # )
    # gdf_renovated = gpd.read_parquet(renovated_buildingstock_path)

    # # import the data with the unrenovated buildingstock
    # unrenovated_buildingstock_path = Path(
    #     "../building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results.parquet"
    # )
    # gdf_unrenovated = gpd.read_parquet(unrenovated_buildingstock_path)

    # renovation_costs = renovation_costs_iwu(gdf_renovated)
    # renovation_costs.to_csv("renovation_costs.csv")

    # savings_df = energy_savings(gdf_renovated, gdf_unrenovated, rel_path=True)
    # savings_df.to_csv("energy_savings_renovated.csv")

    # I already ran the code above and save the results to csv. I will now load the results and calculate the cash flows
    renovation_costs = pd.read_csv("renovation_costs.csv")
    savings_df = pd.read_csv("energy_savings_renovated.csv", index_col=0)

    # the energy prices are in euros per kWh. But they also change according to user type
    # and annual energy demand.
    # so we take the energy consumption data from the buildingstock results we have already calculated
    unrenovated_buildingstock_path = Path(
        "../building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results.parquet"
    )
    unrenovated_buildingstock = pd.read_parquet(unrenovated_buildingstock_path)
    renovated_buildingstock_path = Path(
        "../building_analysis/results/renovated_whole_buildingstock/buildingstock_renovated_results.parquet"
    )
    renovated_buildingstock = pd.read_parquet(renovated_buildingstock_path)
    # set the first column as the index. read_parquet does not have index_col function
    # unrenovated_buildingstock.set_index(unrenovated_buildingstock.columns[0], inplace=True)

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
    year_consumption["unrenovated_total_demand"] = (
        year_consumption["unrenovated_yearly_space_heating"]
        + year_consumption["yearly_DHW_energy_demand"]
    )
    year_consumption["renovated_total_demand"] = (
        year_consumption["renovated_yearly_space_heating"]
        + year_consumption["yearly_DHW_energy_demand"]
    )

    small_consumer_threshold = 20  # GJ per year
    gj_to_kwh = 1 / 3600 * 1000000  # 1 GJ = 1/3600 * 1000000 kwh - conversion factor
    threshold_small_consumer_kwh = (
        small_consumer_threshold * gj_to_kwh
    )  # convert to kwh
    mask_small_consumer = (
        year_consumption["unrenovated_total_demand"] < threshold_small_consumer_kwh
    )

    medium_consumer_threshold = 200  # GJ per year
    medium_consumer_threshold_kwh = medium_consumer_threshold * gj_to_kwh
    mask_medium_consumer = np.logical_and(
        (year_consumption["unrenovated_total_demand"] >= threshold_small_consumer_kwh),
        (year_consumption["unrenovated_total_demand"] < medium_consumer_threshold_kwh),
    )

    mask_large_consumer = (
        year_consumption["unrenovated_total_demand"] >= medium_consumer_threshold_kwh
    )

    print(
        f"small consumers: {mask_small_consumer.sum()}, medium consumers: {mask_medium_consumer.sum()}, large consumers: {mask_large_consumer.sum()}"
    )
    ##
    small_residential_prices = set_energy_prices(
        base_energy_price=0.1405,  # household gas price per kwh 2023- Semester 2 https://ec.europa.eu/eurostat/databrowser/view/nrg_pc_202/default/table?lang=en&category=nrg.nrg_price.nrg_pc
        n_years=25,
        inflation_rate=0.02,
    )

    starting_energy_prices = {
        "r0": 0.1405,  # eurostat data
        "r1": 0.1145,
        "r2": 0.1054,
        "nr0": 0.1312,  # non residential data are Residential prices without VAT and other recoverable taxes
        "nr1": 0.1070,
        "nr2": 0.0985,
    }
    # energy_prices_non_residential = {0: 0.1312, 1: 0.1070, 2: 0.0985} # these are the same as residential but without VAT and other recoverable taxes
    res_types = ["mfh", "sfh", "ab", "th"]

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
    npv_data["initial_gas_price"] = npv_data["consumer_size"].map(
        starting_energy_prices
    )
    npv_data["renovation_costs"] = renovation_costs["total_cost"]
    npv_data["yearly_demand_renovated"] = year_consumption["renovated_total_demand"]

    n_years = 30

    # calculate the energy prices for each type of customer
    energy_prices_future = pd.DataFrame(
        columns=["r0", "r1", "r2", "nr0", "nr1", "nr2"], index=np.arange(n_years)
    )

    for starting_price in starting_energy_prices:
        energy_prices_future[starting_price] = set_energy_prices(
            base_energy_price=starting_energy_prices[starting_price],
            n_years=n_years,
            inflation_rate=0.02,
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

    interest_rate = 0.03
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
