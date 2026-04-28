import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from typing import Union, List
import numpy_financial as npf

# we will use data from the IWU study to calculate the costs of each building's renovation


def renovation_costs_iwu(gdf: gpd.GeoDataFrame, update_costs_factor: float):
    """
    Calculate the costs of a renovation based on the insulation thickness and the area of the building.

    Args:
        insulation_thickness (float): The thickness of the insulation in meters.
        area (float): The area of the building in square meters.
        update_costs_factor (float): The factor by which the costs are multiplied to update to a different year.

    Returns:
        float: The cost of the renovation.
    """
    # we need to make sure that the insulation thickness is in meters
    gdf_cost = gdf.copy(deep=True)
    gdf_cost.fillna(0, inplace=True)
    renovation_mask = gdf_cost["insulation_thickness"] > 0

    # calculate the costs of renovating the walls
    gdf_cost.loc[renovation_mask, "costs_walls"] = gdf_cost["walls_area"] * (
        112.18 + 3.25 * gdf_cost["insulation_thickness"] / 10
    )  # convert mm to cm

    # there is a difference in cost between a sloped roof and a flat roof
    # we assume that anything below 25 degrees is a flat roof and anything above is a sloped roof
    mask_slope = gdf_cost["roof_slope"] >= 25
    gdf_cost.loc[renovation_mask & mask_slope, "costs_roof"] = gdf_cost["roof_area"] * (
        178.48 + 3.27 * gdf_cost["insulation_thickness"] / 10
    )  # convert mm to cm
    gdf_cost.loc[renovation_mask & ~mask_slope, "costs_roof"] = gdf_cost[
        "roof_area"
    ] * (
        123.29 + 4.87 * gdf_cost["insulation_thickness"] / 10
    )  # convert mm to cm

    # calculating the cost of insulating the ground contact floor
    gdf_cost.loc[renovation_mask, "cost_ground_contact"] = gdf_cost[
        "ground_contact_area"
    ] * (
        10.27 + 1.86 * gdf_cost["insulation_thickness"] / 10
    )  # convert mm to cm

    # we assume that all windows are 2 sqm each. The cost of the window varies depending on its size.
    # meaning it is not a linear relationship like the other costs
    sqm_window_cost = (658.86 * 2 ** (-0.257) * 1.116) / 2
    gdf_cost.loc[renovation_mask, "cost_windows"] = (
        gdf_cost["windows_area"] * sqm_window_cost
    )

    # cost of updating the door:
    sfh_mask = gdf_cost["building_usage"] == "sfh"
    gdf_cost.loc[renovation_mask & sfh_mask, "cost_door"] = (
        1612.41 * gdf_cost["door_area"]
    )
    gdf_cost.loc[renovation_mask & np.logical_not(sfh_mask), "cost_door"] = (
        1374.99 * gdf_cost["door_area"]
    )

    gdf_cost["total_cost"] = (
        gdf_cost["costs_walls"]
        + gdf_cost["costs_roof"]
        + gdf_cost["cost_ground_contact"]
        + gdf_cost["cost_windows"]
        + gdf_cost["cost_door"]
    ) * update_costs_factor

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


def apply_inflation(
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

    # Calculate the energy prices for each year
    inflated_prices = base_energy_price * cumulative_inflation

    return inflated_prices


def calculate_energy_prices_future(starting_energy_prices, n_years):
    """
    Creates energy prices for the duration of the NPV analysis based on the inflation rate.
    The energy prices are yearly.


    Args:
        base_energy_price (float): The energy price at the base_year
        inflation_rate (float): The inflation rate. If input is a float then it is expanded to a numpy array to fill the timespan

    Returns:
    """
    energy_prices_future = pd.DataFrame(
        {key: [value] * n_years for key, value in starting_energy_prices.items()},
        index=range(n_years),
    )
    return energy_prices_future


def consumer_size(
    npv_base_data,
    small_threshold: float,
    medium_threshold: float,
    res_types: List[str],
    delivered_energy_column: str,
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
        npv_base_data[delivered_energy_column] < threshold_small_consumer_kwh
    )
    medium_consumer_threshold_kwh = medium_threshold * gj_to_kwh
    mask_medium_consumer = np.logical_and(
        (npv_base_data[delivered_energy_column] >= threshold_small_consumer_kwh),
        (npv_base_data[delivered_energy_column] < medium_consumer_threshold_kwh),
    )

    mask_large_consumer = (
        npv_base_data[delivered_energy_column] >= medium_consumer_threshold_kwh
    )

    consumer_size = pd.Series(index=npv_base_data.index, dtype="object")
    consumer_size[res_mask & mask_small_consumer] = "r0"
    consumer_size[res_mask & mask_medium_consumer] = "r1"
    consumer_size[res_mask & mask_large_consumer] = "r2"
    consumer_size[np.logical_not(res_mask) & mask_small_consumer] = "nr0"
    consumer_size[np.logical_not(res_mask) & mask_medium_consumer] = "nr1"
    consumer_size[np.logical_not(res_mask) & mask_large_consumer] = "nr2"
    return consumer_size


def calculate_expenses(
    npv_data: pd.DataFrame,
    future_energy_prices: pd.DataFrame,
    column_demand: str,
    n_years: int = 25,
    system_efficiency: float = 0.9,
    building_state: str = "",
):
    """
    Calculate the energy costs for each building over the 25 years.

    Args:
        npv_data (pd.DataFrame): The DataFrame with the building data.
        future_energy_prices (pd.DataFrame): The DataFrame with the future energy prices.
        column_demand (str): The name of the column with the demand of the building.
        system_efficiency (float): The efficiency of the energy system.

    Returns:
        pd.DataFrame: A DataFrame with the energy costs.
    """

    # Initialize the energy_costs DataFrame with the correct shape and columns
    energy_costs = pd.DataFrame(index=np.arange(n_years), columns=npv_data["full_id"])

    # Set the first row to the renovation costs
    # energy_costs.loc[0] = npv_data["renovation_costs"]

    if building_state == "unrenovated":
        consumer_column = f"consumer_size_{building_state}"
    elif building_state == "renovated":
        consumer_column = f"consumer_size_{building_state}"
    elif building_state == "":
        consumer_column = "consumer_size"

    # Iterate over the rows to calculate and set the yearly energy expenses
    for idx, row in npv_data.iterrows():
        building_id = row["full_id"]

        consumer_size = row[consumer_column]

        energy_prices = future_energy_prices[consumer_size] / system_efficiency
        yearly_demand = row[column_demand]

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

    # Convert expenses and incomes to numpy arrays and ensure they are 1D
    expenses_array = expenses.values.flatten()
    incomes_array = incomes.values.flatten()

    cash_flow = incomes_array - expenses_array

    cash_flow = np.insert(cash_flow, 0, year_0)

    npv_value = npf.npv(i, cash_flow)
    return npv_value


def npv_2(year_0, expenses, incomes, i):
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

    # Convert expenses and incomes to numpy arrays and ensure they are 1D
    expenses_array = expenses.values.flatten()
    incomes_array = incomes.values.flatten()

    cash_flow = incomes_array - expenses_array

    cash_flow = np.insert(cash_flow, 0, year_0)

    npv_value = npf.npv(i, cash_flow)
    df = pd.DataFrame({"cash_flow": cash_flow})
    return npv_value, df


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


def calculate_npv_savings(
    npv_data,
    energy_expenses_unrenovated,
    energy_expenses_renovated,
    n_years,
    interest_rate,
):
    """
    Calculate the Net Present Value (NPV) savings for renovated and unrenovated buildings.

    This function computes the NPV for both unrenovated and renovated scenarios for each building,
    and then calculates the NPV savings as the difference between the two.

    Args:
        npv_data (pd.DataFrame): DataFrame containing building data and renovation costs.
        energy_expenses_unrenovated (pd.DataFrame): Energy expenses for unrenovated buildings over time.
        energy_expenses_renovated (pd.DataFrame): Energy expenses for renovated buildings over time.
        n_years (int): Number of years for the NPV calculation.
        interest_rate (float): Annual interest rate for NPV calculations.

    Returns:
        pd.DataFrame: Updated npv_data DataFrame with additional columns for unrenovated NPV,
                      renovated NPV, and NPV savings for each building.
    """
    npv_data[f"npv_savings_{n_years}years_ir_{interest_rate}"] = np.nan

    for idx, row in npv_data.iterrows():
        building_id = row["full_id"]

        # Unrenovated NPV
        energy_costs_original = energy_expenses_unrenovated[building_id]
        npv_unrenovated = npv(0, energy_costs_original, 0, interest_rate)
        npv_data.loc[idx, f"npv_unrenovated_{n_years}years_ir_{interest_rate}"] = (
            npv_unrenovated
        )

        # Renovated NPV
        energy_costs_new = energy_expenses_renovated[building_id]
        year_0_renovation = row["renovation_costs"]
        npv_renovated = npv(-year_0_renovation, energy_costs_new, 0, interest_rate)
        npv_data.loc[idx, f"npv_renovated_{n_years}years_ir_{interest_rate}"] = (
            npv_renovated
        )

        # NPV of savings
        npv_savings = npv_renovated - npv_unrenovated
        npv_data.loc[idx, f"npv_savings_{n_years}years_ir_{interest_rate}"] = (
            npv_savings
        )

    return npv_data

