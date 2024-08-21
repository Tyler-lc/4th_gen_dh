import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import warnings

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
    # we assume that anything below 15 degrees is a flat roof and anything above is a sloped roof
    mask_slope = gdf_cost["roof_slope"] >= 15
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


if __name__ == "__main__":
    import os
    import sys
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    from pathlib import Path

    # import the data with the renovation measures
    renovated_buildingstock_path = Path(
        "../building_analysis/results/renovated_whole_buildingstock/buildingstock_renovated_results.parquet"
    )
    gdf_renovated = gpd.read_parquet(renovated_buildingstock_path)

    # import the data with the unrenovated buildingstock
    unrenovated_buildingstock_path = Path(
        "../building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results.parquet"
    )
    gdf_unrenovated = gpd.read_parquet(unrenovated_buildingstock_path)

    savings_df = energy_savings(gdf_renovated, gdf_unrenovated, rel_path=True)
