import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
from shapely import wkb

# TODO: since we have two different databases for u-values in the data (Tabula for residentia, BSO for non-res)
# we basically need to separate these two building types. Because the residential buildings have ages from 1 to 12,
# while the non residential buildings have ages from 1 to 7. So we need to separate these two types of buildings.
# TODO: check if the u-values fetching mechanism is "hard coded" or if it can handle different age ranges
# TODO: we need to kind fix the data with the windows, we have changed it again now. Before it was nested in a json i think
#       now we are doing something different with the new datastructure we are using. We need to fix this.
#       Also we now want to use the same fenestration on all for sides basically and use an average value for the irradiation
#       So we will be using the window to wall ration to calculate the window areas. And we will just say that windows
#       are on all sides and it wont matter really. Because now we use the average values.


def generate_building(
    building_usage: str,  # mfh, sfh, th, ab. Where x is a integer from 1 to 12 and indicates the age
    age_code: int,  # age of the building
    building_id: int,  # id of the building. this should be the full_id from the geometry data
    plot_area: float,  # area of the plot the building is on
    roof_area: float,  # area of the roof of the building
    wall_area: float,  # area of the walls of the building. This is the NET area (minus shared walls with other buildings)
    volume: float,  # volume of the building
    building_height: float,  # height of the building
    ceiling_height: float,  # height of the ceiling
    u_value_path: str,  # path to the csv file with the u-values
    geometry: gpd.GeoDataFrame,  # geometry of the building. This is a geopandas dataframe. We need the geometry column
    random_factor: float = 0.15,  # 15% random factor for u-values
    convert_wkb: bool = True,  # convert the geometry from wkb to shapely. this allows usage in gdf
    verbose: bool = False,  # print warnings if data is missing when True
) -> gpd.GeoDataFrame:  # TODO: we need to define the return type. Could be a gpd
    """
    This function generates a building based on the provided parameters and u-value template.

    :param building_usage: Usage of the building (e.g., mfh, sfh, th, ab)
    :param age_code: Age code of the building
    :param building_id: ID of the building
    :param plot_area: Plot Area of the building
    :param roof_area: Area of the roof of the building
    :param wall_area: Area of the walls of the building
    :param volume: Volume of the building
    :param n_neighbors: Number of neighboring buildings
    :param building_height: Height of the building
    :param ceiling_height: Height of the ceiling
    :param u_value_path: Path to the CSV containing u-values for different building types
    :param geometry: Geometry of the building
    :param random_factor: Random factor for u-values (default: 0.15)
    :param convert_wkb: Convert the geometry from WKB to Shapely (default: True)
    :param verbose: Print warnings if data is missing (default: False)
    :return: A GeoDataFrame containing the updated values for the building
    """

    building_usage = building_usage.lower()
    building_type = building_usage + str(age_code)
    # Read the U-values from the provided CSV path

    template_df = pd.read_csv(u_value_path)

    # Filter for the specific building type
    template_df = template_df[template_df["building_type"] == building_type]

    # Check if the building type exists in the template
    assert not template_df.empty, f"No entries found for building type: {building_type}"

    # Extract and randomize u-values
    roof_u_value = template_df["roof_uvalue"].values[0]
    wall_u_value = template_df["wall_uvalue"].values[0]
    floor_u_value = template_df["floor_uvalue"].values[0]
    window_u_value = template_df["window_uvalue"].values[0]
    door_u_value = template_df["door_uvalue"].values[0]

    #  acquire door's area from Tabula templates. Only surface we take from tabula data
    door_area = template_df["door_surface"].values[0]

    roof_u_value *= 1 + np.random.uniform(-random_factor, random_factor)
    wall_u_value *= 1 + np.random.uniform(-random_factor, random_factor)
    floor_u_value *= 1 + np.random.uniform(-random_factor, random_factor)
    window_u_value *= 1 + np.random.uniform(-random_factor, random_factor)
    door_u_value *= 1 + np.random.uniform(-random_factor, random_factor)

    # the number of sides with windows depends on the number of neighboring buildings
    # so the number of sides will be 4 - n_neighboring_buildings

    windows_to_walls_ratio: float = (template_df["windows_to_walls_ratio"]).values[
        0
    ]  # ratio of windows to walls

    # retrieve window areas from Tabula templates
    windows_area = windows_to_walls_ratio * wall_area

    # retrieve window shgc from Tabula templates
    windows_shgc = template_df["window_shgc"].values[0]

    # calculate the number of floors:
    n_floors = np.floor(building_height / ceiling_height)

    # calculate the GFA
    gfa = plot_area * n_floors

    door_area = template_df["door_surface"].values[0]
    door_u_value = template_df["door_uvalue"].values[0]

    # if no data is entered about the door, we set the area and u-value to 0
    if np.isnan(door_area):
        if verbose:
            warnings.warn("door_area is NaN. Setting door_area and door_u_value to 0")
        door_area = 0
        door_u_value = 0

    if convert_wkb:
        if isinstance(geometry, bytes):
            geometry = wkb.loads(geometry)

    export_data = {
        "building_id": building_id,
        "building_usage": building_usage,
        "age_code": age_code,
        "roof_area": roof_area,
        "roof_u_value": roof_u_value,
        "walls_area": wall_area,
        "wall_u_value": wall_u_value,
        "ground_contact_area": plot_area,
        "ground_contact_u_value": floor_u_value,
        "door_area": door_area,
        "door_u_value": door_u_value,
        "wall_area": wall_area,
        "windows_area": windows_area,
        "windows_shgc": windows_shgc,
        "volume": volume,
        "building_height": building_height,
        "ceiling_height": ceiling_height,
        "n_floors": n_floors,
        "GFA": gfa,
        "geometry": geometry,
    }

    gdf = gpd.GeoDataFrame(export_data, index=[0])

    return gdf


if __name__ == "__main__":
    import os
    import sys
    from shapely import wkb
    from tqdm import tqdm

    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.dirname(current_dir)
    # utils_dir = os.path.join(parent_dir, "utils")
    # sys.path.append(utils_dir)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.building_utilities import process_data

    # import data from QGIS parquet file

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Import data from QGIS parquet file
    path_geometry_data = os.path.join(
        base_dir, "Building", "building_generator_data", "frankfurt.parquet"
    )
    path_ceiling_heights = os.path.join(
        base_dir, "Building", "building_generator_data", "ceiling_heights.csv"
    )
    ceiling_heights = pd.read_csv(path_ceiling_heights)
    age_path = os.path.join(
        base_dir, "Building", "building_generator_data", "buildings_age.csv"
    )
    age_distr = pd.read_csv(age_path)
    res_types = ["mfh", "sfh", "th", "ab"]
    building_data = process_data(
        path_geometry_data, "parquet", age_distr, ceiling_heights, res_types
    )

    u_value_path = os.path.join(
        base_dir, "Building", "building_generator_data", "archetype_u_values.csv"
    )

    # Initialize an empty list to store the results
    results_list = []

    for idx, row in tqdm(building_data.iterrows(), total=building_data.shape[0]):
        building_usage = row["building_usage"]
        age_code = row["age_code"]
        building_id = row["full_id"]
        plot_area = row["plot_area"]
        roof_area = row["roof_surface"]
        wall_area = row["wall_surface"]
        volume = row["volume"]
        n_neighbors = row["neighbors_count"]
        building_height = row["height"]
        geometry = row["geometry"]
        ceiling_height = row["ceiling_height"]

        # if isinstance(geometry, bytes):
        #     geometry = wkb.loads(geometry)

        # Call the generate_building function for each row
        result = generate_building(
            building_usage,
            age_code,
            building_id,
            plot_area,
            roof_area,
            wall_area,
            volume,
            building_height,
            ceiling_height,
            u_value_path,
            geometry,
        )
        results_list.append(result)

    # Convert the results list to a DataFrame
    results_df = pd.concat(results_list, ignore_index=True)

    # Convert the DataFrame to a GeoDataFrame
    building_data_gdf = gpd.GeoDataFrame(results_df, geometry="geometry")
    mask_res = building_data_gdf["building_usage"].isin(["mfh", "sfh", "th", "ab"])
    print(building_data_gdf[mask_res].head())
