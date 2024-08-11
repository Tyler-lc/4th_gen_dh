import numpy as np
import pandas as pd

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
    building_usage: str,  # mfhx, sfhx, thx, abx. Where x is a integer from 1 to 12 and indicates the age
    age_code: int,  # age of the building
    plot_area: float,  # area of the plot the building is on
    roof_area: float,  # area of the roof of the building
    wall_area: float,  # area of the walls of the building
    volume: float,  # volume of the building
    # GFA: float,  # gross flooar area #TODO i think we need to calculate this one
    # n_floors: int,  # number of floors in the building #TODO i think we need to calculate this one
    n_neighbors: int,  # number of neighboring buildings
    n_sides: int,  # total number of sides of the building
    building_height: float,  # height of the building
    u_value_path: str,  # path to the csv file with the u-values
    random_factor: float = 0.15,  # 15% random factor for u-values
) -> pd.DataFrame:  # TODO: we need to define the return type. Could be a gpd
    """
    This function generates a building based on the provided parameters and u-value template.

    :param building_type: Building type to determine which u-values to pull from the u_value_path
    :param plot_area: Area of the plot the building is on
    :param roof_area: Area of the roof of the building
    :param wall_area: Area of the walls of the building
    :param volume: Volume of the building
    :param GFH: Gross floor height of the building
    :param n_floors: Number of floors in the building
    :param u_value_path: Path to the CSV containing u-values for different building types
    :return: A DataFrame containing the updated values for the building
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
    windows_sides = 4 - n_neighbors  # TODO: we need to define n_neighbors now

    # retrieve window areas from Tabula templates
    south_window_area = template_df["window_south_surface"].values[0]
    north_window_area = template_df["window_north_surface"].values[0]
    east_window_area = template_df["window_east_surface"].values[0]
    west_window_area = template_df["window_west_surface"].values[0]

    # TODO: All this window section will be changed to use the window to wall ration. MUCH EASIER.
    total_window_area = (
        south_window_area + north_window_area + east_window_area + west_window_area
    )

    # Constructing the windows list based on number of sides and areas
    orientations = ["south"] + np.random.choice(
        ["east", "west", "north"], size=windows_sides - 1, replace=False
    ).tolist()

    windows = []
    windows.append(
        {
            "building type": building_type,
            "surface type": "transparent",
            "surface name": "window1",
            "total surface": south_window_area,
            "u-value": window_u_value,
            "orientation": "south",
            "SHGC": 0.65,
            "number of floors": n_floors,
        }
    )

    # Other window orientations
    other_orientations = np.random.choice(
        ["east", "west", "north"], size=windows_sides - 1, replace=False
    ).tolist()
    for idx, area in enumerate(other_windows_area):
        windows.append(
            {
                "building type": building_type,
                "surface type": "transparent",
                "surface name": f"window{idx + 2}",
                "total surface": area,
                "u-value": window_u_value,
                "orientation": other_orientations[idx],
                "SHGC": 0.65,
                "number of floors": n_floors,
            }
        )

    # Ensure no window is too small
    min_window_area = 0.5  # m^2
    for window_dict in windows:
        if window_dict["total surface"] < min_window_area:
            window_dict["total surface"] = min_window_area

    # Construct dictionaries
    roof_dict = {
        "building type": building_type,
        "surface type": "opaque",
        "surface name": "roof",
        "total surface": roof_area,
        "u-value": roof_u_value,
        "orientation": np.nan,
        "SHGC": np.nan,
        "number of floors": n_floors,
    }

    wall_dict = {
        "building type": building_type,
        "surface type": "opaque",
        "surface name": "wall",
        "total surface": wall_area,
        "u-value": wall_u_value,
        "orientation": np.nan,
        "SHGC": np.nan,
        "number of floors": n_floors,
    }

    floor_dict = {
        "building type": building_type,
        "surface type": "ground contact",
        "surface name": "floor",
        "total surface": plot_area,
        "u-value": floor_u_value,
        "orientation": np.nan,
        "SHGC": np.nan,
        "number of floors": n_floors,
    }

    door_dict = {
        "building type": building_type,
        "surface type": "opaque",
        "surface name": "door",
        "total surface": door_area,
        "u-value": door_u_value,
        "orientation": np.nan,
        "SHGC": np.nan,
        "number of floors": n_floors,
    }
    # calculate the number of floors:
    n_floors = np.floor(plot_area / building_height)
    # calculate the GFA
    gfa = plot_area * n_floors
    # Convert dictionaries to a dataframe
    df = pd.DataFrame([roof_dict, wall_dict, floor_dict, door_dict] + windows)

    return df


if __name__ == "__main__":
    import os
    import sys

    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.dirname(current_dir)
    # utils_dir = os.path.join(parent_dir, "utils")
    # sys.path.append(utils_dir)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.building_utilities import process_data

    # import data from QGIS parquet file

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Import data from QGIS parquet file
    path_geometry_data = os.path.join(base_dir, "Building", "data", "frankfurt.parquet")
    path_ceiling_heights = os.path.join(
        base_dir, "Building", "data", "ceiling_heights.csv"
    )
    ceiling_heights = pd.read_csv(path_ceiling_heights)
    age_path = os.path.join(base_dir, "Building", "data", "buildings_age.csv")
    age_distr = pd.read_csv(age_path)
    res_types = ["mfh", "sfh", "th", "ab"]
    building_data = process_data(
        path_geometry_data, "parquet", age_distr, ceiling_heights, res_types
    )

    u_value_path = os.path.join(base_dir, "Building", "data", "archetype_u_values.csv")

    plot_area = building_data["plot_area"][1]
    roof_area = building_data["roof_surface"][1]
    wall_area = building_data["wall_surface"][1]
    volume = building_data["volume"][1]
    n_neighbors = building_data["neighbors_count"][1]
    n_sides = building_data["n_sides"][1]
    building_height = building_data["height"][1]
    data_frame = generate_building(
        "sfh",
        1,
        plot_area,
        roof_area,
        wall_area,
        volume,
        n_neighbors,
        n_sides,
        building_height,
        u_value_path,
    )

    # building_1 = generate_building("MFH1", 200, 220, 450, 500, 600, 2, path)

    # building_1 = generate_building(
    #     "mfh1",
    # )
    # print(building_1)
