import numpy as np
import pandas as pd


def generate_building(
    building_type: str,  # mfhx, sfhx, thx, abx. Where x is a integer from 1 to 12 and indicates the age
    plot_area: float,  # area of the plot the building is on
    roof_area: float,  # area of the roof of the building
    wall_area: float,  # area of the walls of the building
    volume: float,  # volume of the building
    GFA: float,  # gross flooar area #TODO i think we need to calculate this one
    n_floors: int,  # number of floors in the building #TODO i think we need to calculate this one
    n_neighbors: int,  # number of neighboring buildings
    n_sides: int,  # total number of sides of the building
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
    building_type = building_type.lower()
    # Read the U-values from the provided CSV path
    template_df = pd.read_csv(u_value_path)

    # Filter for the specific building type
    template_df = template_df[template_df["building_type"] == building_type]

    # Check if the building type exists in the template
    assert not template_df.empty, f"No entries found for building type: {building_type}"

    # Extract and randomize u-values
    roof_u_value = template_df["roof"].values[0]
    wall_u_value = template_df["wall"].values[0]
    floor_u_value = template_df["floor"].values[0]
    window_u_value = template_df["window"].values[0]
    door_u_value = template_df["door"].values[0]

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

    # Allocate window areas
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

    # Convert dictionaries to a dataframe
    df = pd.DataFrame([roof_dict, wall_dict, floor_dict, door_dict] + windows)

    return df


if __name__ == "__main__":
    path = "input_building_gen/archetype_u_values.csv"
    # building_1 = generate_building("MFH1", 200, 220, 450, 500, 600, 2, path)
    building_1 = generate_building(
        "mfh1",
        plot_area=200,
        roof_area=220,
        wall_area=450,
        volume=500,
        GFA=280,
        n_floors=2,
        n_neighbors=2,
        u_value_path=path,
    )
    print(building_1)
