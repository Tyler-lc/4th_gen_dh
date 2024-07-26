import numpy as np
import pandas as pd


def generate_building(building_type, plot_area, roof_area, wall_area, volume, GFH, n_floors, u_value_path):
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

    # Read the U-values from the provided CSV path
    template_df = pd.read_csv(u_value_path)

    # Filter for the specific building type
    template_df = template_df[template_df['building type'] == building_type]

    # Check if the building type exists in the template
    assert not template_df.empty, f"No entries found for building type: {building_type}"

    # Extract and randomize u-values
    roof_u_value = template_df.loc[template_df['surface name'] == 'roof', 'u-value'].values[0]
    wall_u_value = template_df.loc[template_df['surface name'] == 'wall', 'u-value'].values[0]
    floor_u_value = template_df.loc[template_df['surface name'] == 'floor', 'u-value'].values[0]
    window_u_value = template_df[template_df['surface type'] == 'transparent']['u-value'].values[0]
    door_u_value = template_df[template_df["surface name"] == "door"]["u-value"].values[0]

    #  acquire door's area from Tabula templates. Only surface we take from tabula data
    door_area = template_df[template_df["surface name"] == "door"]["total surface"].values[0]

    roof_u_value *= (1 + np.random.uniform(-0.3, 0.3))
    wall_u_value *= (1 + np.random.uniform(-0.3, 0.3))
    floor_u_value *= (1 + np.random.uniform(-0.3, 0.3))
    window_u_value *= (1 + np.random.uniform(-0.3, 0.3))
    door_u_value *= (1 + np.random.uniform(-0.3, 0.3))

    # Decide the number of sides with windows
    num_sides = np.random.choice([2, 3, 4])

    # Allocate window areas
    total_fenestration_area = template_df[template_df['surface type'] == 'transparent']['total surface'].sum() * \
                              np.random.uniform(0.7, 1.3)
    south_window_area = total_fenestration_area * np.random.uniform(0.3, 0.4)
    remaining_area = total_fenestration_area - south_window_area
    other_windows_area = list(np.random.dirichlet(np.ones(num_sides - 1)) * remaining_area)

    # Constructing the windows list based on number of sides and areas
    orientations = ['south'] + np.random.choice(['east', 'west', 'north'], size=num_sides - 1, replace=False).tolist()

    windows = []
    windows.append({
        "building type": building_type,
        "surface type": "transparent",
        "surface name": "window1",
        "total surface": south_window_area,
        "u-value": window_u_value,
        "orientation": "south",
        "SHGC": 0.65,
        "number of floors": n_floors
    })

    # Other window orientations
    other_orientations = np.random.choice(['east', 'west', 'north'], size=num_sides - 1, replace=False).tolist()
    for idx, area in enumerate(other_windows_area):
        windows.append({
            "building type": building_type,
            "surface type": "transparent",
            "surface name": f"window{idx + 2}",
            "total surface": area,
            "u-value": window_u_value,
            "orientation": other_orientations[idx],
            "SHGC": 0.65,
            "number of floors": n_floors
        })

    # Ensure no window is too small
    min_window_area = 0.5  # m^2
    for window_dict in windows:
        if window_dict["total surface"] < min_window_area:
            window_dict["total surface"] = min_window_area

    # Construct dictionaries
    roof_dict = {
        "building type": building_type, "surface type": "opaque", "surface name": "roof", "total surface": roof_area,
        "u-value": roof_u_value, "orientation": np.nan, "SHGC": np.nan, "number of floors": n_floors
    }

    wall_dict = {
        "building type": building_type, "surface type": "opaque", "surface name": "wall", "total surface": wall_area,
        "u-value": wall_u_value, "orientation": np.nan, "SHGC": np.nan, "number of floors": n_floors
    }

    floor_dict = {
        "building type": building_type, "surface type": "ground contact", "surface name": "floor",
        "total surface": plot_area,
        "u-value": floor_u_value, "orientation": np.nan, "SHGC": np.nan, "number of floors": n_floors
    }

    door_dict = {
        "building type": building_type, "surface type": "opaque", "surface name": "door", "total surface": door_area,
        "u-value": door_u_value, "orientation": np.nan, "SHGC": np.nan, "number of floors": n_floors
    }

    # Convert dictionaries to a dataframe
    df = pd.DataFrame([roof_dict, wall_dict, floor_dict, door_dict] + windows)

    return df

# path = "buildings_data/mfh_small_u-values.csv"
# building_1 = generate_building("MFH1", 200, 220, 450, 500, 600, 2, path )