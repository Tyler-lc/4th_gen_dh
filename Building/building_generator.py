import numpy as np
import pandas as pd
import random

# TODO: we need to create the standard for the input geometry of the building first
#


def generate_building(building_type, plot_area, gross_floor_area, base_file_path):
    # Load the template
    template_df = pd.read_csv(base_file_path)

    # Decide on the roof type
    pitched_roof = random.choice([True, False])

    # Calculate building dimensions and surfaces
    dL = np.random.uniform(1, 1.6)
    l = np.sqrt(plot_area)
    l1 = l * (1 + dL)
    l2 = plot_area / l1  # l2 is always the longer side of the building

    height = (
        gross_floor_area / plot_area * 2.5
    )  # the height of the ceiling should change according to age of building
    if pitched_roof:
        roof_height = 2.0
        area_triangle = l2 * roof_height / 2
        volume_roof = area_triangle * l1
        roof_surface = np.sqrt(2.5**2 + (l1 / 2) ** 2) * 2 * l2
        volume = plot_area * height + volume_roof
        ext_surface = (l1 * height + l2 * height + area_triangle) * 2 + roof_surface
    else:
        roof_surface = plot_area
        volume = plot_area * height
        ext_surface = plot_area + (l1 + l2) * height * 2

    total_window_area = (
        np.random.uniform(0.05, 0.12) * ext_surface
    )  # TODO: fix the max and min numbers. max c.ca 18
    door_area = 2.0
    ext_wall_surface = ext_surface - total_window_area - door_area - roof_surface

    south_window = np.random.uniform(0.33, 0.4) * total_window_area
    amount_windows = random.randint(3, 4)
    if (
        amount_windows == 3
    ):  # TODO: minimum number of façades with windows could also be 2!!!!
        west_window = np.random.uniform(0.4, 1) * south_window
        east_window = total_window_area - south_window - west_window
    else:
        west_window = np.random.uniform(0.4, 0.8) * south_window
        east_window = np.random.uniform(0.35, 0.7) * west_window
        north_window = total_window_area - south_window - west_window - east_window

    # Extract and randomize u-values
    roof_u_value = template_df[
        template_df["surface name"].str.contains("roof", case=False, na=False)
    ]["u-value"].values[0]
    wall_u_value = template_df[
        template_df["surface name"].str.contains("wall", case=False, na=False)
    ]["u-value"].values[0]
    floor_u_value = template_df[
        template_df["surface name"].str.contains("floor", case=False, na=False)
    ]["u-value"].values[0]
    window_u_value = template_df[template_df["surface type"] == "transparent"][
        "u-value"
    ].values[0]

    roof_u_value *= 1 + np.random.uniform(-0.3, 0.3)
    wall_u_value *= 1 + np.random.uniform(-0.3, 0.3)
    floor_u_value *= 1 + np.random.uniform(-0.3, 0.3)
    window_u_value *= 1 + np.random.uniform(-0.3, 0.3)

    # Construct dictionaries
    roof_dict = {
        "building type": building_type,
        "surface type": "opaque",
        "surface name": "roof",
        "total surface": roof_surface,
        "u-value": roof_u_value,
        "orientation": np.nan,
        "SHGC": np.nan,
        "reduction factor": 1,
        "number of floors": 2,
    }

    wall_dict = {
        "building type": building_type,
        "surface type": "opaque",
        "surface name": "wall",
        "total surface": ext_wall_surface,
        "u-value": wall_u_value,
        "orientation": np.nan,
        "SHGC": np.nan,
        "reduction factor": 1,
        "number of floors": 2,
    }

    floor_dict = {
        "building type": building_type,
        "surface type": "ground contact",
        "surface name": "floor",
        "total surface": plot_area,
        "u-value": floor_u_value,
        "orientation": np.nan,
        "SHGC": np.nan,
        "reduction factor": 1,
        "number of floors": 2,
    }

    windows = [
        {
            "building type": building_type,
            "surface type": "transparent",
            "surface name": "south window",
            "total surface": south_window,
            "u-value": window_u_value,
            "orientation": "south",
            "SHGC": 0.65,
            "reduction factor": 1,
            "number of floors": 2,
        },
        {
            "building type": building_type,
            "surface type": "transparent",
            "surface name": "west window",
            "total surface": west_window,
            "u-value": window_u_value,
            "orientation": "west",
            "SHGC": 0.65,
            "reduction factor": 1,
            "number of floors": 2,
        },
        {
            "building type": building_type,
            "surface type": "transparent",
            "surface name": "east window",
            "total surface": east_window,
            "u-value": window_u_value,
            "orientation": "east",
            "SHGC": 0.65,
            "reduction factor": 1,
            "number of floors": 2,
        },
    ]
    # TODO: make this print the correct amount of Floors!
    # TODO: this should also reflect the number of facades with windows!
    if amount_windows == 4:
        windows.append(
            {
                "building type": building_type,
                "surface type": "transparent",
                "surface name": "north window",
                "total surface": north_window,
                "u-value": window_u_value,
                "orientation": "north",
                "SHGC": 0.65,
                "reduction factor": 1,
                "number of floors": 2,
            }
        )

    # Convert dictionaries to a dataframe
    df = pd.DataFrame([roof_dict, wall_dict, floor_dict] + windows)

    return df


if __name__ == "__main__":
    path = "C:/VSCode_python/4th_gen_dh/tests/sfh/sfh_sample.csv"
    building = generate_building("sfh1", 100, 200, path)
