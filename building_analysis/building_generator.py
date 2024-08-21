import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
from shapely import wkb
from typing import List, Tuple
import json
from tqdm import tqdm
import copy

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.building_utilities import convert_angle_to_cardinal
from building_analysis.Building import Building

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
    fid: int,  # fid of the building. this should be the fid full id from QGIS from the geometry data
    osm_id: int,  # osm_id of the building. this should be the osm_id from the geometry data
    plot_area: float,  # area of the plot the building is on
    roof_area: float,  # area of the roof of the building
    wall_area: float,  # area of the walls of the building. This is the NET area (minus shared walls with other buildings)
    volume: float,  # volume of the building
    building_height: float,  # height of the building
    ceiling_height: float,  # height of the ceiling
    angles_shared_borders: List[float],  # angles of the shared borders
    cardinal_directions_shared_borders: List[
        str
    ],  # cardinal directions of the shared borders
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
    :param angles_shared_borders: Angles of the shared borders to determine free sides
    :param cardinal_directions_shared_borders: Cardinal directions of the shared borders to determine free sides
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
    window_shgc = template_df["window_shgc"].values[0]
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
    # here we will assign first the sides that are available to use:

    available_angles, available_cardinals = remaining_angles(angles_shared_borders)
    available_angles = available_angles.tolist()

    # retrieve window areas from Tabula templates
    windows_to_walls_ratio: float = (template_df["windows_to_walls_ratio"]).values[
        0
    ]  # ratio of windows to walls
    windows_area = windows_to_walls_ratio * wall_area

    # retrieve window shgc from Tabula templates
    windows_shgc = template_df["window_shgc"].values[0]

    windows_json = assign_windows(
        available_cardinals, windows_area, window_u_value, windows_shgc
    )
    # calculate the number of floors:
    n_floors = calculate_n_floors(building_height, ceiling_height)

    # calculate the GFA
    gfa = plot_area * n_floors

    # calculate the NFA :TODO in the future we can assign a different ratio according to the building type
    nfa = calculate_nfa(gfa)

    # door data evaluation. If no door data are available then we set the area and u-value to 0
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
        "fid": int(fid),
        "full_id": building_id,
        "osm_id": int(osm_id),
        "building_usage": building_usage,
        "age_code": age_code,
        "roof_area": roof_area,
        "roof_u_value": roof_u_value,
        "walls_area": wall_area,
        "walls_u_value": wall_u_value,
        "ground_contact_area": plot_area,
        "ground_contact_u_value": floor_u_value,
        "door_area": door_area,
        "door_u_value": door_u_value,
        "windows_area": windows_area,
        "windows_shgc": windows_shgc,
        "window_u_value": window_u_value,
        "windows": windows_json,
        "available_angles": [
            available_angles
        ],  # these are lists. So we need to pass them as list
        "available_cardinals": [
            available_cardinals
        ],  # otherwise it will try to put each value in a separate row but we only have one row
        "volume": volume,
        "building_height": building_height,
        "ceiling_height": ceiling_height,
        "n_floors": n_floors,
        "GFA": gfa,
        "NFA": nfa,
        "geometry": geometry,
    }
    gdf = gpd.GeoDataFrame(export_data, index=[0])

    return gdf


def calculate_n_floors(building_height: float, ceiling_height: float) -> int:
    """Calculates the number of floors in a building based on the total building height and the ceiling height.
    :param building_height: The total height of the building
    :param ceiling_height: The height of the ceiling
    """
    if building_height <= ceiling_height:
        n_floors = 1
    else:
        n_floors = np.floor(building_height / ceiling_height)
    return n_floors


def assign_people_id(gdf: gpd.GeoDataFrame, res_types: List[str]) -> pd.Series:
    """Assigns a unique ID to each person in the building. The ID is a combination of the building_id
    and the number of the person in the building. For example if the building ID is w123 and there are 10 people
    the IDs will be w123_0, w123_1, w123_2, ..., w123_9. We only assign IDs to residential buildings.
    :param building_id: The ID of the building
    :param n_people: The number of people in the building"""
    people_id = []
    res_mask = gdf["building_usage"].isin(res_types)
    gdf["people_id"] = None

    for idx, row in gdf[res_mask].iterrows():
        building_id = row["full_id"]
        n_people = row["n_people"]
        for i in range(n_people):
            people_id.append(f"{building_id}_{i}")
        gdf.at[idx, "people_id"] = people_id
        people_id = []

    return gdf["people_id"]


def people_in_building(
    gdf: gpd.GeoDataFrame, res_types: List[str], total_people: int
) -> pd.Series:
    """Calculates the amount of people in each building. It is based on the GFA of the building and the
    population density in terms of Gross Floor Area (GFA) per person.
    We only assign people to residential buildings.
    :param gdf: The GeoDataFrame containing the building data
    :res_types: The list of residential building types
    :total_people: The total number of people in the area
    """

    res_mask = gdf["building_usage"].isin(res_types)
    total_GFA = gdf[res_mask]["GFA"].sum()
    people_per_GFA = total_people / total_GFA
    gdf.loc[res_mask, "n_people"] = round(gdf["GFA"] * people_per_GFA)
    gdf["n_people"] = gdf["n_people"].fillna(0).infer_objects(copy=False)
    gdf["n_people"] = gdf["n_people"].astype(int)

    return gdf["n_people"]


def readjust_angle(angle: float) -> float:
    """Adjusts the angle to ensure it's within the 0-360 range."""
    return angle % 360


def is_angle_close(angle1: float, angle2: float, tolerance: float = 30) -> bool:
    """Checks if two angles are within a certain tolerance."""
    return abs(angle1 - angle2) <= tolerance or abs(angle1 - angle2) >= 360 - tolerance


def remaining_angles(angle: List[float]) -> Tuple[np.ndarray, List[str]]:
    """
    This function calculates the remaining angles that are available in a building that has
    one or more borders shared with another building. It is used to determine the orientation of
    façades that are available for windows. We only deal with main cardinal directions (N, E, S, W).

    :param angle: List of angles of the shared borders.
    :param cardinal: List of cardinal directions of the shared borders.
    :return: The available angles and their corresponding cardinal directions.
    """

    if len(angle) == 0:
        # Base case: no shared borders, all main directions are available
        available_angles = np.array([0, 90, 180, 270])
    else:
        # Generate potential available angles
        potential_angles = []
        for ang in angle:
            candidates = [
                readjust_angle(ang + 90),
                readjust_angle(ang - 90),
                readjust_angle(ang + 180),
            ]
            potential_angles.extend(candidates)

        # Remove duplicates and sort the angles
        potential_angles = sorted(set(potential_angles))

        # Remove angles that are within 15 degrees of any shared border angle
        filtered_angles = []
        for ang in potential_angles:
            if not any(is_angle_close(ang, shared_ang) for shared_ang in angle):
                filtered_angles.append(ang)

        # Remove any duplicates by checking angles close to each other
        final_angles = []
        for ang in filtered_angles:
            if not any(
                is_angle_close(ang, existing_ang) for existing_ang in final_angles
            ):
                final_angles.append(ang)

        available_angles = np.array(final_angles)

    return available_angles, convert_angle_to_cardinal(available_angles)


def assign_windows(
    available_cardinals: List[str],
    windows_tot_area: float,
    window_uvalue: float,
    window_shgc: float,
):
    """Assigns the windows to the available angles and cardinals. It creates two lists with the
    angles and the surface per cardinal."""
    windows_area = windows_tot_area / len(available_cardinals)
    windows_dict = {}
    for cardinal in available_cardinals:
        if isinstance(cardinal, str):  # Ensure each cardinal is a string
            windows_dict[cardinal] = {
                "area": windows_area,
                "u_value": window_uvalue,
                "shgc": window_shgc,
            }
        else:
            raise TypeError(
                f"Expected string in available_cardinals, got {type(cardinal)}"
            )
    return json.dumps(windows_dict)


def iterator_generate_buildings(
    building_data,  # the building data that we want to iterate over (GDF). This should come from the process_data function
    u_value_path,  # the path to the csv file with the u-values. If relative paths don't work, use absolute paths.
    convert_wkb=True,  # convert the geometry from wkb to shapely. this allows usage in gdf
    randomization_factor: float = 0.15,  # the randomization factor for the u-values. Default is 0.15
    verbose: bool = False,  # prints warnings if data is missing when True. Default is False
) -> gpd.GeoDataFrame:
    """a small utility that will iterate over the buildings and generate the buildings.
    This is useful when we want to generate the buildings in a loop. The function will return eventually a
    GeoDataFrame with all the buildings generated.
    :param building_data: the building data that we want to iterate over. This should come from the process_data function
    :param u_value_path: the path to the csv file with the u-values. If relative paths don't work, use absolute paths.
    :param convert_wkb: convert the geometry from wkb to shapely. this allows usage in gdf
    :param randomization_factor: the randomization factor for the u-values. Default is 0.15
    :param verbose: prints warnings if data is missing when True. Default is False
    """

    results_list = []

    for idx, row in tqdm(building_data.iterrows(), total=building_data.shape[0]):
        # print(row)
        building_usage = row["building_usage"]
        age_code = row["age_code"]
        building_id = row["full_id"]
        full_id = row["full_id"]
        fid = row["fid"]
        osm_id = row["osm_id"]
        plot_area = row["plot_area"]
        roof_area = row["roof_surface"]
        wall_area = row["wall_surface"]
        volume = row["volume"]
        n_neighbors = row["neighbors_count"]
        building_height = row["height"]
        geometry = row["geometry"]
        ceiling_height = row["ceiling_height"]
        angles_shared_borders = row["angles_shared_borders_standard"]
        cardinal_directions = row["cardinal_dir_shared_borders"]

        # if isinstance(geometry, bytes):
        #     geometry = wkb.loads(geometry)
        # Call the generate_building function for each row
        result = generate_building(
            building_usage,
            age_code,
            building_id,
            fid,
            osm_id,
            plot_area,
            roof_area,
            wall_area,
            volume,
            building_height,
            ceiling_height,
            angles_shared_borders,
            cardinal_directions,
            u_value_path,
            geometry,
            randomization_factor,
            convert_wkb,
            verbose,
        )
        results_list.append(result)
    results_df = pd.concat(results_list, ignore_index=True)

    # Convert the results list to a DataFrame
    return results_df


def add_insulation(
    mm_insulation: float,
    thermal_conductivity: float,
    original_u_value: float,
    rs_out: float = 0.04,
    rs_in: float = 0.13,
):
    """
    --VVVV--o--VVVV--
    orig R added R

    Rtot = orig R + added R

    we assume that the insulation is added to the outside of the building and uniformly. All buildings' elements will have the same insulation thickness and type.
    the orig R of the element is calculated using the U-value. being the u-value = 1/Rtot -> Rtot = 1/u-value.
    We will just add the R of the insulation to the original R of the element and the re-calculate the u-value.
    We also need to remove the Thermal resistance of the surfaces (internal and external) first
    :param mm_insulation: the thickness of the insulation in mm
    :param thermal_conductivity: the thermal conductivity of the insulation material
    :param original_u_value: the original u-value of the building element
    :param rs_out: the thermal resistance of the outside surface of the building element
    :param rs_in: the thermal resistnace of the inside surface of the building element
    """
    if not all(
        isinstance(x, (int, float))
        for x in [mm_insulation, thermal_conductivity, original_u_value, rs_out, rs_in]
    ):
        raise TypeError("All input values must be integers or floats")

    if mm_insulation < 0:
        raise ValueError("Insulation thickness cannot be negative")
    if thermal_conductivity < 0:
        raise ValueError("Thermal conductivity cannot be negative")
    if original_u_value < 0:
        raise ValueError("U-value cannot be negative")
    if rs_out < 0:
        raise ValueError("Outside surface thermal resistance cannot be negative")
    if rs_in < 0:
        raise ValueError("Inside surface thermal resistance cannot be negative")

    m_insulation = mm_insulation / 1000  # in m
    Rins = m_insulation / thermal_conductivity  # in m2 K / W
    R_orig = 1 / original_u_value
    R_tot = R_orig + Rins
    new_u_value = 1 / R_tot
    return new_u_value


def need_insulation(
    gdf: gpd.GeoDataFrame,
    buffer: float = 20,
    thresholds_dict: dict = None,
) -> pd.Series:
    """
    This functions determins whether a building needs to undergo renovations or not to be able to
    properly heat at low temperature.
    :param gdf: the GeoDataFrame containing the specific heat demand ["specific_ued"] and "building_usage"
    :param thresholds_dict: a dictionary containing the threshold values for each building type above which we need to renovate
    :param buffer: the buffer value to add to the threshold value
    """
    if thresholds_dict == None:
        thresholds_dict = {
            "mfh": 60,
            "ab": 50,
            "th": 65,
            "sfh": 70,
            "trade": 100,
            "office": 100,
            "other": 100,
            "education": 100,
            "health": 100,
        }

    for types in thresholds_dict.keys():
        mask = create_mask(gdf, types)
        gdf.loc[mask, "needs_insulation"] = (
            gdf[mask]["specific_ued"] > thresholds_dict[types] + buffer
        )
    return gdf["needs_insulation"]


def create_mask(gdf: gpd.GeoDataFrame, building_type: str):
    """
    This function will create a mask that will be used to filter the buildings that need insulation.
    It will use the need_insulation function to determine if the building needs insulation.
    """
    mask = gdf["building_usage"] == building_type
    return mask


def calculate_nfa(gfa: float, building_type: str = None):
    """we take that 85% of the Gross Floor Area is Net Floor Area
    for now is constant across all building types"""
    return gfa * 0.85


def apply_renovations(
    gdf: gpd.GeoDataFrame,
    insulation_thickness: float,
    thermal_conductivity: float,
    thresholds_dict: dict = None,
):
    """
    A utility to apply insulation and new windows to the buildings that need renovations.
    It updates also the column called "insulation_thickness"
    :param gdf: the GeoDataFrame containing the buildings that need renovations
    :param thresholds: a dictionary containing the threshold values for each building type above which we need to renovate. should not include buffer
    :param insulation_thickness: the thickness of the insulation in mm
    :param thermal_conductivity: the thermal conductivity of the insulation material in W/mK
    """
    gdf = gdf.copy()
    if thresholds_dict == None:
        thresholds_dict = {
            "mfh": 60,
            "ab": 50,
            "th": 65,
            "sfh": 70,
            "trade": 100,
            "office": 100,
            "other": 100,
            "education": 100,
            "health": 100,
        }

    for idx, row in gdf[gdf["needs_insulation"]].iterrows():

        old_roof = row["roof_u_value"]
        old_walls = row["walls_u_value"]
        old_ground_contact = row["ground_contact_u_value"]
        old_door = row["door_u_value"]
        old_windows = row["windows"]
        old_windows = json.loads(old_windows)

        new_roof = add_insulation(insulation_thickness, thermal_conductivity, old_roof)
        new_walls = add_insulation(
            insulation_thickness, thermal_conductivity, old_walls
        )
        new_ground_contact = add_insulation(
            insulation_thickness, thermal_conductivity, old_ground_contact
        )
        new_door = 0.8
        new_windows_u_value = 0.8
        new_windows = copy.deepcopy(old_windows)
        for sides in new_windows:
            new_windows[sides]["u_value"] = new_windows_u_value
        new_windows = json.dumps(new_windows)

        gdf.at[idx, "roof_u_value"] = new_roof
        gdf.at[idx, "walls_u_value"] = new_walls
        gdf.at[idx, "ground_contact_u_value"] = new_ground_contact
        gdf.at[idx, "door_u_value"] = new_door
        gdf.at[idx, "windows"] = new_windows
        gdf.at[idx, "window_u_value"] = new_windows_u_value
        gdf.at[idx, "space_heating_path"] = 0
        gdf.at[idx, "yearly_space_heating"] = 0
        gdf.at[idx, "insulation_thickness"] = insulation_thickness

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
        base_dir, "building_analysis", "building_generator_data", "frankfurt_v3.parquet"
    )
    path_ceiling_heights = os.path.join(
        base_dir, "building_analysis", "building_generator_data", "ceiling_heights.csv"
    )
    ceiling_heights = pd.read_csv(path_ceiling_heights)
    age_path = os.path.join(
        base_dir, "building_analysis", "building_generator_data", "buildings_age.csv"
    )
    age_distr = pd.read_csv(age_path)
    res_types = ["mfh", "sfh", "th", "ab"]
    building_data = process_data(
        path_geometry_data, "parquet", age_distr, ceiling_heights, res_types
    )

    u_value_path = os.path.join(
        base_dir,
        "building_analysis",
        "building_generator_data",
        "archetype_u_values.csv",
    )

    # testing the iterator function
    buildings_input = iterator_generate_buildings(building_data, u_value_path)
    # Initialize an empty list to store the results
    results_list = []

    buildings_input["n_people"] = people_in_building(buildings_input, res_types, 9500)
    buildings_input["people_id"] = assign_people_id(buildings_input, res_types)

    # test the add_insulation function
    mm_insulation = int(100)
    thermal_conductivity = 0.02  # W/mK like phenolyc foam
    original_u_value = 2.427  # W/m2K

    new_u_value = add_insulation(mm_insulation, thermal_conductivity, original_u_value)
    print(f"new u_value is: {new_u_value}")

    # testing the apply_renovation function
    sim = "unrenovated"
    size = "whole_buildingstock"

    path_load_results = (
        f"../building_analysis/results/{sim}_{size}/buildingstock_results.parquet"
    )
    gdf_buildingstock_results = gpd.read_parquet(path_load_results)

    gdf_buildingstock_results["needs_renovations"] = need_insulation(
        gdf_buildingstock_results, buffer=20
    )
    gdf_renovated = apply_renovations(gdf_buildingstock_results, 10, 0.02)
