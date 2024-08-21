import geopandas as gpd
import pandas as pd
import numpy as np
from typing import List, Dict
import os
import sys
from tqdm import tqdm


def get_building_data(path: str, filetype: str) -> gpd.GeoDataFrame:
    """
    This function reads the building data from the provided geopackage and returns a DataFrame.

    :param path: Path to the building data
    :param filetype: Type of the file, either 'gpkg' or 'parquet'
    :return: DataFrame containing the building data
    """
    try:
        if filetype == "gpkg":
            building_data = gpd.read_file(path)
        if filetype == "parquet":
            building_data = pd.read_parquet(path)
        else:
            raise ValueError(
                f"Invalid file type {filetype} . Please provide either 'gpkg' or 'parquet'."
            )
    except Exception as e:
        print(f"Error while reading the file: {e}")
        return None
    fid: int = building_data["fid"].apply(int)
    full_id: str = building_data["full_id"]
    osm_id: int = building_data["osm_id"].apply(int)
    building_type: str = building_data["building"]
    roof_slope: float = building_data["slope_mean"].apply(float)
    roof_surface: float = building_data["roof_sum"].apply(float)
    building_height: float = building_data["height_mean_point"].apply(float)
    is_isolated: int = building_data["is_isolated"].apply(int)
    wall_surface: float = building_data["vertical_s"].apply(float)
    plot_area: float = building_data["area"].apply(float)
    building_perimiter: float = building_data["perimeter"].apply(float)
    volume: float = building_data["height_sum_raster"].apply(float)
    n_sides: int = building_data["n_all_sides"].apply(int)
    building_use: str = building_data["B_REALXX"]
    angles_shared_walls = building_data["angles_shared_borders_standard"]
    geometry = building_data["geometry"]

    # QGIS writes these lists as strings, so I need to convert them into lists before i proceed
    building_neighbors = building_data["neighbors"].apply(convert_to_list)
    border_length = building_data["border_len"].apply(convert_to_float_list)
    angles_shared_walls = building_data["angles_shared_borders_standard"].apply(
        convert_to_float_list
    )
    # angles
    cardinal_directions_shared_borders = angles_shared_walls.apply(
        convert_angle_to_cardinal
    )

    data = {
        "fid": fid,
        "full_id": full_id,
        "osm_id": osm_id,
        "type": building_type,
        "roof_slope": roof_slope,
        "roof_surface": roof_surface,
        "height": building_height,
        "is_isolated": is_isolated,
        "wall_surface": wall_surface,
        "plot_area": plot_area,
        "perimeter": building_perimiter,
        "volume": volume,
        "n_sides": n_sides,
        "neighbors_count": building_neighbors,
        "border_length": border_length,
        "building_use": building_use,
        "angles_shared_borders_standard": angles_shared_walls,
        "cardinal_dir_shared_borders": cardinal_directions_shared_borders,
        "geometry": geometry,
    }

    export_data = gpd.GeoDataFrame(data)

    return export_data


def convert_angle_to_cardinal(angle: List[float]) -> List[str]:
    angle_list = []
    for angles in angle:
        if angles < 0:
            angles += 360
        if angles >= 360:
            angles -= 360

        if 315 <= angles < 360 or 0 <= angles < 45:
            angle_list.append("north")
        elif 45 <= angles < 135:
            angle_list.append("east")
        elif 135 <= angles < 225:
            angle_list.append("south")
        elif 225 <= angles < 315:
            angle_list.append("west")

    return angle_list


def convert_to_list(value):
    """As QGIS writes lists as strings, this function converts them back to lists.
    Sample usage, where building_data is a geopandas dataframe and QGIs stored a list of neighboring
    buildings as a string in the neighbors column:
    Use the "apply" method to convert the string to a list:

    building_data["neighbors"] = building_data["neighbors"].apply(convert_to_list)

    """
    if pd.isnull(value):
        return []
    return value.split(",")


def convert_to_float_list(string):
    """
    Convert a string representation of a list to an actual list of floats.

    Parameters:
    string (str): String representation of a list (e.g., "[1.0, 2.0, 3.0]").

    Returns:
    List[float]: List of floats.
    """
    if pd.isnull(string):
        return []

    return [float(x) for x in string.strip("[]").split(",")]


def define_building_type(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    This function defines the building type for each building in the building data.
    The data frame must contain the columns 'height', 'neighbors', 'type' and 'building_type'.
    the difference between type and building type is that the first contains the data from
    OpenStreetMap. This way we can exclude already the buildings that OSM knows are non-residential.
    The second is the  colum that will be used to store the data after this function.


    :param gdf: GeoDataFrame containing the building data with columns 'height', 'neighbors_count', and 'building_use'
    :return: Series containing the building type assigned to each building
    """

    list_non_residential_types = [
        "1650 Baumarkt",
        "1610 Möbelmarkt",
        "1320 Handel und Dienstl.",
        "2204 Bürgerhs.,Stadthalle",
        "1300 Industrie u. Gewerbe",
        "1670 Lebensmittelmarkt",
        "1310 Lagerfläche",
        "9990 Freifläche",
        "2440 Sonst. religiöse Zwecke",
        "7000 V+E allg.",
        "2210 Kinderbetreuung",
        "5306 Fußballplatz",
        "2860 Gesamtschule",
        "1910 Garten-/Landschaftsb",
        "6380 Garage",
        "2230 Alteneinrichtung",
        "5500 Nutz-/Freizeitgärten",
        "5360 Kleintierzucht",
        "5900 Verkehrsgrün",
        "5110 Grünanlage",
    ]
    # Translated the german types in english
    translation_dict = {  # TODO: we need to actually change these values into something we can use with BSO database
        "1650 Baumarkt": "trade",  # trade
        "1610 Möbelmarkt": "trade",  # trade
        "1320 Handel und Dienstl.": "trade",  # trade
        "2204 Bürgerhs.,Stadthalle": "office",  # offices
        "1300 Industrie u. Gewerbe": "other",  # other non-residential buildings
        "1670 Lebensmittelmarkt": "trade",  # trade
        "1310 Lagerfläche": "other",  # other non-residential buildings
        "9990 Freifläche": "other",  # other non-residential buildings
        "2440 Sonst. religiöse Zwecke": "other",  # other non-residential buildings
        "7000 V+E allg.": "other",  # other non-residential buildings
        "2210 Kinderbetreuung": "education",  # education
        "5306 Fußballplatz": "other",  # other non-residential buildings
        "2860 Gesamtschule": "education",  # education
        "1910 Garten-/Landschaftsb": "other",  # other non-residential buildings
        "6380 Garage": "other",  # other non-residential buildings
        "2230 Alteneinrichtung": "health",  # health
        "5500 Nutz-/Freizeitgärten": "other",  # other non-residential buildings
        "5360 Kleintierzucht": "other",  # other non-residential buildings
        "5900 Verkehrsgrün": "other",  # other non-residential buildings
        "5110 Grünanlage": "other",  # other non-residential buildings
    }

    # Translate the building_use column
    gdf["building_use"] = gdf["building_use"].replace(translation_dict)

    defined_typologies = pd.Series(index=gdf.index, dtype="object")
    mask_non_residential = gdf["building_use"].isin(translation_dict.values())

    # Define masks
    mask_sfh = (
        (
            gdf["neighbors_count"].str.len() == 0
        )  # Check if neighbors_count is an empty list
        & (gdf["height"] <= 10)
        & ~mask_non_residential
    )
    mask_ab = (
        (
            gdf["neighbors_count"].str.len() == 0
        )  # Check if neighbors_count is an empty list
        & (gdf["height"] > 6)
        & ~mask_non_residential
    )
    mask_th = (
        (
            gdf["neighbors_count"].str.len() > 0
        )  # Check if neighbors_count is not an empty list
        & (gdf["height"] <= 6)
        & ~mask_non_residential
    )
    mask_mfh = (
        (
            gdf["neighbors_count"].str.len() > 0
        )  # Check if neighbors_count is not an empty list
        & (gdf["height"] > 6)
        & ~mask_non_residential
    )
    # Apply masks
    defined_typologies[mask_sfh] = "sfh"
    defined_typologies[mask_ab] = "ab"
    defined_typologies[mask_th] = "th"
    defined_typologies[mask_mfh] = "mfh"

    # Define other types
    mask_other = ~defined_typologies.isin(["sfh", "ab", "th", "mfh"])
    defined_typologies[mask_other] = gdf["building_use"][mask_other]

    return defined_typologies


def define_building_age(
    gdf: gpd.GeoDataFrame, age_distr: pd.DataFrame, res_types: List[str], verbose=False
) -> pd.Series:
    """
    This function defines the age of the buildings. The age is randomly assigned to the buildings
    based on the statistics that we have on the building stock. This function takes the
    age of the buildings. Then it assigns that percentage to all the buildings.
    E.g. if 10% of the buildings are 100 years old, then all building types will have 10% of the total
    buildings assigned to be 100 years old.
    :param gdf: GeoDataFrame containing the building data with columns 'height', 'neighbors_count', and 'building_use'
    :param age_distr: dictionary containing the age distribution of the buildings
    :res_types: a list of strings that identifies residential buildings in age_distr
    :param verbose: If True, prints the number of buildings assigned to each building type (default: False)
    :return: Series containing the age of all the buildings.
    """
    # Create a series to store the age of the buildings
    defined_age = pd.Series(index=gdf.index, dtype="object")

    df = pd.DataFrame(columns=["random_numbers"])
    df["building_usage"] = gdf["building_usage"]
    # df["random_numbers"] = random_assignment(gdf)
    df["random_numbers"] = np.random.uniform(low=0.0, high=1, size=len(df))
    df["age_code"] = None

    # Define building types
    building_types = gdf["building_usage"].unique()
    # res_mask = df["building_usage"].isin(building_types)
    # non_res_mask = ~df["building_usage"].isin(building_types)

    # here we take the non_residential buildings and we change their
    # designation from whatever it is to "non_res"
    res_mask = df["building_usage"].isin(res_types)
    non_res_mask = ~df["building_usage"].isin(res_types)
    df.loc[non_res_mask, "building_usage"] = "non_res"
    building_types = res_types + ["non_res"]

    # Process only residential buildings for now
    # Cycle through the residential building types contained in the building_types list
    for buildings in building_types:
        mask_types = df["building_usage"] == buildings
        mask_ages = age_distr["building_type"] == buildings

        # Sort age distribution by percentage
        sorted_age_distr = age_distr[mask_ages].sort_values(by="percentage")
        cumulative_percentage = 0

        for percentage, id_code in zip(
            sorted_age_distr["percentage"], sorted_age_distr["tabula_ID"]
        ):
            cumulative_percentage += percentage
            percentage_mask = (
                (df["random_numbers"] <= cumulative_percentage)
                & (df["age_code"].isnull())
                & (df["building_usage"] == buildings)
            )
            if verbose:
                print(
                    f"number of {buildings} assigned to category {id_code}: {percentage_mask.sum()}"
                )
            df.loc[percentage_mask, "age_code"] = id_code

    defined_age = df["age_code"]
    return defined_age


def assign_ceiling_height(
    building_age: pd.Series,
    building_types: pd.Series,
    height_distr: pd.DataFrame,
    res_types: List[str],
) -> pd.Series:
    """This function assigns the ceiling height to the buildings based on the building type and the age.
    :param gdf: GeoDataFrame containing the building data with columns 'building_use' and 'age_code"
    :param height_distr: the DataFrame containing the information about the ceiling height of each combination of
                         building type and age code"""

    # create a series to store the ceiling height of the buildings
    ceiling_height = pd.Series(index=building_age.index, dtype="object")
    df = pd.DataFrame({"age_code": building_age, "building_usage": building_types})

    # we are going to change the building types that are not residential to "non_res"
    non_res_mask = ~df["building_usage"].isin(res_types)
    df.loc[non_res_mask, "building_usage"] = "non_res"
    building_types = df["building_usage"].unique()

    for buildings in building_types:
        mask_ages = height_distr["building_type"] == buildings

        for age, ceiling_height in zip(
            height_distr["age_code"], height_distr[mask_ages]["ceiling_height"]
        ):
            df_age_type = (df["building_usage"] == buildings) & (df["age_code"] == age)
            df.loc[df_age_type, "ceiling_height"] = ceiling_height

    ceiling_height = df["ceiling_height"]

    return ceiling_height


def random_assignment(gdf: gpd.GeoDataFrame):
    """assigns a randomly generated number to each building type in the GeoDataFrame.
    :param gdf: GeoDataFrame containing the building data with columns 'building_use'"""

    building_types_list = gdf["building_usage"].unique()
    random_numbers = pd.Series(index=gdf.index, dtype="object")

    for buildings in building_types_list:
        mask = gdf["building_usage"] == buildings
        random_numbers[mask] = np.random.uniform(low=0.0, high=1, size=len(gdf[mask]))

    return random_numbers


def check_percent_types(gdf: gpd.GeoDataFrame, res_types: List[str]) -> pd.DataFrame:
    """
    A utility to check if the buildings have been assigned a somewhat correct
    percentage. This function subdivides the residential buildings in sub-categories
    (that are selected in res_types). The rest is given as "non_res".

    :param gdf: GeoDataFrame that contains the "building_usage" and "age_code" columns
    :param res_types: List of strings that contains the list of all residential buildings
    :return: DataFrame with the percentage of each combination of building type and age code
    """
    # Filter the relevant columns
    df = gdf.filter(["building_usage", "age_code"])

    # Assign non-residential buildings to "non_res"
    non_res_mask = ~df["building_usage"].isin(res_types)
    df.loc[non_res_mask, "building_usage"] = "non_res"

    # Calculate the counts for each combination of building type and age code
    counts = (
        df.groupby(["building_usage", "age_code"]).size().reset_index(name="counts")
    )

    # Calculate the total number of buildings for each building type
    total_counts = df.groupby("building_usage").size().reset_index(name="total_counts")

    # Merge the total counts with the counts DataFrame
    counts = counts.merge(total_counts, on="building_usage")

    # Calculate the percentage for each combination relative to the total number of buildings of that type
    counts["percentage"] = counts["counts"] / counts["total_counts"]

    return counts


def process_data(
    gdf_path: str,
    filetype: str,
    age_distr: pd.DataFrame,
    ceiling_height_distr: pd.DataFrame,
    res_types: List[str],
) -> gpd.GeoDataFrame:
    # from qgis_utils import get_building_data

    """
    This function processes the data from the GeoDataFrame and assigns the building type, age and ceiling height to the buildings.
    :param gdf: GeoDataFrame containing the building data with columns 'height', 'neighbors_count', 'building_use'
    :param age_distr: DataFrame containing the age distribution of the buildings
    :param height_distr: DataFrame containing the ceiling height distribution of the buildings
    :res_types: a list of strings that identifies residential buildings in age_distr
    :return: GeoDataFrame containing the updated values for the building
    """

    # Retrieve the buildings data from the parquet or gpkg file
    building_data = get_building_data(path=gdf_path, filetype=filetype)

    # save the building usage in a new column
    building_data["building_usage"] = define_building_type(building_data)

    # save the building age in a new column called "age_code"
    building_data["age_code"] = define_building_age(building_data, age_distr, res_types)

    # save the ceiling heights in a new column called "ceiling_height"
    building_types = building_data["building_usage"]
    building_age = building_data["age_code"]

    building_data["ceiling_height"] = assign_ceiling_height(
        building_age=building_age,
        building_types=building_types,
        height_distr=ceiling_height_distr,
        res_types=res_types,
    )

    return building_data


def net_vertical_surface(
    building_id: pd.Series,
    perimeter: pd.Series,
    neighbors: pd.Series,
    border_length: pd.Series,
    building_height: pd.Series,
):
    print("calculating net vertical surface")
    df_data = pd.DataFrame(
        {
            "full_id": building_id,
            "perimeter": perimeter,
            "neighbors": neighbors,
            "border_length": border_length,
            "building_height": building_height,
        }
    )
    net_surfaces = []

    for idx, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        full_id = row["full_id"]
        neighbors_id = row["neighbors"]
        border_lengths = row["border_length"]
        current_building_height = row["building_height"]
        current_perimeter = row["perimeter"]
        gross_surface = current_building_height * current_perimeter
        neighbors_heights = []
        for neighbor in neighbors_id:
            neighbor_height = df_data[df_data["full_id"] == neighbor][
                "building_height"
            ].values[0]
            neighbors_heights.append(neighbor_height)

        shared_surface = []
        for heights, borders in zip(neighbors_heights, border_lengths):
            min_height = min(heights, current_building_height)
            shared_surface.append(min_height * borders)

        net_surface_current = gross_surface - np.sum(shared_surface)
        net_surfaces.append(net_surface_current)

    df_data["net_vertical_surface"] = net_surfaces

    return df_data["net_vertical_surface"]


if __name__ == "__main__":
    from qgis_utils import convert_to_list
    import os
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import sys

    np.random.seed(42)

    # insure we can import the qgis_utils module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    path_parquet = "../building_analysis/building_generator_data/frankfurt_v3.parquet"
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path_parquet))

    # Retrieve the buildings data from the parquet file
    data_gdf = get_building_data(path=abs_path, filetype="parquet")
    raw_data = gpd.read_parquet(path_parquet)

    # save the building useage in a new column
    data_gdf["building_usage"] = define_building_type(data_gdf)
    df_data = pd.DataFrame(data_gdf)

    # create the masks and print the number of buildings of each type
    all_res = df_data["building_usage"].isin(["sfh", "ab", "th", "mfh"])
    mask_sfh = df_data["building_usage"] == "sfh"
    mask_ab = df_data["building_usage"] == "ab"
    mask_th = df_data["building_usage"] == "th"
    mask_mfh = df_data["building_usage"] == "mfh"

    # print(
    #     "Single Family Homes: ",
    #     mask_sfh.sum(),
    #     "share:",
    #     mask_sfh.sum() / all_res.sum(),
    # )
    # print("Terraced Houses: ", mask_th.sum(), "share:", mask_th.sum() / all_res.sum())
    # print(
    #     "Multi Family Homes: ", mask_mfh.sum(), "share:", mask_mfh.sum() / all_res.sum()
    # )
    # print(
    #     "Apartment Buildings: ", mask_ab.sum(), "share:", mask_ab.sum() / all_res.sum()
    # )
    # print("all residential buildings", all_res.sum())

    # Age of the building assignment

    path_age = "../building_analysis/building_generator_data/buildings_age.csv"
    abs_age = os.path.abspath(os.path.join(os.path.dirname(__file__), path_age))

    age_look_up = pd.read_csv(abs_age)

    res_types = ["sfh", "ab", "th", "mfh"]
    df_data["age_code"] = define_building_age(df_data, age_look_up, res_types)

    # this is a check to compare to the original data. Not really the best because
    # AB has different percentages and so do all the non_res_buildings
    # I should do this on the different building types specifically. Maybe make a function
    # that takes the building type and the age distribution and then returns the age distribution
    age_dict = {}
    for age in range(1, 13):
        mask_age = df_data["age_code"] == age
        age_dict.update({age: mask_age.sum()})

    percentage_dict = {}
    total_residential = df_data["age_code"].count()
    for age in range(1, 13):
        percentage = age_dict[age] / total_residential
        percentage_dict.update({age: percentage})
        print(f"percentage of buildings in age category {age}: {percentage_dict[age]}")

    non_res_mask = np.logical_not(df_data["building_usage"].isin(res_types))

    percent_types = check_percent_types(df_data, res_types)
    print(percent_types)

    # Ceiling height assignment

    building_types = df_data["building_usage"]
    building_age = df_data["age_code"]
    path_heights = "../building_analysis/building_generator_data/ceiling_heights.csv"
    abs_heights = os.path.abspath(os.path.join(os.path.dirname(__file__), path_heights))

    ceiling_data = pd.read_csv(abs_heights)
    res_types = ["sfh", "ab", "th", "mfh"]

    df_data["ceiling_height"] = assign_ceiling_height(
        building_age=building_age,
        building_types=building_types,
        height_distr=ceiling_data,
        res_types=res_types,
    )

    # checking now if the assignment of the ceiling height was correct
    df_test_heights = df_data.filter(["building_usage", "age_code", "ceiling_height"])
    non_res_mask = np.logical_not(df_test_heights["building_usage"].isin(res_types))
    df_test_heights.loc[non_res_mask, "building_usage"] = "non_res"
    for building in df_test_heights["building_usage"].unique():
        for ages in range(1, 13):
            mask_age = df_test_heights["age_code"] == ages
            mask_type = df_test_heights["building_usage"] == building
            current_height = df_data[(mask_age) & (mask_type)][
                "ceiling_height"
            ].unique()
            height_from_data = ceiling_data[
                (ceiling_data["building_type"] == building)
                & (ceiling_data["age_code"] == ages)
            ]["ceiling_height"].unique()

            print(
                f" building type {building} and age {ages}, assigned to buildings: {current_height}. Height from data  {height_from_data}"
            )
            # print(f"Building type {building} and age {ages}, ceiling height: {df_data[(mask_age) & (mask_type)]['ceiling_height'].unique()}")

    processed_data = process_data(
        gdf_path=abs_path,
        filetype="parquet",
        age_distr=age_look_up,
        ceiling_height_distr=ceiling_data,
        res_types=res_types,
    )

    building_id = processed_data["full_id"]
    perimeter = processed_data["perimeter"]
    neighbors = processed_data["neighbors_count"]
    border_length = processed_data["border_length"]
    building_height = processed_data["height"]

    wall_surface = net_vertical_surface(
        building_id, perimeter, neighbors, border_length, building_height
    )
    # TODO: https://mapview.region-frankfurt.de/maps4.16/resources/apps/RegioMap/index.html?lang=de&vm=2D&s=2543.2619432300075&r=0&c=471481.09638917685%2C5549632.605473744&l=siedlungsflaechentypologie%280%29%2Cstaedtekartetoc%2C-poi_3d%2C-windmills%2C-gebaeude_1
    # this link has a list of all the building types that we can use to define the building types that are not residential.
