import geopandas as gpd
import pandas as pd
import numpy as np
from typing import List, Dict


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
        "2204 Bürgerhs.,Stadthalle": "offices",  # offices
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
    gdf: gpd.GeoDataFrame, age_distr: pd.DataFrame, res_types: List[str]
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
            print(
                f"number of {buildings} assigned to category {id_code}: {percentage_mask.sum()}"
            )
            df.loc[percentage_mask, "age_code"] = id_code

    defined_age = df["age_code"]
    return defined_age


def random_assignment(gdf: gpd.GeoDataFrame):
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


if __name__ == "__main__":
    from qgis_utils import get_building_data
    from qgis_utils import convert_to_list
    import os
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import sys

    np.random.seed(42069)

    # insure we can import the qgis_utils module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    path_parquet = "../Building/data/frankfurt.parquet"
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path_parquet))

    # Retrieve the buildings data from the parquet file
    building_data = get_building_data(path=abs_path, filetype="parquet")

    # save the building useage in a new column
    building_data["building_usage"] = define_building_type(building_data)
    df_data = pd.DataFrame(building_data)

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

    path_age = "../Building/data/buildings_age.csv"
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

    # TODO: https://mapview.region-frankfurt.de/maps4.16/resources/apps/RegioMap/index.html?lang=de&vm=2D&s=2543.2619432300075&r=0&c=471481.09638917685%2C5549632.605473744&l=siedlungsflaechentypologie%280%29%2Cstaedtekartetoc%2C-poi_3d%2C-windmills%2C-gebaeude_1
    # this link has a list of all the building types that we can use to define the building types that are not residential.
