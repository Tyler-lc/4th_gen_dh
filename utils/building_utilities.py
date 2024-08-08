import geopandas as gpd
import pandas as pd
import numpy as np


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
    translation_dict = {
        "1650 Baumarkt": "1650 DIY Store",  # trade
        "1610 Möbelmarkt": "1610 Furniture Store",  # trade
        "1320 Handel und Dienstl.": "1320 Trade and Services",  # trade
        "2204 Bürgerhs.,Stadthalle": "2204 Community Center, Town Hall",  # offices
        "1300 Industrie u. Gewerbe": "1300 Industry and Commerce",  # other non-residential buildings
        "1670 Lebensmittelmarkt": "1670 Grocery Store",  # trade
        "1310 Lagerfläche": "1310 Storage Area",  # other non-residential buildings
        "9990 Freifläche": "9990 Open Space",  # other non-residential buildings
        "2440 Sonst. religiöse Zwecke": "2440 Other Religious Purposes",  # other non-residential buildings
        "7000 V+E allg.": "7000 V+E General",  # other non-residential buildings
        "2210 Kinderbetreuung": "2210 Childcare",  # education
        "5306 Fußballplatz": "5306 Soccer Field",  # other non-residential buildings
        "2860 Gesamtschule": "2860 Comprehensive School",  # education
        "1910 Garten-/Landschaftsb": "1910 Gardening/Landscaping",  # other non-residential buildings
        "6380 Garage": "6380 Garage",  # other non-residential buildings
        "2230 Alteneinrichtung": "2230 Elderly Care Facility",  # health
        "5500 Nutz-/Freizeitgärten": "5500 Utility/Leisure Gardens",  # other non-residential buildings
        "5360 Kleintierzucht": "5360 Small Animal Breeding",  # other non-residential buildings
        "5900 Verkehrsgrün": "5900 Traffic Green",  # other non-residential buildings
        "5110 Grünanlage": "5110 Green Area",  # other non-residential buildings
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


def define_building_age(gdf: gpd.GeoDataFrame, age_distr: dict) -> pd.Series:
    """
    this function defines the age of the buildings. The age is randomly assigned to the buildings
    based on the statistics that we have on the building stock. this function takes the
    age of the buildings. Then it assigns that percentage to all the buildings.
    E.g. if 10% of the buildings are 100 years old, then all building types will have 10% of the total
    buildings assigned to be 100 years old.
    :param gdf: GeoDataFrame containing the building data with columns 'height', 'neighbors_count', and 'building_use'
    :param age_distr: dictionary containing the age distribution of the buildings
    :return: Series containng the age of all the buildings.
    """
    # create a series to store the age of the buildings
    defined_typologies = pd.Series(index=gdf.index, dtype="object")

    return None


if __name__ == "__main__":
    from qgis_utils import get_building_data
    from qgis_utils import convert_to_list
    import os
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import sys

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

    all_res = df_data["building_usage"].isin(["sfh", "ab", "th", "mfh"])
    mask_sfh = df_data["building_usage"] == "sfh"
    mask_ab = df_data["building_usage"] == "ab"
    mask_th = df_data["building_usage"] == "th"
    mask_mfh = df_data["building_usage"] == "mfh"

    print(
        "Single Family Homes: ",
        mask_sfh.sum(),
        "share:",
        mask_sfh.sum() / all_res.sum(),
    )
    print("Terraced Houses: ", mask_th.sum(), "share:", mask_th.sum() / all_res.sum())
    print(
        "Multi Family Homes: ", mask_mfh.sum(), "share:", mask_mfh.sum() / all_res.sum()
    )
    print(
        "Apartment Buildings: ", mask_ab.sum(), "share:", mask_ab.sum() / all_res.sum()
    )
    print("all residential buildings", all_res.sum())

    # TODO: https://mapview.region-frankfurt.de/maps4.16/resources/apps/RegioMap/index.html?lang=de&vm=2D&s=2543.2619432300075&r=0&c=471481.09638917685%2C5549632.605473744&l=siedlungsflaechentypologie%280%29%2Cstaedtekartetoc%2C-poi_3d%2C-windmills%2C-gebaeude_1
    # this link has a list of all the building types that we can use to define the building types that are not residential.
