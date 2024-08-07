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

    defined_typologies = pd.Series(index=gdf.index, dtype="object")
    mask_non_residential = gdf["building_use"].isin(list_non_residential_types)

    # Define masks
    mask_sfh = (
        (len(gdf["neighbors_count"]) == 0)
        & (gdf["height"] <= 6)
        & ~mask_non_residential
    )
    mask_ab = (
        (len(gdf["neighbors_count"]) == 0) & (gdf["height"] > 6) & ~mask_non_residential
    )
    mask_th = (
        (len(gdf["neighbors_count"]) > 0) & (gdf["height"] <= 6) & ~mask_non_residential
    )
    mask_mfh = (
        (len(gdf["neighbors_count"]) > 0) & (gdf["height"] > 6) & ~mask_non_residential
    )

    # Apply masks
    defined_typologies[mask_sfh] = "sfh"
    defined_typologies[mask_ab] = "ab"
    defined_typologies[mask_th] = "th"
    defined_typologies[mask_mfh] = "mfh"

    # Define other types
    mask_other = ~defined_typologies.isin(["sfh", "ab", "th", "mfh"])
    defined_typologies[mask_other] = "other"

    return defined_typologies


if __name__ == "__main__":
    from qgis_utils import get_building_data
    from qgis_utils import convert_to_list
    import os
    import pandas as pd
    import geopandas as gpd
    import numpy as np

    path_parquet = "../Building/data/frankfurt.parquet"
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path_parquet))
    building_data = get_building_data(path=abs_path, filetype="parquet")

    # Define building type
    buildings_defined = define_building_type(building_data)

    # find out differences between the two dataframes

    # TODO: https://mapview.region-frankfurt.de/maps4.16/resources/apps/RegioMap/index.html?lang=de&vm=2D&s=2543.2619432300075&r=0&c=471481.09638917685%2C5549632.605473744&l=siedlungsflaechentypologie%280%29%2Cstaedtekartetoc%2C-poi_3d%2C-windmills%2C-gebaeude_1
    # this link has a list of all the building types that we can use to define the building types that are not residential.

    # Now building_data has the building_type column with the assigned types
