import geopandas as gpd
import pandas as pd
import numpy as np


def get_building_data(path: str) -> gpd.GeoDataFrame:
    """
    This function reads the building data from the provided geopackage and returns a DataFrame.

    :param path: Path to the building data
    :return: DataFrame containing the building data
    """
    building_data = gpd.read_file(path)
    full_id: str = building_data["full_id"]
    osm_id: int = building_data["osm_id"]
    building_type: str = building_data["building"]
    roof_slope: float = building_data["slope_mean"]
    roof_surface: float = building_data["roof_sum"]
    building_height: float = building_data["height_mean_point"]
    is_isolated: int = building_data["is_isolated"]
    wall_surface: float = building_data["vertical_s"]
    plot_area: float = building_data["area"]
    building_perimiter: float = building_data["perimeter"]
    volume: float = building_data["height_sum_raster"]
    n_sides: int = building_data["n_all_sides"]
    geometry = building_data["geometry"]

    # QGIS writes these lists as strings, so I need to convert them into lists before i proceed
    building_neighbors = building_data["neighbors"].apply(convert_to_list)
    border_length = building_data["border_len"].apply(convert_to_list)

    data = {
        "full_id": full_id,
        "osm_id": osm_id,
        "roof_slope": roof_slope,
        "roof_surface": roof_surface,
        "building_height": building_height,
        "is_isolated": is_isolated,
        "wall_surface": wall_surface,
        "plot_area": plot_area,
        "building_perimiter": building_perimiter,
        "volume": volume,
        "n_sides": n_sides,
        "building_neighbors": building_neighbors,
        "border_length": border_length,
        "geometry": geometry,
    }

    export_data = gpd.GeoDataFrame(data)

    return export_data


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


if __name__ == "__main__":
    path = "../Building/data/geometry_data_sides_neighbors.gpkg"
    building_data = get_building_data(path)
    print("Building data loaded successfully!")
