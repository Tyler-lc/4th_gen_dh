import geopandas as gpd
import pandas as pd
import numpy as np


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

    full_id: str = building_data["full_id"]
    osm_id: int = building_data["osm_id"]
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
    geometry = building_data["geometry"]

    # QGIS writes these lists as strings, so I need to convert them into lists before i proceed
    building_neighbors = building_data["neighbors"].apply(convert_to_list)
    border_length = building_data["border_len"].apply(convert_to_list)

    data = {
        "full_id": full_id,
        "osm_id": osm_id,
        "type": building_type,
        "roof_slope": roof_slope,
        "roof_surface": roof_surface,
        "height": building_height,
        "is_isolated": is_isolated,
        "wall_surface": wall_surface,
        "plot_area": plot_area,
        "perimiter": building_perimiter,
        "volume": volume,
        "n_sides": n_sides,
        "neighbors_count": building_neighbors,
        "border_length": border_length,
        "building_use": building_use,
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
    import os
    import pandas as pd
    import geopandas as gpd
    import numpy as np

    path = "../Building/data/frankfurt.parquet"
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    original_data = gpd.read_parquet(abs_path)
    building_data = get_building_data(path=abs_path, filetype="parquet")

    print("Building data loaded successfully!")
