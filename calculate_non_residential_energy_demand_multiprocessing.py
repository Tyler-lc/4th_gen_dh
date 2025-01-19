import numpy as np
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool
import os
from tqdm import tqdm

from building_analysis.Building import Building
from Person.Person import Person


# Refactor the building processing code into a function for multiprocessing
def process_building(
    row,
    temperature,
    irradiation,
    df_soil_temp,
    inside_temp,
    dhw_volumes_folder,
    dir_dhw_volumes,
    dir_dhw_energy,
    dir_space_heating,
):
    building_id = row["full_id"]
    building_type = row["building_usage"] + str(row["age_code"])
    components = row.to_frame().T  # Convert row to DataFrame

    building = Building(
        building_id,
        building_type,
        components,
        temperature,
        irradiation,
        df_soil_temp["V_TE0052"],
        inside_temp["inside_temp"],
        year_start=2019,
    )

    # Perform calculations
    building.thermal_balance()
    building.add_people()
    building.append_water_usage(dhw_volumes_folder)
    building.people_dhw_energy()

    # Collect results
    dhw_volume = building.building_dhw_volume()
    dhw_energy = building.building_dhw_energy()
    space_heating = building.get_useful_demand()

    # Save the results
    dhw_volume_path = os.path.join(dir_dhw_volumes, f"dhw_volume_{building_id}.csv")
    dhw_energy_path = os.path.join(dir_dhw_energy, f"dhw_energy_{building_id}.csv")
    space_heating_path = os.path.join(
        dir_space_heating, f"space_heating_{building_id}.csv"
    )

    dhw_volume.to_csv(dhw_volume_path)
    dhw_energy.to_csv(dhw_energy_path)
    space_heating.to_csv(space_heating_path)

    return dhw_volume, dhw_energy, space_heating


if __name__ == "__main__":
    # Load the buildingstock data
    buildingstock_path = "building_analysis/buildingstock/buildingstock.parquet"
    gdf_buildingstock = gpd.read_parquet(buildingstock_path)

    # Create a new GeoDataFrame for storing the results
    gdf_buildingstock_results = gdf_buildingstock.copy(deep=True)

    # Load irradiation data
    city_name = "Frankfurt_Griesheim_Mitte"
    year_start = 2019
    year_end = 2019
    path_weather = f"irradiation_data/{city_name}_{year_start}_{year_end}/{city_name}_irradiation_data_{year_start}_{year_end}.csv"
    temperature = pd.read_csv(path_weather, usecols=["T2m"])
    irradiation = pd.read_csv(path_weather)
    irradiation = irradiation.filter(regex="G\(i\)")
    dhw_volumes_folder = "building_analysis/dhw_profiles"

    # Load and process soil temperature data
    soil_temp_path = "irradiation_data/Frankfurt_Griesheim_Mitte_2019_2019/Frankfurt_Griesheim_Mitte_soil_temperature_2019_2019.csv"
    df_soil_temp = pd.read_csv(soil_temp_path)
    df_soil_temp.replace(-99.9, np.nan, inplace=True)
    df_soil_temp["V_TE0052"] = df_soil_temp["V_TE0052"].interpolate()

    # Define inside temperature based on time of day
    time_index = pd.date_range(start="2019-01-01", periods=8760, freq="h")
    inside_temp = pd.DataFrame(index=time_index)
    inside_temp["inside_temp"] = 20
    mask_heating = inside_temp.index.hour.isin(range(8, 22))
    inside_temp.loc[~mask_heating, "inside_temp"] = 16

    # Directories for saving results
    dir_dhw_volumes = (
        "building_analysis/results_unrenovated__non_residential/dhw_volumes"
    )
    dir_dhw_energy = "building_analysis/results_unrenovated__non_residential/dhw_energy"
    dir_space_heating = (
        "building_analysis/results_unrenovated__non_residential/space_heating"
    )
    res_mask = gdf_buildingstock_results["building_usage"].isin(
        ["sfh", "mfh", "ab", "th"]
    )

    area_results = pd.DataFrame(
        0,
        index=inside_temp.index,
        columns=["dhw_volume", "dhw_energy", "space_heating"],
    )

    directories = [dir_dhw_volumes, dir_dhw_energy, dir_space_heating]
    for dirs in directories:
        if not os.path.exists(dirs):
            os.makedirs(dirs, exist_ok=True)

    # Use multiprocessing to process each building
    num_cores = os.cpu_count()  # Number of available cores
    with Pool(num_cores) as pool:
        # Prepare the arguments for each row
        args = [
            (
                row,
                temperature,
                irradiation,
                df_soil_temp,
                inside_temp,
                dhw_volumes_folder,
                dir_dhw_volumes,
                dir_dhw_energy,
                dir_space_heating,
            )
            for idx, row in tqdm(
                gdf_buildingstock_results[np.logical_not(res_mask)].iterrows()
            )
        ]

        # Use tqdm to track the progress of starmap
        results = list(
            tqdm(
                pool.starmap(process_building, args),
                total=len(gdf_buildingstock_results),
            )
        )
    i = 0
    tot = len(results)
    n_nan = 0
    for dhw_volume, dhw_energy, space_heating in results:
        i += 1
        dhw_volume_df = pd.DataFrame(dhw_volume.sum(axis=1), columns=["dhw_volume"])
        dhw_volume_df.index = area_results.index
        dhw_energy_df = pd.DataFrame(dhw_energy.sum(axis=1), columns=["dhw_energy"])
        dhw_energy_df.index = area_results.index
        space_heating_df = pd.DataFrame(
            space_heating.sum(axis=1), columns=["space_heating"]
        )
        space_heating.index = area_results.index

        n_nan += dhw_volume_df.isna().sum().values[0]

        area_results["dhw_volume"] = (
            area_results["dhw_volume"] + dhw_volume_df["dhw_volume"]
        )
        area_results["dhw_energy"] = (
            area_results["dhw_energy"] + dhw_energy_df["dhw_energy"]
        )
        area_results["space_heating"] = (
            area_results["space_heating"] + space_heating_df["space_heating"]
        )

        if i % 50 == 0:
            print(f"Processing completed for {i} out of {tot}.")
    print(f"Number of NaN values in dhw_volume: {n_nan}")

    area_results.to_csv(
        "building_analysis/results_unrenovated__non_residential/area_results.csv"
    )
    print(f"Processing completed for {len(results)} buildings.")
    print(f"Processing completed for {len(results)} buildings.")
