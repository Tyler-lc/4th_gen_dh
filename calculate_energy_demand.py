import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import os
import sys
from multiprocessing import Pool
from tqdm import tqdm

from building_analysis.Building import Building
from Person.Person import Person
from utils.misc import get_mask

# first we load the buildingstock data that we generated in the create_buildingstock.py script
buildingstock_path = "building_analysis/buildingstock/buildingstock.parquet"
gdf_buildingstock = gpd.read_parquet(buildingstock_path)

# we now create a new geodataframe that will contain the results of the energy demand calculations
gdf_buildingstock_results = gdf_buildingstock.copy(deep=True)

# let's get the irradiation data first
city_name = "Frankfurt_Griesheim_Mitte"
year_start = 2019
year_end = 2019
path_weather = f"irradiation_data/{city_name}_{year_start}_{year_end}/{city_name}_irradiation_data_{year_start}_{year_end}.csv"
temperature = pd.read_csv(path_weather, usecols=["T2m"])
irradiation = pd.read_csv(path_weather)
irradiation = irradiation.filter(regex="G\(i\)")

# now the soil temperature
soil_temp_path = "irradiation_data/Frankfurt_Griesheim_Mitte_2019_2019/Frankfurt_Griesheim_Mitte_soil_temperature_2019_2019.csv"
df_soil_temp = pd.read_csv(soil_temp_path)

# we know the data for the soil temperature is lacking some data, so we will fill it with the interpolate
soil_temp_path = "irradiation_data/Frankfurt_Griesheim_Mitte_2019_2019/Frankfurt_Griesheim_Mitte_soil_temperature_2019_2019.csv"
df_soil_temp = pd.read_csv(soil_temp_path)

# missing data are represented by -99.9. So we replace them with NaN values. This allows fill by interpolation
df_soil_temp.replace(-99.9, np.nan, inplace=True)
print(
    f"total number of NaN values in soil temperature before fix: {df_soil_temp['V_TE0052'].isna().sum()}"
)

# now interpolate the missing values
df_soil_temp["V_TE0052"] = df_soil_temp["V_TE0052"].interpolate()
print(
    f"total number of NaN values in soil temperature after fix: {df_soil_temp['V_TE0052'].isna().sum()}"
)

# we are also setting the inside temperature to be a bit variable. We set it to be 20 °C from 8 am to 10pm
# and 17 °C anywhere else (mostly night time)
time_index = pd.date_range(start="2019-01-01", periods=8760, freq="h")
inside_temp = pd.DataFrame(index=time_index)
inside_temp["inside_temp"] = 20
mask_heating = inside_temp.index.hour.isin(range(8, 22))
inside_temp.loc[np.logical_not(mask_heating), "inside_temp"] = 17

# setting up the folder where the dhw_profiles are stored
dhw_volumes_folder = "building_analysis/dhw_profiles"

res_mask = gdf_buildingstock_results["building_usage"].isin(["sfh", "mfh", "ab", "th"])

# Initialize DataFrames for storing results
space_heating_df = pd.DataFrame()
dhw_volume_df = pd.DataFrame()
dhw_energy_df = pd.DataFrame()

# we only have DHW for residential buildings. Hence we apply the residential mask to check
# if all buildings actually have dhw_profiles for each person inside.
dont_exist = []
exist = []
for idx, building in gdf_buildingstock_results[res_mask].iterrows():
    people_ids = building["people_id"]
    for person_id in people_ids:
        file_path = os.path.join(dhw_volumes_folder, f"{person_id}.csv")
        if os.path.exists(file_path):
            # print(
            #     f"File exists for person {person_id} in building {building['full_id']}."
            # )
            exist.append(person_id)
        else:
            # print(
            #     f"File does not exist for person {person_id} in building {building['full_id']}."
            # )
            dont_exist.append(person_id)
print(f"Total number of people that exist: {len(exist)}")
print(f"Total number of people that do not exist: {len(dont_exist)}")
if len(dont_exist) > 0:
    sys.exit("There are people that do not have a dhw profile. Please check the data.")

sim = "unrenovated"
size = "whole_buildingstock"


mask = get_mask(size, res_mask)

dir_dhw_volumes = f"building_analysis/results/{sim}_{size}/dhw_volumes"
dir_dhw_energy = f"building_analysis/results/{sim}_{size}/dhw_energy"
dir_space_heating = f"building_analysis/results/{sim}_{size}/space_heating"

directories = [dir_dhw_volumes, dir_dhw_energy, dir_space_heating]

for dirs in directories:
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)


area_results = pd.DataFrame(
    0,
    index=inside_temp.index,
    columns=["dhw_volume", "dhw_energy", "space_heating"],
)

for idx, row in tqdm(gdf_buildingstock_results[mask].iterrows(), total=mask.sum()):

    building_id = row["full_id"]
    building_type = row["building_usage"] + str(row["age_code"])
    components = row.to_frame().T  # we need to pass the row as a dataframe
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

    building.thermal_balance()
    building.add_people()
    building.append_water_usage(dhw_volumes_folder)
    building.people_dhw_energy()
    dhw_volume_df = building.building_dhw_volume()
    dhw_energy_df = building.building_dhw_energy()
    space_heating_df = building.get_useful_demand()

    dhw_volume_path = os.path.join(dir_dhw_volumes, f"dhw_volume_{building_id}.csv")
    dhw_energy_df_path = os.path.join(dir_dhw_energy, f"dhw_energy_{building_id}.csv")
    space_heating_path = os.path.join(
        dir_space_heating, f"space_heating_{building_id}.csv"
    )

    dhw_volume_df.to_csv(dhw_volume_path)
    dhw_energy_df.to_csv(dhw_energy_df_path)
    space_heating_df.to_csv(space_heating_path)

    # adding the path of the results to the gdf_buildingstock_results for easier retrieval later on
    gdf_buildingstock_results.loc[idx, "dhw_volume_path"] = dhw_volume_path
    gdf_buildingstock_results.loc[idx, "dhw_energy_path"] = dhw_energy_df_path
    gdf_buildingstock_results.loc[idx, "space_heating_path"] = space_heating_path

    # adding the yearly results to the gdf_buildingstock_results for each building
    gdf_buildingstock_results.loc[idx, "yearly_dhw_volume"] = (
        building.get_dhw_sum_volume()
    )
    gdf_buildingstock_results.loc[idx, "yearly_dhw_energy"] = (
        building.get_dhw_sum_energy()
    )
    gdf_buildingstock_results.loc[idx, "yearly_space_heating"] = (
        building.get_sum_useful_demand()
    )

    gdf_buildingstock_results.loc[idx, "specific_ued"] = building.get_specific_ued()

    # creating a csv with the overall results of the area
    # we make sure that the indexes of all DFs are the same
    # otherwise the code breaks
    dhw_volume_df.index = area_results.index
    dhw_energy_df.index = area_results.index
    space_heating_df.index = area_results.index

    # now we save the results in the area_results dataframe
    area_results["dhw_volume"] += dhw_volume_df.sum(axis=1)
    area_results["dhw_energy"] += dhw_energy_df.sum(axis=1)
    area_results["space_heating"] += space_heating_df.sum(axis=1)


path_save_results = (
    f"building_analysis/results/{sim}_{size}/buildingstock_results.parquet"
)
gdf_buildingstock_results.to_parquet(path_save_results)
print(f"Results saved to {path_save_results}")

path_area_results = f"building_analysis/results/{sim}_{size}/area_results.csv"
area_results.to_csv(path_area_results)
