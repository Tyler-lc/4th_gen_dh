import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import os
import sys
import shutil
from multiprocessing import Pool
from tqdm import tqdm

from building_analysis.Building import Building
from Person.Person import Person
from utils.misc import get_mask
from building_analysis.building_generator import apply_renovations, need_insulation

########################################################
# in this file we are going to calculate the energy demand for the booster scenario
# we will use the COP method for the heat pump energy demand. The temperatures will be:
# 50 °C at the inlet and 85 (?) at the outlet. The approach temperature will be 5 °C.
########################################################


# first we load the energy demand data that we generated in the calculate_energy_demand.py script
path_load_results = "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results.parquet"
gdf_buildingstock_results = gpd.read_parquet(path_load_results)


# Now we know that the DHW doesn't change across scenarios. Also we don't want to recalculate buildings that
# are alraedy efficient enough. So we will copy from the "unrenovated scenario" the data we need.

# Define paths where unrenovated data is stored
unrenovated_dhw_energy_path = (
    "building_analysis/results/unrenovated_whole_buildingstock/dhw_energy"
)
unrenovated_dhw_volume_path = (
    "building_analysis/results/unrenovated_whole_buildingstock/dhw_volumes"
)
unrenovated_space_heating_path = (
    "building_analysis/results/unrenovated_whole_buildingstock/space_heating"
)

# Define the paths where the renovated data will be stored
booster_dhw_energy_path = (
    "building_analysis/results/booster_whole_buildingstock/dhw_energy"
)
booster_dhw_volume_path = (
    "building_analysis/results/booster_whole_buildingstock/dhw_volume"
)
booster_space_heating_path = (
    "building_analysis/results/booster_whole_buildingstock/space_heating"
)

# Ensure the destination directories exist
os.makedirs(booster_dhw_energy_path, exist_ok=True)
os.makedirs(booster_dhw_volume_path, exist_ok=True)
os.makedirs(booster_space_heating_path, exist_ok=True)

# Copy DHW data
print(f"Moving dhw_energy data from unrenovated scenario")
for file_name in tqdm(os.listdir(unrenovated_dhw_energy_path)):
    shutil.copy(
        os.path.join(unrenovated_dhw_energy_path, file_name), booster_dhw_energy_path
    )

print(f"Moving dhw_volume data from unrenovated scenario")
for file_name in tqdm(os.listdir(unrenovated_dhw_volume_path)):
    shutil.copy(
        os.path.join(unrenovated_dhw_volume_path, file_name), booster_dhw_volume_path
    )


for idx, row in gdf_buildingstock_results.iterrows():
    building_id = row["full_id"]
    if not row["needs_insulation"]:
        # Copy space heating data
        source_file = os.path.join(
            booster_space_heating_path, f"space_heating_{building_id}.csv"
        )
        dest_file = os.path.join(
            booster_space_heating_path, f"space_heating_{building_id}.csv"
        )
        shutil.copy(source_file, dest_file)

        # Update DataFrame paths
        gdf_buildingstock_results.at[idx, "dhw_volume_path"] = os.path.join(
            booster_dhw_volume_path, f"dhw_volume_{building_id}.csv"
        )
        gdf_buildingstock_results.at[idx, "dhw_energy_path"] = os.path.join(
            booster_dhw_energy_path, f"dhw_energy_{building_id}.csv"
        )
        gdf_buildingstock_results.at[idx, "space_heating_path"] = dest_file


# setting up some hyperparameters and directories
res_mask = gdf_buildingstock_results["building_usage"].isin(["sfh", "mfh", "ab", "th"])
sim = "booster"
size = "whole_buildingstock"
mask = get_mask(size, res_mask)  # type:ignore

dir_space_heating = f"building_analysis/results/{sim}_{size}/space_heating"
os.makedirs(dir_space_heating, exist_ok=True)


# importing the weather data (temperature, irradiation, soil temperature)
city_name = "Frankfurt_Griesheim_Mitte"
year_start = 2019
year_end = 2019
path_weather = f"irradiation_data/{city_name}_{year_start}_{year_end}/{city_name}_irradiation_data_{year_start}_{year_end}.csv"
temperature = pd.read_csv(path_weather, usecols=["T2m"])
irradiation = pd.read_csv(path_weather)
irradiation = irradiation.filter(regex="G\(i\)")

# the soil temperature has some missing data. We will interpolate it
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
# and 17 °C anywhere else (basically night time)
time_index = pd.date_range(start="2019-01-01", periods=8760, freq="h")
inside_temp = pd.DataFrame(index=time_index)
inside_temp["inside_temp"] = 20
mask_heating = inside_temp.index.hour.isin(range(8, 22))
inside_temp.loc[np.logical_not(mask_heating), "inside_temp"] = 17

# in this case we will not need to change the dhw profiles, since that remains the same
# across the renovated and unrenovated scenario.

# create an empty dataframe to store area results
area_results = pd.DataFrame(
    0,
    index=inside_temp.index,
    columns=["dhw_volume", "dhw_energy", "space_heating"],
)


# now we can save the results to a file
gdf_buildingstock_results.to_parquet(
    "building_analysis/results/renovated_whole_buildingstock/buildingstock_renovated_results.parquet"
)

area_results_urenovated_path = (
    "building_analysis/results/unrenovated_whole_buildingstock/area_results.csv"
)
area_results_unrenovated = pd.read_csv(area_results_urenovated_path, index_col=0)
area_results_unrenovated.index = pd.to_datetime(area_results_unrenovated.index)

area_results["dhw_energy"] = area_results_unrenovated["dhw_energy"]
area_results["dhw_volume"] = area_results_unrenovated["dhw_volume"]

area_results.to_csv(
    "building_analysis/results/renovated_whole_buildingstock/area_results_renovated.csv"
)
