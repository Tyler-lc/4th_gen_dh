import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import os
import shutil

from building_analysis.Building import Building
from Person.Person import Person
from utils.misc import get_mask
from building_analysis.building_generator import apply_renovations, need_insulation
from config import (
    RESIDENTIAL_BUILDING_TYPES,
    YEAR_START,
    weather_data_path,
    soil_temperature_path,
    results_dir,
    buildingstock_results_path,
    area_results_path,
)


# first we load the energy demand data that we generated in the calculate_energy_demand.py script
gdf_buildingstock_results = gpd.read_parquet(buildingstock_results_path("unrenovated"))

# from https://doi.org/10.1016/j.enbuild.2024.114324 we know that to accept low-temperature heating
# buildings must achieve a certain energy demand, according to their  type. This information thresholds are
# already baked-in the method need_insulation. We will use this method to determine which buildings need insulation
# the buffer is used so that buildings that are close enough to the target energy demand are not going to renovate
# otherwise we would have a renovation for very few centimeters of insulation
gdf_buildingstock_results["needs_insulation"] = need_insulation(
    gdf_buildingstock_results, buffer=20
)


# Now we know that the DHW doesn't change across scenarios. Also we don't want to recalculate buildings that
# are alraedy efficient enough. So we will copy from the "unrenovated scenario" the data we need.

# Define paths where unrenovated data is stored
_unren = results_dir("unrenovated")
unrenovated_dhw_energy_path = str(_unren / "dhw_energy")
unrenovated_dhw_volume_path = str(_unren / "dhw_volumes")
unrenovated_space_heating_path = str(_unren / "space_heating")

# Define the paths where the renovated data will be stored
_ren = results_dir("renovated")
renovated_dhw_energy_path = str(_ren / "dhw_energy")
renovated_dhw_volume_path = str(_ren / "dhw_volume")
renovated_space_heating_path = str(_ren / "space_heating")

# Ensure the destination directories exist
os.makedirs(renovated_dhw_energy_path, exist_ok=True)
os.makedirs(renovated_dhw_volume_path, exist_ok=True)
os.makedirs(renovated_space_heating_path, exist_ok=True)

# Copy DHW data
print(f"Moving dhw_volume data from unrenovated scenario")
for file_name in tqdm(os.listdir(unrenovated_dhw_energy_path)):
    shutil.copy(
        os.path.join(unrenovated_dhw_energy_path, file_name), renovated_dhw_energy_path
    )

print("Moving dhw energy demand from unrenovated scenario")
for file_name in tqdm(os.listdir(unrenovated_dhw_volume_path)):
    shutil.copy(
        os.path.join(unrenovated_dhw_volume_path, file_name), renovated_dhw_volume_path
    )

print(
    "Moving space heating data from unrenovated scenario, for buildings that do not need renovation \
    and updating the paths in the GeoDataFrame"
)
for idx, row in gdf_buildingstock_results.iterrows():
    building_id = row["full_id"]
    if not row["needs_insulation"]:
        # Copy space heating data
        source_file = os.path.join(
            unrenovated_space_heating_path, f"space_heating_{building_id}.csv"
        )
        dest_file = os.path.join(
            renovated_space_heating_path, f"space_heating_{building_id}.csv"
        )
        shutil.copy(source_file, dest_file)

        # Update DataFrame paths
        gdf_buildingstock_results.at[idx, "dhw_volume_path"] = os.path.join(
            renovated_dhw_volume_path, f"dhw_volume_{building_id}.csv"
        )
        gdf_buildingstock_results.at[idx, "dhw_energy_path"] = os.path.join(
            renovated_dhw_energy_path, f"dhw_energy_{building_id}.csv"
        )
        gdf_buildingstock_results.at[idx, "space_heating_path"] = dest_file


# setting up some hyperparameters and directories
res_mask = gdf_buildingstock_results["building_usage"].isin(RESIDENTIAL_BUILDING_TYPES)
sim = "renovated"
size = "whole_buildingstock"
mask = get_mask(size, res_mask)  # type:ignore

dir_space_heating = str(results_dir(sim, size) / "space_heating")
os.makedirs(dir_space_heating, exist_ok=True)

# this is the initial insulation thickness. The renovation will add this thickness to the
# buildings that need insulation. If after application the building still does not meet the energy demand
# the thickness will be increased by incremental_insulation
insulation_thickness = 50  # mm
thermal_conductivity = 0.02  # W/m2K typical of phenolic foams
incremental_insulation = 50  # mm


# importing the weather data (temperature, irradiation, soil temperature)
path_weather = weather_data_path()
temperature = pd.read_csv(path_weather, usecols=["T2m"])
irradiation = pd.read_csv(path_weather)
irradiation = irradiation.filter(regex="G\(i\)")

# the soil temperature has some missing data. We will interpolate it
df_soil_temp = pd.read_csv(soil_temperature_path())

# missing data are represented by -99.9. So we replace them with NaN values. This allows fill by interpolation
df_soil_temp.replace(-99.9, np.nan, inplace=True)

# now interpolate the missing values
df_soil_temp["V_TE0052"] = df_soil_temp["V_TE0052"].interpolate()

# we are also setting the inside temperature to be a bit variable. We set it to be 20 °C from 8 am to 10pm
# and 17 °C anywhere else (basically night time)
time_index = pd.date_range(start=f"{YEAR_START}-01-01", periods=8760, freq="h")
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


# BUG: area_results["space_heating"] only accumulates demand from buildings
# that need insulation (inside the while loop). Buildings already below the
# threshold are never added, resulting in ~30% undercount (37,614 vs 54,184 MWh).
# The downstream scenario scripts no longer depend on this CSV (they use
# utils/area_demand.py instead), so this is non-critical. Fix by initialising
# area_results with the unrenovated demand and replacing only renovated buildings.
# now we will apply the renovation to the buildings that need it.
while gdf_buildingstock_results["needs_insulation"].sum() > 0:

    gdf_buildingstock_results = apply_renovations(
        gdf_buildingstock_results, insulation_thickness, thermal_conductivity
    )

    to_renovate = gdf_buildingstock_results["needs_insulation"].sum()
    mask = gdf_buildingstock_results["needs_insulation"]

    for idx, row in tqdm(gdf_buildingstock_results[mask].iterrows(), total=to_renovate):
        if row["needs_insulation"]:

            building_id = row["full_id"]
            building_type = row["building_usage"] + str(row["age_code"])
            components_df = row.to_frame().T
            building = Building(
                building_id,
                building_type,
                components_df,
                temperature,
                irradiation,
                df_soil_temp["V_TE0052"],
                inside_temp["inside_temp"],
                YEAR_START,
            )
            # perform calculation for useful space heating demand and saving it in a dataframe
            building.thermal_balance()
            space_heating_df = building.get_useful_demand()

            # saving the data to csv
            space_heating_path = os.path.join(
                dir_space_heating, f"space_heating_{building_id}.csv"
            )
            space_heating_df.to_csv(space_heating_path)

            # saving the path to csv in the dataframe for easier retrieval later
            gdf_buildingstock_results.loc[idx, "space_heating_path"] = (
                space_heating_path
            )

            # saving the yearly space heating demand in the DF
            gdf_buildingstock_results.loc[idx, "yearly_space_heating"] = (
                building.get_sum_useful_demand()
            )

            # saving the specific UED in the DF [kWh/(m2year)]
            gdf_buildingstock_results.loc[idx, "specific_ued"] = (
                building.get_specific_ued()
            )

            # adding the space heating demand to the area_results
            space_heating_df.index = area_results.index
            area_results["space_heating"] += space_heating_df.sum(axis=1)

    # now we check again if all buildings need renovation or not. If they do, we increase the insulation thickness
    gdf_buildingstock_results["needs_insulation"] = need_insulation(
        gdf_buildingstock_results, buffer=20
    )
    insulation_thickness += incremental_insulation

# now we can save the results to a file
gdf_buildingstock_results.to_parquet(buildingstock_results_path("renovated"))

area_results_unrenovated = pd.read_csv(area_results_path("unrenovated"), index_col=0)
area_results_unrenovated.index = pd.to_datetime(area_results_unrenovated.index)

area_results["dhw_energy"] = area_results_unrenovated["dhw_energy"]
area_results["dhw_volume"] = area_results_unrenovated["dhw_volume"]

area_results.to_csv(area_results_path(sim))
