import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import os

from building_analysis.Building import Building
from Person.Person import Person

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
dhw_volumes_folder = "building_analysis/dhw_profiles"

res_mask = gdf_buildingstock_results["building_usage"].isin(["sfh", "mfh", "ab", "th"])


dont_exist = []
exist = []
for idx, building in gdf_buildingstock_results[res_mask].iterrows():
    people_ids = building["people_id"]
    for person_id in people_ids:
        file_path = os.path.join(dhw_volumes_folder, f"{person_id}.csv")
        if os.path.exists(file_path):
            print(
                f"File exists for person {person_id} in building {building['full_id']}."
            )
            exist.append(person_id)
        else:
            print(
                f"File does not exist for person {person_id} in building {building['full_id']}."
            )
            dont_exist.append(person_id)
print(f"Total number of people that exist: {len(exist)}")
print(f"Total number of people that do not exist: {len(dont_exist)}")

for idx, row in tqdm(gdf_buildingstock_results.iterrows()):
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
    building.building_dhw_volume()
    building.building_dhw_energy()

# TODO we need to save the results somehow now.
