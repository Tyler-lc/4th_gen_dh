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
from heat_supply.carnot_efficiency import carnot_cop


########################################################
# in this file we are going to calculate the energy demand for the booster scenario
# we will use the COP method for the heat pump energy demand. The temperatures will be:
# 50 °C at the inlet and 85 (?) at the outlet. The approach temperature will be 5 °C.
########################################################

### safety factor for the heat pump size. This is just an oversizing factor really.
safety_factor = 1.2

### we set the supply temperature to the buildings (i.e. outlet temperature)
t_supply = 85  # °C

### and we also set the inlet temperature to the heat pump. Which in this case
### is the temperature coming from the district heating network
t_grid = 50  # °C


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


# setting up some hyperparameters and directories
res_mask = gdf_buildingstock_results["building_usage"].isin(["sfh", "mfh", "ab", "th"])
sim = "booster"
size = "whole_buildingstock"
mask = get_mask(size, res_mask)  # type:ignore

dir_space_heating = f"building_analysis/results/{sim}_{size}/space_heating"
os.makedirs(dir_space_heating, exist_ok=True)


# for now we can put a pause on the area results. Let's do that later

### what we need now is to calculate the heat pump electricity demand:
### we have the COP function calculator, so we can use that.
### we first need to retrieve the data from the unrenovated scenario.
### inside our gdf_buildingstock_results we have the path to the space heating results.
### let's start by iterating over the gdf and iterate over the space heating data.

### First we need to create a new column in the gdf_buildingstock_results to store the heat pump size
gdf_buildingstock_results["heat_pump_size [kW]"] = 0
gdf_buildingstock_results["peak_demand_on_dh_grid [kW]"] = 0

### we also need to create a new file to store the heat pump electricity demand
### along with the folder path
booster_space_heating_path = (
    f"building_analysis/results/{sim}_{size}/space_heating_{sim}"
)
os.makedirs(booster_space_heating_path, exist_ok=True)

booster_dhw_path = f"building_analysis/results/{sim}_{size}/dhw_energy_{sim}"
os.makedirs(booster_dhw_path, exist_ok=True)

### we create a temporary dataframe to store the heat pump electricity demand
### the thermal demand of the HP and its thermal output.
### We export this to a csv file later in the loop.
### First let's retrieve a single space heating file to get the index
space_heating_path = gdf_buildingstock_results.loc[0, "space_heating_path"]
space_heating = pd.read_csv(space_heating_path, index_col=0)

temp_df = pd.DataFrame(
    index=space_heating.index,
    columns=[
        "cop_hourly",
        "el_demand [kWh]",
        "demand_on_dh_grid [kWh]",
        "thermal_output [kWh]",
    ],
)

### we create a temporary dataframe to store the heat pump electricity demand
### that we will export to a csv file

for idx, row in gdf_buildingstock_results.iterrows():

    ### initialize the temp_df with zeros
    temp_df.iloc[:] = 0

    building_id = row["full_id"]

    ### loading the space heating data
    space_heating_path = row["space_heating_path"]
    space_heating = pd.read_csv(space_heating_path, index_col=0)
    space_heating.index = pd.to_datetime(space_heating.index)

    ### loading the dhw data
    dhw_energy_path = row["dhw_energy_path"]
    dhw_energy = pd.read_csv(dhw_energy_path, index_col=0)
    dhw_energy.index = pd.to_datetime(dhw_energy.index)

    ### let's sum the dhw and SH demands together to assess total
    ### HP size needed.
    total_demand = space_heating + dhw_energy
    max_demand = total_demand.max()
    hp_size = max_demand * safety_factor
    gdf_buildingstock_results.loc[idx, "heat_pump_size [kW]"] = hp_size

    ### now we need to calculate the COP for these heat pumps
    ### luckily we have the COP function calculator, so we can use that.
    ### we need to iterate over the space heating data and calculate the COP for each hour
    COP_hourly = carnot_cop(t_supply, t_grid, approach_temperature=5)
    el_demand = space_heating.values / COP_hourly.values

    # original COP eq is COP = Qh / (Qh - Qc)
    # solving for Qc ->
    # Qh*COP - Qc*COP = Qh ->   Qc = Qh - Qh/COP -> Qc = Qh * (1- 1/COP)
    # this is the demand that the District Heating grid needs to cover.
    # The rest will be provided by the electricity of the booster.
    # because of how EMBERS module works, we need to calculate the maximum
    # value of this demand. This will be then the "demand" in embers.
    dh_grid_demand = space_heating.values * (1 - 1 / COP_hourly.values)
    gdf_buildingstock_results.loc[idx, "peak_demand_on_dh_grid [kW]"] = (
        dh_grid_demand.max()
    )

    ### we can now build and then store the temp_df
    temp_df["cop_hourly"] = COP_hourly
    temp_df["el_demand [kWh]"] = el_demand
    temp_df["thermal_output [kWh]"] = space_heating
    temp_df["demand_on_dh_grid [kWh]"] = dh_grid_demand

    temp_df.to_csv(f"{booster_space_heating_path}/{building_id}_{sim}.csv")

    ### also we need to be able to possibly recover these data later on.
    ### so we will store the path to the csv files in the gdf_buildingstock_results
    gdf_buildingstock_results.loc[idx, "space_heating_booster_path"] = (
        f"{booster_space_heating_path}/{building_id}_{sim}.csv"
    )

#### the rest of the code needs to be double checked right now.


# now we can save the results to a file
gdf_buildingstock_results.to_parquet(
    f"building_analysis/results/{sim}_{size}/buildingstock_{sim}_{size}_results.parquet"
)

area_results_urenovated_path = (
    "building_analysis/results/unrenovated_whole_buildingstock/area_results.csv"
)

# area_results = pd.DataFrame(
#     0,
#     index=inside_temp.index,
#     columns=["dhw_volume", "dhw_energy", "space_heating"],
# )

area_results_unrenovated = pd.read_csv(area_results_urenovated_path, index_col=0)
area_results_unrenovated.index = pd.to_datetime(area_results_unrenovated.index)

area_results["dhw_energy"] = area_results_unrenovated["dhw_energy"]
area_results["dhw_volume"] = area_results_unrenovated["dhw_volume"]

area_results.to_csv(
    "building_analysis/results/renovated_whole_buildingstock/area_results_renovated.csv"
)
