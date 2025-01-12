import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import os
import sys
import shutil
from multiprocessing import Pool
from tqdm import tqdm
import warnings

from building_analysis.Building import Building
from Person.Person import Person
from utils.misc import get_mask
from building_analysis.building_generator import apply_renovations, need_insulation
from heat_supply.carnot_efficiency import carnot_cop

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

########################################################
# in this file we are going to calculate the energy demand for the booster scenario
# we will use the COP method for the heat pump energy demand. The temperatures will be:
# 50 °C at the inlet and 85 (?) at the outlet. The approach temperature will be 5 °C.
########################################################
temperatures = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
for t_grid in temperatures:
    ### safety factor for the heat pump size. This is just an oversizing factor really. It may not be needed here
    safety_factor = 1.2

    ### we set the supply temperature to the buildings (i.e. outlet temperature). maybe not needed here
    t_supply = 85  # °C

    ### and we also set the inlet temperature to the booster heat pump. Which in this case
    ### is the temperature coming from the district heating network
    # t_grid = 55  # °C

    # first we load the energy demand data that we generated in the calculate_energy_demand.py script
    path_load_results = "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results_unrenovated.parquet"
    gdf_buildingstock_results = gpd.read_parquet(path_load_results)

    # Now we know that the DHW doesn't change across scenarios. Also we do not need to recalculate the
    # space heating demand because in this case it remains the same.
    # So we will copy from the "unrenovated scenario" the data we need.

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

    # Define the paths where the booster data will be stored
    booster_dhw_energy_path = f"building_analysis/results/booster/booster_whole_buildingstock_{t_grid}/dhw_energy"
    booster_dhw_volume_path = f"building_analysis/results/booster/booster_whole_buildingstock_{t_grid}/dhw_volume"
    booster_space_heating_path = f"building_analysis/results/booster/booster_whole_buildingstock_{t_grid}/space_heating"

    def verify_files_match(source_path: str, dest_path: str) -> bool:
        """
        Verify that all files from source directory exist in destination directory
        with matching names and count.

        Parameters:
        -----------
        source_path : str
            Path to source directory
        dest_path : str
            Path to destination directory

        Returns:
        --------
        bool
            True if all files match, False otherwise
        """
        # Get sets of filenames from both directories
        source_files = set(os.listdir(source_path))
        if not os.path.exists(dest_path):
            return False
        dest_files = set(os.listdir(dest_path))

        # Check if all source files exist in destination
        return len(source_files) > 0 and source_files == dest_files

    # DHW Energy files
    print("Checking/copying DHW energy data from unrenovated scenario")
    if verify_files_match(unrenovated_dhw_energy_path, booster_dhw_energy_path):
        print("DHW energy files already present, skipping copy")
    else:
        print("Copying DHW energy files...")
        os.makedirs(booster_dhw_energy_path, exist_ok=True)
        for file_name in tqdm(os.listdir(unrenovated_dhw_energy_path)):
            shutil.copy(
                os.path.join(unrenovated_dhw_energy_path, file_name),
                booster_dhw_energy_path,
            )

    # DHW Volume files
    print("Checking/copying DHW volume data from unrenovated scenario")
    if verify_files_match(unrenovated_dhw_volume_path, booster_dhw_volume_path):
        print("DHW volume files already present, skipping copy")
    else:
        print("Copying DHW volume files...")
        os.makedirs(booster_dhw_volume_path, exist_ok=True)
        for file_name in tqdm(os.listdir(unrenovated_dhw_volume_path)):
            shutil.copy(
                os.path.join(unrenovated_dhw_volume_path, file_name),
                booster_dhw_volume_path,
            )

    # setting up some hyperparameters and directories
    res_mask = gdf_buildingstock_results["building_usage"].isin(
        ["sfh", "mfh", "ab", "th"]
    )
    sim = "booster"
    size = "whole_buildingstock"
    mask = get_mask(size, res_mask)  # type:ignore

    dir_space_heating = f"building_analysis/results/{sim}_{size}_{t_grid}/space_heating"
    os.makedirs(dir_space_heating, exist_ok=True)

    # for now we can put a pause on the area results. Let's do that later

    ### what we need now is to calculate the heat pump electricity demand:
    ### we have the COP function calculator, so we can use that.
    ### we first need to retrieve the data from the unrenovated scenario.
    ### inside our gdf_buildingstock_results we have the path to the space heating results.
    ### let's start by iterating over the gdf and iterate over the space heating data.

    ### First we need to create a new column in the gdf_buildingstock_results to store the heat pump size
    gdf_buildingstock_results["heat_pump_size [kW]"] = 0.0
    gdf_buildingstock_results["peak_demand_on_dh_grid [kW]"] = 0.0

    ### we also need to create a new file to store the heat pump electricity demand
    ### along with the folder path
    booster_space_heating_path = f"building_analysis/results/booster/{sim}_{size}_{t_grid}/space_heating_{sim}_{t_grid}"
    os.makedirs(booster_space_heating_path, exist_ok=True)

    booster_dhw_path = f"building_analysis/results/booster/{sim}_{size}_{t_grid}/dhw_energy_{sim}_{t_grid}"
    os.makedirs(booster_dhw_path, exist_ok=True)

    ### we create a temporary dataframe to store the heat pump electricity demand
    ### the thermal demand of the HP and its thermal output.
    ### We export this to a csv file later in the loop.
    ### First let's retrieve a single space heating file to get the index
    space_heating_path = gdf_buildingstock_results.loc[0, "space_heating_path"]
    space_heating = pd.read_csv(space_heating_path, index_col=0)
    space_heating.index = pd.to_datetime(space_heating.index)

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
    ### the carnot_cop method requires a series to work. Hence we need to create a series
    # with the correct length
    index = pd.to_datetime(space_heating.index)
    t_supply_series = pd.Series(t_supply, index=index)
    t_grid_series = pd.Series(t_grid, index=index)

    ### we also want to store the results from all of this in some dataframes because
    # we will need it to create the area_results dataframe later on.

    area_grid_demand_df = pd.DataFrame(index=index)
    area_space_heating_df = pd.DataFrame(index=index)
    booster_el_demand_df = pd.DataFrame(index=index)
    area_dhw_volume = pd.DataFrame(index=index)
    area_dhw_energy = pd.DataFrame(index=index)

    for idx, row in tqdm(
        gdf_buildingstock_results.iterrows(), total=len(gdf_buildingstock_results)
    ):
        ### initialize the temp_df with zeros
        temp_df.iloc[:] = 0

        building_id = row["full_id"]

        ### loading the space heating data
        space_heating_path = row["space_heating_path"]
        space_heating = pd.read_csv(space_heating_path, index_col=0)
        space_heating.index = pd.to_datetime(space_heating.index)

        ### loading the dhw energy data
        dhw_energy_path = row["dhw_energy_path"]
        dhw_energy = pd.read_csv(dhw_energy_path, index_col=0)
        dhw_energy.index = pd.to_datetime(dhw_energy.index)

        ### loading the dhw volume data
        dhw_volume_path = row["dhw_volume_path"]
        dhw_volume = pd.read_csv(dhw_volume_path, index_col=0)
        dhw_volume.index = index

        ### let's sum the dhw and SH demands together to assess total
        ### HP size needed.
        total_demand = space_heating["net useful hourly demand [kWh]"] + dhw_energy.sum(
            axis=1
        )
        max_demand = total_demand.max()
        hp_size = float(max_demand * safety_factor)
        gdf_buildingstock_results.loc[idx, "heat_pump_size [kW]"] = hp_size

        ### now we need to calculate the COP for these heat pumps
        ### luckily we have the COP function calculator, so we can use that.
        ### we need to iterate over the space heating data and calculate the COP for each hour
        COP_hourly = carnot_cop(t_supply_series, t_grid_series, approach_temperature=5)
        el_demand = total_demand / COP_hourly.values
        el_demand_series = pd.Series(el_demand)
        el_demand_series.index = index

        ### we need to store the total demand on the grid and the total demand in electricity
        gdf_buildingstock_results.loc[idx, "total_demand_electricity [kWh]"] = (
            el_demand.sum()
        )

        # original COP eq is COP = Qh / (Qh - Qc)
        # solving for Qc ->
        # Qh*COP - Qc*COP = Qh ->   Qc = Qh - Qh/COP -> Qc = Qh * (1- 1/COP)
        # this is the demand that the District Heating grid needs to cover.
        # The rest will be provided by the electricity of the booster.
        # because of how EMBERS module works, we need to calculate the maximum
        # value of this demand. This will be then the "demand" in embers.
        dh_grid_demand = total_demand * (1 - 1 / COP_hourly.values)
        gdf_buildingstock_results.loc[idx, "peak_demand_on_dh_grid [kW]"] = float(
            dh_grid_demand.max()
        )

        ### we can now build and then store the temp_df
        temp_df["cop_hourly"] = COP_hourly
        temp_df["el_demand [kWh]"] = el_demand
        temp_df["thermal_output [kWh]"] = space_heating
        temp_df["demand_on_dh_grid [kWh]"] = dh_grid_demand
        gdf_buildingstock_results.loc[idx, "total_demand_on_grid [kWh]"] = (
            dh_grid_demand.sum()
        )

        temp_df.to_csv(f"{booster_space_heating_path}/{building_id}_{sim}_{t_grid}.csv")

        ### also we need to be able to possibly recover these data later on.
        ### so we will store the path to the csv files in the gdf_buildingstock_results
        gdf_buildingstock_results.loc[idx, "booster_path"] = (
            f"{booster_space_heating_path}/{building_id}_{sim}_{t_grid}.csv"
        )

        ### before we exit the loop, let's save the data in the dataframe that we will use
        # later on to save the area_results

        area_grid_demand_df[building_id] = dh_grid_demand
        area_space_heating_df[building_id] = space_heating
        booster_el_demand_df[building_id] = el_demand
        area_dhw_volume[building_id] = dhw_volume.sum(axis=1)
        area_dhw_energy[building_id] = dhw_energy.sum(axis=1)

    # now we can save the results to a file
    gdf_buildingstock_results.to_parquet(
        f"building_analysis/results/booster/{sim}_{size}_{t_grid}/buildingstock_{sim}_{size}_{t_grid}_results.parquet"
    )

    area_results_booster_path = f"building_analysis/results/booster/{sim}_{size}_{t_grid}/area_results_{sim}_{size}_{t_grid}.csv"

    area_results = pd.DataFrame(
        columns=[
            "area grid demand [kWh]",
            "area total boosters demand [kWh]",
            "area space heating demand [kWh]",
            "area dhw energy demand [kWh]",
            "area dhw volume demand [l]",
        ],
        index=index,
    )
    area_results["area grid demand [kWh]"] = area_grid_demand_df.sum(axis=1)
    area_results["area total boosters demand [kWh]"] = booster_el_demand_df.sum(axis=1)
    area_results["area space heating demand [kWh]"] = area_space_heating_df.sum(axis=1)
    area_results["area dhw energy demand [kWh]"] = area_dhw_energy.sum(axis=1)
    area_results["area dhw volume demand [l]"] = area_dhw_volume.sum(axis=1)

    folder_area_results_path = (
        f"building_analysis/results/booster/{sim}_{size}_{t_grid}/area_results"
    )
    os.makedirs(folder_area_results_path, exist_ok=True)

    area_results_booster_path = f"building_analysis/results/booster/{sim}_{size}_{t_grid}/area_results/area_results_{sim}_{size}_{t_grid}.csv"

    area_results.to_csv(area_results_booster_path)
