import requests
import pandas as pd
import json
import os
from pathlib import Path


# Set the API endpoint URL
url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?"
# url = "https://re.jrc.ec.europa.eu/api/v5_2/tool_name?param1=value1&param2=value2&..."

# set-up the azimuths

azimuths = {
    "south": 0,
    "south_east": -45,
    "east": -90,
    "north_east": -135,
    "north": 180,
    "north_west": 135,
    "west": 90,
    "south_west": 45,
}

# set up some parameters that we would like to change
lat: float = 50.098
lon: float = 8.600
start_year: int = 2020
end_year: int = 2020
database: str = "PVGIS-SARAH2"
pv_calc: int = 0  # we do not want to calculate pv production
angle: int = 90  # we want a vertical facade
city: str = "Frankfurt_Griesheim_Mitte"
global_radiation: int = 0  # Output the global, direct and diffuse in-plane irradiances.
# Value of 1 for "yes". All other values (or no value) mean "no".
# This is the default value. Also we'd have to modify the code to accept
# the other values.
dfs = {}
for direction, azimuth in azimuths.items():
    # Define the parameters for the API request
    params = {
        "lat": lat,  # latitude
        "lon": lon,  # longitude
        "raddatabase": database,
        "startyear": start_year,  # we are only using 2015 as test
        "endyear": end_year,  # hence also here 2015
        "pvcalculation": pv_calc,  # we are not performing PV, we only want radiation
        "angle": angle,  # we want a vertical facade
        "aspect": azimuth,  # set the azimuth direction
        "components": global_radiation,
        "outputformat": "json",
    }

    # Send a GET request to the API
    response = requests.get(url, params=params)

    # Parse the response JSON into a Python dictionary
    data = response.json()

    dfs[direction] = pd.DataFrame(data["outputs"]["hourly"])


# Saving the data and setting up paths
path = Path(f"../irradiation_data/{city}")
path.mkdir(parents=True, exist_ok=True)

# First we save each column in a separate CSV file
for direction in dfs:
    file_path = path / f"{city}_{direction}_{start_year}_{end_year}.csv"
    print(f"Saving data to {file_path}")
    dfs[direction].to_csv(file_path, sep=",")

irradiation_df = pd.DataFrame()
for direction, df in dfs.items():
    irradiation_df[direction + " G(i) [kWh/m2]"] = df["G(i)"]
irradiation_df["T2m"] = dfs["south"][
    "T2m"
]  # saving the temperature at 2m height from soil

final_file_path = path / f"{city}_irradiation_data_{start_year}_{end_year}.csv"
print(f"Saving final data to {final_file_path}")
irradiation_df.to_csv(final_file_path, sep=",")
