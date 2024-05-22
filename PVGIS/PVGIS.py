import requests
import pandas as pd
import json


# Set the API endpoint URL
url = "https://re.jrc.ec.europa.eu/api/seriescalc?"
# url = "https://re.jrc.ec.europa.eu/api/v5_2/tool_name?param1=value1&param2=value2&..."

azimuths = {
    'south': 0,
    'south_east': -45,
    'east': -90,
    'north_east': -135,
    'north': 180,
    'north_west': 135,
    'west': 90,
    'south_west': 45
}

dfs = {}
for direction, azimuth in azimuths.items():
    # Define the parameters for the API request
    params = {
        'lat': 48.210,        # latitude
        'lon': 16.372,      # longitude
        'raddatabase': 'PVGIS-SARAH',
        'startyear': 2015,  # we are only using 2015 as test
        'endyear': 2015,    # hence also here 2015
        'pvcalculation': 0, # we are not performing PV, we only want radiation
        'angle': 90,        # we want a vertical facade
        'aspect': azimuth,  # set the azimuth direction
        'outputformat': 'json'
    }

    # Send a GET request to the API
    response = requests.get(url, params=params)

    # Parse the response JSON into a Python dictionary
    data = response.json()

    dfs[direction] = pd.DataFrame(data['outputs']["hourly"])

for directions in dfs:
    directory = "../Irradiation_Data/"+directions+".csv"
    dfs[str(directions)].to_csv(directory, sep = ",")

irradiation_df = pd.DataFrame()
for direction, df in dfs.items():
    irradiation_df[direction + ' G(i) [kWh/m2]'] = df['G(i)']

irradiation_df.to_csv('../Irradiation_Data/irradiation_data.csv', sep=',')