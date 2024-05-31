# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:07:07 2023
it seems this funcrion is not used in the project
@author: lucas
"""
import pandas as pd
import numpy as np 

weather_data = pd.read_csv("Vienna_WeatherData2022.csv", usecols=[0,4], comment = "#")
weather_data.columns = ["timestamp", "temperature"]
weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])
weather_data.set_index("timestamp", inplace=True)
weather_data_resampled = weather_data.interpolate(method='linear')

def generate_occupancy_profile(num_hours, occupied_probability):
    occupancy_profile = []
    for _ in range(num_hours):
        if np.random.rand() < occupied_probability:
            occupancy_profile.append(1)  # Occupied
        else:
            occupancy_profile.append(0)  # Unoccupied
    return occupancy_profile

def total_heat(heat_hourly):
    return sum(heat_hourly)

occupancy_profile = generate_occupancy_profile(len(weather_data_resampled), 0.5)