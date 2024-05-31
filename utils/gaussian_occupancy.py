# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:49:40 2023

@author: lucas
"""

import numpy as np
import pandas as pd


def occupancy_distribution(x=np.linspace(0, 23, 24), min_probability=0.05):
    """ this function creates a probability profile to the occupancy probability
    by default it is with two peak spots at 6 and 18, with a std dev 3 and weight 0.8 for both peaks
    x is a np.linspace type. by default is a 24-hour total with hourly resolution.
     min_probability is the minimum chance of being at home and awake. Doesn't matter if you are at home but sleep
      this value is set by default at 0.05"""
    import numpy as np
    import pandas as pd

    # Parameters for the Gaussian distributions
    mean_1 = 6  # Early morning peak mean
    mean_2 = 18  # Late afternoon peak mean
    std_dev_1 = 3  # Early morning peak standard deviation
    std_dev_2 = 3  # Late afternoon peak standard deviation
    weight_1 = 0.8  # Weight for the early morning peak
    weight_2 = 0.8  # Weight for the late afternoon peak

    # Compute the occupancy at each x value
    occupancy_1 = weight_1 * np.exp(-((x - mean_1) / std_dev_1) ** 2)
    occupancy_2 = weight_2 * np.exp(-((x - mean_2) / std_dev_2) ** 2)
    occupancy = occupancy_1 + occupancy_2

    # Set the minimum probability of being home
    occupancy = np.maximum(occupancy, min_probability)

    return occupancy


# Compute the occupancy profile values for each x value
# probabilities = occupancy_profile(x, 0.05)

def generate_occupancy_profile(probabilities, min_hours, max_hours):
    """it generates the occupancy profile 24 hours at the time."""
    import numpy as np

    # Generate random values between 0 and 1
    random_values = np.random.random(len(probabilities))

    # Determine occupancy based on probabilities
    occupancy = np.where(random_values < probabilities, 1, 0)

    # Create random number between min_hours and max_hours to determine each day
    # the minimum amount of hours spent at home awake

    min_occupancy = np.random.randint(min_hours, max_hours)

    # Ensure minimum occupancy of min_occupancy hours per day
    # total_hours = len(occupancy)
    occupied_hours = np.sum(occupancy)
    if occupied_hours < min_occupancy:
        remaining_hours = min_occupancy - occupied_hours
        available_indices = np.where(occupancy == 0)[0]
        if remaining_hours > len(available_indices):
            remaining_hours = len(available_indices)
        selected_indices = np.random.choice(
            available_indices, size=remaining_hours, replace=False)
        occupancy[selected_indices] = 1

    return occupancy


def defined_time_occupancy(occupancy_distr, days=365, min_hours_daily=6, max_hours_daily=16,
                           start_year="01/01/2021"):
    """it takes the occupancy distribution, the amount of days it should create the occupancy for. Based on the minimum
    and maximum amount of hours spent at home awake, it generates a random number of hours. Additionally it is possible
    to change the year. 2021 is deault as this is still not tested for leap years. it returns a list of lists. Each
    element of the list is a list of 24 hours occupancy profile. Needs to be flattened using misc.flatten"""
    import pandas as pd

    occupancy_year_daily = []
    for day in pd.date_range(start=start_year, periods=days, freq="1D"):
        occupancy_profile_day = generate_occupancy_profile(
            occupancy_distr, min_hours_daily, max_hours_daily)
        occupancy_year_daily.append(occupancy_profile_day)

    return occupancy_year_daily
