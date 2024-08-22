import pandas as pd
from typing import List


def df_index_timestamp(start_date="01/01/2021", periods=8760, frequency="1H"):
    """Creates a dataframe with timestamps only. By default is the year 2021, 8760 hours with 1 hour resolution"""
    import pandas as pd

    # Adjusted frequency to '1H'
    timestamps = pd.date_range(start="1/1/2021", periods=periods, freq=frequency)
    df = pd.DataFrame(index=timestamps)
    return df


def safe_min_ones(a, b, min_one):
    import numpy as np

    """# this script takes a given array A and then generates a second array B of the same 
    # size. The array B will always have value 0 where A is 0. The array B will have 
    # a 50% chance of having a 1 or a 0 where A is 1. It also insures a minimum 
    # amount of 1 in array B """

    num_ones = np.count_nonzero(a)

    if num_ones < min_one:
        return a
    else:
        indices = np.random.choice(np.where(a == 1)[0], size=num_ones, replace=False)
        b[indices] = np.random.choice([0, 1], size=num_ones)
        c = (b ^ 1) * a

        while np.count_nonzero(b) < min_one:
            random_index = np.random.choice(np.where(c == 1)[0])
            b[random_index] = 1
            c = (b ^ 1) * a

        return b


def dhw_input_generator(occupancy_distribution):
    """is to quickly generate the inputs for the domestic hot water demand profile generator. Takes the occupancy
    distribution so to package the whole thing together nicely. Don't have to do it, it is just nice to have
    """
    import numpy as np

    daily_water_consumption = 100
    randomisation_factor = np.random.uniform(0, 0.4)
    active_hours = len(occupancy_distribution)
    min_large = 30
    max_large = 60
    min_draws = 3
    min_lt = 1
    max_lt = 10

    input_params = {
        "occupancy_distribution": occupancy_distribution,
        "daily_amount": daily_water_consumption,
        "random_factor": randomisation_factor,
        "active_hours": active_hours,
        "min_large": min_large,
        "max_large": max_large,
        "min_draws": min_draws,
        "min_lt": min_lt,
        "max_lt": max_lt,
    }

    return input_params


def flatten(any_list_of_lists):
    """flattens a list of lists into one single list"""
    return [item for sublist in any_list_of_lists for item in sublist]


def calculate_dwellings(plot_area):
    pass


def calculate_number_people(dwelling_area):
    import math
    import numpy as np

    average = 4 / 100  # 4 people every 100 sqm
    # randomly varies the average amount of people by a maximum of 50%
    random_modifier = np.random.randint(-50, 50) / 100
    # print(random_modifier)
    number_people = math.ceil(average * dwelling_area * (1 + random_modifier))
    return number_people


def is_weekend(date, weekend_days=[6, 7]):
    """
    Check if a given date falls on a weekend.

    Args:
        date (datetime.date): The date to check.
        weekend_days (list, optional): List of weekend days. Defaults to [6, 7].

    Returns:
        bool: True if the date falls on a weekend, False otherwise.
    """
    import datetime

    if date.weekday() == weekend_days[0] or date.weekday() == weekend_days[1]:
        return True
    else:
        return False


def select_random_hour(day: pd.DataFrame):
    """
    Select a random hour from a given day.

    Args:
        day (pd.DataFrame): A DataFrame with a datetime index.

    Returns:
        pd.Timestamp: A random hour from the given day.
    """
    import numpy as np

    return np.random.choice(day.index)


import numpy as np

# Assuming res_mask is already defined
# res_mask = gdf_buildingstock_results["building_usage"].isin(["sfh", "mfh", "ab", "th"])


def get_mask(size: str, res_mask: str):
    """Small utility to automatically generate the mask in one line. This is useful when you want to
    select the whole buildingstock, only the residential buildings or only the non-residential buildings.
    :param size: str: "whole_buildingstock", "residential" or "non_residential
    :param res_mask: a list of the acronyms used to identify the residential buildings
    """

    if size == "whole_buildingstock":
        return np.ones(len(res_mask), dtype=bool)
    elif size == "residential":
        return res_mask
    elif size == "non_residential":
        return np.logical_not(res_mask)
    else:
        raise ValueError("Invalid size parameter")
