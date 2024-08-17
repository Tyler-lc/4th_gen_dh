import numpy as np
import pandas as pd
import icecream as ic


class Person:
    def __init__(self, building_id, person_id):
        """This class generates a person with a specific age and building id.
        In this class we generate DHW and Occupancy profile. In this case occupancy is defined as the probability of
        being at home and awake. We do consider sleeping time as occupancy = 0.
         Domestic hot water (DHW), is generated based on the occupancy profile.
         In this class we assign a wake-up category and a sleep category based on the percentage of Germans that wake up
         at a certain time. [Schlaf gut, Deutschland - TK-Schlafstudie 2017]
         DHW is generated based on the occupancy profile.
         It also changes based on the ages of the people
        """
        self.building_id = building_id
        self.person_id = person_id
        self.workday_wakeup_category = self.assign_wakeup_category(workday=True)
        self.freeday_wakeup_category = self.assign_wakeup_category(workday=False)
        self.workday_sleep_category = self.assign_sleep_category(
            self.workday_wakeup_category, workday=True
        )
        self.freeday_sleep_category = self.assign_sleep_category(
            self.freeday_wakeup_category, workday=False
        )
        self.workday_occupancy_pdf = self.occupancy_distribution(workday=True)
        self.freeday_occupancy_pdf = self.occupancy_distribution(workday=False)
        self.occupancy_year = self.defined_time_occupancy()
        self.dhw_year = None

    def assign_wakeup_category(self, workday=True):
        """Assign wakeup category based on the given probabilities."""
        wakeup_probs = {
            "workday": [9, 18, 32, 24, 10, 6],
            "free day": [2, 2, 13, 23, 29, 30],
        }
        categories = ["0-5", "5-6", "6-7", "7-8", "8-9", "9 and later"]
        probs = wakeup_probs["workday"] if workday else wakeup_probs["free day"]
        probs = [p / sum(probs) for p in probs]  # Normalize probabilities
        return np.random.choice(categories, p=probs)

    def assign_sleep_category(self, wakeup_category, workday=True):
        """Assign sleep category based on the wake-up category."""
        sleep_mapping = {
            "0-5": "before 22",
            "5-6": "before 22",
            "6-7": "22 and 23",
            "7-8": "23 and 00:00",
            "8-9": "00:00 and 01:00",
            "9 and later": "1 or later",
        }
        return sleep_mapping[wakeup_category]

    def occupancy_distribution(
        self, workday=True, x=np.linspace(0, 23, 24), min_probability=0.2
    ):
        """Creates an occupancy probability profile based on the assigned wake-up and sleep categories."""

        wake_up_times = {
            "0-5": (3, 1.5),
            "5-6": (5.5, 1.5),
            "6-7": (6.5, 1.5),
            "7-8": (7.5, 1.5),
            "8-9": (8.5, 1.5),
            "9 and later": (10.5, 1.5),
        }

        sleep_times = {
            "before 22": (21, 1),
            "22 and 23": (22.5, 1),
            "23 and 00:00": (23.5, 1),
            "00:00 and 01:00": (0.5, 1),
            "1 or later": (2, 1.5),
        }

        if workday:
            wakeup_category = self.workday_wakeup_category
            sleep_category = self.workday_sleep_category
        else:
            wakeup_category = self.freeday_wakeup_category
            sleep_category = self.freeday_sleep_category

        wake_mean, wake_std = wake_up_times[wakeup_category]
        sleep_mean, sleep_std = sleep_times[sleep_category]

        # Compute the Gaussian distribution for the wake-up time
        occupancy_pdf = np.exp(-(((x - wake_mean) / wake_std) ** 2))

        # Compute the Gaussian distribution for the sleep time
        occupancy_pdf += np.exp(-(((x - sleep_mean) / sleep_std) ** 2))

        # Parameters for the second Gaussian distribution
        time_afternoon = [14, 15, 16, 17, 18, 19, 20]
        mean_2 = np.random.choice(time_afternoon)  # Late afternoon peak mean
        std_dev_2 = 4  # Late afternoon peak standard deviation
        weight_2 = np.random.uniform(
            min_probability, 0.6
        )  # Weight for the late afternoon peak

        # Compute the occupancy at each x value for the second peak
        occupancy_2 = weight_2 * np.exp(-(((x - mean_2) / std_dev_2) ** 2))

        # Add the second Gaussian to the occupancy profile
        occupancy_pdf += occupancy_2

        # Set the minimum probability of being home
        occupancy_pdf = np.maximum(occupancy_pdf, min_probability)
        occupancy_pdf = np.minimum(occupancy_pdf, 1)

        return occupancy_pdf

    def defined_time_occupancy(
        self,
        wd_occupancy_distr=None,  # occupancy distribution for workdays
        fd_occupancy_distr=None,  # occupancy distribution for free days
        days=365,
        min_hours_daily=6,
        max_hours_daily=16,
        start_year="01/01/2021",
    ):
        """Generates occupancy profile for each day over a specified number of days.
        Returns a DataFrame with timestamps and occupancy profiles.
        """
        if wd_occupancy_distr is None:
            wd_occupancy_distr = self.workday_occupancy_pdf
        if fd_occupancy_distr is None:
            fd_occupancy_distr = self.freeday_occupancy_pdf

        timestamps = pd.date_range(start=start_year, periods=days * 24, freq="h")
        occupancy_df = pd.DataFrame(index=timestamps, columns=["occupancy"])
        occupancy_df["weekday"] = occupancy_df.index.weekday

        # Create a mask for workdays and free days
        workdays_mask = occupancy_df["weekday"] < 5
        freedays_mask = ~workdays_mask

        # Generate random values for the entire DataFrame
        random_values = np.random.rand(len(occupancy_df))

        # Create the initial occupancy profile based on the minimum probability
        occupancy_df.loc[workdays_mask, "occupancy"] = np.where(
            random_values[workdays_mask] < wd_occupancy_distr.min(), 1, 0
        )
        occupancy_df.loc[freedays_mask, "occupancy"] = np.where(
            random_values[freedays_mask] < fd_occupancy_distr.min(), 1, 0
        )

        occupancy_df.drop(columns=["weekday"], inplace=True)
        return occupancy_df

    def dhw_profile(self):
        """Generate a DHW profile based on occupancy."""
        timestamps = self.occupancy_year.index
        dhw_df = pd.DataFrame(
            index=timestamps, columns=["shower", "bath", "cooking", "handwash"]
        )
        dhw_df[:] = 0  # Initialize all values to 0

        # Vectorized operation for occupancy == 1
        occupancy_mask = self.occupancy_year["occupancy"] == 1

        days = pd.date_range(
            start=timestamps.min().floor("D"), end=timestamps.max().floor("D"), freq="D"
        ).date

        # Precompute shower and bath probabilities
        shower_prob = np.random.uniform(size=len(days)) < 0.7
        bath_prob = np.random.uniform(size=len(days)) < 0.044

        for i, day in enumerate(days):
            day_mask = occupancy_mask.loc[occupancy_mask.index.date == day]
            if day_mask.sum() == 0:
                continue

            # Shower
            if shower_prob[i]:
                shower_lt = max(np.random.normal(loc=170, scale=40), 40)
                morning_shower = np.random.choice([True, False])
                morning_mask = (day_mask.index.hour < 12) & day_mask

                if morning_shower and morning_mask.any():
                    draw_times = np.random.choice(
                        day_mask.index[morning_mask], size=1, replace=False
                    )
                    dhw_df.loc[draw_times, "shower"] += shower_lt
                else:
                    evening_mask = (day_mask.index.hour >= 12) & day_mask
                    if evening_mask.any():
                        draw_times = np.random.choice(
                            day_mask.index[evening_mask], size=1, replace=False
                        )
                        dhw_df.loc[draw_times, "shower"] += shower_lt

            # Bath
            if bath_prob[i]:
                bath_lt = max(np.random.normal(115, 5), 100)
                draw_times = np.random.choice(day_mask.index, size=1, replace=False)
                dhw_df.loc[draw_times, "bath"] += bath_lt

            # Hand washing and cooking water usage
            n_handwash = min(np.random.randint(1, 5), day_mask.sum())
            handwash_water = np.random.uniform(0.25, 1.5)
            draw_times = np.random.choice(
                day_mask.index, size=n_handwash, replace=False
            )
            dhw_df.loc[draw_times, "handwash"] += handwash_water

            n_cooking = min(np.random.randint(0, 3), day_mask.sum())
            cooking_lt = np.random.uniform(0.25, 10)
            draw_times = np.random.choice(day_mask.index, size=n_cooking, replace=False)
            dhw_df.loc[draw_times, "cooking"] += cooking_lt

        self.dhw_year = dhw_df
        return dhw_df

    def set_dhw_profile(self, dhw_profile):
        """utility to add pre-calculated dhw_profile to the person object. The dhw volume must be in liters
        and 8760 hours long or 8761 hours long if the year is a leap year.

        :param dhw_profile: pd.DataFrame with columns ['shower', 'bath', 'cooking', 'handwash']
        """
        required_columns = ["shower", "bath", "cooking", "handwash"]

        # Check length of dhw_profile
        if len(dhw_profile) not in [8760, 8761]:
            raise ValueError("dhw_profile must be 8760 or 8761 hours long.")

        # Check for required columns
        for column in required_columns:
            if column not in dhw_profile:
                raise KeyError(f"dhw_profile must contain the column '{column}'")


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # instantiate a Person
    luca = Person(building_id=1, person_id=1)

    # generate the probability distribution for luca
    occupancy_probabilities = luca.occupancy_distribution()
    # generate the occupancy for the whole year
    luca_occupancy_year = luca.defined_time_occupancy()
    # luca_dhw = luca.dhw_profile2()
    plot = True

    if plot == True:
        start_date = "2021-01-01"
        end_date = "2021-01-09 23:59:59"
        days_df = luca_occupancy_year[start_date:end_date]

        # Plot the occupancy profile
        plt.figure(figsize=(30, 10))

        plt.bar(range(len(days_df)), days_df.occupancy)
        plt.show()

        plt.plot(range(len(occupancy_probabilities)), luca.freeday_occupancy_pdf)
        plt.plot(range(len(occupancy_probabilities)), luca.workday_occupancy_pdf)
        plt.legend(["Free day", "Work day"])
        plt.show()
    import timeit

    def test_time():
        tina = Person(building_id=1, person_id=2)
        dhw_luca = luca.dhw_profile()

    def test_dhw2():
        paolo = Person(building_id=1, person_id=2)
        dhw_paolo = paolo.dhw_profile3()

    def test_dhw3():
        gianni = Person(building_id=1, person_id=2)
        dhw_gianni = gianni.dhw_profile4()

    print(timeit.timeit(test_time, number=1))
    print(timeit.timeit(test_dhw2, number=1))
    print(timeit.timeit(test_dhw3, number=1))
    from pyinstrument import Profiler

    # profiler = Profiler()
    # profiler.start()
    # gianni = Person(building_id=1, name=2)
    # dhw = luca.dhw_profile2()
    # profiler.stop()
    # output = profiler.output_text(unicode=True, color=True)
    # # filtered_output = "\n".join(
    # #         line for line in output.split("\n") if "pandas" not in line
    # # )
    # print(output)

    import os
    import sys

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Ensure the parent directory is in the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    buildings_path = "../building_analysis/building_input.parquet"
    buildings_data = pd.read_parquet(buildings_path)

    res_mask = buildings_data["building_usage"].isin(["sfh", "mfh", "ab", "th"])
    total_GFA = buildings_data[res_mask]["GFA"].sum()
    maximum_people = 9500
    people_per_GFA = maximum_people / total_GFA
    buildings_data.loc[res_mask, "n_people"] = round(
        buildings_data["GFA"] * people_per_GFA
    )
    buildings_data["n_people"] = (
        buildings_data["n_people"].fillna(0).infer_objects(copy=False)
    )

    # conver n_people to integer
    buildings_data["n_people"] = buildings_data["n_people"].astype(int)

    i = 0
    for idx, row in buildings_data[res_mask].iterrows():
        fid = row["fid"]
        full_id = row["full_id"]
        osmid = row["osm_id"]
        n_people = row["n_people"]
        print(f"Building {full_id} has {n_people} people")
        for people in range(n_people):
            print(f"analyising person {person} in building {full_id}")
            person = Person(full_id, people)
            dhw_data = person.dhw_profile()
            os.makedirs("../building_analysis/dhw_profiles", exist_ok=True)
            dhw_data.to_csv(f"../building_analysis/dhw_profiles/{full_id}_{people}.csv")
