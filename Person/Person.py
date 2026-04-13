import numpy as np
import pandas as pd

import warnings


class Person:
    def __init__(self, building_id, person_id, start_year="01/01/2019", seed=None):
        """This class generates a person with a specific age and building id.
        In this class we generate DHW and Occupancy profile. In this case occupancy is defined as the probability of
        being at home and awake. We do consider sleeping time as occupancy = 0.
         Domestic hot water (DHW), is generated based on the occupancy profile.
         In this class we assign a wake-up category and a sleep category based on the percentage of Germans that wake up
         at a certain time. [Schlaf gut, Deutschland - TK-Schlafstudie 2017]
         DHW is generated based on the occupancy profile.
         It also changes based on the ages of the people

        Parameters
        ----------
        seed : int, optional
            If provided, creates a dedicated RandomState for reproducible
            results.  When *None* (default) ``np.random`` is used, preserving
            the original stochastic behaviour.
        """
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
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
        self.start_year = start_year
        self.occupancy_year = self.defined_time_occupancy()
        self.dhw_year = None
        self.dhw_energy_demand = pd.DataFrame()

    def assign_wakeup_category(self, workday=True):
        """Assign wakeup category based on the given probabilities."""
        wakeup_probs = {
            "workday": [9, 18, 32, 24, 10, 6],
            "free day": [2, 2, 13, 23, 29, 30],
        }
        categories = ["0-5", "5-6", "6-7", "7-8", "8-9", "9 and later"]
        probs = wakeup_probs["workday"] if workday else wakeup_probs["free day"]
        probs = [p / sum(probs) for p in probs]  # Normalize probabilities
        return self.rng.choice(categories, p=probs)

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
        mean_2 = self.rng.choice(time_afternoon)  # Late afternoon peak mean
        std_dev_2 = 4  # Late afternoon peak standard deviation
        weight_2 = self.rng.uniform(
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
        start_year=None,
    ):
        """Generates occupancy profile for each day over a specified number of days.
        Returns a DataFrame with timestamps and occupancy profiles.
        """
        if start_year == None:
            start_year = self.start_year
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
        random_values = self.rng.rand(len(occupancy_df))

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
        shower_prob = self.rng.uniform(size=len(days)) < 0.7
        bath_prob = self.rng.uniform(size=len(days)) < 0.044

        for i, day in enumerate(days):
            day_mask = occupancy_mask.loc[occupancy_mask.index.date == day]
            if day_mask.sum() == 0:
                continue

            # Shower
            if shower_prob[i]:
                shower_lt = max(self.rng.normal(loc=170, scale=40), 40)
                morning_shower = self.rng.choice([True, False])
                morning_mask = (day_mask.index.hour < 12) & day_mask

                if morning_shower and morning_mask.any():
                    draw_times = self.rng.choice(
                        day_mask.index[morning_mask], size=1, replace=False
                    )
                    dhw_df.loc[draw_times, "shower"] += shower_lt
                else:
                    evening_mask = (day_mask.index.hour >= 12) & day_mask
                    if evening_mask.any():
                        draw_times = self.rng.choice(
                            day_mask.index[evening_mask], size=1, replace=False
                        )
                        dhw_df.loc[draw_times, "shower"] += shower_lt

            # Bath
            if bath_prob[i]:
                bath_lt = max(self.rng.normal(115, 5), 100)
                draw_times = self.rng.choice(day_mask.index, size=1, replace=False)
                dhw_df.loc[draw_times, "bath"] += bath_lt

            # Hand washing and cooking water usage
            n_handwash = min(self.rng.randint(1, 5), day_mask.sum())
            handwash_water = self.rng.uniform(0.25, 1.5)
            draw_times = self.rng.choice(
                day_mask.index, size=n_handwash, replace=False
            )
            dhw_df.loc[draw_times, "handwash"] += handwash_water

            n_cooking = min(self.rng.randint(0, 3), day_mask.sum())
            cooking_lt = self.rng.uniform(0.25, 10)
            draw_times = self.rng.choice(day_mask.index, size=n_cooking, replace=False)
            dhw_df.loc[draw_times, "cooking"] += cooking_lt

        self.dhw_year = dhw_df
        return dhw_df

    def dhw_energy(self, cold_water_temp: float = 8, hot_water_temp: dict = None):
        """calculate the energy needed for the domestic hot water demand in kWh
        cold_water_temp: float, optional. Default is 8 °C
        hot_water_temp: dict, optional. Default is None. If None, the default values are used
        default values are: {"shower": 38, "bath": 40, "cooking": 35, "handwash": 37}"""
        if hot_water_temp is None:
            # values come from http://dx.doi.org/10.1016/j.apenergy.2016.02.107
            hot_water_temp = {
                "shower": 40,
                "bath": 40,
                "cooking": 35,
                "handwash": 35,
            }  # °C
            # TODO: we can add a function that varies the cold water temperature based on the hour of the year
            # TODO: we can add a function that varies the hot water temperature based on the hour of the year (some cosin or sin function)
        if self.dhw_year.empty:
            warnings.warn(
                "No hot water demand. Make sure to run Building.dhw_volume() first"
            )
            return
        hot_water_temp_df = pd.DataFrame(
            {
                key: [value] * len(self.dhw_year)
                for key, value in hot_water_temp.items()
            },
            index=self.dhw_year.index,
        )
        # E = m * c_p * DT. 4.182 is the specific heat of water (c_p) in kJ/kg°C
        # 3600 is to convert kJ to kWh
        self.dhw_energy_demand = (
            self.dhw_year * 4.182 * (hot_water_temp_df - cold_water_temp) / 3600  #
        )

        return self.dhw_energy_demand

    def set_dhw_profile(self, dhw_profile_path, index=None):
        """utility to add pre-calculated dhw_profile to the person object. The dhw volume must be in liters
        and 8760 hours long or 8761 hours long if the year is a leap year.

        :param dhw_profile: pd.DataFrame with columns ['shower', 'bath', 'cooking', 'handwash']
        """
        if index is None:
            index = self.dhw_year.index

        required_columns = ["shower", "bath", "cooking", "handwash"]
        dhw_df = pd.read_csv(dhw_profile_path, index_col=0, parse_dates=True)
        # Check length of dhw_profile
        if len(dhw_df) not in [8760, 8761]:
            raise ValueError("dhw_profile must be 8760 or 8761 hours long.")

        # Check for required columns
        for column in required_columns:
            if column not in dhw_df:
                raise KeyError(f"dhw_profile must contain the column '{column}'")

        self.dhw_year = dhw_df

    def get_dhw_profile(self):
        if self.dhw_year is None:
            raise ValueError(
                "DHW profile not generated. Generate DHW first by using Person.dhw_profile() method or append a pre-calculated DHW profile using Person.set_dhw_profile() method."
            )
        return self.dhw_year

    def get_dhw_energy_demand(self):
        if self.dhw_energy_demand.empty:
            raise ValueError(
                "DHW energy demand not calculated. Calculate DHW energy demand first by using Person.dhw_energy() method."
            )
        return self.dhw_energy_demand


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
    luca_dhw = luca.dhw_profile()
    plot = True

    if plot == True:
        start_date = "2021-01-01"
        end_date = "2021-01-09 23:59:59"
        days_df = luca_occupancy_year[start_date:end_date]

        # Plot the occupancy profile

        plt.bar(range(len(days_df)), days_df.occupancy)
        plt.show()

        plt.figure(figsize=(20, 10))
        plt.plot(
            range(len(occupancy_probabilities)), luca.freeday_occupancy_pdf, linewidth=5
        )
        plt.plot(
            range(len(occupancy_probabilities)), luca.workday_occupancy_pdf, linewidth=5
        )
        plt.legend(["Free day", "Work day"], fontsize=22)
        plt.xlabel("Time of day (hours)", fontsize=28)
        plt.ylabel("Probability of occupancy", fontsize=28)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title("Occupancy probability distribution", fontsize=28)
        plt.show()
