import numpy as np
import pandas as pd
import icecream as ic


class Person:
    def __init__(self, building_id, name):
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

        timestamps = pd.date_range(start=start_year, periods=days * 24, freq="1H")
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

    def select_dhw_times2(
        self,
        mask_shower,
        mask_bath,
        mask_handwash,
        mask_cooking,
        n_shower,
        n_bath,
        n_handwash,
        n_cooking,
    ):
        result_df = pd.DataFrame(
            0,
            index=mask_shower.index,
            columns=["shower", "bath", "handwash", "cooking"],
        )

        days = pd.date_range(
            start=mask_shower.index.min().floor("D"),
            end=mask_shower.index.max().floor("D"),
            freq="D",
        ).date

        n_shower = (
            np.full(len(days), n_shower) if isinstance(n_shower, int) else n_shower
        )
        n_bath = np.full(len(days), n_bath) if isinstance(n_bath, int) else n_bath
        n_handwash = (
            np.full(len(days), n_handwash)
            if isinstance(n_handwash, int)
            else n_handwash
        )
        n_cooking = (
            np.full(len(days), n_cooking) if isinstance(n_cooking, int) else n_cooking
        )

        for i, day in enumerate(days):
            day_mask = mask_shower.index.date == day
            possible_hours_shower = mask_shower[day_mask & (mask_shower == 1)].index
            possible_hours_bath = mask_bath[day_mask & (mask_bath == 1)].index
            possible_hours_handwash = mask_handwash[
                day_mask & (mask_handwash == 1)
            ].index
            possible_hours_cooking = mask_cooking[day_mask & (mask_cooking == 1)].index

            if not possible_hours_shower.empty:
                selected_hours_shower = np.random.choice(
                    possible_hours_shower,
                    size=min(n_shower[i], len(possible_hours_shower)),
                    replace=False,
                )
                result_df.loc[selected_hours_shower, "shower"] = 1

            if not possible_hours_bath.empty:
                selected_hours_bath = np.random.choice(
                    possible_hours_bath,
                    size=min(n_bath[i], len(possible_hours_bath)),
                    replace=False,
                )
                result_df.loc[selected_hours_bath, "bath"] = 1

            if not possible_hours_handwash.empty:
                selected_hours_handwash = np.random.choice(
                    possible_hours_handwash,
                    size=min(n_handwash[i], len(possible_hours_handwash)),
                    replace=False,
                )
                result_df.loc[selected_hours_handwash, "handwash"] = 1

            if not possible_hours_cooking.empty:
                selected_hours_cooking = np.random.choice(
                    possible_hours_cooking,
                    size=min(n_cooking[i], len(possible_hours_cooking)),
                    replace=False,
                )
                result_df.loc[selected_hours_cooking, "cooking"] = 1

        return result_df

    def dhw_profile3(self, occupancy_profile=None):
        """Generate a DHW profile based on occupancy. The objective is to make the function faster."""
        if occupancy_profile is None:
            occupancy_profile = self.occupancy_year
            if occupancy_profile is None:
                raise ValueError(
                    "No occupancy profile provided. Please generate one first. Using self.defined_time_occupancy()"
                )

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

        # setting up masks and draw counts
        shower_days = np.random.uniform(size=len(days)) < 0.7
        shower_days_hourly = np.repeat(shower_days, 24)
        mask_select_shower = occupancy_mask & shower_days_hourly

        bath_days = np.random.uniform(size=len(days)) < 0.044
        bath_days = np.where(shower_days, False, bath_days)
        bath_days_hourly = np.repeat(bath_days, 24)
        mask_select_bath = occupancy_mask & bath_days_hourly

        n_handwash = np.random.randint(1, 5, size=len(days))
        n_handwash = np.minimum(
            n_handwash, self.occupancy_year.resample("D").sum()["occupancy"].values
        )

        n_cooking = np.random.randint(0, 3, size=len(days))
        n_cooking = np.minimum(
            n_cooking, self.occupancy_year.resample("D").sum()["occupancy"].values
        )

        dhw_times = self.select_dhw_times2(
            mask_shower=mask_select_shower,
            mask_bath=mask_select_bath,
            mask_handwash=occupancy_mask,
            mask_cooking=occupancy_mask,
            n_shower=1,
            n_bath=1,
            n_handwash=n_handwash,
            n_cooking=n_cooking,
        )

        dhw_df["shower"] = dhw_times["shower"]
        dhw_df["bath"] = dhw_times["bath"]
        dhw_df["handwash"] = dhw_times["handwash"]
        dhw_df["cooking"] = dhw_times["cooking"]

        return dhw_df

    def dhw_profile4(self, occupancy_df=None):
        """Generate a DHW profile based on occupancy. The objective is to make the function faster."""
        if occupancy_profile is None:
            occupancy_profile = self.occupancy_year
            if occupancy_profile is None:
                raise ValueError(
                    "No occupancy profile provided. Please generate one first. Using self.defined_time_occupancy()"
                )

        # amount of hours at home awake per day
        occupancy_resampled = occupancy_df.resample("D").sum()["occupancy"].values

        # add a column for the date
        occupancy_df["date"] = occupancy_df.index.date
        # filter only rows where occupancy is == 1
        occupied_hours_df = occupancy_df[occupancy_df["occupancy"] == 1].copy()

        # add a column for the hour
        occupied_hours_df["hour"] = occupied_hours_df.index.hour

        # Using pivot_table and melt
        # pivot the table to have dates as rows and hours as columns
        pivoted = occupied_hours_df.pivot_table(
            index="date",
            columns="hour",
            values="occupancy",
            aggfunc="size",
            fill_value=0,
        )

        # melt the table to have the date and hour as columns
        melted = pivoted.melt(ignore_index=False).reset_index()

        # Drop rows with value == 0 and reset index
        occupied_hours_by_day = (
            melted[melted["value"] > 0].drop(columns="value").reset_index(drop=True)
        )

        # group by date and create list of hours
        occupied_hours_by_day = occupied_hours_by_day.groupby("date")["hour"].apply(
            list
        )

        # length of year
        n_days = int(len(self.occupancy_year) / 24)

        # setting up masks and draw counts
        shower_days = np.random.uniform(size=(n_days)) < 0.7
        # shower_days_hourly = np.repeat(shower_days, 24)
        # mask_select_shower = occupancy_mask & shower_days_hourly
        # shower_lt = max(np.random.normal(loc=170, scale=40), 40)

        # find out how many showers in the year.
        showers_in_year = np.sum(shower_days)

        # create a list of water amount of each shower
        shower_water = np.maximum(
            np.random.normal(loc=170, scale=40, size=showers_in_year), 40
        )

        bath_days = np.random.uniform(size=n_days) < 0.044
        bath_days = np.where(shower_days, False, bath_days)
        # bath_days_hourly = np.repeat(bath_days, 24)

        # find out how many baths in a year
        baths_in_year = np.sum(bath_days)

        # mask_select_bath = occupancy_mask & bath_days_hourly
        bath_lt = max(np.random.normal(115, 5), 100)

        n_handwash = np.random.randint(1, 5, size=n_days)
        n_handwash = np.minimum(n_handwash, occupancy_resampled)
        handwash_lt = np.random.uniform(0.25, 1.5, size=n_days)

        n_cooking = np.random.randint(0, 3, size=n_days)
        n_cooking = np.minimum(n_cooking, occupancy_resampled)
        cooking_lt = np.random.uniform(0.25, 10, size=n_days)

        def select_hours():
            selected_hours = []

            # Loop through each day's occupied hours list
            for hours in occupied_hours_by_day["hour"]:
                if len(hours) > 0:
                    selected_hour = np.random.choice(hours)
                    selected_hours.append(selected_hour)
                else:
                    selected_hours.append(None)
            return selected_hours

        #   this will return a random pick of the hours when the person is at home

        return None


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # instantiate a Person
    luca = Person(building_id=1, name=1)

    # generate the probability distribution for luca
    occupancy_probabilities = luca.occupancy_distribution()
    # generate the occupancy for the whole year
    luca_occupancy_year = luca.defined_time_occupancy()
    luca_dhw = luca.dhw_profile2()
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
        tina = Person(building_id=1, name=2)
        dhw_luca = luca.dhw_profile()

    def test_dhw2():
        paolo = Person(building_id=1, name=2)
        dhw_paolo = paolo.dhw_profile2()

    def test_dhw3():
        gianni = Person(building_id=1, name=2)
        dhw_gianni = gianni.dhw_profile3()

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
