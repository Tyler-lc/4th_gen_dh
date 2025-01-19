import numpy as np
import pandas as pd


class Person:
    def __init__(self, age, building_id):
        self.age = age
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
            "0-5": (3, 1),
            "5-6": (5.5, 1),
            "6-7": (6.5, 1),
            "7-8": (7.5, 1),
            "8-9": (8.5, 1),
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
        occupandy_pdf = np.minimum(occupancy_pdf, 1)

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


# Example usage
if __name__ == "__main__":
    person = Person(age=30, building_id=1)
    occupancy_df = person.defined_time_occupancy(days=365)

    print(occupancy_df.head())

    import matplotlib.pyplot as plt

    plt.plot(
        occupancy_df.index[: 24 * 7], occupancy_df.occupancy[: 24 * 7]
    )  # Plot first week
    plt.xlabel("Time")
    plt.ylabel("Occupancy")
    plt.title("Occupancy Profile for First Week")
    plt.show()

    person = Person(age=30, building_id=1)

    print(f"Workday wake-up category: {person.workday_wakeup_category}")
    print(f"Workday sleep category: {person.workday_sleep_category}")
    print(f"Free day wake-up category: {person.freeday_wakeup_category}")
    print(f"Free day sleep category: {person.freeday_sleep_category}")

    x = np.linspace(0, 23, 24)
    workday_occupancy = person.workday_occupancy_pdf
    freeday_occupancy = person.freeday_occupancy_pdf
    daily_awake_hours = occupancy_df["occupancy"].resample("D").sum()

    import matplotlib.pyplot as plt

    plt.plot(x, workday_occupancy, label="Workday")
    plt.plot(x, freeday_occupancy, label="Free Day")
    plt.xlabel("Hour")
    plt.ylabel("Occupancy Probability")
    plt.title("Occupancy Distribution")
    plt.legend()
    plt.show()

    plt.plot(range(len(daily_awake_hours)), daily_awake_hours)
    plt.title("Daily Awake Hours")
    plt.show()

    import timeit

    def test_speed():
        test = Person(age=30, building_id=1)
        test.defined_time_occupancy(days=365)

    print(timeit.timeit(test_speed, number=100))
