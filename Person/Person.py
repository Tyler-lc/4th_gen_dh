import numpy as np
import pandas as pd


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

    def dhw_profile(
        self,
        occupancy_distr=None,  # occupancy over the year
        shower_min=40,  # minimum amount of water used for a shower (water flow = 8lt/min, 5 min shower)
        shower_max=300,  # maximum amount of water used for a shower (water flow = 15 lt/min, 20 min shower)
        bath_min=100,  # minimum amount of water used for a bath (100 lt)
        bath_max=130,  # maximum amount of water used for a bath (130 lt)
        **kwargs
    ):
        from utils import misc

        if kwargs:
            occupancy_distr = kwargs.get("occupancy_distribution", occupancy_distr)
            shower_min = kwargs.get("shower_min", shower_min)
            shower_max = kwargs.get("shower_max", shower_max)
            bath_min = kwargs.get("bath_min", bath_min)
            bath_max = kwargs.get("bath_max", bath_max)

        if occupancy_distr is None:
            # it is possible to give it a different occupancy over the year
            # if not given, it will use the one generated when the class is instantiated
            occupancy_distr = self.occupancy_year

        # data for these are taken from the literature
        # "In-building waste water heat recovery: An urban-scale method for the
        #   characterisation of water streams and the assessment of energy savings
        #   and costs" - Alexandre Bertrande, Riad, Aggoune, François Maréchal - 2017
        # http://dx.doi.org/10.1016/j.apenergy.2017.01.096
        # TODO: the other study is on my tablet, I will add it later

        # amount of water used for a shower

        timestamps = self.occupancy_distribution.index
        dhw_df = pd.DataFrame(index=timestamps, columns=["water_amount", "draw_type"])

        # create daily masks
        days = pd.date_range(
            start=timestamps.min().floor("D"), end=timestamps.max().floor("D"), freq="D"
        )
        daily_shower_taken = pd.Series(Falsem, index=days)
        daily_bath_taken = pd.Series(False, index=days)
        daily_handwash = pd.Series(0, index=days)
        daily_cooking = pd.Series(0, index=days)

        # create mask for occupancy
        # it makes sure the DHW is used only when person is home

        occupancy_mask = self.occupancy_distribution["occupancy"] == 1
        times = timestamps[occupancy_mask]  # only the times when the person is home

        def assign_draws(num_draws, water_amount, draw_type, daily_mask):
            """this method makes sure that all the draws in the day are assigned correctly
            e.g. if the person washes their hands 4 times in one day, it will make sure
            that 4 water draws are made, assigned at different hours of the day and
            tagged accordingly as handwash"""

        shower_lt = np.random.normal(loc=170, scale=40)
        shower_lt = max(shower_lt, shower_min)
        bath_lt = np.random.normal(115, 5)
        bath_lt = max(bath_lt, bath_min)
        if np.random.uniform() < 0.7:
            take_shower = True
        if np.random.uniform() < 0.044:
            take_bath = True

        n_handwash = np.random.randint(
            1, 5
        )  # number of times a person washes their hands. Average is 3 times
        handwash_water = np.random.uniform(0.25, 1.5)  #
        n_cooking = np.random.randint(
            0, 3
        )  # number of times a person cooks. Average is 1.5 times. It also includes some other activities
        cooking_lt = np.random.uniform(0.25, 10)

        draw_amount = take_shower + take_bath + n_handwash + n_cooking
        morning_shower = np.random.choice([True, False])

        return None

        # Calculation for the DHW profiles

    # # First, this function generates the input needed to calculate the profile
    # def dhw_input_generator(occupancy_distribution):
    #     """is to quickly generate the inputs for the domestic hot water demand profile generator. Takes the occupancy
    #     distribution so to package the whole thing together nicely. Don't have to do it, it is just nice to have
    #     """
    #     shower_time = np.random.rand()
    #     if shower_time < 0.4:  # 40% of germans shower in the morning
    #         pass

    #     daily_water_consumption = 100
    #     randomisation_factor = np.random.uniform(0, 0.4)
    #     active_hours = len(occupancy_distribution)
    #     min_large = 30
    #     max_large = 60
    #     min_draws = 3
    #     min_lt = 1
    #     max_lt = 10

    #     input_params = {
    #         "occupancy_distribution": occupancy_distribution,
    #         "daily_amount": daily_water_consumption,
    #         "random_factor": randomisation_factor,
    #         "active_hours": active_hours,
    #         "min_large": min_large,
    #         "max_large": max_large,
    #         "min_draws": min_draws,
    #         "min_lt": min_lt,
    #         "max_lt": max_lt,
    #     }

    #     return input_params


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # instantiate a Person
    luca = Person(building_id=1, name=1)

    # generate the probability distribution for luca
    occupancy_probabilities = luca.occupancy_distribution()
    # generate the occupancy for the whole year
    luca_occupancy_year = luca.defined_time_occupancy()

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
        luca = Person(building_id=1, name=2)
        luca_one_year = luca.defined_time_occupancy()

    # print(timeit.timeit(test_time, number=10000))
