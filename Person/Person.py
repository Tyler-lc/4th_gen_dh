import numpy as np
import pandas as pd


class Person:
    def __init__(self, age: int, building_id):
        self.age = age
        self.building_id = building_id
        self.dhw_profile = None
        self.occupancy = None
        self.occupancy_pdf = None

    def get_age(self):
        return self.age

    def get_building_id(self):
        return self.building_id

    def get_occupancy_pdf(self):
        if occupancy_pdf is None:
            print(
                "Occupancy PDF is not yet defined. Please call the occupancy_distribution method first."
            )
        else:
            return self.occupancy_pdf

    @staticmethod
    def occupancy_distribution(x=np.linspace(0, 23, 24), min_probability=0.05):
        """this function creates a probability profile to the occupancy probability
        by default it is with two peak spots at 6 and 18, with a std dev 3 and weight 0.8 for both peaks
        x is a np.linspace type. by default is a 24-hour total with hourly resolution.
        min_probability is the minimum chance of being at home and awake. Doesn't matter if you are at home but sleep
        this value is set by default at 0.05"""
        import numpy as np
        import pandas as pd

        # TODO: these values should change according to age. We can do that quite easily

        # Parameters for the Gaussian distributions
        mean_1 = 6  # Early morning peak mean
        mean_2 = 18  # Late afternoon peak mean
        std_dev_1 = 3  # Early morning peak standard deviation
        std_dev_2 = 3  # Late afternoon peak standard deviation
        weight_1 = 0.8  # Weight for the early morning peak
        weight_2 = 0.8  # Weight for the late afternoon peak

        # Compute the occupancy at each x value
        occupancy_1 = weight_1 * np.exp(-(((x - mean_1) / std_dev_1) ** 2))
        occupancy_2 = weight_2 * np.exp(-(((x - mean_2) / std_dev_2) ** 2))
        occupancy_pdf = occupancy_1 + occupancy_2

        # Set the minimum probability of being home
        occupancy_pdf = np.maximum(occupancy_pdf, min_probability)

        return occupancy_pdf

    def generate_occupancy_profile(
        self, min_hours, max_hours, occupancy_probability=None
    ):
        """it generates the occupancy profile 24 hours at the time.
        it takes the probability distribution created in def occupancy_distribution
        and generates the occupancy profile for one day."""
        import numpy as np

        if occupancy_probability is None:
            occupancy_probability = Person.occupancy_distribution()

        # Generate random values between 0 and 1
        # print(occupancy_probability)
        # print(f"length of occupancy probability is: {len(occupancy_probability)}")
        random_values = np.random.random(len(occupancy_probability))

        # Determine occupancy based on probabilities
        self.occupancy = np.where(random_values < occupancy_probability, 1, 0)

        # Create random number between min_hours and max_hours to determine each day
        # the minimum amount of hours spent at home awake

        min_occupancy = np.random.randint(min_hours, max_hours)

        # Ensure minimum occupancy of min_occupancy hours per day
        # total_hours = len(occupancy)
        occupied_hours = np.sum(self.occupancy)
        if occupied_hours < min_occupancy:
            remaining_hours = min_occupancy - occupied_hours
            available_indices = np.where(self.occupancy == 0)[0]
            if remaining_hours > len(available_indices):
                remaining_hours = len(available_indices)
            selected_indices = np.random.choice(
                available_indices, size=remaining_hours, replace=False
            )
            self.occupancy[selected_indices] = 1

        return self.occupancy

    def defined_time_occupancy(
        self,
        occupancy_distr=None,
        days=365,
        min_hours_daily=6,
        max_hours_daily=16,
        start_year="01/01/2021",
    ):
        """Generates occupancy profile for each day over a specified number of days.
        Returns a NumPy array of occupancy profiles, which can be flattened if needed.
        """
        if occupancy_distr is None:
            occupancy_distr = self.occupancy_distribution()

        occupancy_year_daily = np.zeros((days, 24), dtype=int)

        for i, day in enumerate(
            pd.date_range(start=start_year, periods=days, freq="1D")
        ):
            occupancy_profile_day = self.generate_occupancy_profile(
                min_hours_daily, max_hours_daily, occupancy_distr
            )
            occupancy_year_daily[i, :] = occupancy_profile_day

        return occupancy_year_daily

    # Calculation for the DHW profiles
    # First, this function generates the input needed to calculate the profile
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


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # instantiate a Person
    luca = Person(age=30, building_id=1)

    # generate the probability distribution for luca
    occupancy_probabilities = luca.occupancy_distribution()
    # generate the occupancy profile for luca
    lucas_one_day = luca.generate_occupancy_profile(7, 16, occupancy_probabilities)
    luca_occupancy_year = luca.defined_time_occupancy().flatten()

    howmanydays = 10
    total_days = howmanydays * 24
    # Plot the occupancy profile
    plt.bar(
        range(len(luca_occupancy_year[:total_days])), luca_occupancy_year[:total_days]
    )

    # add red dashed lines to show the end of each day
    for day in range(howmanydays):
        plt.axvline(x=day * 24, color="red", linestyle="--", linewidth=0.8)

    plt.title("One day occupancy profile")
    plt.show()
