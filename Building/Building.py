import numpy as np
import pandas as pd
import warnings
import json
import warnings


class Building:

    def __init__(
        self,
        name,
        building_type,
        components,
        outside_temperature,
        irradiation_data,
        soil_temp=8,
        year_start=2015,
        summer_start=6,
        summer_end=9,
    ):
        self.name = name
        self.building_type = building_type
        self.components = components
        self.outside_temperature = outside_temperature
        self.soil_temp = soil_temp
        self.summer_start = summer_start
        self.summer_end = summer_end

        # Ensure the outside_temperature has a DatetimeIndex and is a Dataframe
        # if not a dataframe then create one with DateTimeIndex
        if not isinstance(outside_temperature, pd.DataFrame):
            start = f"{year_start}-01-01"
            period = len(outside_temperature)
            weather_index = pd.date_range(start=start, periods=period, freq="1H")
            self.outside_temperature = pd.DataFrame(
                outside_temperature, index=weather_index, columns=["Outside T °C"]
            )
        # if it is a dataframe but does not have a DatetimeIndex then set it
        elif not isinstance(outside_temperature.index, pd.DatetimeIndex):
            start = f"{year_start}-01-01"
            period = len(outside_temperature)
            weather_index = pd.date_range(start=start, periods=period, freq="1H")
            outside_temperature.set_index(weather_index, inplace=True)
            self.outside_temperature = outside_temperature
        # if it is a dataframe and has a DatetimeIndex then just use input data
        else:
            self.outside_temperature = outside_temperature
            weather_index = outside_temperature.index

        self.list_temperature = self.outside_temperature.iloc[
            :, 0
        ].values.tolist()  # not used for now?
        self.irradiation_data = pd.DataFrame(index=weather_index)
        self.irradiation_data = irradiation_data

        # if soil temperature is single value then expand it to be as long as the outside_temperature
        # Ensure soil_temp is a pandas Series with the same index as outside_temperature
        if isinstance(self.soil_temp, (int, float)):
            self.soil_temp = pd.Series(
                [self.soil_temp] * len(self.outside_temperature),
                index=self.outside_temperature.index,
            )
        elif isinstance(self.soil_temp, list):
            self.soil_temp = pd.Series(
                self.soil_temp, index=self.outside_temperature.index
            )
        elif isinstance(self.soil_temp, pd.Series):
            if not self.soil_temp.index.equals(self.outside_temperature.index):
                self.soil_temp = pd.Series(
                    self.soil_temp.values, index=self.outside_temperature.index
                )
        else:
            raise TypeError("soil_temp must be an int, float, list, or pandas Series")

        # gather geometric data from the components dataframe
        self.n_floors = self.components.loc[0, "n_floors"]
        self.volume = self.components.loc[0, "volume"]
        self.ground_contact_area = self.components.loc[0, "ground_contact_area"]

        # from here on we are creating the dataframes for the different surfaces
        self.transparent_surfaces = self.parse_transparent_surfaces(
            self.components.loc[0, "windows"]
        )
        self.opaque_surfaces = self.parse_opaque_surfaces(components)
        self.ground_contact_surfaces = self.parse_ground_contact_surfaces(components)

        # create global u value variable and total area
        self.global_uvalue = 0

        # self.number_floors = self.components.loc[1, "number of floors"]
        self.net_floor_area = self.ground_contact_area * self.n_floors * 0.85

        # setting the infiltration coefficient to 0.4 for old windows and 0.2 for newer windows
        # first get the u-values of the windows
        u_val_windows = self.transparent_surfaces["uvalue"].values[0]

        # now check if u-value is smaller than 1.4 then it is not an old window
        if u_val_windows <= 1.4:
            self.unwanted_vent_coeff = 0.2
        else:
            self.unwanted_vent_coeff = 0.4

        # create dataframes where we store the various type of losses and gains
        self.heat_losses = pd.DataFrame(
            index=weather_index, columns=["Net Transmission Losses [kW"]
        )
        self.opaque_losses = pd.DataFrame(index=weather_index)
        self.transparent_losses = pd.DataFrame(index=weather_index)
        self.ground_losses = pd.DataFrame(index=weather_index)
        self.total_useful_energy_demand = 0
        self.specific_losses = 0
        self.solar_gain = pd.DataFrame(
            index=weather_index, columns=self.transparent_surfaces["surface_name"]
        )
        self.ventilation_losses = pd.DataFrame(
            index=weather_index, columns=["ventilation losses [kWh]"]
        )
        self.hourly_useful_demand = pd.DataFrame(
            index=weather_index, columns=["net useful hourly demand [kWh]"]
        )
        self.summer_months = self.is_summer(self.outside_temperature.index)

    # in this first section we parse the data from the components dataframe to create
    # the data structure we use in the calculations
    def parse_opaque_surfaces(self, components):
        # Extract opaque surfaces data
        return pd.DataFrame(
            {
                "surface_name": ["roof", "wall", "door"],
                "total_surface": [
                    components.loc[0, "roof_area"],
                    components.loc[0, "walls_area"],
                    components.loc[0, "door_area"],
                ],
                "uvalue": [
                    components.loc[0, "roof_u_value"],
                    components.loc[0, "walls_u_value"],
                    components.loc[0, "door_u_value"],
                ],
            }
        )

    def parse_transparent_surfaces(self, windows_json):
        windows_dict = json.loads(windows_json)
        windows_list = [
            {
                "surface_name": orientation,
                "total_surface": data["area"],
                "uvalue": data["u_value"],
                "SHGC": data["shgc"],
                "orientation": orientation,
            }
            for orientation, data in windows_dict.items()
            if data["area"] > 0
        ]
        return pd.DataFrame(windows_list)

    def parse_ground_contact_surfaces(self, components):
        return pd.DataFrame(
            {
                "surface_name": ["ground_contact"],
                "total_surface": [components.loc[0, "ground_contact_area"]],
                "uvalue": [components.loc[0, "ground_contact_u_value"]],
            }
        )

    # this first section is for all thermal calculations. All the function here calculate the thermal losses due to
    # transmission, ventilation and also the solar gain (Based on PVGIS data).
    def compute_losses(self, surfaces_df, outside_temp, inside_temp):
        u_values = surfaces_df["uvalue"].values[:, np.newaxis]
        areas = surfaces_df["total_surface"].values[:, np.newaxis]
        losses_matrix = (
            u_values * areas * (inside_temp.values - outside_temp.values) / 1000
        )
        losses_matrix[losses_matrix < 0] = 0
        losses_df = pd.DataFrame(
            losses_matrix.T,
            index=outside_temp.index,
            columns=surfaces_df["surface_name"],
        )
        losses_df.loc[self.summer_months] = 0
        return losses_df

    def transmission_losses_opaque(
        self, outside_temp=None, inside_temp=20, temp_col_index=0
    ):
        """by default the outside temperature is the temperature passed when creating the Building instance
        Also the inside temperature is by default 20 °C constant. It is possible to also pass a list
        """

        if outside_temp is None:
            outside_temp = self.outside_temperature.iloc[:, temp_col_index]

        if isinstance(inside_temp, int):
            inside_temp = [inside_temp] * len(self.outside_temperature)

        # convert inside_temp to a pandas Series for vectorized operation
        inside_temp = pd.Series(inside_temp, index=self.outside_temperature.index)

        # call the helper function
        self.opaque_losses = self.compute_losses(
            self.opaque_surfaces, outside_temp, inside_temp
        )

    def transmission_losses_transparent(
        self, outside_temp=None, inside_temp=20, temp_col_index=0
    ):
        """by default the outside temperature is the temperature passed when creating the Building instance
        Also the inside temperature is by default 20 °C constant. It is possible to also pass a list
        """

        if outside_temp is None:
            outside_temp = self.outside_temperature.iloc[:, temp_col_index]

        if isinstance(inside_temp, int):
            inside_temp = [inside_temp] * len(self.outside_temperature)

        # convert inside_temp to a pandas Series for vectorized operation
        inside_temp = pd.Series(inside_temp, index=self.outside_temperature.index)

        # call the helper function
        self.transparent_losses = self.compute_losses(
            self.transparent_surfaces, outside_temp, inside_temp
        )

    def transmission_losses_ground(
        self, soil_temp=None, inside_temp=20, temp_col_index=0
    ):
        """by default the soil temperature is the temperature passed when creating the Building instance
        Also the inside temperature is by default 20 °C constant. It is possible to also pass a list
        """

        if soil_temp is None:
            soil_temp = self.soil_temp

        if isinstance(inside_temp, int):
            inside_temp = [inside_temp] * len(self.outside_temperature)

        # convert inside_temp to a pandas Series for vectorized operation
        inside_temp = pd.Series(
            inside_temp, index=self.outside_temperature.index
        )  # convert inside_temp to a pandas Series
        soil_temp = pd.Series(
            soil_temp, index=self.outside_temperature.index
        )  # convert soil_temp to a pandas Series

        # call the helper function
        self.ground_losses = self.compute_losses(
            self.ground_contact_surfaces, soil_temp, inside_temp
        )

    # this function simply groups all type of transmission losses at once
    # so that we do not have to call them one by one

    def all_transmission_losses(self):
        self.transmission_losses_opaque()
        self.transmission_losses_transparent()
        self.transmission_losses_ground()

    def sol_gain(self):
        for _, row in self.transparent_surfaces.iterrows():
            window_area = row["total_surface"]
            window_SHGC = row["SHGC"]
            window_orientation = row["orientation"]
            irradiation_name = window_orientation + " G(i) [kWh/m2]"
            global_irradiation = self.irradiation_data[irradiation_name]
            global_irradiation.index = self.solar_gain.index

            # compute gain for all hours in one vectorized operation
            gains = global_irradiation * window_SHGC * window_area / 1000
            # Set gains to zero during summer months
            gains[self.summer_months] = 0

            self.solar_gain[row["surface_name"]] = gains

    def vent_loss(self, inside_temp=20):
        # Qv = 0.34 * (0.4 + 0.2) * V

        vent_coeff = 0.34 * (self.unwanted_vent_coeff + 0.2) * self.volume

        # compute loss for all hours in one vectorized operation
        vent_losses = (
            vent_coeff
            * np.maximum(0, inside_temp - self.outside_temperature.iloc[:, 0])
            / 1000
        )

        vent_losses[vent_losses < 0] = 0  # set negative values to zero

        # Set losses to zero during summer months
        vent_losses[self.summer_months] = 0

        self.ventilation_losses["ventilation losses [kWh]"] = vent_losses

    # this function automatically calls all the relevant functions to perform the thermal balance calculations

    def thermal_balance(self):
        self.transmission_losses_opaque()  # ok
        self.transmission_losses_transparent()  # ok
        self.transmission_losses_ground()  # ok
        self.sol_gain()  # ok
        self.vent_loss()  # ok
        self.useful_demand()  # ok
        self.total_use_energy_demand()  # ok

    # from here on we are making other calculations that are not strictly related to the thermal losses

    # this first function computes the global tranmission coefficient. This is a quick calculation tool but it is not
    # used further in the code.
    def global_transmission_coeff(self):
        """calculates the global transmission coefficient
        this was based on another set of data. Not useful for now"""
        self.global_uvalue = (
            self.components["u-value"]
            * self.components["total surface"]
            * self.components["reduction factor"]
        ).sum()

    def total_use_energy_demand(self):
        """calculates the total amount of useful heating energy demand"""
        if self.hourly_useful_demand.isna().any().any():
            warnings.warn(
                "The method 'useful_demand' should be called before 'year_losses'."
            )
            return
        total_loss_value = self.hourly_useful_demand.sum().values[0]
        self.total_useful_energy_demand = pd.Series(
            total_loss_value, index=["Total Useful Energy Demand [kWh]"]
        )

    def total_specific_losses(self):
        self.specific_losses = self.year_losses() / self.net_floor_area

    def useful_demand(self):
        total_losses = (
            self.opaque_losses.sum(axis=1)
            + self.transparent_losses.sum(axis=1)
            + self.ground_losses.sum(axis=1)
            + self.ventilation_losses.sum(axis=1)
        )
        total_gains = self.solar_gain.sum(axis=1)
        # total_gains[self.outside_temperature.iloc[:, 0] >= 20] = (
        #     0  # TODO: i'm not sure this is the correct way to address the
        # )

        net_result = total_losses - total_gains
        mask = net_result < 0
        net_result[mask] = 0

        self.hourly_useful_demand["net useful hourly demand [kWh]"] = net_result

    def is_summer(self, date_index):
        return (date_index.month >= self.summer_start) & (
            date_index.month <= self.summer_end
        )

    def get_useful_demand(self):
        return self.hourly_useful_demand

    def get_sum_useful_demand(self):
        return self.hourly_useful_demand.sum().values[0]

    def get_components(self):
        return self.components

    def get_global_uvalue(self):
        return self.global_uvalue

    def get_outside_temp(self):
        return self.outside_temperature

    def get_opaque_losses(self):
        return self.opaque_losses

    def get_transparent_losses(self):
        return self.transparent_losses

    def get_ground_losses(self):
        return self.ground_losses

    def get_solar_gain(self):
        return self.solar_gain

    def get_ventilation_losses(self):
        return self.ventilation_losses

    def get_hourly_useful_demand(self):
        return self.hourly_useful_demand

    def get_total_useful_energy_demand(self):
        return self.total_useful_energy_demand

    def get_total_walls_surface(self):
        return self.opaque_surfaces.loc[
            self.opaque_surfaces["surface_name"] == "wall", "total_surface"
        ].sum()

    def get_total_windows_surface(self):
        return self.transparent_surfaces["total_surface"].sum()

    def get_total_ground_surface(self):
        return self.ground_contact_surfaces["total_surface"].sum()

    def get_total_roof_surface(self):
        return self.opaque_surfaces.loc[
            self.opaque_surfaces["surface_name"] == "roof", "total_surface"
        ].sum()

    def get_total_door_surface(self):
        return self.opaque_surfaces.loc[
            self.opaque_surfaces["surface_name"] == "door", "total_surface"
        ].sum()

    def get_total_volume(self):
        return self.volume


if __name__ == "__main__":
    import os
    import sys
    import timeit

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Ensure the parent directory is in the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from Databases.mysql_utils.mysql_utils import create_connection, fetch_data

    # import the weather and irradiation data
    path_weather = "../Irradiation_Data/east.csv"
    weather = pd.read_csv(path_weather, usecols=["T2m"])
    irradiation_path = "../Irradiation_Data/irradiation_data.csv"
    irradiation = pd.read_csv(irradiation_path)

    # on/off toggle some tests and debugging options
    losses_outputs = 0
    plot = 0
    test_speed = 1
    profiler = 0

    from Databases.mysql_utils.mysql_utils import create_connection, fetch_data

    # test building results
    data = fetch_data(1)

    if profiler == 1:
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        sfh = Building("test", "SFH1", data, weather, irradiation)
        sfh.thermal_balance()
        profiler.stop()
        output = profiler.output_text(unicode=True, color=True)
        filtered_output = "\n".join(
            line for line in output.split("\n") if "pandas" not in line
        )
        print(filtered_output)

    if profiler == 0:
        sfh = Building("test", "SFH1", data, weather, irradiation)
        sfh.thermal_balance()

    total_ued = sfh.get_sum_useful_demand()
    original_value = 100452.379147

    print(
        f"Total useful energy demand per year = {total_ued} kWh. Should be: {original_value} kWh"
    )

    if test_speed == 1:
        # test the speed of the code
        def new_losses():
            sfh.thermal_balance()

        n_runs = 1000
        time = timeit.timeit(lambda: new_losses(), number=n_runs)

        print(f"Time for {n_runs} runs: {time} seconds")

    if plot == 1:
        import matplotlib.pyplot as plt

        # Assuming `df` is your DataFrame and 'Net Useful Hourly Demand [kWh]' is the column to be plotted
        df = sfh.get_useful_demand()

        # Calculate the 12-hour rolling average
        rolling_window = 96
        df["Smoothed"] = (
            df["net useful hourly demand [kWh]"].rolling(window=rolling_window).mean()
        )

        # Plot the original data
        plt.figure(figsize=(10, 6))
        plt.plot(
            df.index,
            df["net useful hourly demand [kWh]"],
            label="Original Data",
            alpha=0.5,
        )

        # Plot the smoothed data
        plt.plot(df.index, df["Smoothed"], label="12-hour Rolling Average", color="red")

        # Add title and labels
        plt.title(
            f"Net Useful Hourly Demand with {rolling_window}-hour Rolling Average"
        )
        plt.xlabel("Time")
        plt.ylabel("Net Useful Hourly Demand [kWh]")
        plt.legend()

        # Show the plot
        plt.show()

    if losses_outputs == 1:
        # results from each category of losses and gains are saved in csv files.
        test_results = df.to_csv("test_results.csv")
        solar_gain = sfh.get_solar_gain()
        ground_losses = sfh.get_ground_losses()
        opaque_losses = sfh.get_opaque_losses()
        transparent_losses = sfh.get_transparent_losses()
        ventilation_losses = sfh.get_ventilation_losses()
        useful_demand = sfh.get_hourly_useful_demand()
        total_useful_energy_demand = sfh.get_total_useful_energy_demand()

        from pathlib import Path

        Path("test_results").mkdir(parents=True, exist_ok=True)
        solar_gain.to_csv("test_results/solar_gain.csv")
        ground_losses.to_csv("test_results/ground_losses.csv")
        opaque_losses.to_csv("test_results/opaque_losses.csv")
        transparent_losses.to_csv("test_results/transparent_losses.csv")
        ventilation_losses.to_csv("test_results/ventilation_losses.csv")
        useful_demand.to_csv("test_results/useful_demand.csv")
        total_useful_energy_demand.to_csv("test_results/total_useful_energy_demand.csv")
        print("Results saved in test_results folder.")

    # def thermal_balance(self):
    #     self.transmission_losses_opaque()  # ok
    #     self.transmission_losses_transparent()  # ok
    #     self.transmission_losses_ground()  # ok
    #     self.sol_gain()  # ok
    #     self.vent_loss()  # ok
    #     self.useful_demand()  # ok
    #     self.total_use_energy_demand()  # ok
