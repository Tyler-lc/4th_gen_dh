import numpy as np
import pandas as pd
import warnings
import json


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

        # uses same indexing as the outside_temperature input given and creates a list of the temperatures to be used
        # easily in for loops
        if isinstance(outside_temperature, pd.DataFrame) and not isinstance(
            outside_temperature.index, pd.DatetimeIndex
        ):
            start = str(year_start) + "-01-01"
            period = len(outside_temperature)
            weather_index = pd.date_range(start=start, periods=period, freq="1H")
            outside_temperature.set_index(weather_index, inplace=True)
            self.outside_temperature = outside_temperature

        # if the weather data is not a Pandas DataFrame then it will create a dataframe using the default start year
        else:
            start = str(year_start) + "-01-01"
            period = len(outside_temperature)
            weather_index = pd.date_range(start=start, periods=period, freq="1H")
            self.outside_temperature = pd.DataFrame(
                index=weather_index, columns=["Outside T °C"]
            )
            self.outside_temperature["Outside T °C"] = outside_temperature

        self.list_temperature = self.outside_temperature.iloc[:, 0].values.tolist()
        self.irradiation_data = pd.DataFrame(index=weather_index)
        self.irradiation_data = irradiation_data

        # if soil temperature is single value then expand it to be as long as the outside_temperature
        if type(self.soil_temp) == int:
            self.soil_temp = [soil_temp] * len(self.outside_temperature)

        # filter the components by building type in the CSV
        self.n_floors = self.components.loc[0, "n_floors"]
        self.volume = self.components.loc[0, "volume"]
        self.ground_contact_area = self.components.loc[0, "ground_contact_area"]
        # create the data frames for each surface type
        # self.roof_area = self.components.loc[0, "roof_area"]
        # self.roof_uvalue = self.components.loc[0, "roof_u_value"]
        # self.wall_area = self.components.loc[0, "walls_area"]
        # self.wall_uvalue = self.components.loc[0, "walls_u_value"]
        # self.ground_contact_area = self.components.loc[0, "ground_contact_area"]
        # self.ground_contact_uvalue = self.components.loc[0, "ground_contact_u_value"]
        # self.door_area = self.components.loc[0, "door_area"]
        # self.door_uvalue = self.components.loc[0, "door_u_value"]
        
        # from here on we are creating the dataframes for the different surfaces
        self.transparent_surfaces = self.parse_transparent_surfaces(self.components.loc[0, "windows"])
        self.opaque_surfaces = self.parse_opaque_surfaces(components)
        self.ground_contact_surfaces = self.parse_ground_contact_surfaces(components)

        # create global u value variable and total area
        self.global_uvalue = 0
        
        # self.number_floors = self.components.loc[1, "number of floors"]
        self.net_floor_area = self.ground_contact_area* self.n_floors* 0.85
        
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
    # TODO: we now need to change all the calculation types to be based on the new data structure
        self.heating_months = self.is_summer(self.outside_temperature.index)
    
    # this first section is for all thermal calculations. All the function here calculate the thermal losses due to
    # transmission, ventilation and also the solar gain (Based on PVGIS data).
    def parse_opaque_surfaces(self, components):
        # Extract opaque surfaces data
        return pd.DataFrame({
            "surface_name": ["roof", "wall", "door"],
            "total_surface": [components.loc[0,"roof_area"], 
                              components.loc[0, "walls_area"], 
                              components.loc[0,"door_area"]],
            "uvalue": [components.loc[0, "roof_u_value"], 
                       components.loc[0, "walls_u_value"],
                        components.loc[0, "door_u_value"]]
        })

    def parse_transparent_surfaces(self, windows_json):
        windows_dict = json.loads(windows_json)
        windows_list = [
            {"surface_name": orientation, "total_surface": data["area"], "uvalue": data["u_value"], "SHGC": data["shgc"], "orientation": orientation}
            for orientation, data in windows_dict.items() if data["area"] > 0
        ]
        return pd.DataFrame(windows_list)

    def parse_ground_contact_surfaces(self, components):
        return pd.DataFrame({
            "surface_name": ["ground_contact"],
            "total_surface": [components.loc[0, "ground_contact_area"]],
            "uvalue": [components.loc[0, "ground_contact_u_value"]]
        })

    def compute_losses(self, surfaces_df, outside_temp, inside_temp):
        losses_df = pd.DataFrame(index=self.outside_temperature.index)
        # TODO: change the is_summer to be run only once! 

        for _, row in surfaces_df.iterrows():
            u_value = row["uvalue"]
            area = row["total_surface"]
            surface_name = row["surface_name"]

            losses = u_value* area * (inside_temp - outside_temp) / 1000  # kWh

            losses[losses < 0] = 0  # set negative values to zero

            # Set losses to zero during summer months
            losses[self.heating_months] = 0

            losses_df[surface_name] = losses

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
# TODO: we want to change the data input structure.
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
        inside_temp = pd.Series(inside_temp, index=self.outside_temperature.index) # convert inside_temp to a pandas Series
        soil_temp = pd.Series(soil_temp, index=self.outside_temperature.index) # convert soil_temp to a pandas Series

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
            gains[self.heating_months] = 0

            self.solar_gain[row["surface_name"]] = gains

    def vent_loss(self, inside_temp=20):
        #Qv = 0.34 * (0.4 + 0.2) * V
        
        vent_coeff = 0.34 * (self.unwanted_vent_coeff + 0.2) * self.volume

        # compute loss for all hours in one vectorized operation
        vent_losses = (
            vent_coeff
            * np.maximum(0, inside_temp - self.outside_temperature.iloc[:, 0])
            / 1000
        )

        vent_losses[vent_losses < 0] = 0  # set negative values to zero

        # Set losses to zero during summer months
        vent_losses[self.heating_months] = 0

        self.ventilation_losses["ventilation losses [kWh]"] = vent_losses

    # this function automatically calls all the relevant functions to perform the thermal balance calculations

    def thermal_balance(self):
        self.transmission_losses_opaque() # ok
        self.transmission_losses_transparent() # ok
        self.transmission_losses_ground() #ok 
        self.sol_gain()# ok
        self.vent_loss() # ok
        self.useful_demand() # ok
        self.total_use_energy_demand() # ok

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
        total_gains[self.outside_temperature.iloc[:, 0] >= 20] = 0 #TODO: i'm not sure this is the correct way to address the
        net_result = total_losses - total_gains 
        net_result[net_result < 0] = 0
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
            self.opaque_surfaces["surface name"] == "wall", "total surface"
        ].sum()

    def get_total_windows_surface(self):
        return self.transparent_surfaces["total surface"].sum()

    def get_total_ground_surface(self):
        return self.ground_contact_surfaces["total surface"].sum()

    def get_total_roof_surface(self):
        return self.opaque_surfaces.loc[
            self.opaque_surfaces["surface name"] == "roof", "total surface"
        ].sum()

    def get_total_door_surface(self):
        return self.opaque_surfaces.loc[
            self.opaque_surfaces["surface name"] == "door", "total surface"
        ].sum()

    def get_total_volume(self):
        return self.volume


# path_weather = "../Irradiation_Data/east.csv"
# weather = pd.read_csv(path_weather, usecols=["T2m"])
# irradiation_path = "../Irradiation_Data/irradiation_data.csv"
# irradiation = pd.read_csv(irradiation_path)
#
# path = "../SFHs.csv"
# components_df = pd.read_csv(path)
# building1 = Building("name", "SFH1", components_df, weather, irradiation)
# building1.global_transmission_coeff()
# # building1.transmission_losses_opaque()
# # building1.transmission_losses_transparent()
# # building1.transmission_losses_ground()
# # building1.total_specific_losses()
# # building1.all_transmission_losses()
# # building1.sol_gain()
# # building1.vent_loss()
# building1.thermal_balance()
#
# # i would like to eventually change how the internal temperature is managed in the code. It can't just always be 20 °C
# # and I do not want it to be passed in every function. I would rather have it as a part of the building. This
# # way one wouldn't accidentally use the wrong inside temperature for different building components.
#
#
# import matplotlib.pyplot as plt
#
# # assuming building1 is your Building instance
# # building1.useful_demand()
# # building1.year_losses()
#
# global_uvalue=building1.get_global_uvalue()
# print(global_uvalue)
#
#
# building1.get_useful_demand().plot()
# plt.title('Net Useful Hourly Demand')
# plt.xlabel('Time')
# plt.ylabel('Net Useful Hourly Demand [kWh]')
# plt.show()


if __name__ == "__main__":
    import os
    from pyinstrument import Profiler
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    path_weather = "../Irradiation_Data/east.csv"
    weather = pd.read_csv(path_weather, usecols=["T2m"])
    irradiation_path = "../Irradiation_Data/irradiation_data.csv"
    irradiation = pd.read_csv(irradiation_path)

    # profiler = Profiler()
    # profiler.start()

    # # test for unrenovated SFH building
    # path_unrenovated = "../tests/sfh/sfh_sample.csv"
    # components_df_unrenovated = pd.read_csv(path_unrenovated)
    # building_unrenovated = Building("name", "SFH1", components_df_unrenovated, weather, irradiation)
    # building_unrenovated.thermal_balance()


    # # test for renovated SFH building
    # path_renovated = "../tests/sfh/sfh_sample_renpack3.csv"
    # components_df_renpack3 = pd.read_csv(path_renovated)
    # building_renovated = Building("building_renpack3", "SFH1", components_df_renpack3, weather, irradiation)
    # building_renovated.thermal_balance()

    # #print results
    # print(f"SFH unrenovated Specific HD = {building_unrenovated.get_total_useful_energy_demand()}")
    # print(f"SFH renpack3 Specific HD = {building_renovated.get_total_useful_energy_demand()}")

    # # for reference this was the initial result
    # #     SFH unrenovated Specific HD = Total Useful Energy Demand [kWh]    100452.379147
    # # dtype: float64
    # # SFH renpack3 Specific HD = Total Useful Energy Demand [kWh]    32857.385152

    # # now testing on MFH_large buildings
    # #setting up directories and files with test data
    # path_mfh_large_unrenovated = "../tests/mfh_large/mfh_large_unrenovated.csv"
    # path_mfh_large_renpack3 = "../tests/mfh_large/mfh_large_renpack3.csv"

    # # reading the data to create components dataframes
    # components_mfh_large_unrenovated = pd.read_csv(path_mfh_large_unrenovated)
    # components_mfh_large_renpack3 = pd.read_csv(path_mfh_large_renpack3)

    # # creating the building instances
    # mfh_large_unrenovated = Building("mfh_large_unrenovated", "MFH_LARGE1", components_mfh_large_unrenovated, weather, irradiation)
    # mfh_large_renpack3 = Building("mfh_large_renpack3", "MFH_LARGE1", components_mfh_large_renpack3, weather, irradiation)

    # # calculate the total useful heat demand for both buildings:
    # mfh_large_unrenovated.thermal_balance()
    # mfh_large_renpack3.thermal_balance()

    # #print the results 
    # print(f"MFH Large unrenovated Specific HD = {mfh_large_unrenovated.get_total_useful_energy_demand()}")
    # print(f"MFH Large renpack3 Specific HD = {mfh_large_renpack3.get_total_useful_energy_demand()}")

    # # for reference this was the previous result 
    # # MFH Large unrenovated Specific HD = Total Useful Energy Demand [kWh]    588576.661496
    # # dtype: float64
    # # MFH Large renpack3 Specific HD = Total Useful Energy Demand [kWh]    259179.226384
    # # dtype: float64

    # profiler.stop()
    # output = profiler.output_text(unicode=True, color=True)
    # filtered_output = "\n".join(line for line in output.split("\n") if "pandas" not in line)

    # b1_opaque_loss = mfh_large_renpack3.opaque_losses
    # b1_ground_contact_loss = mfh_large_renpack3.ground_losses
    # b1_transparent_losses = mfh_large_renpack3.transparent_losses
    # b1_ventilation_losses = mfh_large_renpack3.ventilation_losses
    # b1_solar_gain = mfh_large_renpack3.solar_gain
    # b1_wall_losses = b1_opaque_loss.sum().loc["wall"]
    # b1_roof_losses = b1_opaque_loss.sum().loc["roof"]


    # # self.opaque_losses.sum(axis=1)
    #         + self.transparent_losses.sum(axis=1)
    #         + self.ground_losses.sum(axis=1)
    #         + self.ventilation_losses.sum(axis=1)

    # print(filtered_output)
    import sys
    import os
    import timeit

    # Ensure the parent directory is in the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from Databases.mysql_utils.mysql_utils import create_connection, fetch_data

    # test building results
    data = fetch_data(1)
    sfh = Building("test", "SFH1", data, weather, irradiation)
    sfh.thermal_balance()
    total_ued = sfh.get_sum_useful_demand()
    origina_value = 100452.379147
    print(f"Total useful energy demand per year = {total_ued} kWh. Should be: {origina_value} kWh")
    
    def new_losses():
        sfh.thermal_balance()
    n_runs = 1000
    time = timeit.timeit(lambda: new_losses(), number=n_runs)

    print(f"Time for {n_runs} runs: {time} seconds")
