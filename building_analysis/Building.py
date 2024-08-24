import numpy as np
import pandas as pd
import warnings
import json
import warnings
import os
import sys
from typing import Union

# Get the directory of the current script. Withot this line the script can't import the Person class
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Person.Person import Person

#       from ISEK_Integriertes_Städtebauliches_Entwicklungskonzept_für_Griesheim-Mitte_Stand_08_2019_.pdf
#       we know that there are on average 3.36m2/person. So we can calculate the number of people based on the floor area
#       Stadtumbau Griesheim-Mitte _ Stadtplanungsamt Frankfurt am Main.pdf says that there are about 8000 people in Grihesheim mitte


class Building:

    def __init__(
        self,
        building_id: Union[str, int],
        building_type: str,
        components: pd.DataFrame,
        outside_temperature: pd.Series,
        irradiation_data: pd.DataFrame,
        soil_temp: Union[int, list, pd.Series] = 8,
        inside_temp: Union[int, list, pd.Series] = 20,
        year_start: int = 2019,
        summer_start: int = 6,
        summer_end: int = 9,
        verbose: bool = False,
    ):
        self.building_id = building_id
        self.building_type = building_type
        self.components = components
        self.outside_temperature = outside_temperature
        self.soil_temp = soil_temp
        self.inside_temp = inside_temp
        self.summer_start = summer_start
        self.summer_end = summer_end
        self.people = []
        self.building_water_usage = pd.DataFrame()
        self.total_dhw_energy = pd.DataFrame()
        self.verbose = verbose
        self.year_start = year_start

        # Ensure the outside_temperature has a DatetimeIndex and is a Dataframe
        # if not a dataframe then create one with DateTimeIndex
        if not isinstance(outside_temperature, pd.DataFrame):
            start = f"{year_start}-01-01"
            period = len(outside_temperature)
            weather_index = pd.date_range(start=start, periods=period, freq="h")
            self.outside_temperature = pd.DataFrame(
                outside_temperature, index=weather_index, columns=["Outside T °C"]
            )
        # if it is a dataframe but does not have a DatetimeIndex then set it
        elif not isinstance(outside_temperature.index, pd.DatetimeIndex):
            start = f"{year_start}-01-01"
            period = len(outside_temperature)
            weather_index = pd.date_range(start=start, periods=period, freq="h")
            outside_temperature.set_index(weather_index, inplace=True)
            self.outside_temperature = outside_temperature
        # if it is a dataframe and has a DatetimeIndex then just use input data
        else:
            self.outside_temperature = outside_temperature
            weather_index = outside_temperature.index

        self.irradiation_data = pd.DataFrame(index=weather_index)
        self.irradiation_data = irradiation_data

        # if soil temperature is single value then expand it to be as long as the outside_temperature
        # Ensure soil_temp is a pandas Series with the same index as outside_temperature

        self.soil_temp = self.convert_temps(self.soil_temp, self.outside_temperature)
        # convert the inside temperature to a pandas Series
        self.inside_temp = self.convert_temps(
            self.inside_temp, self.outside_temperature
        )

        # set the index for the building_water_usage and total_dhw_energy
        self.building_water_usage.index = weather_index
        self.total_dhw_energy.index = weather_index
        self.building_water_usage["dhw_volume"] = 0
        self.total_dhw_energy["dhw_energy"] = 0

        # gather geometric data from the components dataframe
        self.n_floors = self.components["n_floors"].values[0]
        self.volume = self.components["volume"].values[0]
        self.ground_contact_area = self.components["ground_contact_area"].values[0]
        self.n_people = self.components["n_people"].values[0]
        self.people_id = self.components["people_id"].values[0]
        self.nfa = self.components["NFA"].values[0]
        self.gfa = self.components["GFA"].values[0]

        # from here on we are creating the dataframes for the different surfaces
        self.transparent_surfaces = self.parse_transparent_surfaces(
            self.components["windows"].values[0]
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
        self.internal_heat_sources = pd.DataFrame(
            index=weather_index, columns=["internal_gains [kWh]"]
        )
        self.ventilation_losses = pd.DataFrame(
            index=weather_index, columns=["ventilation losses [kWh]"]
        )
        self.hourly_useful_demand = pd.DataFrame(
            index=weather_index, columns=["net useful hourly demand [kWh]"]
        )
        self.summer_months = self.is_summer(self.outside_temperature.index)

    def convert_temps(self, temp_data, outside_temperature):

        if isinstance(temp_data, (int, float)):
            temp_data = pd.Series(
                [temp_data] * len(outside_temperature),
                index=self.outside_temperature.index,
            )

        elif isinstance(temp_data, list):
            temp_data = pd.Series(temp_data, index=outside_temperature.index)

        elif isinstance(temp_data, pd.Series):
            if not temp_data.index.equals(outside_temperature.index):
                temp_data = pd.Series(temp_data.values, index=outside_temperature.index)

        else:
            raise TypeError("soil_temp must be an int, float, list, or pandas Series")
        return temp_data

    # in this first section we parse the data from the components dataframe to create
    # the data structure we use in the calculations
    def parse_opaque_surfaces(self, components):
        # Extract opaque surfaces data
        return pd.DataFrame(
            {
                "surface_name": ["roof", "wall", "door"],
                "total_surface": [
                    components["roof_area"].values[0],
                    components["walls_area"].values[0],
                    components["door_area"].values[0],
                ],
                "uvalue": [
                    components["roof_u_value"].values[0],
                    components["walls_u_value"].values[0],
                    components["door_u_value"].values[0],
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
                "total_surface": components["ground_contact_area"].values[0],
                "uvalue": components["ground_contact_u_value"].values[0],
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

    def transmission_losses_opaque(self, outside_temp=None, temp_col_index=0):
        """by default the outside temperature is the temperature passed when creating the Building instance
        Also the inside temperature is by default 20 °C constant. It is possible to also pass a list
        """

        if outside_temp is None:
            outside_temp = self.outside_temperature.iloc[:, temp_col_index]

        # convert inside_temp to a pandas Series for vectorized operation
        inside_temp = self.inside_temp

        # call the helper function
        self.opaque_losses = self.compute_losses(
            self.opaque_surfaces, outside_temp, inside_temp
        )

    def transmission_losses_transparent(self, outside_temp=None, temp_col_index=0):
        """by default the outside temperature is the temperature passed when creating the Building instance
        Also the inside temperature is by default 20 °C constant. It is possible to also pass a list
        """

        if outside_temp is None:
            outside_temp = self.outside_temperature.iloc[:, temp_col_index]

        # convert inside_temp to a pandas Series for vectorized operation
        inside_temp = self.inside_temp

        # call the helper function
        self.transparent_losses = self.compute_losses(
            self.transparent_surfaces, outside_temp, inside_temp
        )

    def transmission_losses_ground(self, soil_temp=None, temp_col_index=0):
        """by default the soil temperature is the temperature passed when creating the Building instance
        Also the inside temperature is by default 20 °C constant. It is possible to also pass a list
        """

        if soil_temp is None:
            soil_temp = self.soil_temp

        # convert inside_temp to a pandas Series for vectorized operation
        inside_temp = self.inside_temp

        soil_temp = pd.Series(
            soil_temp, index=self.outside_temperature.index
        )  # convert soil_temp to a pandas Series

        # call the helper function
        self.ground_losses = self.compute_losses(
            self.ground_contact_surfaces, soil_temp, inside_temp
        )

        soil_temp_mask = soil_temp > inside_temp
        self.ground_losses.loc[soil_temp_mask] = 0

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

            mask_inside_temp = self.outside_temperature["T2m"] >= self.inside_temp
            gains.loc[mask_inside_temp] = 0
            self.solar_gain[row["surface_name"]] = gains

    def internal_gains(self, watts_per_sqm: pd.DataFrame = None):
        """calculate the internal gains in the building
        watts_per_sqm: int, optional. Default is 3. The internal gains in the building in W/m2
        """
        if watts_per_sqm is None:
            watts_per_sqm = pd.DataFrame(
                [3] * len(self.outside_temperature),
                index=self.outside_temperature.index,
            )

        internal_heat_sources = pd.DataFrame(index=self.outside_temperature.index)

        internal_heat_sources["internal_gains"] = (
            watts_per_sqm / 1000 * self.net_floor_area
        )

        # set internal gains to 0 during summer months
        internal_heat_sources[self.summer_months] = 0

        # set internal gains to zero when the inside temperature is equal or higher than the setpoint
        inside_temp_mask = self.outside_temperature["T2m"] >= self.inside_temp
        internal_heat_sources.loc[inside_temp_mask, "internal_gains"] = 0

        self.internal_heat_sources = internal_heat_sources

    def vent_loss(self):
        # Qv = 0.34 * (0.4 + 0.2) * V

        inside_temp = self.inside_temp
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
        self.internal_gains()  # ok
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
        total_gains = self.solar_gain.sum(axis=1) + self.internal_heat_sources.sum(
            axis=1
        )
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

    # here we start adding people to the building.

    def add_people(self, n_people: int = None):
        """add people in the building based on the Person.py class.
        To calculate domestic hot water call the method 'domestic_hot_water'"""
        if n_people == None:
            n_people = self.n_people

        if self.n_people == 0:  # if there are no people do not add any
            return

        for ids in self.people_id:
            person_id = ids
            start_year = f"01/01/{self.year_start}"
            self.people.append(
                Person(
                    building_id=self.building_id,
                    person_id=person_id,
                    start_year=start_year,
                )
            )

    def append_water_usage(self, profiles_folder):
        """add a person to the building based on the Person.py class.
        To calculate domestic hot water call the method 'domestic_hot_water'
        This is in case the user wants to add a person that has already been created
        and pre-calculated the hot water demand
        """
        if self.people == []:
            return
        for person in self.people:
            path = os.path.join(profiles_folder, person.person_id + ".csv")
            person.set_dhw_profile(path, self.outside_temperature.index)

    def people_dhw_energy(self):
        """calculate the domestic hot water demand in kWh for each person in the building
        The results are stored in the instance of the Person class
        """
        if self.people == []:
            return
        for person in self.people:
            person.dhw_energy()

    def building_dhw_volume(self):
        """calculate the domestic hot water demand in liters
        to calculate the energy needed call the method dhw_demand
        This function returns the total volume of hot water used in the building
        If you want to check the volume used by each person, you can check Building.people[n].dhw_year
        """
        if not self.people:
            if self.verbose:
                warnings.warn("No people in the building.")
            return self.building_water_usage
        self.building_water_usage = sum([person.dhw_year for person in self.people])
        return self.building_water_usage

    def building_dhw_energy(self):
        """calculate the domestic hot water demand in kWh
        to calculate the volume needed call the method dhw_volume
        This function returns the total energy used by the building
        If you want to check the energy used by each person, you can check Building.people[n].dhw_year
        """
        if not self.people:
            if self.verbose:
                warnings.warn("No people in the building.")
            return self.total_dhw_energy

        self.total_dhw_energy = sum(
            [person.dhw_energy_demand for person in self.people]
        )
        return self.total_dhw_energy

    def get_useful_demand(self):
        return self.hourly_useful_demand

    def get_sum_useful_demand(self):
        return self.hourly_useful_demand.sum().values[0]

    def get_specific_ued(self):
        self.specific_losses = self.get_sum_useful_demand() / self.nfa
        return self.specific_losses

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

    def get_internal_heat_sources(self):
        return self.internal_heat_sources

    def get_ventilation_losses(self):
        return self.ventilation_losses

    def get_hourly_useful_demand(self):
        """the same as get_useful_demand"""
        return self.hourly_useful_demand

    def get_total_useful_energy_demand(self):
        """same results as get_hourly_useful_deamdn and get_sum_useful_demand
        but it comes in pandas.core.series.Series format"""
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

    def get_dhw_sum_volume(self):
        return self.building_dhw_volume().sum().sum()

    def get_dhw_sum_energy(self):
        return self.building_dhw_energy().sum().sum()


if __name__ == "__main__":
    import os
    import sys
    import timeit
    import pandas as pd
    from pathlib import Path

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Ensure the parent directory is in the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    # from Databases.mysql_utils.mysql_utils import create_connection, fetch_data

    # import the weather and irradiation data
    # path_weather = "../Irradiation_Data/east.csv"
    # weather = pd.read_csv(path_weather, usecols=["T2m"])
    # irradiation_path = "../Irradiation_Data/irradiation_data.csv"
    # irradiation = pd.read_csv(irradiation_path)

    # import the weather and irradiation data. This time we will use the same file for both
    city_name = "Frankfurt_Griesheim_Mitte"
    year_start = 2019
    year_end = 2019
    path_weather = f"../irradiation_data/{city_name}_{year_start}_{year_end}/{city_name}_irradiation_data_{year_start}_{year_end}.csv"
    temperature = pd.read_csv(path_weather, usecols=["T2m"])
    irradiation = pd.read_csv(path_weather)
    irradiation = irradiation.filter(regex="G\(i\)")

    # fetching building data. I need to add the
    from utils.building_utilities import process_data

    current_dir = os.path.dirname(os.path.abspath(__file__))
    qgis_path = "building_generator_data/frankfurt_v3.parquet"
    age_distr_path = "building_generator_data/buildings_age.csv"
    height_distr_path = "building_generator_data/ceiling_heights.csv"

    age_distr_abs = os.path.abspath(os.path.join(current_dir, age_distr_path))
    qgis_abs = os.path.abspath(os.path.join(current_dir, qgis_path))
    height_distr_abs = os.path.abspath(os.path.join(current_dir, height_distr_path))

    age_distr = pd.read_csv(age_distr_abs)
    height_distr = pd.read_csv(height_distr_abs)
    qgis_data = pd.read_parquet(qgis_abs)

    res_types = ["mfh", "sfh", "ab", "th"]
    building_data = process_data(
        qgis_path, "parquet", age_distr, height_distr, res_types
    )

    from building_analysis.building_generator import iterator_generate_buildings

    # As I have already run this code, I saved the output to a parquet file. But I left this
    # code here commented as an example.

    # u_value_path = "building_generator_data/archetype_u_values.csv"
    # abs_path = os.path.abspath(os.path.join(current_dir, u_value_path))
    # building_input = iterator_generate_buildings(building_data, abs_path)
    # building_input.to_parquet("building_input.parquet")

    # load up the data about the building stock in Grihesheim Mitte
    building_input = pd.read_parquet("buildingstock/buildingstock.parquet")
    building_values = building_input[building_input["building_usage"] == "sfh"][
        building_input["fid"] == 30
    ]
    building_id = building_values["full_id"]
    building_type = building_values["building_usage"].values[0] + str(
        building_values["age_code"].values[0]
    )

    # adding soil temperatures from https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/soil_temperature/historical/
    # station number 01420.

    soil_temp_path = "../irradiation_data/Frankfurt_Griesheim_Mitte_2019_2019/Frankfurt_Griesheim_Mitte_soil_temperature_2019_2019.csv"
    soil_path_abs = os.path.abspath(os.path.join(current_dir, soil_temp_path))
    df_soil_temp = pd.read_csv(soil_path_abs, usecols=["V_TE0052"])

    # cleaning the data a little. The dwd uses -99.9 to indicate missing data
    # first replace the -99.9 with np.nan
    df_soil_temp.replace(-99.9, np.nan, inplace=True)
    print(
        f"total number of NaN values in soil temperature before fix: {df_soil_temp['V_TE0052'].isna().sum()}"
    )

    # now interpolate the missing values
    df_soil_temp["V_TE0052"] = df_soil_temp["V_TE0052"].interpolate()
    print(
        f"total number of NaN values in soil temperature after fix: {df_soil_temp['V_TE0052'].isna().sum()}"
    )

    # creating a dataframe with the inside temperatures to be used throughout the year
    # it is set to be 20 °c from 8am to 10pm and 17°C at any other time.

    time_index = pd.date_range(start="2019-01-01", periods=8760, freq="h")
    inside_temp = pd.DataFrame(index=time_index)
    inside_temp["inside_temp"] = 20
    mask_heating = inside_temp.index.hour.isin(range(8, 22))

    inside_temp.loc[np.logical_not(mask_heating), "inside_temp"] = 17

    test_sfh = Building(
        building_id,
        building_type,
        building_values,
        temperature,
        irradiation,
        df_soil_temp["V_TE0052"],
        inside_temp["inside_temp"],
    )

    test_sfh.thermal_balance()

    # extracting the hourly data of all the components

    solar_gain = test_sfh.get_solar_gain()
    internal_gains = test_sfh.get_internal_heat_sources()
    ground_losses = test_sfh.get_ground_losses()
    opaque_losses = test_sfh.get_opaque_losses()
    transparent_losses = test_sfh.get_transparent_losses()
    ventilation_losses = test_sfh.get_ventilation_losses()
    useful_demand = test_sfh.get_hourly_useful_demand()

    # saving the data to csv files
    df_results = pd.concat(
        [
            solar_gain,
            internal_gains,
            ground_losses,
            opaque_losses,
            transparent_losses,
            ventilation_losses,
            useful_demand,
        ],
        axis=1,
    )

    ## testing for the people functions
    test_sfh.add_people()

    # check if names are actually somewhat correct
    print(
        f" there are {len(test_sfh.people)} person(s) in the building. There should be: {test_sfh.n_people}"
    )

    # check the if the ids are correct

    i = 0
    for person in test_sfh.people:
        from_class = person.person_id
        manual_name = test_sfh.building_id.values[0] + f"_{i}"
        print(f"person id: {from_class}. It should be {manual_name}")
        print(f" is name correct: {from_class == manual_name}")
        i += 1

    # append the water usage to each person
    test_sfh.append_water_usage("dhw_profiles")
    print("water usage appended correctly")

    # calculate the dhw energy expenditure for each person
    test_sfh.people_dhw_energy()
    print("dhw energy calculated correctly")

    # calculate the total dhw volume used in the building
    dhw_volume = test_sfh.building_dhw_volume()
    print(f"total dhw volume used in the building: {dhw_volume.sum().sum()}")

    # calculate the total dhw energy used in the building
    dhw_energy = test_sfh.building_dhw_energy()
    print(f"total dhw energy used in the building: {dhw_energy.sum().sum()}")

    ## now i want to test a non residential building
    non_res_building = building_input.loc[0, :].to_frame().T
    non_res_id = non_res_building["full_id"]
    non_res_type = non_res_building["building_usage"].values[0] + str(
        non_res_building["age_code"].values[0]
    )

    building_non_res = Building(
        non_res_id,
        non_res_type,
        non_res_building,
        temperature,
        irradiation,
        df_soil_temp["V_TE0052"],
        inside_temp["inside_temp"],
        verbose=True,
    )

    building_non_res.thermal_balance()
    building_non_res.add_people()
    building_non_res.append_water_usage("dhw_profiles")
    dhw_volume_non_res = building_non_res.building_dhw_volume()
    dhw_energy_non_res = building_non_res.building_dhw_energy()
    non_res_space_heating = building_non_res.get_useful_demand()

    # df_results.to_csv("sht_test_results.csv")

    # # on/off toggle some tests and debugging options
    # losses_outputs = 0
    # plot = 0
    # test_speed = 0
    # profiler = 0

    # from Databases.mysql_utils.mysql_utils import create_connection, fetch_data

    # # test building results
    # data = fetch_data(1)

    # if profiler == 0:
    #     sfh = Building("test", "SFH1", data, weather, irradiation)
    #     sfh.thermal_balance()
    #     sfh.add_people(2)
    #     dhw_volume = sfh.dhw_volume()
    #     dhw_energy = sfh.dhw_energy()

    # if profiler == 1:
    #     from pyinstrument import Profiler

    #     profiler = Profiler()
    #     profiler.start()
    #     sfh = Building("test", "SFH1", data, weather, irradiation)
    #     sfh.thermal_balance()
    #     profiler.stop()
    #     output = profiler.output_text(unicode=True, color=True)
    #     filtered_output = "\n".join(
    #         line for line in output.split("\n") if "pandas" not in line
    #     )
    #     print(filtered_output)

    # total_ued = sfh.get_sum_useful_demand()
    # original_value = 100452.379147

    # print(
    #     f"Total useful energy demand per year = {total_ued} kWh. Should be: {original_value} kWh"
    # )

    # if test_speed == 1:
    #     # test the speed of the code
    #     def new_losses():
    #         sfh.thermal_balance()

    #     n_runs = 1000
    #     time = timeit.timeit(lambda: new_losses(), number=n_runs)

    #     print(f"Time for {n_runs} runs: {time} seconds")

    # if plot == 1:
    #     import matplotlib.pyplot as plt

    #     # Assuming `df` is your DataFrame and 'Net Useful Hourly Demand [kWh]' is the column to be plotted
    #     df = sfh.get_useful_demand()

    #     # Calculate the 12-hour rolling average
    #     rolling_window = 96
    #     df["Smoothed"] = (
    #         df["net useful hourly demand [kWh]"].rolling(window=rolling_window).mean()
    #     )

    #     # Plot the original data
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(
    #         df.index,
    #         df["net useful hourly demand [kWh]"],
    #         label="Original Data",
    #         alpha=0.5,
    #     )

    #     # Plot the smoothed data
    #     plt.plot(
    #         df.index,
    #         df["Smoothed"],
    #         label=f"{rolling_window}-hour Rolling Average",
    #         color="red",
    #     )

    #     # Add title and labels
    #     plt.title(
    #         f"Net Useful Hourly Demand with {rolling_window}-hour Rolling Average"
    #     )
    #     plt.xlabel("Time")
    #     plt.ylabel("Net Useful Hourly Demand [kWh]")
    #     plt.legend()

    #     # Show the plot
    #     plt.show()

    # if losses_outputs == 1:
    #     # results from each category of losses and gains are saved in csv files.
    #     test_results = df.to_csv("test_results.csv")
    #     solar_gain = sfh.get_solar_gain()
    #     ground_losses = sfh.get_ground_losses()
    #     opaque_losses = sfh.get_opaque_losses()
    #     transparent_losses = sfh.get_transparent_losses()
    #     ventilation_losses = sfh.get_ventilation_losses()
    #     useful_demand = sfh.get_hourly_useful_demand()
    #     total_useful_energy_demand = sfh.get_total_useful_energy_demand()

    #     from pathlib import Path

    #     Path("test_results").mkdir(parents=True, exist_ok=True)
    #     solar_gain.to_csv("test_results/solar_gain.csv")
    #     ground_losses.to_csv("test_results/ground_losses.csv")
    #     opaque_losses.to_csv("test_results/opaque_losses.csv")
    #     transparent_losses.to_csv("test_results/transparent_losses.csv")
    #     ventilation_losses.to_csv("test_results/ventilation_losses.csv")
    #     useful_demand.to_csv("test_results/useful_demand.csv")
    #     total_useful_energy_demand.to_csv("test_results/total_useful_energy_demand.csv")
    #     print("Results saved in test_results folder.")

    # # def thermal_balance(self):
    # #     self.transmission_losses_opaque()  # ok
    # #     self.transmission_losses_transparent()  # ok
    # #     self.transmission_losses_ground()  # ok
    # #     self.sol_gain()  # ok
    # #     self.vent_loss()  # ok
    # #     self.useful_demand()  # ok
    # #     self.total_use_energy_demand()  # ok
