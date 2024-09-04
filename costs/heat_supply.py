import pandas as pd
import numpy as np


def lorenz_cop(
    t_sink_in: pd.Series,
    t_sink_out: pd.Series,
    t_source_in: pd.Series,
    t_source_out: pd.Series,
    source_type: str,
    installed_thermal_capacity: float,
    approach_temperature: float = 5,
) -> float:
    """
    Calculate the COP of a heat pump using the Lorentz formula.
    COPlorentz = Tlm_sink/(Tlm_sink - Tlm_source) * eta_lorenz
    The lorentz COP is more accurate than Carnot COP in the case of multistage heat pumps.
    https://ens.dk/sites/ens.dk/files/Analyser/technology_data_catalogue_for_el_and_dh.pdf (pag 288)
    Input temperatures are in °C.

    Args:
    t_sink_in: temperature of the sink fluid at the inlet of the heat pump [C]
    t_sink_out: temperature of the sink fluid at the outlet of the heat pump [C]
    t_source_in: temperature of the source fluid at the inlet of the heat pump [C]
    t_source_out: temperature of the source fluid at the outlet of the heat pump [C]
    source_type : type of the source fluid. It can be 'air' or 'excess_heat'
    installed_thermal_capacity: the installed thermal capacity of the heat pump [MW]
    approach_temperature: the additional temperature difference between the sink or source and the heat pump fluid [C]
    """
    t_sink_in = t_sink_in + 273.15 + approach_temperature
    t_sink_out = t_sink_out + 273.15 + approach_temperature
    t_source_in = t_source_in + 273.15 + approach_temperature
    t_source_out = t_source_out + 273.15 + approach_temperature

    cop_lorenz = log_temp(t_sink_in, t_sink_out) / (
        log_temp(t_sink_in, t_sink_out) - log_temp(t_source_in, t_source_out)
    )
    if cop_lorenz > 4:
        cop_lorenz = 4

    elif source_type == "air":
        eta_lorenz = 5.6485 * np.log(installed_thermal_capacity) + 46.929

    elif source_type == "excess_heat":
        eta_lorenz = 4.3399 * np.log(installed_thermal_capacity) + 40.08

    else:
        raise ValueError("source_type must be 'air' or 'excess_heat'")
    return cop_lorenz * eta_lorenz


def log_temp(t_in, t_out):
    return (t_in - t_out) / np.log(t_in / t_out)


def capital_costs_hp(installed_thermal_capacity: float, source_type: str) -> float:
    """
    Calculate the installation cost of a heat pump based on the heat source type and the installed thermal capacity.
    The cost is in Million EUR. Startup costs are ignored.
    Args:
    installed_thermal_capacity: the installed thermal capacity of the heat pump [MW]
    source_type : type of the source fluid. It can be 'air' or 'excess_heat'
    """
    if source_type == "air":
        nominal_investment_total = 1.435 * installed_thermal_capacity ** (-0.219)
    elif source_type == "excess_heat":
        nominal_investment_total = 1.2858 * installed_thermal_capacity ** (-0.266)
    else:
        raise ValueError("source_type must be 'air' or 'excess_heat'")

    return nominal_investment_total


def var_oem_hp(
    installed_thermal_capacity: float,
    source_type: str,
    yearly_thermal_energy_produced: float,
) -> float:
    """
    Calculate the variable operation and maintenance (O&M) costs of a heat pump based on the heat source type and the amount of Energy produced in the time frame.
    The cost is in Million EUR/year (assuming we a whole year of thermal_ernergy_produced)
    Args:
    installed_thermal_capacity: the installed thermal capacity of the heat pump [MW]
    source_type : type of the source fluid. It can be 'air' or 'excess_heat'
    thermal_energy_produced: the thermal energy produced by the heat pump [MWh/year]
    """

    if source_type == "air":
        var_oem = (-0.461 * np.log(installed_thermal_capacity)) + 2.852

    elif source_type == "excess_heat":
        var_oem = (-0.461 * np.log(installed_thermal_capacity)) + 2.852
    else:
        raise ValueError("source_type must be 'air' or 'excess_heat'")

    return var_oem * yearly_thermal_energy_produced


def fixed_oem_hp(installed_thermal_capacity: float, source_type: str) -> float:
    """
    Calculate the fixed operation and maintenance (O&M) costs of a heat pump based on the heat source type and the installed thermal capacity.
    The cost is in Million EUR/year.
    Args:
    installed_thermal_capacity: the installed thermal capacity of the heat pump [MW]
    source_type : type of the source fluid. It can be 'air' or 'excess_heat'
    """
    if source_type == "air":
        fixed_oem = 2126.75

    elif source_type == "excess_heat":
        fixed_oem = 2126.75

    else:
        raise ValueError("source_type must be 'air' or 'excess_heat'")

    return fixed_oem


def calculate_lcoh(
    investment_costs,
    fixed_om_series,
    variable_om_series,
    electricity_costs_series,
    heat_output_series,
    discount_rate,
):
    if (
        len(fixed_om_series) != len(variable_om_series)
        or len(fixed_om_series) != len(electricity_costs_series)
        or len(fixed_om_series) != len(heat_output_series)
    ):
        raise ValueError("All input series must have the same length")
    years = fixed_om_series.index
    denominator = 0
    numerator = 0

    for t in years:
        discount_factor = (1 + discount_rate) ** t
        # Sum the discounted costs
        numerator += (
            fixed_om_series.iloc[t, 0]
            + variable_om_series.iloc[t, 0]
            + electricity_costs_series.iloc[t, 0]
        ) / discount_factor
        # Sum the discounted heat output
        denominator += heat_output_series.iloc[t, 0] / discount_factor

    # Add investment cost as a one-time cost in the first year (not discounted if it's paid upfront)
    numerator += investment_costs

    lcoh = numerator / denominator

    return lcoh


def calculate_revenues(delivered_heat_demand, heat_prices):
    """
    Calculate the revenues from the heat supply. The two input must be the same length. Alternatively, the heat_demand can be a pd.Series and the heat_prices can be a float.
    Args:
    delivered_heat_demand: the heat demand of the customer [MWh/year]
    heat_prices: the prices of the heat supply [EUR/MWh]
    """
    return delivered_heat_demand * heat_prices


def calculate_future_values(base_values: dict, n_years: int):
    """
    Creates a DataFrame of yearly values (e.g., revenues or expenses) for the duration of the analysis.

    Args:
        base_values (dict): A dictionary containing the base values for each stream (e.g., revenue streams or expense categories).
        n_years (int): The number of years to project the values.

    Returns:
        pd.DataFrame: A DataFrame containing the projected yearly values for each stream over the specified number of years.
    """
    yearly_values = pd.DataFrame(
        {key: [value] * n_years for key, value in base_values.items()},
        index=range(n_years),
    )
    return yearly_values
