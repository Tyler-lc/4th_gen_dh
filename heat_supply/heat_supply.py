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
    The sour
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
    The cost is in EUR. Startup costs are ignored.
    Args:
    installed_thermal_capacity: the installed thermal capacity of the heat pump [MW]
    source_type : type of the source fluid. It can be 'air' or 'excess_heat'
    """
    if source_type == "air":
        nominal_investment_total = 1.435 * installed_thermal_capacity ** (-0.219)
    if source_type == "excess_heat":
        nominal_investment_total = 1.2858 * installed_thermal_capacity ** (-0.266)
    else:
        raise ValueError("source_type must be 'air' or 'excess_heat'")

    return nominal_investment_total


def var_oem_hp(
    installed_thermal_capacity: float,
    source_type: str,
    thermal_energy_produced: pd.Series,
) -> pd.Series:
    """
    Calculate the variable operation and maintenance (O&M) costs of a heat pump based on the heat source type and the installed thermal capacity.
    The cost is in EUR/year (assuming we a whole year of thermal_ernergy_produced)
    Args:
    installed_thermal_capacity: the installed thermal capacity of the heat pump [MW]
    source_type : type of the source fluid. It can be 'air' or 'excess_heat'
    thermal_energy_produced: the thermal energy produced by the heat pump [MWh/year]
    """

    var_oem_year = pd.Series(index=thermal_energy_produced.index)

    if source_type == "air":
        var_oem = -0.461 * np.log(installed_thermal_capacity) + 2.852

    if source_type == "excess_heat":
        var_oem = -0.461 * np.log(installed_thermal_capacity) + 2.852
    else:
        raise ValueError("source_type must be 'air' or 'excess_heat'")

    var_oem_year = var_oem * thermal_energy_produced
    return var_oem_year


def fixed_oem_hp(installed_thermal_capacity: float, source_type: str) -> float:
    """
    Calculate the fixed operation and maintenance (O&M) costs of a heat pump based on the heat source type and the installed thermal capacity.
    The cost is in EUR/year.
    Args:
    installed_thermal_capacity: the installed thermal capacity of the heat pump [MW]
    source_type : type of the source fluid. It can be 'air' or 'excess_heat'
    """
    if source_type == "air":
        fixed_oem = 2126.75

    if source_type == "excess_heat":
        fixed_oem = 2126.75

    else:
        raise ValueError("source_type must be 'air' or 'excess_heat'")

    return fixed_oem


# TODO: Variable O&M are calculated based on the thermal energy produced by the heat pump \
# Also they need to have the inflation applied.
# TODO: the capital costs are calculated based on the installed thermal capacity of the heat pump. Only once.
# TODO: Fixed O&M are calculated based the thermal installed capacity and are applied every year. We need to apply inflation to them.

# TODO: create a function to calculate the size of the heat pump based on the thermal energy demand of the buildings in the area.
# TODO: We might have to set a limit and split the heat pump in two or more heat pumps.
