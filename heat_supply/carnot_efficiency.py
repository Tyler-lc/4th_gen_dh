import pandas as pd
import numpy as np
from typing import Union


def carnot_cop(
    T_hot: pd.Series,
    T_cold: pd.Series,
    approach_temperature: float,
    carnot_efficiency: float = 0.524,
    COP_max: float = 4,
) -> pd.Series:
    """
    Calculate the COP of a heat pump using the Carnot formula.

    Args:
        T_hot (pd.Series): Hot temperature in Celsius
        T_cold (pd.Series): Cold temperature in Celsius
        approach_temperature (float): Approach temperature to be added.
        carnot_efficiency (float): Carnot efficiency factor. Default is 0.524.

    Returns:
        pd.Series: COP values.
    """
    COP = pd.DataFrame()
    T_hot = T_hot + 273.15 + approach_temperature
    T_cold = T_cold + 273.15 + approach_temperature
    COP["hot"] = T_hot
    COP["cold"] = T_cold
    COP["COP_hourly"] = COP["hot"] / (COP["hot"] - COP["cold"]) * carnot_efficiency

    # Limit the COP to a maximum value of 4
    COP = COP.clip(upper=COP_max)

    return COP["COP_hourly"]


def hp_el_demand(Qhot, COP):
    """
    Calculates the electricity demand of a heat pump give the heat demand and the COP.
    """
    Qel = Qhot / COP
    return Qel
