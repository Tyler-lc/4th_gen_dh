"""Heat supply economics: heat-pump CAPEX/OPEX power-law fits, LCOH.

Cost coefficients in :func:`capital_costs_hp`, :func:`var_oem_hp` and
:func:`fixed_oem_hp` are power-law fits to the Danish Energy Agency
"Technology Data for Heating Plants and District Heating" catalogue
(2020 EUR, VAT-exclusive).
"""

import numpy as np
import pandas as pd


def capital_costs_hp(installed_thermal_capacity: float, source_type: str) -> float:
    """Heat-pump nominal investment cost from a DEA power-law fit.

    Parameters
    ----------
    installed_thermal_capacity : float
        Installed thermal capacity of the heat pump, MW.
    source_type : {"air", "excess_heat"}
        Heat source category. Selects the relevant DEA fit.

    Returns
    -------
    float
        Nominal investment cost in million EUR (2020, VAT-exclusive).
        Startup costs are not included.

    Raises
    ------
    ValueError
        If ``source_type`` is not one of the supported categories.
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
    """Variable O&M cost per year, from a DEA power-law fit.

    Parameters
    ----------
    installed_thermal_capacity : float
        Installed thermal capacity of the heat pump, MW.
    source_type : {"air", "excess_heat"}
        Heat source category.
    yearly_thermal_energy_produced : float
        Annual thermal energy delivered, MWh/year.

    Returns
    -------
    float
        Variable O&M cost in million EUR/year.

    Raises
    ------
    ValueError
        If ``source_type`` is not one of the supported categories.
    """
    if source_type == "air":
        var_oem = (-0.461 * np.log(installed_thermal_capacity)) + 2.852

    elif source_type == "excess_heat":
        var_oem = (-0.461 * np.log(installed_thermal_capacity)) + 2.852
    else:
        raise ValueError("source_type must be 'air' or 'excess_heat'")

    return var_oem * yearly_thermal_energy_produced


def fixed_oem_hp(installed_thermal_capacity: float, source_type: str) -> float:
    """Fixed O&M cost per year (independent of capacity in the current fit).

    Parameters
    ----------
    installed_thermal_capacity : float
        Installed thermal capacity of the heat pump, MW. Currently
        unused — the DEA fit is flat across capacities — but kept in
        the signature for symmetry with :func:`capital_costs_hp` and
        :func:`var_oem_hp`.
    source_type : {"air", "excess_heat"}
        Heat source category.

    Returns
    -------
    float
        Fixed O&M cost in EUR/year.

    Raises
    ------
    ValueError
        If ``source_type`` is not one of the supported categories.
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
    r"""Levelised Cost of Heat over the supplied operating horizon.

    Implements the standard discounted-cash-flow LCOH:

    .. math::

       \text{LCOH} = \frac{C_\text{inv} + \sum_{t=1}^{N}
                            \frac{C_\text{fix}^{t} + C_\text{var}^{t}
                            + C_\text{el}^{t}}{(1+r)^{t}}}
                          {\sum_{t=1}^{N}
                            \frac{Q_\text{out}^{t}}{(1+r)^{t}}}

    Investment is taken at year 0 (undiscounted); operating costs and
    heat output start at year 1 (so the loop variable ``t`` is
    one-indexed via ``(t + 1)`` in the discount factor).

    Parameters
    ----------
    investment_costs : float
        Up-front investment cost in EUR (or any currency, consistently
        applied to the operating series).
    fixed_om_series : pandas.DataFrame
        One column, indexed 0..N-1, with annual fixed O&M cost.
    variable_om_series : pandas.DataFrame
        Same shape as ``fixed_om_series``; annual variable O&M cost.
    electricity_costs_series : pandas.DataFrame
        Same shape; annual electricity cost.
    heat_output_series : pandas.DataFrame
        Same shape; annual delivered heat in MWh (or kWh — the unit of
        the result follows: EUR/MWh or EUR/kWh).
    discount_rate : float
        Real discount rate, e.g. ``0.05`` for 5 %.

    Returns
    -------
    float
        Levelised cost of heat in EUR per unit of ``heat_output_series``.

    Raises
    ------
    ValueError
        If any of the four time series have differing lengths.
    """
    if (
        len(fixed_om_series) != len(variable_om_series)
        or len(fixed_om_series) != len(electricity_costs_series)
        or len(fixed_om_series) != len(heat_output_series)
    ):
        raise ValueError("All input series must have the same length")

    # Start with investment costs at year 0 (not discounted)
    numerator = investment_costs
    denominator = 0

    # The series index starts at 0; the first operating year is year 1,
    # so the discount factor uses (t + 1).
    years = fixed_om_series.index
    for t in years:
        discount_factor = (1 + discount_rate) ** (t + 1)
        numerator += (
            fixed_om_series.iloc[t, 0]
            + variable_om_series.iloc[t, 0]
            + electricity_costs_series.iloc[t, 0]
        ) / discount_factor

        denominator += heat_output_series.iloc[t, 0] / discount_factor

    lcoh = numerator / denominator
    return lcoh


def compute_ouc_residual(discount_rate, npv_years, lcoh_years):
    """Outstanding Unrecovered Capital as a fraction of the DHG investment.

    When LCOH is amortised over a longer horizon (``lcoh_years``) than
    the NPV evaluation period (``npv_years``), the operator
    under-recovers on the investment by year ``npv_years``. The OUC
    quantifies that gap as an *undiscounted* fraction of the original
    investment, suitable to plug in as a residual-value term in the
    NPV calculation.

    Parameters
    ----------
    discount_rate : float
        Real discount rate (e.g. ``0.05``).
    npv_years : int
        Length of the NPV evaluation horizon, years.
    lcoh_years : int
        Length of the LCOH amortisation horizon, years.
        Typically ``lcoh_years > npv_years``.

    Returns
    -------
    float
        Undiscounted residual fraction of investment at year
        ``npv_years``. Multiply by ``investment_costs`` to recover the
        residual value.
    """
    r = discount_rate
    pv_lcoh = sum(1 / (1 + r) ** (t + 1) for t in range(lcoh_years))
    pv_npv = sum(1 / (1 + r) ** (t + 1) for t in range(npv_years))
    recovery_fraction = pv_npv / pv_lcoh
    ouc_pv = 1 - recovery_fraction
    return ouc_pv * (1 + r) ** npv_years


def calculate_revenues(delivered_heat_demand, heat_prices):
    """Revenue from heat sales, element-wise multiplication.

    Parameters
    ----------
    delivered_heat_demand : pandas.Series or float
        Heat delivered (MWh/year, or any time-aligned shape).
    heat_prices : pandas.Series or float
        Heat price (EUR/MWh). When both arguments are series they must
        be the same length.

    Returns
    -------
    pandas.Series or float
        Revenue, same shape as the inputs.
    """
    return delivered_heat_demand * heat_prices


def calculate_future_values(base_values: dict, n_years: int):
    """Project a dict of base values into a flat ``n_years``-row DataFrame.

    Parameters
    ----------
    base_values : dict
        Mapping from stream name (e.g. ``"electricity_cost"``) to its
        base annual value. Each value is broadcast to all years.
    n_years : int
        Number of years to project.

    Returns
    -------
    pandas.DataFrame
        ``n_years`` rows indexed ``0..n_years-1``, one column per key
        in ``base_values``. All rows hold the base value (no escalation
        applied here — escalation is layered in by callers).
    """
    yearly_values = pd.DataFrame(
        {key: [value] * n_years for key, value in base_values.items()},
        index=range(n_years),
    )
    return yearly_values
