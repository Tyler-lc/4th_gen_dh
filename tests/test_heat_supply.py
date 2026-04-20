"""Behavioural invariants for heat-supply economics.

Carnot COP (heat_supply.carnot_efficiency.carnot_cop):
  - COP > 1 across realistic operating lifts (air-source HP supplying DH)
  - COP decreases monotonically as the lift (T_hot - T_cold) grows, up to
    the COP_max clip

LCOH (costs.heat_supply.calculate_lcoh):
  - LCOH increases strictly with CAPEX
  - LCOH increases strictly with discount rate in this regime (CAPEX is
    paid at t=0 and not discounted; future costs are discounted; future
    heat output is discounted. Higher r shrinks the denominator faster
    than the numerator, pushing LCOH up).
"""

import numpy as np
import pandas as pd
import pytest

from costs.heat_supply import calculate_lcoh
from heat_supply.carnot_efficiency import carnot_cop


# ---------------------------------------------------------------------------
# Carnot COP invariants
# ---------------------------------------------------------------------------
@pytest.mark.equivalence
def test_carnot_cop_above_one_realistic_range():
    """Air-source HP: cold = 0..10 °C outside, hot = 50..90 °C supply.
    Carnot COP with η=0.524 should stay above 1.
    """
    index = pd.RangeIndex(24)
    t_hot = pd.Series([70] * 24, index=index)
    t_cold = pd.Series(np.linspace(-5, 15, 24), index=index)
    cop = carnot_cop(t_hot, t_cold, approach_temperature=5)
    assert (cop > 1.0).all()


@pytest.mark.equivalence
def test_carnot_cop_monotonic_decreasing_in_lift():
    """At fixed T_cold, increasing T_hot widens the lift and must not
    increase COP (may plateau at COP_max clip).
    """
    supplies = [50, 60, 70, 80, 90]
    cops = []
    for t_hot_val in supplies:
        t_hot = pd.Series([t_hot_val])
        t_cold = pd.Series([5.0])
        cops.append(float(carnot_cop(t_hot, t_cold, approach_temperature=5).iloc[0]))
    # Non-increasing (allow equality for the COP_max clip)
    for a, b in zip(cops, cops[1:]):
        assert b <= a + 1e-9, f"COP increased with lift: {cops}"
    # And at least one strict drop between the extremes (not fully clipped)
    assert cops[-1] < cops[0], "COP should drop over a 40 K lift span"


@pytest.mark.equivalence
def test_carnot_cop_respects_max_clip():
    """When the lift is tiny (hot≈cold), Carnot would blow up; the
    implementation clips to COP_max=4.
    """
    t_hot = pd.Series([30.0])
    t_cold = pd.Series([28.0])
    cop = carnot_cop(t_hot, t_cold, approach_temperature=0)
    assert cop.iloc[0] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# LCOH invariants
# ---------------------------------------------------------------------------
@pytest.mark.equivalence
def test_lcoh_strictly_increases_with_capex(hp_lcoh_inputs):
    d = hp_lcoh_inputs
    lcoh_low = calculate_lcoh(
        d["investment_costs"],
        d["fixed_om_series"], d["variable_om_series"],
        d["electricity_costs_series"], d["heat_output_series"],
        d["discount_rate"],
    )
    lcoh_high = calculate_lcoh(
        d["investment_costs"] * 2,
        d["fixed_om_series"], d["variable_om_series"],
        d["electricity_costs_series"], d["heat_output_series"],
        d["discount_rate"],
    )
    assert lcoh_high > lcoh_low


@pytest.mark.equivalence
def test_lcoh_increases_with_discount_rate(hp_lcoh_inputs):
    d = hp_lcoh_inputs
    lcohs = []
    for r in [0.02, 0.05, 0.08, 0.12]:
        lcohs.append(calculate_lcoh(
            d["investment_costs"],
            d["fixed_om_series"], d["variable_om_series"],
            d["electricity_costs_series"], d["heat_output_series"],
            r,
        ))
    for a, b in zip(lcohs, lcohs[1:]):
        assert b > a, f"LCOH should grow with r; got {lcohs}"


@pytest.mark.equivalence
def test_lcoh_mismatched_series_raises(hp_lcoh_inputs):
    d = hp_lcoh_inputs
    short_fixed = d["fixed_om_series"].iloc[:10]
    with pytest.raises(ValueError, match="same length"):
        calculate_lcoh(
            d["investment_costs"],
            short_fixed, d["variable_om_series"],
            d["electricity_costs_series"], d["heat_output_series"],
            d["discount_rate"],
        )
