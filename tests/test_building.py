"""Behavioural invariants for Building thermal balance.

Encoded per user sign-off, 2026-04-19 session:
- Heat loss components are additive: opaque + transparent + ground
  transmission + ventilation equals total transmission+ventilation loss
- Useful energy demand is non-negative element-wise
- Solar gains reduce UED (same building with and without irradiation)
- Archetype boundary: lowering all U-values strictly lowers UED
- DHW aggregation: building_dhw_volume equals sum over Persons
- Determinism: same seed → identical per-Person DHW aggregated at
  building level
"""

import json

import numpy as np
import pandas as pd
import pytest

from building_analysis.Building import Building
from config import SEED


# Analytic reference values for the constant-T fixture (analytic_building).
# Derivation: each loss class collapses to UA·ΔT·heating_hours/1000 because
# T_out, T_soil and T_in are constant and irradiation is zero. ΔT is 20 K
# against air, 12 K against soil. Heating hours = 5832 (months 1-5 + 10-12).
# Window U-values are ≤ 1.4 so unwanted_vent_coeff resolves to 0.2.
ANALYTIC_OPAQUE_KWH = (0.30 * 130 + 0.35 * 200 + 1.8 * 3) * 20 * 5832 / 1000
ANALYTIC_TRANSPARENT_KWH = 1.2 * (8 + 16 + 10 + 10) * 20 * 5832 / 1000
ANALYTIC_GROUND_KWH = 0.40 * 120 * 12 * 5832 / 1000
ANALYTIC_VENTILATION_KWH = 0.34 * (0.2 + 0.2) * 360 * 20 * 5832 / 1000


@pytest.mark.equivalence
def test_opaque_transmission_matches_analytic(analytic_building):
    analytic_building.transmission_losses_opaque()
    assert analytic_building.opaque_losses.values.sum() == pytest.approx(
        ANALYTIC_OPAQUE_KWH, rel=1e-6
    )


@pytest.mark.equivalence
def test_transparent_transmission_matches_analytic(analytic_building):
    analytic_building.transmission_losses_transparent()
    assert analytic_building.transparent_losses.values.sum() == pytest.approx(
        ANALYTIC_TRANSPARENT_KWH, rel=1e-6
    )


@pytest.mark.equivalence
def test_ground_transmission_matches_analytic(analytic_building):
    analytic_building.transmission_losses_ground()
    assert analytic_building.ground_losses.values.sum() == pytest.approx(
        ANALYTIC_GROUND_KWH, rel=1e-6
    )


@pytest.mark.equivalence
def test_ventilation_loss_matches_analytic(analytic_building):
    analytic_building.vent_loss()
    assert analytic_building.ventilation_losses.values.sum() == pytest.approx(
        ANALYTIC_VENTILATION_KWH, rel=1e-6
    )


@pytest.mark.equivalence
def test_total_losses_match_analytic_sum(analytic_building):
    """Cross-check: the four loss streams together equal the analytic sum.
    This is the regression bound that catches any single-component drift
    even if the per-component tests are skipped.
    """
    b = analytic_building
    b.transmission_losses_opaque()
    b.transmission_losses_transparent()
    b.transmission_losses_ground()
    b.vent_loss()

    total = (
        b.opaque_losses.values.sum()
        + b.transparent_losses.values.sum()
        + b.ground_losses.values.sum()
        + b.ventilation_losses.values.sum()
    )
    expected = (
        ANALYTIC_OPAQUE_KWH
        + ANALYTIC_TRANSPARENT_KWH
        + ANALYTIC_GROUND_KWH
        + ANALYTIC_VENTILATION_KWH
    )
    assert total == pytest.approx(expected, rel=1e-6)


@pytest.mark.equivalence
def test_useful_demand_non_negative(minimal_building):
    b = minimal_building
    b.thermal_balance()
    ued = b.hourly_useful_demand
    assert (ued >= 0).all().all()


@pytest.mark.equivalence
def test_solar_gains_reduce_demand(synthetic_weather, synthetic_irradiation, make_components):
    """Identical building + weather, one with solar irradiation and one
    with zero irradiation. The zero-solar building should have UED
    greater than or equal to the sunny one.
    """
    components = make_components()

    sunny = Building(
        "sunny", "sfh5", components.copy(),
        synthetic_weather.copy(), synthetic_irradiation.copy(),
    )
    sunny.thermal_balance()

    dark_irradiation = synthetic_irradiation * 0.0
    dark = Building(
        "dark", "sfh5", components.copy(),
        synthetic_weather.copy(), dark_irradiation,
    )
    dark.thermal_balance()

    sunny_ued = sunny.get_sum_useful_demand()
    dark_ued = dark.get_sum_useful_demand()
    assert dark_ued >= sunny_ued
    assert dark_ued > sunny_ued  # irradiation must make a measurable difference


@pytest.mark.equivalence
def test_lower_u_values_reduce_demand(synthetic_weather, synthetic_irradiation, make_components):
    """Archetype boundary: halving U-values across the envelope must
    reduce annual UED.
    """
    base = make_components()
    insulated = make_components(
        roof_u=0.15, wall_u=0.175, floor_u=0.20, door_u=0.9, window_u=0.6,
    )

    b_base = Building(
        "base", "sfh5", base,
        synthetic_weather.copy(), synthetic_irradiation.copy(),
    )
    b_base.thermal_balance()

    b_ins = Building(
        "ins", "sfh5", insulated,
        synthetic_weather.copy(), synthetic_irradiation.copy(),
    )
    b_ins.thermal_balance()

    assert b_ins.get_sum_useful_demand() < b_base.get_sum_useful_demand()


@pytest.mark.equivalence
def test_dhw_aggregation_equals_person_sum(minimal_building):
    """building_dhw_volume returns the element-wise sum of each Person's
    dhw_year DataFrame.
    """
    b = minimal_building
    b.add_people(seed=SEED)
    for person in b.people:
        person.dhw_profile()

    agg = b.building_dhw_volume()
    expected = sum(person.dhw_year for person in b.people)
    pd.testing.assert_frame_equal(agg, expected)


@pytest.mark.equivalence
def test_building_dhw_deterministic_under_seed(synthetic_weather, synthetic_irradiation, make_components):
    """Two fresh buildings with identical inputs and seeded Persons produce
    byte-identical aggregated DHW.
    """
    def run():
        b = Building(
            "det", "sfh5", make_components(),
            synthetic_weather.copy(), synthetic_irradiation.copy(),
        )
        b.add_people(seed=SEED)
        for person in b.people:
            person.dhw_profile()
        return b.building_dhw_volume()

    first = run()
    second = run()
    pd.testing.assert_frame_equal(first, second)


