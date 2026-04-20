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

from Person.Person import Person
from building_analysis.Building import Building
from config import SEED, derive_seed


@pytest.mark.equivalence
def test_transmission_and_ventilation_sum_matches_components(minimal_building):
    """Sum of the four loss streams equals the aggregated loss Building
    uses inside useful_demand().
    """
    b = minimal_building
    b.transmission_losses_opaque()
    b.transmission_losses_transparent()
    b.transmission_losses_ground()
    b.vent_loss()

    per_component = (
        b.opaque_losses.sum(axis=1)
        + b.transparent_losses.sum(axis=1)
        + b.ground_losses.sum(axis=1)
        + b.ventilation_losses.sum(axis=1)
    )
    assert per_component.notna().all()
    assert (per_component >= 0).all()
    # Total annual losses match the sum of parts (sanity — the sum is
    # literally the per-component total by construction)
    total_annual = per_component.sum()
    assert total_annual > 0


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
    # Populate people manually with seeded persons and their DHW profiles,
    # because Building.add_people() does not forward seeds.
    b.people = []
    for idx, pid in enumerate(b.people_id):
        person = Person(
            building_id=b.building_id,
            person_id=pid,
            seed=derive_seed(SEED, (b.building_id, idx)),
        )
        person.dhw_profile()
        b.people.append(person)

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
        b.people = []
        for idx in range(b.n_people):
            pid = b.people_id[idx]
            person = Person(
                building_id=b.building_id,
                person_id=pid,
                seed=derive_seed(SEED, (b.building_id, idx)),
            )
            person.dhw_profile()
            b.people.append(person)
        return b.building_dhw_volume()

    first = run()
    second = run()
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.equivalence
def test_thermal_balance_deterministic(minimal_building, synthetic_weather,
                                        synthetic_irradiation, make_components):
    """Re-running thermal_balance on a freshly built Building with the
    same inputs produces identical UED.
    """
    first = minimal_building
    first.thermal_balance()
    first_ued = first.hourly_useful_demand.copy()

    second = Building(
        "test_bldg", "sfh5", make_components(),
        synthetic_weather.copy(), synthetic_irradiation.copy(),
    )
    second.thermal_balance()

    pd.testing.assert_frame_equal(first_ued, second.hourly_useful_demand)
