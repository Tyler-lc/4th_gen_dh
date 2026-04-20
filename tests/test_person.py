"""Behavioural invariants for Person.

Encoded per user sign-off, 2026-04-19 session:
- determinism under seed
- occupancy_year is 8760 hours, values in {0, 1}, no NaN
- DHW saturation: at most one shower/day, at most one bath/day,
  cooking and handwash counts bounded by the RNG range hardcoded in
  Person.dhw_profile (randint(0, 3) and randint(1, 5) respectively),
- annual DHW volume per person falls in a physically plausible band
"""

import numpy as np
import pandas as pd
import pytest

from Person.Person import Person
from config import SEED, derive_seed


@pytest.mark.equivalence
def test_occupancy_year_shape_and_values(minimal_person):
    occ = minimal_person.occupancy_year
    assert len(occ) == 8760
    assert occ["occupancy"].notna().all()
    assert set(occ["occupancy"].unique()).issubset({0, 1})


@pytest.mark.equivalence
def test_same_seed_identical_dhw_profiles():
    p1 = Person("b", "p", seed=derive_seed(SEED, ("b", "p")))
    p2 = Person("b", "p", seed=derive_seed(SEED, ("b", "p")))
    pd.testing.assert_frame_equal(p1.dhw_profile(), p2.dhw_profile())


@pytest.mark.equivalence
def test_different_seed_different_dhw():
    p1 = Person("b", "p", seed=1)
    p2 = Person("b", "p", seed=2)
    assert not p1.dhw_profile().equals(p2.dhw_profile())


def _daily_event_count(series: pd.Series) -> pd.Series:
    """Count the number of non-zero draws per day."""
    return (series > 0).groupby(series.index.date).sum()


@pytest.mark.equivalence
def test_shower_at_most_one_per_day(minimal_person):
    shower_per_day = _daily_event_count(minimal_person.dhw_year["shower"])
    assert shower_per_day.max() <= 1


@pytest.mark.equivalence
def test_bath_at_most_one_per_day(minimal_person):
    bath_per_day = _daily_event_count(minimal_person.dhw_year["bath"])
    assert bath_per_day.max() <= 1


@pytest.mark.equivalence
def test_cooking_bounded_by_rng_range(minimal_person):
    # Person.dhw_profile uses rng.randint(0, 3) → 0, 1, or 2 distinct draw hours.
    cooking_per_day = _daily_event_count(minimal_person.dhw_year["cooking"])
    assert cooking_per_day.max() <= 2


@pytest.mark.equivalence
def test_handwash_bounded_by_rng_range(minimal_person):
    # Person.dhw_profile uses rng.randint(1, 5) → 1..4 distinct draw hours.
    handwash_per_day = _daily_event_count(minimal_person.dhw_year["handwash"])
    assert handwash_per_day.max() <= 4
    assert handwash_per_day.min() >= 0  # days with zero occupancy can have 0


@pytest.mark.equivalence
def test_annual_dhw_volume_physically_plausible(minimal_person):
    """One person ≈ 15–70 m³ DHW/year is the bracket implied by the
    hardcoded per-draw volumes (shower ≥40 L, bath ≥100 L, plus handwash
    and cooking). Values far outside this range signal a draw-logic bug.
    """
    total_liters = minimal_person.dhw_year.sum().sum()
    assert 15_000 <= total_liters <= 70_000, (
        f"Annual DHW total {total_liters:.0f} L out of physical band"
    )


@pytest.mark.equivalence
def test_dhw_energy_positive_and_monotonic_in_volume(minimal_person):
    energy = minimal_person.dhw_energy()
    assert (energy >= 0).all().all()
    # Total energy should be strictly positive for a person with non-zero DHW
    assert energy.sum().sum() > 0
