"""Fold B – Deterministic seeding tests.

Verify that every stochastic component produces identical results when
given the same seed and different results when given a different seed.
Also verify that seed=None (the default) still works without crashing.
"""

import numpy as np
import pytest

from Person.Person import Person


# ---------------------------------------------------------------------------
# Person determinism
# ---------------------------------------------------------------------------
class TestPersonDeterminism:
    """Same seed  -> identical profiles; different seed -> different profiles."""

    @pytest.mark.equivalence
    def test_same_seed_same_occupancy(self):
        p1 = Person(building_id="b1", person_id="p1", seed=42)
        p2 = Person(building_id="b1", person_id="p1", seed=42)

        np.testing.assert_array_equal(
            p1.workday_occupancy_pdf, p2.workday_occupancy_pdf
        )
        np.testing.assert_array_equal(
            p1.freeday_occupancy_pdf, p2.freeday_occupancy_pdf
        )
        assert p1.workday_wakeup_category == p2.workday_wakeup_category
        assert p1.freeday_wakeup_category == p2.freeday_wakeup_category

    @pytest.mark.equivalence
    def test_same_seed_same_occupancy_year(self):
        p1 = Person(building_id="b1", person_id="p1", seed=42)
        p2 = Person(building_id="b1", person_id="p1", seed=42)

        np.testing.assert_array_equal(
            p1.occupancy_year.values, p2.occupancy_year.values
        )

    @pytest.mark.equivalence
    def test_same_seed_same_dhw(self):
        p1 = Person(building_id="b1", person_id="p1", seed=42)
        p2 = Person(building_id="b1", person_id="p1", seed=42)

        dhw1 = p1.dhw_profile()
        dhw2 = p2.dhw_profile()

        np.testing.assert_array_equal(dhw1.values, dhw2.values)

    @pytest.mark.equivalence
    def test_different_seed_different_profiles(self):
        p1 = Person(building_id="b1", person_id="p1", seed=42)
        p2 = Person(building_id="b1", person_id="p1", seed=99)

        # At least one array should differ (occupancy or DHW).
        occ_differ = not np.array_equal(
            p1.occupancy_year.values, p2.occupancy_year.values
        )
        dhw1 = p1.dhw_profile()
        dhw2 = p2.dhw_profile()
        dhw_differ = not np.array_equal(dhw1.values, dhw2.values)

        assert occ_differ or dhw_differ, "Different seeds should produce different output"

    @pytest.mark.equivalence
    def test_no_seed_no_crash(self):
        """Default (seed=None) must not raise."""
        p = Person(building_id="b1", person_id="p1")
        dhw = p.dhw_profile()
        assert len(dhw) > 0


# ---------------------------------------------------------------------------
# Building-generator determinism (U-value randomisation)
# ---------------------------------------------------------------------------
class TestBuildingGeneratorDeterminism:

    @pytest.mark.equivalence
    def test_same_rng_same_uvalues(self):
        from building_analysis.building_generator import generate_building

        common_kwargs = dict(
            building_usage="sfh",
            age_code=1,
            building_id=1,
            fid=1,
            osm_id=1,
            plot_area=100.0,
            roof_area=110.0,
            wall_area=200.0,
            volume=300.0,
            building_height=6.0,
            ceiling_height=3.0,
            roof_slope=30.0,
            angles_shared_borders=[],
            cardinal_directions_shared_borders=[],
            u_value_path=str(
                __import__("config").U_VALUES_PATH
            ),
            geometry=None,
            random_factor=0.15,
            convert_wkb=False,
            verbose=False,
        )

        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        gdf1 = generate_building(**common_kwargs, rng=rng1)
        gdf2 = generate_building(**common_kwargs, rng=rng2)

        for col in ["roof_u_value", "walls_u_value", "ground_contact_u_value",
                     "window_u_value", "door_u_value"]:
            assert gdf1[col].values[0] == gdf2[col].values[0], f"{col} mismatch"

    @pytest.mark.equivalence
    def test_no_rng_no_crash(self):
        """Default (rng=None) must not raise."""
        from building_analysis.building_generator import generate_building

        gdf = generate_building(
            building_usage="sfh",
            age_code=1,
            building_id=1,
            fid=1,
            osm_id=1,
            plot_area=100.0,
            roof_area=110.0,
            wall_area=200.0,
            volume=300.0,
            building_height=6.0,
            ceiling_height=3.0,
            roof_slope=30.0,
            angles_shared_borders=[],
            cardinal_directions_shared_borders=[],
            u_value_path=str(
                __import__("config").U_VALUES_PATH
            ),
            geometry=None,
            random_factor=0.15,
            convert_wkb=False,
            verbose=False,
        )
        assert len(gdf) == 1
