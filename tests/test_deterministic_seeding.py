"""Fold B – Deterministic seeding tests.

Verify that every stochastic component produces identical results when
given the same seed and different results when given a different seed.
Also verify that seed=None (the default) still works without crashing.

Three layers:
- derive_seed itself (unit).
- Individual components (Person, generate_building) under fixed seed.
- End-to-end production-entry patterns mimicking the top of
  01_create_people.py and the iterator_generate_buildings call in
  01b_create_buildingstock.py, to catch regressions like the
  original Phase 2 wiring gap that this suite would not have noticed
  before Phase 7a.
"""

import numpy as np
import pandas as pd
import pytest

from Person.Person import Person
from config import SEED, derive_seed


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


# ---------------------------------------------------------------------------
# derive_seed unit tests
# ---------------------------------------------------------------------------
class TestDeriveSeed:
    """config.derive_seed must be deterministic, distinct on distinct keys,
    and return a value in the 32-bit unsigned range accepted by
    numpy.random.RandomState.
    """

    @pytest.mark.equivalence
    def test_same_key_same_seed(self):
        assert derive_seed(SEED, "bldg_abc") == derive_seed(SEED, "bldg_abc")

    @pytest.mark.equivalence
    def test_different_keys_different_seeds(self):
        assert derive_seed(SEED, "bldg_abc") != derive_seed(SEED, "bldg_xyz")

    @pytest.mark.equivalence
    def test_tuple_key_supported(self):
        s = derive_seed(SEED, ("bldg_abc", 3))
        assert s != derive_seed(SEED, "bldg_abc")
        assert 0 <= s < 2**32

    @pytest.mark.equivalence
    def test_range_is_32_bit_unsigned(self):
        for key in ["a", "bldg_123", ("b", 0), ("b", 1)]:
            s = derive_seed(SEED, key)
            assert 0 <= s < 2**32

    @pytest.mark.equivalence
    def test_different_base_different_seed(self):
        assert derive_seed(41, "same_key") != derive_seed(42, "same_key")


# ---------------------------------------------------------------------------
# Production-entry reproducibility
# ---------------------------------------------------------------------------
# These tests mimic the exact call patterns used by 01_create_people.py and
# 01b_create_buildingstock.py. They would have failed before Phase 7a because
# the production callers did not pass seeds into Person or
# iterator_generate_buildings.


class TestPersonProductionPattern:
    """Mirrors 01_create_people.py:49 — Person(full_id, p_idx, seed=derive_seed(SEED, (full_id, p_idx)))."""

    @pytest.mark.equivalence
    def test_two_runs_produce_identical_persons(self):
        full_id = "bldg_test_42"
        person_idx = 3

        s = derive_seed(SEED, (full_id, person_idx))
        p1 = Person(full_id, person_idx, seed=s)
        p2 = Person(full_id, person_idx, seed=s)

        np.testing.assert_array_equal(
            p1.occupancy_year.values, p2.occupancy_year.values
        )
        pd.testing.assert_frame_equal(p1.dhw_profile(), p2.dhw_profile())

    @pytest.mark.equivalence
    def test_different_person_idx_different_profile(self):
        full_id = "bldg_test_42"
        p_a = Person(
            full_id, 0, seed=derive_seed(SEED, (full_id, 0))
        )
        p_b = Person(
            full_id, 1, seed=derive_seed(SEED, (full_id, 1))
        )

        # Overwhelmingly likely to differ in occupancy or DHW
        occ_diff = not np.array_equal(
            p_a.occupancy_year.values, p_b.occupancy_year.values
        )
        dhw_diff = not p_a.dhw_profile().equals(p_b.dhw_profile())
        assert occ_diff or dhw_diff, (
            "Persons at different indices within the same building should "
            "receive distinct derived seeds and produce distinct profiles"
        )


class TestBuildingstockIteratorReproducible:
    """Mirrors 01b_create_buildingstock.py call: iterator_generate_buildings(..., seed=SEED).

    Two calls over the same input rows must produce identical U-value
    jitter for every building. This closes the loop: if someone regresses
    the seeding threading in the iterator, this test catches it.
    """

    @pytest.fixture
    def tiny_building_data(self):
        """Minimum viable buildingstock input with two buildings."""
        return pd.DataFrame(
            [
                {
                    "building_usage": "sfh",
                    "age_code": 8,
                    "full_id": "bldg_aaa",
                    "fid": 1,
                    "osm_id": 101,
                    "plot_area": 120.0,
                    "roof_surface": 130.0,
                    "roof_slope": 30.0,
                    "wall_surface": 220.0,
                    "volume": 360.0,
                    "neighbors_count": 0,
                    "height": 6.0,
                    "geometry": None,
                    "ceiling_height": 2.8,
                    "angles_shared_borders_standard": [],
                    "cardinal_dir_shared_borders": [],
                },
                {
                    "building_usage": "mfh",
                    "age_code": 5,
                    "full_id": "bldg_bbb",
                    "fid": 2,
                    "osm_id": 102,
                    "plot_area": 240.0,
                    "roof_surface": 260.0,
                    "roof_slope": 20.0,
                    "wall_surface": 600.0,
                    "volume": 1800.0,
                    "neighbors_count": 0,
                    "height": 12.0,
                    "geometry": None,
                    "ceiling_height": 2.8,
                    "angles_shared_borders_standard": [],
                    "cardinal_dir_shared_borders": [],
                },
            ]
        )

    @pytest.mark.equivalence
    def test_two_iterations_identical(self, tiny_building_data):
        from building_analysis.building_generator import iterator_generate_buildings
        import config

        out1 = iterator_generate_buildings(
            tiny_building_data.copy(),
            str(config.U_VALUES_PATH),
            convert_wkb=False,
            randomization_factor=0.15,
            seed=SEED,
        )
        out2 = iterator_generate_buildings(
            tiny_building_data.copy(),
            str(config.U_VALUES_PATH),
            convert_wkb=False,
            randomization_factor=0.15,
            seed=SEED,
        )

        for col in [
            "roof_u_value",
            "walls_u_value",
            "ground_contact_u_value",
            "window_u_value",
            "door_u_value",
        ]:
            np.testing.assert_array_equal(
                out1[col].values, out2[col].values,
                err_msg=f"{col} not reproducible across runs with same SEED",
            )

    @pytest.mark.equivalence
    def test_different_seed_different_uvalues(self, tiny_building_data):
        from building_analysis.building_generator import iterator_generate_buildings
        import config

        out_a = iterator_generate_buildings(
            tiny_building_data.copy(),
            str(config.U_VALUES_PATH),
            convert_wkb=False,
            randomization_factor=0.15,
            seed=SEED,
        )
        out_b = iterator_generate_buildings(
            tiny_building_data.copy(),
            str(config.U_VALUES_PATH),
            convert_wkb=False,
            randomization_factor=0.15,
            seed=SEED + 1,
        )

        # At least one stochastic column should differ somewhere
        any_diff = any(
            not np.array_equal(out_a[col].values, out_b[col].values)
            for col in [
                "roof_u_value",
                "walls_u_value",
                "ground_contact_u_value",
                "window_u_value",
                "door_u_value",
            ]
        )
        assert any_diff, "Different seeds should yield different U-value jitter"
