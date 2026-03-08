"""Verify that config.py helpers resolve to the same absolute paths as
the old hardcoded strings that the pipeline scripts used to contain.

This catches typos or naming mismatches introduced during the Phase-4
migration from hardcoded paths → config imports.
"""

from pathlib import Path

import pytest

from config import (
    PROJECT_ROOT,
    BUILDINGSTOCK_DIR,
    RESULTS_DIR,
    DHW_PROFILES_DIR,
    GRID_CALCULATION_DIR,
    IRRADIATION_DIR,
    COSTS_DIR,
    SENSITIVITY_DIR,
    PLOTS_DIR,
    CITY_NAME,
    YEAR_START,
    YEAR_END,
    results_dir,
    buildingstock_results_path,
    area_results_path,
    weather_data_path,
    soil_temperature_path,
    grid_results_parquet,
    booster_buildingstock_results_path,
    booster_area_results_path,
    sensitivity_results_dir,
    grid_results_sensitivity_parquet,
    SENSITIVITY_PARAMS_PATH,
)

ROOT = PROJECT_ROOT


# ── Directory constants ──────────────────────────────────────────────

class TestDirectoryConstants:
    def test_results_dir(self):
        assert RESULTS_DIR == ROOT / "building_analysis" / "results"

    def test_grid_calculation_dir(self):
        assert GRID_CALCULATION_DIR == ROOT / "grid_calculation"

    def test_sensitivity_dir(self):
        assert SENSITIVITY_DIR == ROOT / "sensitivity_analysis"

    def test_plots_dir(self):
        assert PLOTS_DIR == ROOT / "plots"

    def test_irradiation_dir(self):
        assert IRRADIATION_DIR == ROOT / "irradiation_data"


# ── Unrenovated / Renovated scenario paths ───────────────────────────

class TestUnrenovatedPaths:
    """Paths that were hardcoded in 05b_HT_Scenario.py."""

    def test_grid_parquet(self):
        expected = ROOT / "grid_calculation" / "unrenovated_result_df.parquet"
        assert grid_results_parquet("unrenovated") == expected

    def test_area_results(self):
        expected = ROOT / "building_analysis" / "results" / "unrenovated_whole_buildingstock" / "area_results_unrenovated.csv"
        assert area_results_path("unrenovated") == expected

    def test_buildingstock_results(self):
        expected = ROOT / "building_analysis" / "results" / "unrenovated_whole_buildingstock" / "buildingstock_results_unrenovated.parquet"
        assert buildingstock_results_path("unrenovated") == expected

    def test_weather_data(self):
        expected = ROOT / "irradiation_data" / "Frankfurt_Griesheim_Mitte_2019_2019" / "Frankfurt_Griesheim_Mitte_irradiation_data_2019_2019.csv"
        assert weather_data_path() == expected

    def test_soil_temperature(self):
        expected = ROOT / "irradiation_data" / "Frankfurt_Griesheim_Mitte_2019_2019" / "Frankfurt_Griesheim_Mitte_soil_temperature_2019_2019.csv"
        assert soil_temperature_path() == expected


class TestRenovatedPaths:
    """Paths that were hardcoded in 07_LT_Scenario2.py."""

    def test_grid_parquet(self):
        expected = ROOT / "grid_calculation" / "renovated_result_df.parquet"
        assert grid_results_parquet("renovated") == expected

    def test_area_results(self):
        expected = ROOT / "building_analysis" / "results" / "renovated_whole_buildingstock" / "area_results_renovated.csv"
        assert area_results_path("renovated") == expected

    def test_buildingstock_results(self):
        expected = ROOT / "building_analysis" / "results" / "renovated_whole_buildingstock" / "buildingstock_results_renovated.parquet"
        assert buildingstock_results_path("renovated") == expected


# ── Booster scenario paths ───────────────────────────────────────────

class TestBoosterPaths:
    """Paths that were hardcoded in 08_Booster_Scenario.py."""

    def test_grid_parquet(self):
        expected = ROOT / "grid_calculation" / "booster_result_df.parquet"
        assert grid_results_parquet("booster") == expected

    def test_buildingstock_results(self):
        expected = ROOT / "building_analysis" / "results" / "booster_whole_buildingstock" / "buildingstock_booster_whole_buildingstock_results.parquet"
        assert booster_buildingstock_results_path() == expected

    def test_area_results(self):
        expected = ROOT / "building_analysis" / "results" / "booster_whole_buildingstock" / "area_results" / "area_results_booster_whole_buildingstock.csv"
        assert booster_area_results_path() == expected


# ── Sensitivity analysis paths ───────────────────────────────────────

class TestSensitivityPaths:
    """Paths that were hardcoded in 09b–09d scripts."""

    def test_params_xlsx(self):
        expected = ROOT / "sensitivity_analysis" / "sensitivity_analysis_parameters.xlsx"
        assert SENSITIVITY_PARAMS_PATH == expected

    def test_sensitivity_results_dir(self):
        expected = ROOT / "sensitivity_analysis" / "unrenovated" / "gas_price"
        assert sensitivity_results_dir("unrenovated", "gas_price") == expected

    def test_grid_sensitivity_parquet(self):
        expected = ROOT / "grid_calculation" / "sensitivity_analysis" / "booster" / "50" / "booster_result_df_50.parquet"
        assert grid_results_sensitivity_parquet("booster", 50) == expected


# ── Files actually exist on disk ─────────────────────────────────────

class TestFilesExist:
    """Spot-check that key paths resolve to files that actually exist."""

    @pytest.mark.parametrize("path", [
        grid_results_parquet("unrenovated"),
        grid_results_parquet("renovated"),
        grid_results_parquet("booster"),
        buildingstock_results_path("unrenovated"),
        buildingstock_results_path("renovated"),
        booster_buildingstock_results_path(),
        area_results_path("unrenovated"),
        area_results_path("renovated"),
        booster_area_results_path(),
        weather_data_path(),
        soil_temperature_path(),
        SENSITIVITY_PARAMS_PATH,
    ])
    def test_file_exists(self, path):
        assert path.exists(), f"Expected file not found: {path}"
