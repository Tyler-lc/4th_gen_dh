"""Smoke tests: imports work and key files exist."""

import pytest
from pathlib import Path


@pytest.mark.smoke
class TestImports:
    """Verify that all core modules can be imported."""

    def test_import_building(self):
        from building_analysis.Building import Building

    def test_import_person(self):
        from Person.Person import Person

    def test_import_heat_supply(self):
        from costs.heat_supply import calculate_lcoh

    def test_import_renovation_costs(self):
        from costs.renovation_costs import calculate_npv


@pytest.mark.smoke
class TestKeyFilesExist:
    """Verify that key result files exist on disk."""

    KEY_FILES = [
        "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results_unrenovated.parquet",
        "building_analysis/results/renovated_whole_buildingstock/buildingstock_results_renovated.parquet",
        "building_analysis/results/booster_whole_buildingstock/buildingstock_booster_whole_buildingstock_results.parquet",
        "building_analysis/buildingstock/buildingstock.parquet",
        "grid_calculation/unrenovated_result_df.parquet",
        "grid_calculation/renovated_result_df.parquet",
        "grid_calculation/booster_result_df.parquet",
    ]

    @pytest.mark.parametrize("rel_path", KEY_FILES)
    def test_file_exists(self, project_root, rel_path):
        filepath = project_root / rel_path
        assert filepath.exists(), f"Key result file missing: {rel_path}"
