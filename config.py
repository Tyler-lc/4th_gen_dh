"""Central configuration for the 4th Generation District Heating project.

All directory paths, data file paths, and study-area constants live here
so that pipeline scripts never need os.chdir() or sys.path hacks.
"""

from pathlib import Path

# ── Project root (directory that contains this file) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Directory constants ───────────────────────────────────────────────────
BUILDINGSTOCK_DIR = PROJECT_ROOT / "building_analysis" / "buildingstock"
RESULTS_DIR = PROJECT_ROOT / "building_analysis" / "results"
DHW_PROFILES_DIR = PROJECT_ROOT / "building_analysis" / "dhw_profiles"
GRID_CALCULATION_DIR = PROJECT_ROOT / "grid_calculation"
IRRADIATION_DIR = PROJECT_ROOT / "irradiation_data"
COSTS_DIR = PROJECT_ROOT / "costs"
SENSITIVITY_DIR = PROJECT_ROOT / "sensitivity_analysis"
PLOTS_DIR = PROJECT_ROOT / "plots"
BUILDING_GENERATOR_DATA_DIR = (
    PROJECT_ROOT / "building_analysis" / "building_generator_data"
)

# ── Data file constants ──────────────────────────────────────────────────
BUILDINGSTOCK_PATH = BUILDINGSTOCK_DIR / "buildingstock.parquet"
QGIS_DATA_PATH = BUILDING_GENERATOR_DATA_DIR / "frankfurt_v3.parquet"
AGE_DISTRIBUTION_PATH = BUILDING_GENERATOR_DATA_DIR / "buildings_age.csv"
CEILING_HEIGHTS_PATH = BUILDING_GENERATOR_DATA_DIR / "ceiling_heights.csv"
U_VALUES_PATH = BUILDING_GENERATOR_DATA_DIR / "archetype_u_values.csv"

# ── Study-area constants ─────────────────────────────────────────────────
CITY_NAME = "Frankfurt_Griesheim_Mitte"
YEAR_START = 2019
YEAR_END = 2019
MAXIMUM_PEOPLE = 9500
RESIDENTIAL_BUILDING_TYPES = ["sfh", "mfh", "ab", "th"]


# ── Helper functions ─────────────────────────────────────────────────────
def results_dir(sim: str, size: str = "whole_buildingstock") -> Path:
    """Return ``building_analysis/results/<sim>_<size>/``."""
    return RESULTS_DIR / f"{sim}_{size}"


def buildingstock_results_path(sim: str, size: str = "whole_buildingstock") -> Path:
    """Return path to ``buildingstock_results_<sim>.parquet``."""
    return results_dir(sim, size) / f"buildingstock_results_{sim}.parquet"


def area_results_path(sim: str, size: str = "whole_buildingstock") -> Path:
    """Return path to ``area_results_<sim>.csv``."""
    return results_dir(sim, size) / f"area_results_{sim}.csv"


def weather_data_path() -> Path:
    """Return path to the combined irradiation/temperature CSV."""
    folder = f"{CITY_NAME}_{YEAR_START}_{YEAR_END}"
    filename = f"{CITY_NAME}_irradiation_data_{YEAR_START}_{YEAR_END}.csv"
    return IRRADIATION_DIR / folder / filename


def soil_temperature_path() -> Path:
    """Return path to the soil-temperature CSV."""
    folder = f"{CITY_NAME}_{YEAR_START}_{YEAR_END}"
    filename = f"{CITY_NAME}_soil_temperature_{YEAR_START}_{YEAR_END}.csv"
    return IRRADIATION_DIR / folder / filename


def grid_results_path(scenario: str) -> Path:
    """Return path to grid-calculation results for a scenario."""
    return GRID_CALCULATION_DIR / "results" / scenario
