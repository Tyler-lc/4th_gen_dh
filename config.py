"""Central configuration for the 4th Generation District Heating project.

All directory paths, data file paths, and study-area constants live here
so that pipeline scripts never need os.chdir() or sys.path hacks.
"""

import hashlib
import os
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

# ── Stochastic pipeline seeding ──────────────────────────────────────────
# Single source of truth for every randomised step in the pipeline
# (Person occupancy + DHW, building U-value jitter, age-code assignment).
# Every script that introduces randomness reads `SEED` from here and derives
# a per-entity seed via `derive_seed`, so a fresh run reproduces the same
# buildingstock, the same Persons, and the same downstream results bit for bit.
SEED = 42


def derive_seed(base: int, key) -> int:
    """Return a deterministic 32-bit seed from a base integer and an arbitrary key.

    Uses SHA-256 over ``f"{base}|{key}"`` and returns the first 32 bits as an
    integer, giving a stable, cross-process, cross-platform mapping
    ``(base, key) -> seed``. Python's built-in ``hash()`` is not used because
    it is randomised per interpreter session for strings, which would break
    reproducibility across runs.

    Parameters
    ----------
    base : int
        Root seed, typically ``config.SEED``.
    key : Any
        Stable identifier for the entity being seeded (e.g. a building's
        ``full_id``, a ``(building_id, person_index)`` tuple). Converted to
        string via ``str(key)``.

    Returns
    -------
    int
        A 32-bit non-negative integer suitable for ``numpy.random.RandomState``.
    """
    digest = hashlib.sha256(f"{base}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


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


def grid_results_parquet(scenario: str) -> Path:
    """Return path to the grid-optimisation result parquet for a scenario.

    Example: ``grid_calculation/unrenovated_result_df.parquet``
    """
    return GRID_CALCULATION_DIR / f"{scenario}_result_df.parquet"


# ── Booster-specific paths (different naming convention) ─────────────
def booster_buildingstock_results_path(size: str = "whole_buildingstock") -> Path:
    """Return path to the booster buildingstock results parquet.

    Booster uses ``buildingstock_booster_{size}_results.parquet`` instead of
    the ``buildingstock_results_booster.parquet`` pattern used by other scenarios.
    """
    return results_dir("booster", size) / f"buildingstock_booster_{size}_results.parquet"


def booster_area_results_path(size: str = "whole_buildingstock") -> Path:
    """Return path to the booster area-results CSV.

    Booster nests area results in a subdirectory:
    ``area_results/area_results_booster_{size}.csv``
    """
    return results_dir("booster", size) / "area_results" / f"area_results_booster_{size}.csv"


# ── Scenario temperature parameters ───────────────────────────────────
SCENARIO_TEMPERATURES = {
    "unrenovated": {"supply": 90, "return": 65},   # HT
    "renovated":   {"supply": 50, "return": 25},    # LT
    "booster":     {"supply": 50, "return": 25},    # Booster
}

# ── Pipe material roughness [m] ──────────────────────────────────────
PIPE_ROUGHNESS = {
    "steel_new":  0.045e-3,    # 0.045 mm — new pre-insulated DH pipe
    "steel_aged": 0.5e-3,      # 0.5 mm — aged/corroded steel
    "pvc_pe":     0.007e-3,    # 0.007 mm — PVC / polyethylene
    "copper":     0.0015e-3,   # 0.0015 mm
}
DEFAULT_PIPE_MATERIAL = "steel_new"

# ── Pump parameters ──────────────────────────────────────────────────
PUMP_EFFICIENCY_ELECTRIC = 0.90    # motor efficiency
PUMP_EFFICIENCY_HYDRAULIC = 0.80   # pump hydraulic efficiency
K_BEND_90 = 0.3                    # long-radius 90-degree bend, standard DH
ANNUAL_HOURS = 8760                # hours/year (consistent with scenario scripts)

# ── Pumping losses output ────────────────────────────────────────────
PUMPING_LOSSES_DIR = PROJECT_ROOT / "pumping_losses"

# ── Paper figure directory (optional, machine-specific) ────────────
# Set PAPER_FIGURE_DIR environment variable to copy figures to paper dir.
# If unset, figures are only saved to PLOTS_DIR.
_paper_fig = os.environ.get("PAPER_FIGURE_DIR")
PAPER_FIGURE_DIR = Path(_paper_fig) if _paper_fig else None

# ── Sensitivity analysis paths ───────────────────────────────────────
SENSITIVITY_PARAMS_PATH = SENSITIVITY_DIR / "sensitivity_analysis_parameters.xlsx"


# Note: typo in existing directory name ("multitple"), kept for compatibility
MULTIPLE_GRAPHS_SUBDIR = "multitple_graphs"


def sensitivity_results_dir(scenario: str, analysis_type: str) -> Path:
    """Return ``sensitivity_analysis/<scenario>/<analysis_type>/``."""
    return SENSITIVITY_DIR / scenario / analysis_type


def grid_results_sensitivity_parquet(scenario: str, supply_temperature: int) -> Path:
    """Return path to a sensitivity grid-optimisation parquet.

    Example: ``grid_calculation/sensitivity_analysis/booster/50/booster_result_df_50.parquet``
    """
    return (
        GRID_CALCULATION_DIR / "sensitivity_analysis" / scenario
        / str(supply_temperature)
        / f"booster_result_df_{supply_temperature}.parquet"
    )
