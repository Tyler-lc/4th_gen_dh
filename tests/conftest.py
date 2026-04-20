"""Shared test fixtures for the 4th Gen DH test suite."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import PROJECT_ROOT
GOLDEN_BASELINE_PATH = Path(__file__).parent / "golden_baseline.json"

HOURS_PER_YEAR = 8760
DEFAULT_YEAR_START = "2019-01-01"


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def golden_baseline():
    """Load the golden baseline reference data.

    Returns a dict keyed by relative file path with entries containing:
    - path, md5, size_bytes
    - summary: shape, columns, numeric_stats (mean, std, min, max, count)
    """
    if not GOLDEN_BASELINE_PATH.exists():
        pytest.skip(
            "Golden baseline not found. Run: python tests/capture_golden_baseline.py"
        )
    with open(GOLDEN_BASELINE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def key_result_files(golden_baseline):
    """Return only the key aggregate result files from the golden baseline."""
    key_patterns = [
        "buildingstock_results_unrenovated.parquet",
        "buildingstock_results_renovated.parquet",
        "buildingstock_booster_whole_buildingstock_results.parquet",
        "area_results_unrenovated.csv",
        "area_results_renovated.csv",
        "unrenovated_result_df.parquet",
        "renovated_result_df.parquet",
        "booster_result_df.parquet",
    ]
    return {
        path: data
        for path, data in golden_baseline.items()
        if any(path.endswith(p) for p in key_patterns)
    }


# ---------------------------------------------------------------------------
# Synthetic weather and irradiation (Phase 7c behavioural fixtures)
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_weather():
    """8760-hour outside temperature DataFrame with a T2m column.

    Sinusoidal annual cycle roughly matching Frankfurt: 0 °C in January,
    20 °C in July. Column name `T2m` matches the PVGIS convention used by
    ``Building.sol_gain()`` and ``Building.internal_gains()``.
    """
    index = pd.date_range(start=DEFAULT_YEAR_START, periods=HOURS_PER_YEAR, freq="h")
    hours = np.arange(HOURS_PER_YEAR)
    # peak 20 °C around hour-of-year 4380 (early July), min 0 °C around hour 0
    annual = 10 - 10 * np.cos(2 * np.pi * hours / HOURS_PER_YEAR)
    diurnal = 3 * np.sin(2 * np.pi * hours / 24)
    return pd.DataFrame({"T2m": annual + diurnal}, index=index)


@pytest.fixture
def synthetic_irradiation():
    """8760-hour irradiation DataFrame with the four cardinal G(i) columns.

    Values are small positive during daylight hours and zero at night, with
    a crude annual amplitude. Enough to exercise solar-gain code paths;
    not intended to be physically precise.
    """
    index = pd.date_range(start=DEFAULT_YEAR_START, periods=HOURS_PER_YEAR, freq="h")
    hours = np.arange(HOURS_PER_YEAR)
    hour_of_day = hours % 24
    daylight = np.maximum(0.0, np.sin(np.pi * (hour_of_day - 6) / 12))
    annual = 0.5 * (1 - np.cos(2 * np.pi * hours / HOURS_PER_YEAR))  # 0..1
    base = daylight * annual  # kWh/m² scale-ish
    return pd.DataFrame(
        {
            "north G(i) [kWh/m2]": 0.3 * base,
            "south G(i) [kWh/m2]": 1.0 * base,
            "east G(i) [kWh/m2]": 0.7 * base,
            "west G(i) [kWh/m2]": 0.7 * base,
        },
        index=index,
    )


def _make_components_row(roof_u=0.30, wall_u=0.35, floor_u=0.40, door_u=1.8,
                         window_u=1.2, window_shgc=0.5,
                         roof_area=130.0, walls_area=200.0, door_area=3.0,
                         ground_contact_area=120.0, volume=360.0,
                         n_floors=2, n_people=2, people_id=None,
                         nfa=204.0, gfa=240.0):
    """Build the single-row components DataFrame a Building instance expects."""
    windows_payload = json.dumps(
        {
            "north": {"area": 8.0, "u_value": window_u, "shgc": window_shgc},
            "south": {"area": 16.0, "u_value": window_u, "shgc": window_shgc},
            "east": {"area": 10.0, "u_value": window_u, "shgc": window_shgc},
            "west": {"area": 10.0, "u_value": window_u, "shgc": window_shgc},
        }
    )
    if people_id is None:
        people_id = [f"p{i}" for i in range(n_people)]
    return pd.DataFrame(
        [{
            "n_floors": n_floors,
            "volume": volume,
            "ground_contact_area": ground_contact_area,
            "n_people": n_people,
            "people_id": people_id,
            "NFA": nfa,
            "GFA": gfa,
            "windows": windows_payload,
            "roof_area": roof_area,
            "walls_area": walls_area,
            "door_area": door_area,
            "roof_u_value": roof_u,
            "walls_u_value": wall_u,
            "door_u_value": door_u,
            "ground_contact_u_value": floor_u,
        }]
    )


@pytest.fixture
def make_components():
    """Factory fixture for per-test components DataFrames with custom params."""
    return _make_components_row


@pytest.fixture
def minimal_building(synthetic_weather, synthetic_irradiation):
    """Instantiate a Building with synthetic weather and default archetype."""
    from building_analysis.Building import Building

    components = _make_components_row()
    return Building(
        building_id="test_bldg",
        building_type="sfh5",
        components=components,
        outside_temperature=synthetic_weather,
        irradiation_data=synthetic_irradiation,
    )


@pytest.fixture
def minimal_person():
    """Return a seeded Person with a fully generated DHW profile."""
    from Person.Person import Person

    p = Person(building_id="test_bldg", person_id="p0", seed=42)
    p.dhw_profile()
    return p


# ---------------------------------------------------------------------------
# LCOH fixture (Phase 7c behavioural fixtures)
# ---------------------------------------------------------------------------
@pytest.fixture
def hp_lcoh_inputs():
    """Return a dict of constant yearly series suitable for calculate_lcoh.

    Mirrors the pattern in tests/test_lcoh_npv.py: investment costs plus
    single-column DataFrames for fixed O&M, variable O&M, electricity cost,
    and heat output, all indexed 0..n_years-1. Values are chosen so that
    LCOH comes out in a reasonable €/MWh range.
    """
    n_years = 25
    investment_costs = 5_000_000.0  # €
    fixed_om = pd.DataFrame({"fixed": [50_000.0] * n_years})
    variable_om = pd.DataFrame({"variable": [20_000.0] * n_years})
    electricity = pd.DataFrame({"electricity": [200_000.0] * n_years})
    heat_output = pd.DataFrame({"heat": [10_000.0] * n_years})  # MWh/year
    discount_rate = 0.05
    return {
        "investment_costs": investment_costs,
        "fixed_om_series": fixed_om,
        "variable_om_series": variable_om,
        "electricity_costs_series": electricity,
        "heat_output_series": heat_output,
        "discount_rate": discount_rate,
        "n_years": n_years,
    }
