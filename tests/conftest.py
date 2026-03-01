"""Shared test fixtures for the 4th Gen DH test suite."""

import json
from pathlib import Path

import pytest

from config import PROJECT_ROOT
GOLDEN_BASELINE_PATH = Path(__file__).parent / "golden_baseline.json"


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
