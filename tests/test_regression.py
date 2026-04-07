"""Regression tests: verify result files match the golden baseline.

Uses the golden baseline captured by tests/capture_golden_baseline.py.
Each test checks that a result file's shape, columns, and numeric summary
statistics match the baseline within a tolerance (rtol=1e-6).

MD5 hashes are NOT checked here because parquet metadata can vary across
pandas/pyarrow versions without changing the data.  The numeric stats
comparison catches actual data drift.
"""

import numpy as np
import pandas as pd
import pytest

from config import PROJECT_ROOT

# Import KEY_RESULT_FILES from capture script so the parametrization stays
# in sync with what the baseline actually captures.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "capture_golden_baseline",
    str(PROJECT_ROOT / "tests" / "capture_golden_baseline.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
KEY_RESULT_FILES = _mod.KEY_RESULT_FILES


def _read_file(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ── Key result files (aggregate outputs the paper relies on) ────────


@pytest.mark.regression
@pytest.mark.parametrize("rel_path", KEY_RESULT_FILES)
def test_key_file_shape_and_columns(golden_baseline, rel_path):
    """Check that row count and column names match the baseline."""
    if rel_path not in golden_baseline:
        pytest.skip(f"{rel_path} not in baseline")

    baseline = golden_baseline[rel_path]
    if "summary" not in baseline:
        pytest.skip(f"{rel_path} has no summary in baseline")

    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        pytest.fail(f"Result file missing: {rel_path}")

    df = _read_file(full_path)
    expected = baseline["summary"]

    assert list(df.shape) == expected["shape"], (
        f"Shape mismatch: got {list(df.shape)}, expected {expected['shape']}"
    )
    assert list(df.columns) == expected["columns"], (
        f"Column mismatch for {rel_path}"
    )


@pytest.mark.regression
@pytest.mark.parametrize("rel_path", KEY_RESULT_FILES)
def test_key_file_numeric_stats(golden_baseline, rel_path):
    """Check that numeric column statistics match the baseline within tolerance."""
    if rel_path not in golden_baseline:
        pytest.skip(f"{rel_path} not in baseline")

    baseline = golden_baseline[rel_path]
    summary = baseline.get("summary", {})
    expected_stats = summary.get("numeric_stats", {})
    if not expected_stats:
        pytest.skip(f"{rel_path} has no numeric stats in baseline")

    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        pytest.fail(f"Result file missing: {rel_path}")

    df = _read_file(full_path)
    rtol = 1e-6

    for col, exp in expected_stats.items():
        if col not in df.columns:
            pytest.fail(f"Column '{col}' missing from {rel_path}")

        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        actual_mean = float(col_data.mean())
        actual_min = float(col_data.min())
        actual_max = float(col_data.max())

        np.testing.assert_allclose(
            actual_mean, exp["mean"], rtol=rtol,
            err_msg=f"{rel_path}:{col} mean",
        )
        np.testing.assert_allclose(
            actual_min, exp["min"], rtol=rtol,
            err_msg=f"{rel_path}:{col} min",
        )
        np.testing.assert_allclose(
            actual_max, exp["max"], rtol=rtol,
            err_msg=f"{rel_path}:{col} max",
        )


# ── Broad scan: all baseline files exist and have stable shape ──────


@pytest.mark.regression
def test_all_baseline_files_exist(golden_baseline):
    """Every file recorded in the baseline should still exist on disk."""
    missing = []
    for rel_path in golden_baseline:
        if not (PROJECT_ROOT / rel_path).exists():
            missing.append(rel_path)
    assert not missing, f"{len(missing)} baseline files missing:\n" + "\n".join(missing[:20])


@pytest.mark.regression
def test_all_baseline_shapes_stable(golden_baseline):
    """Spot-check that shapes haven't changed for any baseline file."""
    mismatches = []
    for rel_path, entry in golden_baseline.items():
        summary = entry.get("summary")
        if not summary:
            continue
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            continue

        try:
            df = _read_file(full_path)
        except Exception:
            continue

        if list(df.shape) != summary["shape"]:
            mismatches.append(
                f"  {rel_path}: got {list(df.shape)}, expected {summary['shape']}"
            )

    assert not mismatches, (
        f"{len(mismatches)} files with shape changes:\n" + "\n".join(mismatches[:20])
    )
