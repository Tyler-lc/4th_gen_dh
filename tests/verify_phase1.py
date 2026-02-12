"""Phase 1 Verification: Confirm file I/O fixes are correct.

Checks:
1. All input file paths referenced by changed scripts exist on disk
2. All input files load successfully (parquet via geopandas/pandas, csv via pandas)
3. Loaded data matches golden baseline (shape, column names, numeric stats, MD5)
4. Output files on disk still match golden baseline (nothing accidentally modified)
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).parent.parent

# Load golden baseline
BASELINE_PATH = Path(__file__).parent / "golden_baseline.json"
with open(BASELINE_PATH) as f:
    GOLDEN_BASELINE = json.load(f)

# ─── Files referenced by Phase 1 changed scripts ───

# Input files that scripts try to LOAD (the paths we fixed)
INPUT_FILES = {
    # 03_renovate_buildingstock.py line 18
    "03_renovate_buildingstock.py (input)":
        "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results_unrenovated.parquet",

    # 04_calculate_NPV_renovation.py lines 59, 65, 89, 92
    "04_calculate_NPV_renovation.py (unrenovated input)":
        "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results_unrenovated.parquet",
    "04_calculate_NPV_renovation.py (renovated input)":
        "building_analysis/results/renovated_whole_buildingstock/buildingstock_results_renovated.parquet",

    # grid_calculation/01_unrenovated_grid_calculation.py line 24
    "grid_calculation/01_unrenovated_grid_calculation.py (input)":
        "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results_unrenovated.parquet",

    # grid_calculation/create_grid.py line 143
    "grid_calculation/create_grid.py (input)":
        "building_analysis/results/renovated_whole_buildingstock/buildingstock_results_renovated.parquet",

    # costs/renovation_costs.py lines 496, 500 (in commented example code and functions)
    "costs/renovation_costs.py (unrenovated ref)":
        "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results_unrenovated.parquet",
    "costs/renovation_costs.py (renovated ref)":
        "building_analysis/results/renovated_whole_buildingstock/buildingstock_results_renovated.parquet",
}

# Output files produced by the changed scripts (should still match baseline)
OUTPUT_FILES = {
    # 03_renovate_buildingstock.py outputs
    "03 output: renovated buildingstock":
        "building_analysis/results/renovated_whole_buildingstock/buildingstock_results_renovated.parquet",
    "03 output: area results":
        "building_analysis/results/renovated_whole_buildingstock/area_results_renovated.csv",

    # 04_calculate_NPV_renovation.py outputs
    "04 output: npv data":
        "costs/npv_data_renovated_gas.csv",
    "04 output: renovation costs":
        "costs/renovation_costs.csv",
    "04 output: energy savings":
        "costs/energy_savings_renovated.csv",

    # grid_calculation outputs
    "grid output: unrenovated result":
        "grid_calculation/unrenovated_result_df.parquet",
}


def md5_hash(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file_exists(label: str, rel_path: str) -> bool:
    full_path = PROJECT_ROOT / rel_path
    exists = full_path.exists()
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {label}")
    if not exists:
        print(f"         Expected: {full_path}")
    return exists


def check_file_loads(rel_path: str) -> tuple[bool, pd.DataFrame | None]:
    full_path = PROJECT_ROOT / rel_path
    try:
        if rel_path.endswith(".parquet"):
            df = gpd.read_parquet(full_path)
        else:
            df = pd.read_csv(full_path)
        return True, df
    except Exception as e:
        # Try regular pandas for parquet (some may not be geo)
        if rel_path.endswith(".parquet"):
            try:
                df = pd.read_parquet(full_path)
                return True, df
            except Exception as e2:
                print(f"         Load error: {e2}")
                return False, None
        print(f"         Load error: {e}")
        return False, None


def check_against_baseline(rel_path: str, df: pd.DataFrame) -> list[str]:
    """Compare loaded DataFrame against golden baseline. Returns list of discrepancies."""
    issues = []

    if rel_path not in GOLDEN_BASELINE:
        issues.append(f"Not in golden baseline")
        return issues

    baseline = GOLDEN_BASELINE[rel_path]

    # Check MD5
    full_path = PROJECT_ROOT / rel_path
    current_md5 = md5_hash(full_path)
    if current_md5 != baseline.get("md5"):
        issues.append(f"MD5 mismatch: {current_md5} != {baseline['md5']}")

    # Check shape
    if "summary" in baseline:
        expected_shape = baseline["summary"].get("shape")
        if expected_shape and list(df.shape) != expected_shape:
            issues.append(f"Shape mismatch: {list(df.shape)} != {expected_shape}")

        # Check columns
        expected_cols = baseline["summary"].get("columns")
        if expected_cols:
            current_cols = list(df.columns)
            if current_cols != expected_cols:
                missing = set(expected_cols) - set(current_cols)
                extra = set(current_cols) - set(expected_cols)
                if missing:
                    issues.append(f"Missing columns: {missing}")
                if extra:
                    issues.append(f"Extra columns: {extra}")

        # Check numeric stats (with tolerance)
        expected_stats = baseline["summary"].get("numeric_stats", {})
        for col, stats in expected_stats.items():
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    current_mean = float(col_data.mean())
                    expected_mean = stats.get("mean", 0)
                    if expected_mean != 0:
                        rel_diff = abs(current_mean - expected_mean) / abs(expected_mean)
                        if rel_diff > 1e-6:
                            issues.append(f"Column '{col}' mean drift: {current_mean} vs {expected_mean} (rel_diff={rel_diff:.2e})")

    return issues


def main():
    all_ok = True

    print("=" * 70)
    print("PHASE 1 VERIFICATION: File I/O Consistency Check")
    print("=" * 70)

    # ─── Step 1: Check all input files exist ───
    print("\n--- Step 1: Input file existence ---")
    for label, rel_path in INPUT_FILES.items():
        if not check_file_exists(label, rel_path):
            all_ok = False

    # ─── Step 2: Check all output files exist ───
    print("\n--- Step 2: Output file existence ---")
    for label, rel_path in OUTPUT_FILES.items():
        if not check_file_exists(label, rel_path):
            all_ok = False

    # ─── Step 3: Load input files and verify against baseline ───
    print("\n--- Step 3: Load input files & compare to golden baseline ---")
    checked_paths = set()
    for label, rel_path in INPUT_FILES.items():
        if rel_path in checked_paths:
            continue  # Don't check same file twice
        checked_paths.add(rel_path)

        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            continue

        loaded, df = check_file_loads(rel_path)
        if not loaded:
            print(f"  [FAIL] {rel_path} - could not load")
            all_ok = False
            continue

        issues = check_against_baseline(rel_path, df)
        if issues:
            print(f"  [FAIL] {rel_path}")
            for issue in issues:
                print(f"         {issue}")
            all_ok = False
        else:
            print(f"  [OK]   {rel_path} ({df.shape[0]} rows x {df.shape[1]} cols)")

    # ─── Step 4: Check output files against baseline ───
    print("\n--- Step 4: Output files vs golden baseline ---")
    checked_paths = set()
    for label, rel_path in OUTPUT_FILES.items():
        if rel_path in checked_paths:
            continue
        checked_paths.add(rel_path)

        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            continue

        loaded, df = check_file_loads(rel_path)
        if not loaded:
            print(f"  [FAIL] {rel_path} - could not load")
            all_ok = False
            continue

        issues = check_against_baseline(rel_path, df)
        if issues:
            print(f"  [FAIL] {rel_path}")
            for issue in issues:
                print(f"         {issue}")
            all_ok = False
        else:
            print(f"  [OK]   {rel_path} ({df.shape[0]} rows x {df.shape[1]} cols)")

    # ─── Summary ───
    print("\n" + "=" * 70)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED")
        print("All input/output files exist, load correctly, and match golden baseline.")
    else:
        print("RESULT: SOME CHECKS FAILED (see above)")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
