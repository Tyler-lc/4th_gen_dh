"""Capture golden baseline from existing result files.

Walks all result directories and records for every .parquet and .csv file:
- Shape (rows, cols), column names
- Summary statistics (mean, std, min, max) for all numeric columns
- MD5 hash of the file

Saves to tests/golden_baseline.json as the regression reference.

Excludes per-building CSV subdirectories (space_heating/, dhw_volumes/,
dhw_energy/) since these are thousands of files whose data is aggregated
in the buildingstock parquet files.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

# Subdirectory names to skip (per-building intermediate results)
SKIP_SUBDIRS = {"space_heating", "dhw_volumes", "dhw_energy", "dhw_volume",
                 "dhw_profiles"}

# Explicit list of key result files to capture
KEY_RESULT_FILES = [
    # Buildingstock results
    "building_analysis/results/unrenovated_whole_buildingstock/buildingstock_results_unrenovated.parquet",
    "building_analysis/results/unrenovated_whole_buildingstock/area_results_unrenovated.csv",
    "building_analysis/results/renovated_whole_buildingstock/buildingstock_results_renovated.parquet",
    "building_analysis/results/renovated_whole_buildingstock/area_results_renovated.csv",
    "building_analysis/results/booster_whole_buildingstock/buildingstock_booster_whole_buildingstock_results.parquet",
    # Buildingstock input
    "building_analysis/buildingstock/buildingstock.parquet",
    # Grid calculation results
    "grid_calculation/unrenovated_result_df.parquet",
    "grid_calculation/renovated_result_df.parquet",
    "grid_calculation/booster_result_df.parquet",
    "grid_calculation/booster_results.csv",
    "grid_calculation/dh_parameters.csv",
    # Cost results
    "costs/renovation_costs.csv",
    "costs/energy_savings_renovated.csv",
    "costs/npv_data_renovated_gas.csv",
]

# Directories to scan (non-recursively for parquets, with skip logic for CSVs)
SCAN_DIRS = [
    "sensitivity_analysis",
    "grid_calculation/sensitivity_analysis",
    "building_analysis/results/sensitivity_analysis",
    "building_analysis/results/booster_whole_buildingstock_50",
    "building_analysis/results/booster_whole_buildingstock_55",
]


def md5_hash(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Extract shape, columns, and numeric summary stats from a DataFrame."""
    summary = {
        "shape": list(df.shape),
        "columns": list(df.columns),
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()) if len(col_data) > 1 else 0.0,
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "count": int(col_data.count()),
                }
        summary["numeric_stats"] = stats

    return summary


def capture_file(filepath: Path) -> dict:
    """Capture metadata and summary for a single file."""
    entry = {
        "path": str(filepath.relative_to(PROJECT_ROOT)),
        "md5": md5_hash(filepath),
        "size_bytes": filepath.stat().st_size,
    }

    try:
        if filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
            entry["summary"] = summarize_dataframe(df)
        elif filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
            entry["summary"] = summarize_dataframe(df)
    except Exception as e:
        entry["read_error"] = str(e)

    return entry


def should_skip(filepath: Path) -> bool:
    """Check if a file is in a per-building subdirectory that should be skipped."""
    parts = filepath.parts
    for skip_dir in SKIP_SUBDIRS:
        if skip_dir in parts:
            return True
    # Also skip per-building subdirs with percentage suffixes like dhw_energy_50
    for part in parts:
        for skip_dir in SKIP_SUBDIRS:
            if part.startswith(skip_dir + "_"):
                return True
    return False


def collect_result_files() -> list[Path]:
    """Collect result files, skipping per-building CSV subdirectories."""
    files = set()

    # Add explicit key result files
    for rel_path in KEY_RESULT_FILES:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            files.add(full_path)
        else:
            print(f"Warning: Key file {rel_path} does not exist")

    # Scan additional directories
    for scan_dir_rel in SCAN_DIRS:
        scan_dir = PROJECT_ROOT / scan_dir_rel
        if not scan_dir.exists():
            print(f"Warning: {scan_dir_rel} does not exist, skipping")
            continue
        for ext in ("*.parquet", "*.csv"):
            for f in scan_dir.rglob(ext):
                if not should_skip(f):
                    files.add(f)

    return sorted(files)


def main():
    print("Capturing golden baseline...")
    files = collect_result_files()
    print(f"Found {len(files)} result files")

    baseline = {}
    for filepath in files:
        rel_path = str(filepath.relative_to(PROJECT_ROOT))
        print(f"  Processing: {rel_path}")
        baseline[rel_path] = capture_file(filepath)

    output_path = Path(__file__).parent / "golden_baseline.json"
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)

    print(f"\nGolden baseline saved to {output_path}")
    print(f"Total files captured: {len(baseline)}")


if __name__ == "__main__":
    main()
