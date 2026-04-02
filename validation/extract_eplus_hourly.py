"""Extract hourly district heating demand from EnergyPlus SQL output.

Reads the eplusout.sql file for each building, extracts DistrictHeating:Facility
(or DistrictHeatingWater:Facility for older runs), aggregates sub-hourly timesteps
to hourly, and saves a CSV with 8760 rows in kWh.
"""

import sqlite3
import pandas as pd
from pathlib import Path

# --- configuration -----------------------------------------------------------
VALIDATION_DIR = Path(__file__).resolve().parent
EPLUS_DIR = VALIDATION_DIR / "e_plus"

EPLUS_RUNS = {
    "AB_HT": EPLUS_DIR / "AB" / "HT_Rad_outdoorair_V2" / "run" / "eplusout.sql",
    "MFH3_HT": EPLUS_DIR / "MFH1945_1957" / "Final_RadiatorHT" / "run" / "eplusout.sql",
    "MFH5_HT": EPLUS_DIR / "MFH1969_1978" / "MFH 1969-1978_HT_withAutosizing_FinalVersion-v2" / "run" / "eplusout.sql",
    "TH5_HT": EPLUS_DIR / "TH1969_1978" / "baseline_HT_TH1969-1978" / "run" / "eplusout.sql",
}

# Expected annual totals for verification (from direct SQL queries)
EXPECTED_KWH = {
    "AB_HT": 144_403,
    "MFH3_HT": 89_246,
    "MFH5_HT": 65_499,
    "TH5_HT": 29_744,
}

OUT_DIR = VALIDATION_DIR / "results" / "eplus_hourly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

J_TO_KWH = 1 / 3_600_000
YEAR = 2019  # dummy year for datetime index


def extract_hourly_district_heating(sql_path: Path, label: str) -> pd.Series:
    """Extract hourly DistrictHeating from E+ SQL, return 8760-row Series in kWh."""
    conn = sqlite3.connect(str(sql_path))

    # Find the annual run (the EnvironmentPeriodIndex with most rows)
    env = pd.read_sql_query(
        "SELECT EnvironmentPeriodIndex, COUNT(*) as n "
        "FROM Time WHERE WarmupFlag = 0 "
        "GROUP BY EnvironmentPeriodIndex ORDER BY n DESC", conn
    )
    annual_env = int(env.iloc[0]["EnvironmentPeriodIndex"])

    # Find the district heating variable (handle both naming conventions)
    rdd = pd.read_sql_query(
        "SELECT ReportDataDictionaryIndex, Name FROM ReportDataDictionary "
        "WHERE Name LIKE 'DistrictHeating%Facility'", conn
    )
    if len(rdd) == 0:
        print(f"  ERROR: No DistrictHeating variable found")
        conn.close()
        return None

    var_name = rdd.iloc[0]["Name"]
    var_idx = int(rdd.iloc[0]["ReportDataDictionaryIndex"])
    print(f"  Variable: {var_name} (index {var_idx})")

    # Extract all sub-timestep data for the annual run
    data = pd.read_sql_query(f"""
        SELECT t.Month, t.Day, t.Hour, t.Minute, rd.Value
        FROM ReportData rd
        JOIN Time t ON rd.TimeIndex = t.TimeIndex
        WHERE rd.ReportDataDictionaryIndex = {var_idx}
          AND t.EnvironmentPeriodIndex = {annual_env}
          AND t.WarmupFlag = 0
        ORDER BY t.TimeIndex
    """, conn)
    conn.close()

    data["kWh"] = data["Value"] * J_TO_KWH

    # E+ hour convention: Hours 0-24 with sub-timesteps.
    # Hour 0 (min 10-50) + Hour 24 (min 0) = the midnight hour (23:00-00:00).
    # Merge Hour 24 into Hour 0 of the same day.
    data.loc[data["Hour"] == 24, "Hour"] = 0

    # Now aggregate sub-timesteps to hourly: Hours 0-23
    hourly = data.groupby(["Month", "Day", "Hour"])["kWh"].sum().reset_index()
    hourly = hourly.sort_values(["Month", "Day", "Hour"]).reset_index(drop=True)

    n_rows = len(hourly)
    if n_rows != 8760:
        print(f"  WARNING: {n_rows} hourly rows (expected 8760)")

    # Build datetime index and create Series
    time_index = pd.date_range(start=f"{YEAR}-01-01", periods=min(n_rows, 8760), freq="h")
    result = pd.Series(hourly["kWh"].values[:8760], index=time_index, name=label)

    # Save CSV
    out_df = pd.DataFrame({"datetime": time_index, "kWh": result.values})
    out_path = OUT_DIR / f"{label}_hourly.csv"
    out_df.to_csv(out_path, index=False)

    annual_total = result.sum()
    expected = EXPECTED_KWH.get(label, 0)
    match_pct = annual_total / expected * 100 if expected else 0
    print(f"  Annual total: {annual_total:,.0f} kWh (expected {expected:,}, match {match_pct:.1f}%)")
    print(f"  Peak hour: {result.max():.1f} kWh")
    print(f"  Saved to {out_path}")

    return result


if __name__ == "__main__":
    results = {}
    for label, sql_path in EPLUS_RUNS.items():
        print(f"\n=== {label} ===")
        if not sql_path.exists():
            print(f"  File not found: {sql_path}")
            continue
        results[label] = extract_hourly_district_heating(sql_path, label)

    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Label':<12} {'Annual kWh':>12} {'Peak kWh':>10} {'Match%':>8}")
        for label, series in results.items():
            expected = EXPECTED_KWH.get(label, 0)
            match = series.sum() / expected * 100 if expected else 0
            print(f"{label:<12} {series.sum():>12,.0f} {series.max():>10.1f} {match:>7.1f}%")
