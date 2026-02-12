"""Verify 02_calculate_energy_demand.py produces identical results.

Re-runs the energy demand calculation with output to a verification directory,
then compares against the original results. Does NOT overwrite originals.
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

# Add project root to path so imports work (mirrors sys.path hacks used by codebase)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from building_analysis.Building import Building
from Person.Person import Person
from utils.misc import get_mask

# ─── Configuration ───
sim = "unrenovated"
size = "whole_buildingstock"
VERIFY_DIR = f"building_analysis/results/{sim}_{size}_verification"
ORIGINAL_DIR = f"building_analysis/results/{sim}_{size}"


def md5_hash(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_csv(original_path, verify_path, label):
    """Compare two CSV files. Returns (match, details)."""
    if not os.path.exists(original_path):
        return False, f"Original missing: {original_path}"
    if not os.path.exists(verify_path):
        return False, f"Verification missing: {verify_path}"

    df_orig = pd.read_csv(original_path)
    df_verify = pd.read_csv(verify_path)

    if df_orig.shape != df_verify.shape:
        return False, f"Shape mismatch: {df_orig.shape} vs {df_verify.shape}"

    # Compare numeric columns with tolerance
    numeric_cols = df_orig.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in df_verify.columns:
            return False, f"Column '{col}' missing in verification"
        orig_vals = df_orig[col].values
        verify_vals = df_verify[col].values
        if not np.allclose(orig_vals, verify_vals, rtol=1e-10, atol=1e-12, equal_nan=True):
            max_diff = np.nanmax(np.abs(orig_vals - verify_vals))
            return False, f"Column '{col}' differs, max_diff={max_diff:.2e}"

    return True, "Identical"


def compare_parquet(original_path, verify_path, label):
    """Compare two parquet files. Returns (match, details)."""
    if not os.path.exists(original_path):
        return False, f"Original missing: {original_path}"
    if not os.path.exists(verify_path):
        return False, f"Verification missing: {verify_path}"

    df_orig = gpd.read_parquet(original_path)
    df_verify = gpd.read_parquet(verify_path)

    if df_orig.shape != df_verify.shape:
        return False, f"Shape mismatch: {df_orig.shape} vs {df_verify.shape}"

    # Compare numeric columns with tolerance
    numeric_cols = df_orig.select_dtypes(include=[np.number]).columns
    mismatches = []
    for col in numeric_cols:
        if col not in df_verify.columns:
            mismatches.append(f"Column '{col}' missing in verification")
            continue
        orig_vals = df_orig[col].values
        verify_vals = df_verify[col].values
        if not np.allclose(orig_vals, verify_vals, rtol=1e-10, atol=1e-12, equal_nan=True):
            max_diff = np.nanmax(np.abs(orig_vals - verify_vals))
            mismatches.append(f"Column '{col}' max_diff={max_diff:.2e}")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, "Identical"


def main():
    print("=" * 70)
    print("VERIFICATION: Re-running 02_calculate_energy_demand.py")
    print(f"Output dir: {VERIFY_DIR} (originals untouched)")
    print("=" * 70)

    # ─── Load inputs (same as original script) ───
    print("\nLoading buildingstock...")
    buildingstock_path = "building_analysis/buildingstock/buildingstock.parquet"
    gdf_buildingstock = gpd.read_parquet(buildingstock_path)
    gdf_buildingstock_results = gdf_buildingstock.copy(deep=True)

    # Weather data
    city_name = "Frankfurt_Griesheim_Mitte"
    year_start = 2019
    year_end = 2019
    path_weather = f"irradiation_data/{city_name}_{year_start}_{year_end}/{city_name}_irradiation_data_{year_start}_{year_end}.csv"
    temperature = pd.read_csv(path_weather, usecols=["T2m"])
    irradiation = pd.read_csv(path_weather)
    irradiation = irradiation.filter(regex="G\(i\)")

    # Soil temperature
    soil_temp_path = f"irradiation_data/{city_name}_{year_start}_{year_end}/{city_name}_soil_temperature_{year_start}_{year_end}.csv"
    df_soil_temp = pd.read_csv(soil_temp_path)
    df_soil_temp.replace(-99.9, np.nan, inplace=True)
    df_soil_temp["V_TE0052"] = df_soil_temp["V_TE0052"].interpolate()

    # Inside temperature
    time_index = pd.date_range(start="2019-01-01", periods=8760, freq="h")
    inside_temp = pd.DataFrame(index=time_index)
    inside_temp["inside_temp"] = 20
    mask_heating = inside_temp.index.hour.isin(range(8, 22))
    inside_temp.loc[np.logical_not(mask_heating), "inside_temp"] = 17

    dhw_volumes_folder = "building_analysis/dhw_profiles"
    res_mask = gdf_buildingstock_results["building_usage"].isin(["sfh", "mfh", "ab", "th"])
    mask = get_mask(size, res_mask)

    # ─── Create verification output directories ───
    dir_dhw_volumes = f"{VERIFY_DIR}/dhw_volumes"
    dir_dhw_energy = f"{VERIFY_DIR}/dhw_energy"
    dir_space_heating = f"{VERIFY_DIR}/space_heating"

    for d in [dir_dhw_volumes, dir_dhw_energy, dir_space_heating]:
        os.makedirs(d, exist_ok=True)

    # ─── Run computation ───
    area_results = pd.DataFrame(
        0, index=inside_temp.index,
        columns=["dhw_volume", "dhw_energy", "space_heating"],
    )

    n_buildings = mask.sum()
    print(f"\nProcessing {n_buildings} buildings...")

    for idx, row in tqdm(gdf_buildingstock_results[mask].iterrows(), total=n_buildings):
        building_id = row["full_id"]
        building_type = row["building_usage"] + str(row["age_code"])
        components = row.to_frame().T

        building = Building(
            building_id, building_type, components,
            temperature, irradiation,
            df_soil_temp["V_TE0052"], inside_temp["inside_temp"],
            year_start=2019,
        )

        building.thermal_balance()
        building.add_people()
        building.append_water_usage(dhw_volumes_folder)
        building.people_dhw_energy()

        dhw_volume_df = building.building_dhw_volume()
        dhw_energy_df = building.building_dhw_energy()
        space_heating_df = building.get_useful_demand()

        # Save to VERIFICATION directory
        dhw_volume_df.to_csv(os.path.join(dir_dhw_volumes, f"dhw_volume_{building_id}.csv"))
        dhw_energy_df.to_csv(os.path.join(dir_dhw_energy, f"dhw_energy_{building_id}.csv"))
        space_heating_df.to_csv(os.path.join(dir_space_heating, f"space_heating_{building_id}.csv"))

        # Store results in the GeoDataFrame
        gdf_buildingstock_results.loc[idx, "dhw_volume_path"] = os.path.join(
            dir_dhw_volumes, f"dhw_volume_{building_id}.csv"
        )
        gdf_buildingstock_results.loc[idx, "dhw_energy_path"] = os.path.join(
            dir_dhw_energy, f"dhw_energy_{building_id}.csv"
        )
        gdf_buildingstock_results.loc[idx, "space_heating_path"] = os.path.join(
            dir_space_heating, f"space_heating_{building_id}.csv"
        )
        gdf_buildingstock_results.loc[idx, "yearly_dhw_volume"] = building.get_dhw_sum_volume()
        gdf_buildingstock_results.loc[idx, "yearly_dhw_energy"] = building.get_dhw_sum_energy()
        gdf_buildingstock_results.loc[idx, "yearly_space_heating"] = building.get_sum_useful_demand()
        gdf_buildingstock_results.loc[idx, "specific_ued"] = building.get_specific_ued()

        # Area results
        dhw_volume_df.index = area_results.index
        dhw_energy_df.index = area_results.index
        space_heating_df.index = area_results.index

        area_results["dhw_volume"] += dhw_volume_df.sum(axis=1)
        area_results["dhw_energy"] += dhw_energy_df.sum(axis=1)
        area_results["space_heating"] += space_heating_df.sum(axis=1)

    # Save verification results
    verify_buildingstock_path = f"{VERIFY_DIR}/buildingstock_results_{sim}.parquet"
    gdf_buildingstock_results.to_parquet(verify_buildingstock_path)
    verify_area_path = f"{VERIFY_DIR}/area_results_{sim}.csv"
    area_results.to_csv(verify_area_path)

    print(f"\nVerification results saved to {VERIFY_DIR}/")

    # ─── Compare against originals ───
    print("\n" + "=" * 70)
    print("COMPARING VERIFICATION vs ORIGINAL")
    print("=" * 70)

    all_ok = True

    # Compare buildingstock parquet
    orig_bs = f"{ORIGINAL_DIR}/buildingstock_results_{sim}.parquet"
    print(f"\n1. Buildingstock parquet:")
    match, details = compare_parquet(orig_bs, verify_buildingstock_path, "buildingstock")
    status = "OK" if match else "MISMATCH"
    print(f"   [{status}] {details}")
    if not match:
        all_ok = False

    # Compare area results CSV
    orig_area = f"{ORIGINAL_DIR}/area_results_{sim}.csv"
    print(f"\n2. Area results CSV:")
    match, details = compare_csv(orig_area, verify_area_path, "area_results")
    status = "OK" if match else "MISMATCH"
    print(f"   [{status}] {details}")
    if not match:
        all_ok = False

    # Compare per-building CSVs (sample first 20 + last 5)
    print(f"\n3. Per-building CSVs (sampling):")
    building_ids = gdf_buildingstock_results[mask]["full_id"].tolist()
    sample_ids = building_ids[:20] + building_ids[-5:]
    n_checked = 0
    n_ok = 0
    for bid in sample_ids:
        for subdir, prefix in [("space_heating", "space_heating"), ("dhw_energy", "dhw_energy"), ("dhw_volumes", "dhw_volume")]:
            orig_file = f"{ORIGINAL_DIR}/{subdir}/{prefix}_{bid}.csv"
            verify_file = f"{VERIFY_DIR}/{subdir}/{prefix}_{bid}.csv"
            match, details = compare_csv(orig_file, verify_file, f"{prefix}_{bid}")
            n_checked += 1
            if match:
                n_ok += 1
            else:
                print(f"   [MISMATCH] {prefix}_{bid}: {details}")
                all_ok = False

    print(f"   Checked {n_checked} files: {n_ok}/{n_checked} identical")

    # ─── Summary ───
    print("\n" + "=" * 70)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED - Energy demand calculation is reproducible!")
    else:
        print("RESULT: MISMATCHES FOUND (see details above)")
    print("=" * 70)

    # Cleanup hint
    print(f"\nTo remove verification files: rm -rf {VERIFY_DIR}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
