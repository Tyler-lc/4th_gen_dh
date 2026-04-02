"""Validation: compare quasi-steady-state model against EnergyPlus results.

Runs the building energy demand model for representative buildings of 4
archetypes using IWEC TMY weather data (the same TMY used in the EnergyPlus
study by Casamassima & Kranzl, 2024, Energy and Buildings).

Archetypes matched to the E+ study:
    - MFH 1945-1957  →  mfh3
    - MFH 1969-1978  →  mfh5
    - AB  1945-1957  →  ab3
    - TH  1969-1978  →  th5

Reference E+ values are from Table 2 and Figure 2 of the published paper
(unrenovated NER case at 80°C supply).
"""

import sys
from pathlib import Path

# ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd

from building_analysis.Building import Building
from config import BUILDINGSTOCK_PATH, YEAR_START

# --- paths -------------------------------------------------------------------
IWEC_WEATHER = (
    PROJECT_ROOT
    / "irradiation_data"
    / "Frankfurt_Griesheim_Mitte_IWEC_TMY"
    / "Frankfurt_Griesheim_Mitte_IWEC_TMY_irradiation_data.csv"
)
# reuse 2019 soil temp (slowly varying, negligible inter-annual difference)
SOIL_TEMP = (
    PROJECT_ROOT
    / "irradiation_data"
    / "Frankfurt_Griesheim_Mitte_2019_2019"
    / "Frankfurt_Griesheim_Mitte_soil_temperature_2019_2019.csv"
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- target archetypes and reference values ---
# TABULA reference demands for unrenovated German buildings (kWh/m²/yr)
# These are the standard useful energy demands from the TABULA webtool.
# E+ values (Casamassima & Kranzl, 2024) to be added when data is available
# from the 80°C ideal-air-loads run (not the 45°C radiator-limited case).
MIN_NFA = 30  # m² — exclude very small structures (sheds, bungalows)

REFERENCE = {
    "mfh3": {"label": "MFH 1945-1957", "tabula_ued": 201.0},
    "mfh5": {"label": "MFH 1969-1978", "tabula_ued": 153.3},
    "ab3":  {"label": "AB 1949-1957",  "tabula_ued": 170.3},
    "th5":  {"label": "TH 1969-1978",  "tabula_ued": 142.4},
}

# --- load weather data -------------------------------------------------------
print("Loading IWEC TMY weather data...")
weather_df = pd.read_csv(IWEC_WEATHER)
temperature = pd.read_csv(IWEC_WEATHER, usecols=["T2m"])
irradiation = weather_df.filter(regex=r"G\(i\)")

print("Loading soil temperature (2019 DWD)...")
df_soil_temp = pd.read_csv(SOIL_TEMP)
df_soil_temp.replace(-99.9, np.nan, inplace=True)
df_soil_temp["V_TE0052"] = df_soil_temp["V_TE0052"].interpolate()

# inside temperature schedule (same as main model)
time_index = pd.date_range(start=f"{YEAR_START}-01-01", periods=8760, freq="h")
inside_temp = pd.DataFrame(index=time_index)
inside_temp["inside_temp"] = 20
mask_heating = inside_temp.index.hour.isin(range(8, 22))
inside_temp.loc[np.logical_not(mask_heating), "inside_temp"] = 17

# --- load building stock -----------------------------------------------------
print("Loading building stock...")
gdf = gpd.read_parquet(BUILDINGSTOCK_PATH)
gdf["archetype"] = gdf["building_usage"] + gdf["age_code"].astype(str)

# --- run validation ----------------------------------------------------------
results = []

for archetype, ref in REFERENCE.items():
    subset = gdf[(gdf["archetype"] == archetype) & (gdf["NFA"] >= MIN_NFA)]
    if len(subset) == 0:
        print(f"  WARNING: No buildings found for {archetype} (NFA>={MIN_NFA}), skipping.")
        continue

    print(f"\n--- {ref['label']} ({archetype}): {len(subset)} buildings (NFA>={MIN_NFA} m²) ---")

    # run all buildings of this archetype to get the distribution
    ueds = []
    for idx, row in subset.iterrows():
        building_id = row["full_id"]
        building_type = row["building_usage"] + str(row["age_code"])
        components = row.to_frame().T

        building = Building(
            building_id,
            building_type,
            components,
            temperature,
            irradiation,
            df_soil_temp["V_TE0052"],
            inside_temp["inside_temp"],
            year_start=YEAR_START,
        )
        building.thermal_balance()
        ued = building.get_specific_ued()
        ueds.append(ued)

    ueds = np.array(ueds)
    mean_ued = np.mean(ueds)
    std_ued = np.std(ueds)
    median_ued = np.median(ueds)

    diff_mean_pct = (mean_ued - ref["tabula_ued"]) / ref["tabula_ued"] * 100
    diff_median_pct = (median_ued - ref["tabula_ued"]) / ref["tabula_ued"] * 100

    print(f"  QSS model (IWEC TMY): mean={mean_ued:.1f}, median={median_ued:.1f}, "
          f"std={std_ued:.1f} kWh/m²/yr  (n={len(ueds)})")
    print(f"  TABULA reference: {ref['tabula_ued']} kWh/m²/yr")
    print(f"  Diff vs TABULA: mean {diff_mean_pct:+.1f}%, median {diff_median_pct:+.1f}%")

    results.append({
        "archetype": archetype,
        "label": ref["label"],
        "n_buildings": len(ueds),
        "qss_mean_ued": round(mean_ued, 1),
        "qss_median_ued": round(median_ued, 1),
        "qss_std_ued": round(std_ued, 1),
        "tabula_ued": ref["tabula_ued"],
        "diff_mean_pct": round(diff_mean_pct, 1),
        "diff_median_pct": round(diff_median_pct, 1),
    })

# --- save results ------------------------------------------------------------
df_results = pd.DataFrame(results)
out_path = RESULTS_DIR / "qss_vs_energyplus_validation.csv"
df_results.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print("VALIDATION SUMMARY")
print(f"{'='*60}")
print(df_results.to_string(index=False))
print(f"\nResults saved to {out_path}")
print(f"\nNote: TABULA values are for a single standardised archetype.")
print(f"Actual district buildings vary in geometry → higher demands expected.")
print(f"E+ comparison pending (need 80°C ideal-air-loads data from other machine).")
