"""Run QSS energy demand for synthetic E+ buildings using IWEC TMY weather.

Runs the QSS model for each building with constant 20°C setpoint (matching E+),
saves hourly demand profiles for comparison, and outputs summary statistics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from building_analysis.Building import Building
from config import YEAR_START
from validation.build_eplus_buildings import buildings, build_component_row

# --- paths -------------------------------------------------------------------
IWEC_WEATHER = (
    PROJECT_ROOT / "irradiation_data" / "Frankfurt_Griesheim_Mitte_IWEC_TMY"
    / "Frankfurt_Griesheim_Mitte_IWEC_TMY_irradiation_data.csv"
)
SOIL_TEMP = (
    PROJECT_ROOT / "irradiation_data" / "Frankfurt_Griesheim_Mitte_2019_2019"
    / "Frankfurt_Griesheim_Mitte_soil_temperature_2019_2019.csv"
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
HOURLY_DIR = RESULTS_DIR / "qss_hourly"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HOURLY_DIR.mkdir(parents=True, exist_ok=True)

# Map building names to labels matching extract_eplus_hourly.py
LABEL_MAP = {
    "AB 1945-1957 (ab3)": "AB_HT",
    "MFH 1945-1957 (mfh3)": "MFH3_HT",
    "MFH 1969-1978 (mfh5)": "MFH5_HT",
    "TH 1969-1978 (th5)": "TH5_HT",
}

# --- load weather ------------------------------------------------------------
print("Loading IWEC TMY weather data...")
weather_df = pd.read_csv(IWEC_WEATHER)
temperature = pd.read_csv(IWEC_WEATHER, usecols=["T2m"])
irradiation = weather_df.filter(regex=r"G\(i\)")

print("Loading soil temperature...")
df_soil_temp = pd.read_csv(SOIL_TEMP)
df_soil_temp.replace(-99.9, np.nan, inplace=True)
df_soil_temp["V_TE0052"] = df_soil_temp["V_TE0052"].interpolate()

# Constant 20°C schedule — matches EnergyPlus model (no night setback)
time_index = pd.date_range(start=f"{YEAR_START}-01-01", periods=8760, freq="h")
inside_temp = pd.DataFrame(index=time_index)
inside_temp["inside_temp"] = 20

# --- run each synthetic building ---------------------------------------------
results = []

for name, bldg in buildings.items():
    label = LABEL_MAP[name]
    print(f"\n--- {name} ({label}) ---")
    components = build_component_row(name, bldg)

    building_id = components["full_id"].values[0]
    building_type = (components["building_usage"].values[0]
                     + str(components["age_code"].values[0]))

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

    # Get hourly demand
    hourly_demand = building.get_useful_demand()
    if isinstance(hourly_demand, pd.DataFrame):
        hourly_total = hourly_demand.sum(axis=1)
    else:
        hourly_total = hourly_demand

    hourly_series = pd.Series(hourly_total.values, index=time_index, name="kWh")

    # Save hourly CSV
    hourly_out = pd.DataFrame({"datetime": time_index, "kWh": hourly_series.values})
    hourly_path = HOURLY_DIR / f"{label}_hourly.csv"
    hourly_out.to_csv(hourly_path, index=False)

    ued = building.get_specific_ued()
    total_demand = building.get_sum_useful_demand()
    nfa = components["NFA"].values[0]
    peak = hourly_series.max()

    print(f"  Total demand: {total_demand:,.0f} kWh/yr")
    print(f"  Specific UED: {ued:.1f} kWh/m²/yr")
    print(f"  Peak hour: {peak:.1f} kWh")
    print(f"  NFA: {nfa:.0f} m²")
    print(f"  Saved hourly to {hourly_path}")

    results.append({
        "building": name,
        "label": label,
        "qss_total_kwh": round(total_demand, 0),
        "qss_specific_ued": round(ued, 1),
        "qss_peak_kwh": round(peak, 1),
        "NFA_m2": round(nfa, 0),
    })

# --- summary -----------------------------------------------------------------
df = pd.DataFrame(results)
out_path = RESULTS_DIR / "qss_eplus_buildings_validation.csv"
df.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print("QSS RESULTS (constant 20°C, IWEC TMY)")
print(f"{'='*60}")
print(df.to_string(index=False))
print(f"\nSaved to {out_path}")
