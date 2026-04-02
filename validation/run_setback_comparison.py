"""Run QSS with 20/17 C night setback schedule and compare against
the constant-20C QSS results and E+ results.

This tests how the paper's actual heating schedule compares to E+.
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

IWEC = PROJECT_ROOT / "irradiation_data" / "Frankfurt_Griesheim_Mitte_IWEC_TMY" / "Frankfurt_Griesheim_Mitte_IWEC_TMY_irradiation_data.csv"
SOIL = PROJECT_ROOT / "irradiation_data" / "Frankfurt_Griesheim_Mitte_2019_2019" / "Frankfurt_Griesheim_Mitte_soil_temperature_2019_2019.csv"
OUT = Path(__file__).resolve().parent / "results" / "qss_hourly_setback"
OUT.mkdir(parents=True, exist_ok=True)

weather_df = pd.read_csv(IWEC)
temperature = pd.read_csv(IWEC, usecols=["T2m"])
irradiation = weather_df.filter(regex=r"G\(i\)")
df_soil = pd.read_csv(SOIL)
df_soil.replace(-99.9, np.nan, inplace=True)
df_soil["V_TE0052"] = df_soil["V_TE0052"].interpolate()

# 20 C day (8-22h) / 17 C night -- paper's actual schedule
time_index = pd.date_range(start=f"{YEAR_START}-01-01", periods=8760, freq="h")
inside_temp = pd.DataFrame(index=time_index)
inside_temp["inside_temp"] = 20
mask_night = ~inside_temp.index.hour.isin(range(8, 22))
inside_temp.loc[mask_night, "inside_temp"] = 17

LABEL_MAP = {
    "AB 1945-1957 (ab3)": "AB_HT",
    "MFH 1945-1957 (mfh3)": "MFH3_HT",
    "MFH 1969-1978 (mfh5)": "MFH5_HT",
    "TH 1969-1978 (th5)": "TH5_HT",
}

EPLUS_DIR = Path(__file__).resolve().parent / "results" / "eplus_hourly"
QSS20_DIR = Path(__file__).resolve().parent / "results" / "qss_hourly"

print(f"{'Building':<22} {'QSS 20C':>12} {'QSS 20/17':>12} {'E+ 20C':>12} {'vs E+ (20C)':>12} {'vs E+ (20/17)':>14}")
print("-" * 85)

rows = []

for name, bldg in buildings.items():
    label = LABEL_MAP[name]
    components = build_component_row(name, bldg)
    bid = components["full_id"].values[0]
    btype = components["building_usage"].values[0] + str(components["age_code"].values[0])

    building = Building(
        bid, btype, components, temperature, irradiation,
        df_soil["V_TE0052"], inside_temp["inside_temp"], year_start=YEAR_START,
    )
    building.thermal_balance()

    hourly = building.get_useful_demand()
    if isinstance(hourly, pd.DataFrame):
        hourly = hourly.sum(axis=1)
    total_setback = hourly.sum()
    peak_setback = hourly.max()

    # Save hourly
    pd.DataFrame({"datetime": time_index, "kWh": hourly.values}).to_csv(
        OUT / f"{label}_hourly.csv", index=False
    )

    # Load constant-20 QSS and E+ for comparison
    qss20 = pd.read_csv(QSS20_DIR / f"{label}_hourly.csv")
    eplus = pd.read_csv(EPLUS_DIR / f"{label}_hourly.csv")
    total_20 = qss20["kWh"].sum()
    total_ep = eplus["kWh"].sum()

    diff_20 = (total_20 / total_ep - 1) * 100
    diff_sb = (total_setback / total_ep - 1) * 100

    short = name.split("(")[0].strip()
    print(f"{short:<22} {total_20:>12,.0f} {total_setback:>12,.0f} {total_ep:>12,.0f} {diff_20:>+11.1f}% {diff_sb:>+13.1f}%")

    rows.append({
        "Building": short,
        "QSS_20C_kWh": round(total_20),
        "QSS_20_17C_kWh": round(total_setback),
        "EPlus_20C_kWh": round(total_ep),
        "diff_20C_pct": round(diff_20, 1),
        "diff_20_17C_pct": round(diff_sb, 1),
        "QSS_20C_peak": round(qss20["kWh"].max(), 1),
        "QSS_20_17C_peak": round(peak_setback, 1),
        "EPlus_peak": round(eplus["kWh"].max(), 1),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "schedule_comparison_summary.csv", index=False)
print(f"\nSaved to {OUT / 'schedule_comparison_summary.csv'}")
