"""Compare hourly heating demand: QSS model vs EnergyPlus for all 4 buildings.

Produces per building:
  1. Coldest-week time series comparison
  2. Load duration curves
And a combined summary table.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EPLUS_DIR = RESULTS_DIR / "eplus_hourly"
QSS_DIR = RESULTS_DIR / "qss_hourly"
IWEC_WEATHER = (
    Path(__file__).resolve().parent.parent
    / "irradiation_data" / "Frankfurt_Griesheim_Mitte_IWEC_TMY"
    / "Frankfurt_Griesheim_Mitte_IWEC_TMY_irradiation_data.csv"
)

BUILDINGS = {
    "AB_HT":   {"name": "AB 1945-1957",  "eplus_area": 1485, "qss_nfa": 1267},
    "MFH3_HT": {"name": "MFH 1945-1957", "eplus_area": 626,  "qss_nfa": 552},
    "MFH5_HT": {"name": "MFH 1969-1978", "eplus_area": 608,  "qss_nfa": 544},
    "TH5_HT":  {"name": "TH 1969-1978",  "eplus_area": 216,  "qss_nfa": 173},
}

YEAR = 2019
time_index = pd.date_range(start=f"{YEAR}-01-01", periods=8760, freq="h")

# Load outdoor temperature
weather = pd.read_csv(IWEC_WEATHER)
temp = pd.Series(weather["T2m"].values, index=time_index, name="T_outdoor")

# Find coldest week (shared across all buildings — same weather)
rolling_temp = temp.rolling(168, center=True).mean()
coldest_center = rolling_temp.idxmin()
week_start = coldest_center - pd.Timedelta(hours=84)
week_end = coldest_center + pd.Timedelta(hours=84)
print(f"Coldest week: {week_start.strftime('%b %d')} — {week_end.strftime('%b %d')}")

# --- Process each building ---------------------------------------------------
summary_rows = []

for label, info in BUILDINGS.items():
    name = info["name"]
    print(f"\n{'='*60}")
    print(f"  {name} ({label})")
    print(f"{'='*60}")

    # Load data
    eplus_df = pd.read_csv(EPLUS_DIR / f"{label}_hourly.csv")
    qss_df = pd.read_csv(QSS_DIR / f"{label}_hourly.csv")

    eplus = pd.Series(eplus_df["kWh"].values, index=time_index)
    qss = pd.Series(qss_df["kWh"].values, index=time_index)

    # Summary statistics
    ep_total = eplus.sum()
    qss_total = qss.sum()
    ep_peak = eplus.max()
    qss_peak = qss.max()
    ep_hours = (eplus > 0.01).sum()
    qss_hours = (qss > 0).sum()

    print(f"  {'Metric':<30} {'QSS':>10} {'E+':>10} {'Diff':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Annual total [kWh]':<30} {qss_total:>10,.0f} {ep_total:>10,.0f} {(qss_total/ep_total-1)*100:>+9.1f}%")
    print(f"  {'Peak hour [kWh]':<30} {qss_peak:>10.1f} {ep_peak:>10.1f} {(qss_peak/ep_peak-1)*100:>+9.1f}%")
    print(f"  {'Heating hours':<30} {qss_hours:>10d} {ep_hours:>10d}")

    summary_rows.append({
        "Building": name,
        "Label": label,
        "E+ total (kWh)": round(ep_total),
        "QSS total (kWh)": round(qss_total),
        "Total diff (%)": round((qss_total / ep_total - 1) * 100, 1),
        "E+ peak (kWh)": round(ep_peak, 1),
        "QSS peak (kWh)": round(qss_peak, 1),
        "Peak diff (%)": round((qss_peak / ep_peak - 1) * 100, 1),
        "E+ area (m²)": info["eplus_area"],
        "QSS NFA (m²)": info["qss_nfa"],
    })

    # --- Coldest week plot ---
    mask = (eplus.index >= week_start) & (eplus.index <= week_end)
    hours = np.arange(mask.sum())

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(hours, qss[mask].values, label="QSS model", linewidth=1.5, color="C0")
    ax1.plot(hours, eplus[mask].values, label="EnergyPlus", linewidth=1.5, color="C1")
    ax1.set_ylabel("Heating demand [kWh]")
    ax1.set_title(f"{name} — Coldest week ({week_start.strftime('%b %d')}–{week_end.strftime('%b %d')})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(hours, temp[mask].values, color="gray", linewidth=1)
    ax2.set_ylabel("T outdoor [°C]")
    ax2.set_xlabel("Hour")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"{label}_coldest_week.png", dpi=200)
    plt.close(fig)

    # --- Load duration curves ---
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    qss_sorted = np.sort(qss.values)[::-1]
    eplus_sorted = np.sort(eplus.values)[::-1]
    hours_axis = np.arange(1, 8761)

    ax3.plot(hours_axis, qss_sorted, label="QSS model", linewidth=1.5, color="C0")
    ax3.plot(hours_axis, eplus_sorted, label="EnergyPlus", linewidth=1.5, color="C1")
    ax3.set_xlabel("Hours")
    ax3.set_ylabel("Heating demand [kWh]")
    ax3.set_title(f"{name} — Load duration curves")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / f"{label}_load_duration.png", dpi=200)
    plt.close(fig2)

    print(f"  Plots saved.")

# --- Combined summary --------------------------------------------------------
summary_df = pd.DataFrame(summary_rows)
summary_path = RESULTS_DIR / "full_validation_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\n{'='*60}")
print("FULL VALIDATION SUMMARY")
print(f"{'='*60}")
print(summary_df.to_string(index=False))
print(f"\nSaved to {summary_path}")
