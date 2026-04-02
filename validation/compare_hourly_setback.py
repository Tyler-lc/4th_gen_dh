"""Compare hourly heating demand: QSS (20/17C setback) vs EnergyPlus (20C constant).

Same structure as compare_hourly.py but uses the setback schedule QSS results.
Plots saved to a separate directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EPLUS_DIR = RESULTS_DIR / "eplus_hourly"
QSS_DIR = RESULTS_DIR / "qss_hourly_setback"
QSS_20_DIR = RESULTS_DIR / "qss_hourly"  # constant 20C for reference
PLOT_DIR = RESULTS_DIR / "plots_setback"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

IWEC_WEATHER = (
    Path(__file__).resolve().parent.parent
    / "irradiation_data" / "Frankfurt_Griesheim_Mitte_IWEC_TMY"
    / "Frankfurt_Griesheim_Mitte_IWEC_TMY_irradiation_data.csv"
)

BUILDINGS = {
    "AB_HT":   {"name": "AB 1945-1957"},
    "MFH3_HT": {"name": "MFH 1945-1957"},
    "MFH5_HT": {"name": "MFH 1969-1978"},
    "TH5_HT":  {"name": "TH 1969-1978"},
}

YEAR = 2019
time_index = pd.date_range(start=f"{YEAR}-01-01", periods=8760, freq="h")

# Load outdoor temperature
weather = pd.read_csv(IWEC_WEATHER)
temp = pd.Series(weather["T2m"].values, index=time_index)

# Find coldest week
rolling_temp = temp.rolling(168, center=True).mean()
coldest_center = rolling_temp.idxmin()
week_start = coldest_center - pd.Timedelta(hours=84)
week_end = coldest_center + pd.Timedelta(hours=84)

for label, info in BUILDINGS.items():
    name = info["name"]
    print(f"Plotting {name}...")

    eplus = pd.Series(pd.read_csv(EPLUS_DIR / f"{label}_hourly.csv")["kWh"].values, index=time_index)
    qss_sb = pd.Series(pd.read_csv(QSS_DIR / f"{label}_hourly.csv")["kWh"].values, index=time_index)
    qss_20 = pd.Series(pd.read_csv(QSS_20_DIR / f"{label}_hourly.csv")["kWh"].values, index=time_index)

    # --- Coldest week: all three curves ---
    mask = (eplus.index >= week_start) & (eplus.index <= week_end)
    hours = np.arange(mask.sum())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(hours, qss_20[mask].values, linewidth=1.2, color="C0", alpha=0.4,
             linestyle="--", label="QSS (constant 20 C)")
    ax1.plot(hours, qss_sb[mask].values, linewidth=1.5, color="C0",
             label="QSS (20/17 C setback)")
    ax1.plot(hours, eplus[mask].values, linewidth=1.5, color="C1",
             label="EnergyPlus (constant 20 C)")
    ax1.set_ylabel("Heating demand [kWh]")
    ax1.set_title(f"{name} — Coldest week ({week_start.strftime('%b %d')}–{week_end.strftime('%b %d')})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(hours, temp[mask].values, color="gray", linewidth=1)
    ax2.set_ylabel("T outdoor [C]")
    ax2.set_xlabel("Hour")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"{label}_coldest_week_setback.png", dpi=200)
    plt.close(fig)

    # --- Load duration curves: all three ---
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(np.arange(1, 8761), np.sort(qss_20.values)[::-1],
             linewidth=1.2, color="C0", alpha=0.4, linestyle="--",
             label="QSS (constant 20 C)")
    ax3.plot(np.arange(1, 8761), np.sort(qss_sb.values)[::-1],
             linewidth=1.5, color="C0", label="QSS (20/17 C setback)")
    ax3.plot(np.arange(1, 8761), np.sort(eplus.values)[::-1],
             linewidth=1.5, color="C1", label="EnergyPlus (constant 20 C)")
    ax3.set_xlabel("Hours")
    ax3.set_ylabel("Heating demand [kWh]")
    ax3.set_title(f"{name} — Load duration curves")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(PLOT_DIR / f"{label}_load_duration_setback.png", dpi=200)
    plt.close(fig2)

print(f"\nAll plots saved to {PLOT_DIR}")
