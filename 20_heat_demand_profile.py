"""
Generate heat demand profile and load duration curve for the district.

Used for R4-9: Show heat demand profile and capacity for each scenario.
Aggregates hourly SH + DHW for NFA >= 30 buildings only (consistent with TEO analysis).
Produces two figures:
  1. Time-series thermal demand profile (full year, hourly)
  2. Load duration curve (sorted descending)
Both show unrenovated (HT/Booster baseline) and renovated (LT+Reno) demands.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PAPER_FIGURE_DIR, PLOTS_DIR, RESULTS_DIR, results_dir, buildingstock_results_path

# ── Configuration ────────────────────────────────────────────────────
SCENARIOS = {
    "Unrenovated (HT / Booster)": {
        "parquet": buildingstock_results_path("unrenovated"),
        "sh_dir": results_dir("unrenovated") / "space_heating",
        "dhw_dir": results_dir("unrenovated") / "dhw_energy",
        "sh_col": "net useful hourly demand [kWh]",
        "dhw_col": None,  # will check
        "color": "#d62728",
    },
    "Renovated (LT+Reno)": {
        "parquet": buildingstock_results_path("renovated"),
        "sh_dir": results_dir("renovated") / "space_heating",
        "dhw_dir": results_dir("renovated") / "dhw_energy",
        "sh_col": "net useful hourly demand [kWh]",
        "dhw_col": None,
        "color": "#1f77b4",
    },
}

NFA_THRESHOLD = 30  # m², consistent with scenario scripts


def load_hourly_demand(bs_df, sh_dir, dhw_dir, sh_col):
    """Sum hourly SH + DHW for all NFA >= 30 buildings."""
    filtered = bs_df[bs_df["NFA"] >= NFA_THRESHOLD]
    total_demand = None

    for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Loading buildings"):
        full_id = row["full_id"]

        # Space heating
        sh_file = sh_dir / f"space_heating_{full_id}.csv"
        if sh_file.exists():
            sh = pd.read_csv(sh_file, index_col=0, parse_dates=True)[sh_col]
            # Clip negative values (no cooling)
            sh = sh.clip(lower=0)
        else:
            continue

        # DHW energy
        dhw_file = dhw_dir / f"dhw_energy_{full_id}.csv"
        if dhw_file.exists():
            dhw_df = pd.read_csv(dhw_file, index_col=0, parse_dates=True)
            # DHW files may have multiple columns (per occupant); sum across
            dhw = dhw_df.sum(axis=1)
        else:
            dhw = pd.Series(0, index=sh.index)

        building_total = sh + dhw

        if total_demand is None:
            total_demand = building_total.copy()
        else:
            total_demand += building_total

    return total_demand


# ── Load data ────────────────────────────────────────────────────────
demands = {}
for name, cfg in SCENARIOS.items():
    print(f"\nProcessing: {name}")
    bs = pd.read_parquet(cfg["parquet"])
    print(f"  Total buildings: {len(bs)}, NFA >= {NFA_THRESHOLD}: {len(bs[bs['NFA'] >= NFA_THRESHOLD])}")
    demand = load_hourly_demand(bs, cfg["sh_dir"], cfg["dhw_dir"], cfg["sh_col"])
    demands[name] = demand / 1000  # kWh -> MWh
    print(f"  Annual demand: {demands[name].sum():.0f} MWh")
    print(f"  Peak demand: {demands[name].max():.1f} MW")

ROLLING_WINDOW = 24  # hours for moving average

# ── Global font settings ─────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

# ── Month labels for x-axis ──────────────────────────────────────────
import calendar
month_starts = [0]
# 2019 is not a leap year
days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
for d in days_per_month[:-1]:
    month_starts.append(month_starts[-1] + d * 24)
month_midpoints = [month_starts[i] + days_per_month[i] * 12 for i in range(12)]
month_labels = [calendar.month_abbr[i + 1] for i in range(12)]

# ── Combined figure: demand profile (left) + load duration curve (right) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left panel: Time-series with rolling average ─────────────────────
for name, cfg in SCENARIOS.items():
    demand = demands[name]
    hours = np.arange(len(demand))
    # Raw hourly data (thin, semi-transparent)
    ax1.plot(hours, demand.values, color=cfg["color"], alpha=0.25, linewidth=0.4)
    # Rolling average (thick, prominent)
    rolling = demand.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()
    ax1.plot(hours, rolling.values, label=name, color=cfg["color"], linewidth=2)

ax1.set_xlabel("Month")
ax1.set_ylabel("Useful Heat Demand [MW]")
ax1.legend(loc="upper right")
ax1.set_xlim(0, 8760)
ax1.set_ylim(bottom=0)
ax1.set_xticks(month_midpoints)
ax1.set_xticklabels(month_labels)
ax1.grid(True, alpha=0.3)

# ── Right panel: Load duration curve ─────────────────────────────────
for name, cfg in SCENARIOS.items():
    demand = demands[name]
    sorted_demand = np.sort(demand.values)[::-1]
    hours = np.arange(1, len(sorted_demand) + 1)
    ax2.plot(hours, sorted_demand, label=name, color=cfg["color"], linewidth=2.5)

ax2.set_xlabel("Hours")
ax2.set_ylabel("Useful Heat Demand [MW]")
ax2.legend(loc="upper right")
ax2.set_xlim(0, 8760)
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOTS_DIR / "heat_demand_combined.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {PLOTS_DIR / 'heat_demand_combined.png'}")

if PAPER_FIGURE_DIR and PAPER_FIGURE_DIR.exists():
    fig.savefig(PAPER_FIGURE_DIR / "heat_demand_combined.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {PAPER_FIGURE_DIR / 'heat_demand_combined.png'}")

plt.show()
