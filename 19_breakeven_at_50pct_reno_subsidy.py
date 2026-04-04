"""
Break-even gas price multiplier at 50% renovation cost reduction (BEG WG max subsidy).

Used for R1-2.5: cost-sharing/ESCO/subsidy discussion in the sensitivity section.
Extracts the break-even gas price increase and corresponding carbon tax at
renovation cost multiplier = 0.5, for comparison with the existing 0.0, 0.2, 1.0 values.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import sensitivity_results_dir

# ── Load LT+Reno sensitivity data (all building types) ───────────────
analysis_type_lt = "combined_electicity_gas_renovation_costs"
lt_data_path = sensitivity_results_dir("renovated", analysis_type_lt) / "data"
all_lt_files = glob.glob(str(lt_data_path / f"{analysis_type_lt}_gas*_el*_reno*.csv"))

lt_data_list = []
for f in all_lt_files:
    try:
        parts = Path(f).stem.split("_")
        gas_mult = float(parts[-3].replace("gas", ""))
        el_mult = float(parts[-2].replace("el", ""))
        reno_mult = float(parts[-1].replace("reno", ""))

        df_temp = pd.read_csv(f)
        all_types_savings = df_temp["savings_npv_25years_ir_0.05"].mean()

        if not pd.isna(all_types_savings):
            lt_data_list.append({
                "electricity_multiplier": el_mult,
                "gas_multiplier": gas_mult,
                "renovation_cost_multiplier": reno_mult,
                "average_savings": all_types_savings,
            })
    except Exception as e:
        print(f"Warning: Could not process file {f}: {e}")

df_lt = pd.DataFrame(lt_data_list)

print("Available renovation cost multipliers:",
      sorted(df_lt["renovation_cost_multiplier"].unique()))

# ── Compute break-even gas multiplier at el=1.0 for key reno values ──
# Carbon tax conversion: 200 kgCO2/MWh gas intensity
# EUROSTAT gas prices from paper Table (gas_prices), Germany 2023
carbon_intensity_kgco2_per_kwh = 0.200  # 200 kgCO2/MWh = 0.200 kgCO2/kWh

print("\n" + "=" * 70)
print("BREAK-EVEN GAS MULTIPLIERS AT ELECTRICITY MULT = 1.0")
print("(all building types average)")
print("=" * 70)

reno_values = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]

for reno_val in reno_values:
    subset = df_lt[
        (abs(df_lt["electricity_multiplier"] - 1.0) < 0.01)
        & (abs(df_lt["renovation_cost_multiplier"] - reno_val) < 0.01)
    ].sort_values("gas_multiplier")

    if subset.empty:
        print(f"\n  reno_mult={reno_val:.1f} ({(1-reno_val)*100:.0f}% subsidy): NO DATA")
        continue

    gv = subset["gas_multiplier"].values
    sv = subset["average_savings"].values

    found = False
    for i in range(len(sv) - 1):
        if sv[i] <= 0 and sv[i + 1] > 0:
            frac = -sv[i] / (sv[i + 1] - sv[i])
            be = gv[i] + frac * (gv[i + 1] - gv[i])
            pct_increase = (be - 1) * 100

            # Carbon tax range using EUROSTAT prices from paper Table (gas_prices)
            # Range: NR large (lowest) to Res small (highest)
            gas_prices = {
                "Res small":  0.1405,  # <20 GJ/yr
                "Res medium": 0.1145,  # 20-200 GJ/yr
                "Res large":  0.1054,  # >200 GJ/yr
                "NR small":   0.1312,  # <20 GJ/yr
                "NR medium":  0.1070,  # 20-200 GJ/yr
                "NR large":   0.0985,  # >200 GJ/yr
            }
            print(f"\n  reno_mult={reno_val:.1f} ({(1-reno_val)*100:.0f}% subsidy):")
            print(f"    Break-even gas multiplier: {be:.3f}  (+{pct_increase:.1f}%)")
            for tier, price in gas_prices.items():
                ct = (be - 1) * price / carbon_intensity_kgco2_per_kwh * 1000
                print(f"    Carbon tax ({tier} tier): {ct:.0f} EUR/tCO2")
            found = True
            break

    if not found:
        if all(v < 0 for v in sv):
            print(f"\n  reno_mult={reno_val:.1f} ({(1-reno_val)*100:.0f}% subsidy):"
                  f" NEVER breaks even (max gas_mult = {gv[-1]:.1f})")
        elif all(v >= 0 for v in sv):
            print(f"\n  reno_mult={reno_val:.1f} ({(1-reno_val)*100:.0f}% subsidy):"
                  f" ALWAYS positive (min gas_mult = {gv[0]:.2f})")
