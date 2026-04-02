"""Compute ETS2 carbon price equivalents for scenario break-even gas price increases.

This script reads the sensitivity analysis results and calculates what carbon
price (EUR/tCO2) would be needed to reach the break-even gas price multiplier
for each scenario at electricity_multiplier = 1.0.

Reads from: sensitivity_analysis/*/combined_electicity_gas/data/mfh_savings_analysis.csv
Writes nothing — output is printed to console.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────
CARBON_INTENSITY = 200  # kgCO2/MWh of natural gas
ETS2_PRICE_RANGE = (45, 340)  # EUR/tCO2, projected 2027 range (Graichen 2024)

EUROSTAT_GAS_PRICES = {  # EUR/kWh, EUROSTAT 2023
    "r0": 0.1405,   # residential small
    "r1": 0.1145,   # residential medium
    "r2": 0.1054,   # residential large
    "nr0": 0.1312,  # non-residential small
    "nr1": 0.1070,  # non-residential medium
    "nr2": 0.0985,  # non-residential large
}


# ──────────────────────────────────────────────────────────────────────
# Step 1: Extract break-even gas multipliers at electricity = 1.0
# ──────────────────────────────────────────────────────────────────────
def get_breakeven_gas_multiplier(csv_path):
    """Find gas multiplier where average_savings crosses zero at el=1.0."""
    df = pd.read_csv(csv_path)
    el1 = df[np.isclose(df["electricity_multiplier"], 1.0)].sort_values("gas_multiplier")
    savings = el1["average_savings"].values
    gas = el1["gas_multiplier"].values

    for i in range(len(savings) - 1):
        if savings[i] * savings[i + 1] < 0:
            g_be = gas[i] + (gas[i + 1] - gas[i]) * (-savings[i]) / (
                savings[i + 1] - savings[i]
            )
            return g_be
    return None


def get_lt_breakeven(data_dir, reno_mult, gas_values=None):
    """Find break-even gas multiplier for LT+Reno from per-point CSVs."""
    if gas_values is None:
        gas_values = [0.10, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00, 5.00]

    results = []
    for gas_m in gas_values:
        fname = (
            f"combined_electicity_gas_renovation_costs_"
            f"gas{gas_m:.2f}_el1.00_reno{reno_mult:.2f}.csv"
        )
        fpath = data_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            avg_sav = df["savings_npv_25years_ir_0.05"].mean()
            results.append((gas_m, avg_sav))

    if not results:
        return None, []

    results.sort()
    gas_vals = [r[0] for r in results]
    sav_vals = [r[1] for r in results]

    for i in range(len(sav_vals) - 1):
        if sav_vals[i] * sav_vals[i + 1] < 0:
            g_be = gas_vals[i] + (gas_vals[i + 1] - gas_vals[i]) * (
                -sav_vals[i]
            ) / (sav_vals[i + 1] - sav_vals[i])
            return g_be, results
    return None, results


scenarios = {
    "Booster": PROJECT_ROOT
    / "sensitivity_analysis/booster/combined_electicity_gas/data/mfh_savings_analysis.csv",
    "HT": PROJECT_ROOT
    / "sensitivity_analysis/unrenovated/combined_electicity_gas/data/mfh_savings_analysis.csv",
}

print("=" * 70)
print("STEP 1: Break-even gas price multipliers at electricity × 1.0")
print("=" * 70)

breakevens = {}
for name, path in scenarios.items():
    g_be = get_breakeven_gas_multiplier(path)
    breakevens[name] = g_be
    pct = (g_be - 1) * 100 if g_be else None
    print(f"  {name:>10s}: gas × {g_be:.3f}  ({pct:+.1f}% increase)" if g_be else f"  {name}: no break-even")

# LT+Reno
lt_dir = (
    PROJECT_ROOT
    / "sensitivity_analysis/renovated/combined_electicity_gas_renovation_costs/data"
)
LT_RENO_MULTS = [0.0, 0.2, 0.5, 1.0]
print(f"\n  LT+Reno (at electricity × 1.0):")
for reno_mult in LT_RENO_MULTS:
    g_be, _ = get_lt_breakeven(lt_dir, reno_mult)
    if g_be and g_be > 1:
        breakevens[f"LT+Reno (reno×{reno_mult:.1f})"] = g_be
        pct = (g_be - 1) * 100
        print(f"    reno × {reno_mult:.1f}: gas × {g_be:.3f}  ({pct:+.1f}%)")
    elif g_be and g_be <= 1:
        print(f"    reno × {reno_mult:.1f}: already profitable (gas × {g_be:.3f})")
    else:
        print(f"    reno × {reno_mult:.1f}: no break-even in range")


# ──────────────────────────────────────────────────────────────────────
# Step 2: Convert gas price increases to carbon prices
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 2: Carbon price equivalents (EUR/tCO2)")
print("=" * 70)

for name, g_be in breakevens.items():
    if g_be is None:
        continue
    pct = (g_be - 1) * 100
    print(f"\n--- {name}: {pct:.0f}% gas price increase (gas × {g_be:.2f}) ---")

    carbon_prices = []
    for tier, price_kwh in EUROSTAT_GAS_PRICES.items():
        price_mwh = price_kwh * 1000
        increase_mwh = price_mwh * (g_be - 1)
        carbon_price = increase_mwh / (CARBON_INTENSITY / 1000)
        carbon_prices.append(carbon_price)
        print(
            f"  {tier}: gas={price_mwh:.1f} EUR/MWh, "
            f"increase={increase_mwh:.1f} EUR/MWh → "
            f"carbon price={carbon_price:.0f} EUR/tCO2"
        )

    print(f"  Range: {min(carbon_prices):.0f} – {max(carbon_prices):.0f} EUR/tCO2")


# ──────────────────────────────────────────────────────────────────────
# Step 3: Effect of minimum ETS2 price on gas costs
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"STEP 3: Effect of ETS2 price range ({ETS2_PRICE_RANGE[0]}–{ETS2_PRICE_RANGE[1]} EUR/tCO2)")
print("=" * 70)

for ets_price in [ETS2_PRICE_RANGE[0], 100, 150, ETS2_PRICE_RANGE[1]]:
    carbon_surcharge_mwh = ets_price * (CARBON_INTENSITY / 1000)  # EUR/MWh
    print(f"\n  At {ets_price} EUR/tCO2 → +{carbon_surcharge_mwh:.1f} EUR/MWh surcharge:")
    for tier in ["r2", "nr2"]:
        price_mwh = EUROSTAT_GAS_PRICES[tier] * 1000
        pct = carbon_surcharge_mwh / price_mwh * 100
        print(f"    {tier}: +{pct:.1f}% on {price_mwh:.1f} EUR/MWh")
