"""
Extract reduction factor sensitivity slopes for each scenario and building type.

Used for R3-RES-M1: Reviewer asked to evaluate slopes of the linear RF sensitivity
and tabulate them. The slope represents how much customer NPV savings (EUR/m² NFA)
change per unit change in the reduction factor.

A steeper (more negative) slope means the building type is more sensitive to
heat price reductions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

SCENARIOS = {
    "HT": "unrenovated",
    "LT+Reno": "renovated",
    "Booster": "booster",
}

# Building type display order and labels
TYPE_ORDER = ["mfh", "ab", "sfh", "th", "office", "trade", "education", "health", "other"]
TYPE_LABELS = {
    "mfh": "MFH",
    "ab": "AB",
    "sfh": "SFH",
    "th": "TH",
    "office": "Office",
    "trade": "Trade",
    "education": "Education",
    "health": "Health",
    "other": "Other",
}

results = []

for scenario_name, sim_name in SCENARIOS.items():
    path = Path(f"sensitivity_analysis/{sim_name}/reduction_factor/data/multitple_graphs/avg_savings_data_nfa.csv")
    df = pd.read_csv(path, index_col=0)

    # RF values as x, NPV savings as y — compute linear slope for each building type
    x = df.index.values  # reduction factor values

    for col in df.columns:
        y = df[col].values
        # Linear regression: y = mx + b
        slope, intercept = np.polyfit(x, y, 1)

        # Also find break-even RF (where y = 0)
        if slope != 0:
            breakeven_rf = -intercept / slope
        else:
            breakeven_rf = np.nan

        results.append({
            "Scenario": scenario_name,
            "Building Type": TYPE_LABELS.get(col, col),
            "Slope [EUR/m² per RF unit]": round(slope, 1),
            "Intercept [EUR/m²]": round(intercept, 1),
            "Break-even RF": round(breakeven_rf, 3) if not np.isnan(breakeven_rf) else "N/A",
        })

results_df = pd.DataFrame(results)

# Pivot for cleaner display
print("=" * 80)
print("REDUCTION FACTOR SENSITIVITY SLOPES")
print("Slope = change in customer NPV savings (EUR/m² NFA) per unit RF change")
print("=" * 80)

for scenario in SCENARIOS:
    subset = results_df[results_df["Scenario"] == scenario]
    print(f"\n{scenario}:")
    print(subset[["Building Type", "Slope [EUR/m² per RF unit]", "Break-even RF"]].to_string(index=False))

# Also create a pivot table for LaTeX
print("\n\n" + "=" * 80)
print("PIVOT TABLE (for LaTeX)")
print("=" * 80)
pivot = results_df.pivot(index="Building Type", columns="Scenario", values="Slope [EUR/m² per RF unit]")
pivot = pivot[["HT", "LT+Reno", "Booster"]]
# Reorder rows
ordered_types = [TYPE_LABELS[t] for t in TYPE_ORDER if TYPE_LABELS[t] in pivot.index]
pivot = pivot.loc[ordered_types]
print(pivot.to_string())

# Break-even pivot
print("\n\nBREAK-EVEN RF VALUES:")
pivot_be = results_df.pivot(index="Building Type", columns="Scenario", values="Break-even RF")
pivot_be = pivot_be[["HT", "LT+Reno", "Booster"]]
pivot_be = pivot_be.loc[ordered_types]
print(pivot_be.to_string())
