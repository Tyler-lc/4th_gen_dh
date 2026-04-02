import pandas as pd
from config import grid_results_parquet

scenarios = ["unrenovated", "renovated", "booster"]

for scenario in scenarios:
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{'=' * 60}")

    result_df = pd.read_parquet(grid_results_parquet(scenario))

    # Group by Diameter and sum Losses [W] and Length
    grouped = (
        result_df.groupby("Diameter")
        .agg({"Losses [W]": "sum", "Length": "sum"})
        .reset_index()
    )

    # Rename columns for clarity
    grouped.columns = ["Diameter", "Total Losses [W]", "Total Length [m]"]

    # Convert length to km for readability
    grouped["Total Length [km]"] = grouped["Total Length [m]"] / 1000

    # Sort by diameter
    grouped = grouped.sort_values("Diameter")

    # Display
    print(
        grouped[["Diameter", "Total Losses [W]", "Total Length [km]"]].to_string(
            index=False
        )
    )

    # Summary stats
    print(f"\nTotal Losses [W]: {grouped['Total Losses [W]'].sum():,.0f}")
    print(f"Total Length [km]: {grouped['Total Length [km]'].sum():,.2f}")


# ============================================================
# COMPARISON: BOOSTER vs RENOVATED
# ============================================================

print(f"\n\n{'=' * 120}")
print(f"BOOSTER vs RENOVATED SCENARIO COMPARISON (Differences: Booster - Renovated)")
print(f"{'=' * 120}\n")

renovated_result_df = pd.read_parquet(grid_results_parquet("renovated"))
booster_result_df = pd.read_parquet(grid_results_parquet("booster"))

# Group by Diameter for each scenario
renovated_grouped = (
    renovated_result_df.groupby("Diameter")
    .agg({"Losses [W]": "sum", "Length": "sum"})
    .reset_index()
)
renovated_grouped.columns = ["Diameter", "Renovated_Losses_W", "Renovated_Length_m"]

booster_grouped = (
    booster_result_df.groupby("Diameter")
    .agg({"Losses [W]": "sum", "Length": "sum"})
    .reset_index()
)
booster_grouped.columns = ["Diameter", "Booster_Losses_W", "Booster_Length_m"]

# Merge on diameter
comparison = pd.merge(
    renovated_grouped, booster_grouped, on="Diameter", how="outer"
).fillna(0)

# Calculate differences (Booster - Renovated)
comparison["Losses_Diff_W"] = (
    comparison["Booster_Losses_W"] - comparison["Renovated_Losses_W"]
)
comparison["Length_Diff_m"] = (
    comparison["Booster_Length_m"] - comparison["Renovated_Length_m"]
)

# Convert to km for readability
comparison["Renovated_Length_km"] = comparison["Renovated_Length_m"] / 1000
comparison["Booster_Length_km"] = comparison["Booster_Length_m"] / 1000
comparison["Length_Diff_km"] = comparison["Length_Diff_m"] / 1000

# Sort by diameter
comparison = comparison.sort_values("Diameter")

# Select and display columns
display_cols = [
    "Diameter",
    "Renovated_Losses_W",
    "Booster_Losses_W",
    "Losses_Diff_W",
    "Renovated_Length_km",
    "Booster_Length_km",
    "Length_Diff_km",
]

print(comparison[display_cols].to_string(index=False))

print(f"\n\n{'=' * 120}")
print(f"SUMMARY")
print(f"{'=' * 120}")
print(
    f"Total Losses Diff [W]:  {comparison['Losses_Diff_W'].sum():>15,.0f}  (Booster - Renovated)"
)
print(
    f"Total Length Diff [km]: {comparison['Length_Diff_km'].sum():>15,.2f}  (Booster - Renovated)"
)

print(
    f"\nDiameters with higher losses in Booster vs Renovated: {(comparison['Losses_Diff_W'] > 0).sum()}"
)
print(
    f"Diameters with higher losses in Renovated vs Booster: {(comparison['Losses_Diff_W'] < 0).sum()}"
)

print(
    f"\nDiameters with longer pipes in Booster vs Renovated: {(comparison['Length_Diff_km'] > 0).sum()}"
)
print(
    f"Diameters with longer pipes in Renovated vs Booster: {(comparison['Length_Diff_km'] < 0).sum()}"
)
comparison.to_csv("booster_vs_renovated_comparison.csv")
