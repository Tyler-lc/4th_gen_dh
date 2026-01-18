"""
15_iwu_renovation_costs_plot.py

Plot the IWU renovation costs per m² for different insulation thicknesses (1-25 cm).
Based on the formulas from the IWU study used in costs/renovation_costs.py
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt

# Insulation thickness range: 1 cm to 25 cm
thickness_cm = np.arange(1, 26, 1)  # 1 to 25 cm

# IWU cost formulas (cost per m²)
# The original formulas use insulation_thickness in mm, then divide by 10 to get cm
# Since we're already working in cm, we use thickness directly

# Walls: 112.18 + 3.25 * thickness_cm
cost_walls = 112.18 + 3.25 * thickness_cm

# Sloped roof (slope >= 25°): 178.48 + 3.27 * thickness_cm
cost_roof_sloped = 178.48 + 3.27 * thickness_cm

# Flat roof (slope < 25°): 123.29 + 4.87 * thickness_cm
cost_roof_flat = 123.29 + 4.87 * thickness_cm

# Ground contact floor: 10.27 + 1.86 * thickness_cm
cost_ground = 10.27 + 1.86 * thickness_cm

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each component
ax.plot(
    thickness_cm,
    cost_walls,
    "o-",
    label="Walls",
    linewidth=2,
    markersize=6,
    color="#2E86AB",
)
ax.plot(
    thickness_cm,
    cost_roof_sloped,
    "s-",
    label="Sloped Roof (≥25°)",
    linewidth=2,
    markersize=6,
    color="#A23B72",
)
ax.plot(
    thickness_cm,
    cost_roof_flat,
    "^-",
    label="Flat Roof (<25°)",
    linewidth=2,
    markersize=6,
    color="#F18F01",
)
ax.plot(
    thickness_cm,
    cost_ground,
    "D-",
    label="Ground Contact Floor",
    linewidth=2,
    markersize=6,
    color="#C73E1D",
)

# Formatting
ax.set_xlabel("Insulation Thickness (cm)", fontsize=14)
ax.set_ylabel("Renovation Cost (€/m²)", fontsize=14)
ax.set_title("IWU Renovation Costs per m² vs Insulation Thickness", fontsize=16)
ax.legend(fontsize=12, loc="upper left")
ax.grid(True, linestyle="--", alpha=0.7)
ax.tick_params(axis="both", labelsize=12)

# Set x-axis ticks at every 5 cm
ax.set_xticks(np.arange(0, 26, 5))
ax.set_xlim(0, 26)

# Add annotation with the formulas
formula_text = (
    "Cost Formulas (€/m²):\n"
    "• Walls: 112.18 + 3.25 × t\n"
    "• Sloped Roof: 178.48 + 3.27 × t\n"
    "• Flat Roof: 123.29 + 4.87 × t\n"
    "• Ground: 10.27 + 1.86 × t\n"
    "(t = thickness in cm)"
)
ax.text(
    0.98,
    0.35,
    formula_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

plt.tight_layout()
plt.savefig("plots/iwu_renovation_costs_per_sqm.png", dpi=300, bbox_inches="tight")
plt.close()

print("Plot saved to: plots/iwu_renovation_costs_per_sqm.png")

# Print a summary table
print("\n" + "=" * 60)
print("IWU Renovation Costs Summary (€/m²)")
print("=" * 60)
print(
    f"{'Thickness (cm)':<15} {'Walls':<12} {'Sloped Roof':<15} {'Flat Roof':<12} {'Ground':<10}"
)
print("-" * 60)
for t in [1, 5, 10, 15, 20, 25]:
    idx = t - 1
    print(
        f"{t:<15} {cost_walls[idx]:<12.2f} {cost_roof_sloped[idx]:<15.2f} {cost_roof_flat[idx]:<12.2f} {cost_ground[idx]:<10.2f}"
    )
