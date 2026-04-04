"""
Extract break-even line slopes from the gas/electricity contour plot.

The slope represents dGas/dElectricity at customer break-even: how much
gas price increase is needed to compensate for a unit decrease in electricity
price (or vice versa). A steeper slope means the scenario is more sensitive
to electricity prices relative to gas prices.

Used for R3-RES-M1 (appendix table alongside RF slopes).
"""

import glob
import numpy as np
import pandas as pd
from pathlib import Path

from config import sensitivity_results_dir

# ── HT and Booster: single break-even line each ─────────────────────
SCENARIOS_SIMPLE = {
    "HT": ("unrenovated", "combined_electicity_gas"),
    "Booster": ("booster", "combined_electicity_gas"),
}

results = []

for name, (sim, analysis) in SCENARIOS_SIMPLE.items():
    path = sensitivity_results_dir(sim, analysis) / "data" / "mfh_savings_analysis.csv"
    df = pd.read_csv(path)

    # Find break-even points: where average savings crosses zero
    # Group by electricity multiplier, find gas multiplier at break-even
    breakeven_points = []
    for el_val in sorted(df["electricity_multiplier"].unique()):
        subset = df[abs(df["electricity_multiplier"] - el_val) < 0.01].sort_values("gas_multiplier")
        gv = subset["gas_multiplier"].values
        sv = subset["average_savings"].values

        for i in range(len(sv) - 1):
            if sv[i] <= 0 and sv[i + 1] > 0:
                frac = -sv[i] / (sv[i + 1] - sv[i])
                be_gas = gv[i] + frac * (gv[i + 1] - gv[i])
                breakeven_points.append((el_val, be_gas))
                break

    if len(breakeven_points) >= 2:
        pts = np.array(breakeven_points)
        # Linear fit: gas = slope * electricity + intercept
        slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
        results.append({
            "Scenario": name,
            "Reno Cost Mult": "N/A",
            "Slope (dGas/dElec)": round(slope, 3),
            "Intercept": round(intercept, 3),
            "N points": len(breakeven_points),
        })

# ── LT+Reno: family of lines at different renovation cost multipliers ──
analysis_lt = "combined_electicity_gas_renovation_costs"
lt_data_path = sensitivity_results_dir("renovated", analysis_lt) / "data"
all_lt_files = glob.glob(str(lt_data_path / f"{analysis_lt}_gas*_el*_reno*.csv"))

# Parse all data
lt_data_list = []
for f in all_lt_files:
    try:
        parts = Path(f).stem.split("_")
        gas_mult = float(parts[-3].replace("gas", ""))
        el_mult = float(parts[-2].replace("el", ""))
        reno_mult = float(parts[-1].replace("reno", ""))
        df_temp = pd.read_csv(f)
        avg_savings = df_temp["savings_npv_25years_ir_0.05"].mean()
        if not pd.isna(avg_savings):
            lt_data_list.append({
                "electricity_multiplier": el_mult,
                "gas_multiplier": gas_mult,
                "renovation_cost_multiplier": reno_mult,
                "average_savings": avg_savings,
            })
    except Exception:
        pass

df_lt = pd.DataFrame(lt_data_list)

# Extract slopes for representative renovation cost multipliers
for reno_val in [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0]:
    subset_reno = df_lt[abs(df_lt["renovation_cost_multiplier"] - reno_val) < 0.01]
    breakeven_points = []

    for el_val in sorted(subset_reno["electricity_multiplier"].unique()):
        subset = subset_reno[abs(subset_reno["electricity_multiplier"] - el_val) < 0.01].sort_values("gas_multiplier")
        gv = subset["gas_multiplier"].values
        sv = subset["average_savings"].values

        for i in range(len(sv) - 1):
            if sv[i] <= 0 and sv[i + 1] > 0:
                frac = -sv[i] / (sv[i + 1] - sv[i])
                be_gas = gv[i] + frac * (gv[i + 1] - gv[i])
                breakeven_points.append((el_val, be_gas))
                break

    if len(breakeven_points) >= 2:
        pts = np.array(breakeven_points)
        slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
        results.append({
            "Scenario": "LT+Reno",
            "Reno Cost Mult": f"{reno_val:.1f}",
            "Slope (dGas/dElec)": round(slope, 3),
            "Intercept": round(intercept, 3),
            "N points": len(breakeven_points),
        })

results_df = pd.DataFrame(results)

print("=" * 70)
print("CONTOUR PLOT BREAK-EVEN LINE SLOPES")
print("Slope = dGas/dElectricity at customer break-even")
print("A steeper slope → more sensitive to electricity prices")
print("=" * 70)
print(results_df.to_string(index=False))

# ── LaTeX table ──────────────────────────────────────────────────────
print("\n\n% ── LaTeX table ──────────────────────────────────────────")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\caption{Break-even line slopes from the gas and electricity price sensitivity analysis (Figure~\ref{fig:gas_el_sensitivity}). The slope represents the ratio of gas price sensitivity to electricity price sensitivity at customer break-even: a steeper slope indicates greater sensitivity to electricity prices relative to gas prices. For LT+Reno, slopes are reported at selected renovation cost multipliers.}")
print(r"\label{tab:contour_slopes}")
print(r"\begin{tabular}{llrr}")
print(r"\hline")
print(r"\textbf{Scenario} & \textbf{Reno Cost Mult.} & \textbf{Slope} & \textbf{Intercept} \\ \hline")

for _, row in results_df.iterrows():
    reno = row["Reno Cost Mult"] if row["Reno Cost Mult"] != "N/A" else "---"
    print(f"{row['Scenario']} & {reno} & {row['Slope (dGas/dElec)']:.3f} & {row['Intercept']:.3f} \\\\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
