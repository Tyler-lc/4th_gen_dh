"""
Generate LaTeX table of reduction factor sensitivity linear coefficients.

Used for R3-RES-M1: Tabulate slopes and intercepts from Figure reduction_factor.
The customer NPV savings per m² NFA follow a linear relationship with the
reduction factor: NPV = m × RF + q, where m is the slope and q the intercept.

Output: LaTeX table for the appendix.
"""

import pandas as pd
import numpy as np

from config import sensitivity_results_dir, MULTIPLE_GRAPHS_SUBDIR

SCENARIOS = {
    "HT": "unrenovated",
    "LT+Reno": "renovated",
    "Booster": "booster",
}

TYPE_ORDER = ["mfh", "ab", "sfh", "th", "office", "trade", "education", "health", "other"]
TYPE_LABELS = {
    "mfh": "MFH", "ab": "AB", "sfh": "SFH", "th": "TH",
    "office": "Office", "trade": "Trade", "education": "Education",
    "health": "Health", "other": "Other",
}

rows = []

for scenario_name, sim_name in SCENARIOS.items():
    path = sensitivity_results_dir(sim_name, "reduction_factor") / "data" / MULTIPLE_GRAPHS_SUBDIR / "avg_savings_data_nfa.csv"
    df = pd.read_csv(path, index_col=0)
    x = df.index.values

    for col in TYPE_ORDER:
        if col not in df.columns:
            continue
        y = df[col].values
        m, q = np.polyfit(x, y, 1)
        rows.append({
            "type": col,
            "label": TYPE_LABELS[col],
            "scenario": scenario_name,
            "m": round(m, 1),
            "q": round(q, 1),
        })

results = pd.DataFrame(rows)

# ── Print summary ────────────────────────────────────────────────────
print("=" * 70)
print("LINEAR COEFFICIENTS: NPV [EUR/m² NFA] = m × RF + q")
print("=" * 70)
pivot_m = results.pivot(index="label", columns="scenario", values="m")
pivot_q = results.pivot(index="label", columns="scenario", values="q")
pivot_m = pivot_m[["HT", "LT+Reno", "Booster"]]
pivot_q = pivot_q[["HT", "LT+Reno", "Booster"]]

ordered = [TYPE_LABELS[t] for t in TYPE_ORDER]
pivot_m = pivot_m.loc[ordered]
pivot_q = pivot_q.loc[ordered]

print("\nSlopes (m):")
print(pivot_m.to_string())
print("\nIntercepts (q):")
print(pivot_q.to_string())

# ── Generate LaTeX ───────────────────────────────────────────────────
print("\n\n% ── LaTeX table ──────────────────────────────────────────")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\caption{Linear coefficients of customer NPV savings per m$^2$ NFA as a function of the heat price reduction factor (RF). The relationship follows NPV $= m \times \text{RF} + q$, where $m$ is the sensitivity coefficient and $q$ is the intercept. Units: \texteuro/m$^2$ NFA.}")
print(r"\label{tab:rf_slopes}")
print(r"\begin{tabular}{l|rr|rr|rr}")
print(r"\hline")
print(r" & \multicolumn{2}{c|}{\textbf{HT}} & \multicolumn{2}{c|}{\textbf{LT+Reno}} & \multicolumn{2}{c}{\textbf{Booster}} \\")
print(r"\textbf{Building Type} & $m$ & $q$ & $m$ & $q$ & $m$ & $q$ \\ \hline")

for t in TYPE_ORDER:
    label = TYPE_LABELS[t]
    row_data = results[results["type"] == t]
    ht = row_data[row_data["scenario"] == "HT"].iloc[0]
    lt = row_data[row_data["scenario"] == "LT+Reno"].iloc[0]
    bo = row_data[row_data["scenario"] == "Booster"].iloc[0]
    print(f"{label} & {ht['m']:.0f} & {ht['q']:.0f} & {lt['m']:.0f} & {lt['q']:.0f} & {bo['m']:.0f} & {bo['q']:.0f} \\\\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
