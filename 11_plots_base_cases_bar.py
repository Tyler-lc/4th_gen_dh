"""Grouped bar chart of customer NPV savings per building type and scenario.

Replaces the original boxplot (11_plots_base_cases.py) with a cleaner
grouped bar chart showing the mean NPV per m² NFA with 25th-75th
percentile whiskers. No outliers shown.

Addresses R1-6.2 (figure clarity), R4-8 (capitalise building types).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import PLOTS_DIR

# ── Load data ────────────────────────────────────────────────────────────
ht = pd.read_csv(
    PLOTS_DIR / "HighTemperature" / "data_exports_1_dhg_lifetime_50" / "npv_data.csv"
)
lt = pd.read_csv(
    PLOTS_DIR / "LowTemperature" / "data_exports_1_dhg_lifetime_50" / "npv_data.csv"
)
bo = pd.read_csv(
    PLOTS_DIR / "booster" / "data_exports_1_dhg_lifetime_50" / "npv_data.csv"
)

ht["scenario"] = "HT"
lt["scenario"] = "LT + Reno"
bo["scenario"] = "Booster"
combined = pd.concat([ht, lt, bo])

# ── Configuration ────────────────────────────────────────────────────────
res_order = ["mfh", "ab", "sfh", "th"]
nonres_order = ["other", "trade", "education", "health", "office"]
type_order = res_order + nonres_order

type_labels = {
    "mfh": "MFH", "sfh": "SFH", "ab": "AB", "th": "TH",
    "other": "Other", "trade": "Trade", "education": "Education",
    "health": "Health", "office": "Office",
}

scenarios = ["HT", "LT + Reno", "Booster"]
colors = {"HT": "#7aa6c2", "LT + Reno": "#e8a862", "Booster": "#7bc77b"}
bar_width = 0.25
x = np.arange(len(type_order))

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

for i, scenario in enumerate(scenarios):
    sdata = combined[combined["scenario"] == scenario]
    means = []
    q25 = []
    q75 = []
    for bt in type_order:
        bt_data = sdata[sdata["building_usage"] == bt]["npv_per_nfa"]
        means.append(bt_data.mean())
        q25.append(bt_data.quantile(0.25) if len(bt_data) > 1 else bt_data.mean())
        q75.append(bt_data.quantile(0.75) if len(bt_data) > 1 else bt_data.mean())

    means = np.array(means)
    q25 = np.array(q25)
    q75 = np.array(q75)

    yerr_low = means - q25
    yerr_high = q75 - means

    ax.bar(
        x + i * bar_width, means, bar_width,
        label=scenario, color=colors[scenario], alpha=0.85,
        yerr=[yerr_low, yerr_high], capsize=3,
        error_kw={"linewidth": 1.5, "color": "black"},
    )

ax.set_xticks(x + bar_width)
ax.set_xticklabels(
    [type_labels[t] for t in type_order], rotation=45, ha="right", fontsize=18
)
ax.set_ylabel("NPV Savings per Net Floor Area [\u20ac/m\u00b2]", fontsize=20)
ax.set_xlabel("Building Type", fontsize=20)
ax.tick_params(axis="y", labelsize=18)
ax.legend(fontsize=18, loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.axhline(y=0, color="black", linewidth=0.8)

plt.tight_layout()

# ── Save to both locations ───────────────────────────────────────────────
save_filename = "comparison_all_scenarios_all_scenarios.png"
paper_fig = Path(
    "/Users/lucacasamassima/Library/CloudStorage/GoogleDrive-lucasamassima@gmail.com/"
    "Other computers/My laptop/Documents/phd thesis/Possible papers/"
    "District Heating Comparison/paper_git/4th-Gen-Paper/figure"
)

for dest in [PLOTS_DIR / save_filename, paper_fig / save_filename]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest, dpi=300, bbox_inches="tight")
    print(f"Saved: {dest}")

plt.close()
print("Done.")
