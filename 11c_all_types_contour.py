"""Contour plot of gas/electricity/renovation sensitivity — ALL building types.

All three scenarios use all-types average NPV savings for consistency.
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SENSITIVITY_DIR, sensitivity_results_dir

# ── paths ────────────────────────────────────────────────────────────────
analysis_type_lt = "combined_electicity_gas_renovation_costs"
analysis_type_other = "combined_electicity_gas"


def _load_all_types_average(simulation, analysis_type):
    """Load individual CSVs and compute all-types average savings."""
    import re
    data_path = sensitivity_results_dir(simulation, analysis_type) / "data"
    pattern = f"{analysis_type}_gas*_el*.csv"
    all_files = glob.glob(str(data_path / pattern))
    if not all_files:
        raise FileNotFoundError(f"No data files found in {data_path}")

    # Regex to extract gas, el, and optional reno multipliers from filename
    re_pattern = re.compile(r"gas([\d.]+)_el([\d.]+)(?:_reno([\d.]+))?\.csv$")

    data_list = []
    for f in all_files:
        m = re_pattern.search(Path(f).name)
        if not m:
            continue
        try:
            gas_mult = float(m.group(1))
            el_mult = float(m.group(2))
            reno_mult = float(m.group(3)) if m.group(3) else None

            df_temp = pd.read_csv(f)
            avg = df_temp["savings_npv_25years_ir_0.05"].mean()
            if not pd.isna(avg):
                row = {
                    "electricity_multiplier": el_mult,
                    "gas_multiplier": gas_mult,
                    "average_savings": avg,
                }
                if reno_mult is not None:
                    row["renovation_cost_multiplier"] = reno_mult
                data_list.append(row)
        except Exception as e:
            print(f"Warning: Could not process {f}: {e}")

    return pd.DataFrame(data_list)


# ── Load all three scenarios with all-types averaging ────────────────────
df_booster = _load_all_types_average("booster", analysis_type_other)
df_ht = _load_all_types_average("unrenovated", analysis_type_other)

df_lt_combined = _load_all_types_average("renovated", analysis_type_lt)

print(f"Booster DF shape: {df_booster.shape}")
print(f"HT DF shape: {df_ht.shape}")
print(f"Combined LT DF shape (all types): {df_lt_combined.shape}")
print(
    "Unique renovation cost multipliers:",
    sorted(df_lt_combined["renovation_cost_multiplier"].unique()),
)

# ── Report break-even at el=1.0 for comparison with MFH-only ────────────
print("\n=== Break-even gas multipliers at el=1.0 (all-types average) ===")
for reno_val in [0.0, 0.2, 1.0]:
    subset = df_lt_combined[
        (abs(df_lt_combined["electricity_multiplier"] - 1.0) < 0.01)
        & (abs(df_lt_combined["renovation_cost_multiplier"] - reno_val) < 0.01)
    ].sort_values("gas_multiplier")

    if subset.empty:
        print(f"  reno={reno_val}: no data")
        continue

    gv = subset["gas_multiplier"].values
    sv = subset["average_savings"].values

    found = False
    for i in range(len(sv) - 1):
        if sv[i] <= 0 and sv[i + 1] > 0:
            frac = -sv[i] / (sv[i + 1] - sv[i])
            be = gv[i] + frac * (gv[i + 1] - gv[i])
            print(f"  reno={reno_val}: break-even gas mult = {be:.3f}  ({(be-1)*100:.1f}% increase)")
            found = True
            break
    if not found:
        if all(v < 0 for v in sv):
            print(f"  reno={reno_val}: never breaks even (max gas_mult tested = {gv[-1]:.1f})")
        elif all(v >= 0 for v in sv):
            print(f"  reno={reno_val}: always positive (min gas_mult tested = {gv[0]:.2f})")


# ── Plotting (same logic as 11c, unchanged) ─────────────────────────────
def _build_extended_grid(df, val_col="average_savings"):
    """Build meshgrid with boundary extrapolation for contour plotting."""
    el_mults = sorted(df["electricity_multiplier"].unique())
    gas_mults = sorted(df["gas_multiplier"].unique())

    if len(el_mults) < 2 or len(gas_mults) < 2:
        return None, None, None

    ext_el = [0.0] + el_mults
    ext_gas = [0.0] + gas_mults
    X, Y = np.meshgrid(ext_el, ext_gas)

    pivot = df.pivot_table(
        index="gas_multiplier", columns="electricity_multiplier", values=val_col
    )
    Z_orig = pivot.reindex(index=gas_mults, columns=el_mults).values

    Z = np.zeros((len(ext_gas), len(ext_el)))
    Z[1:, 1:] = Z_orig

    if len(gas_mults) >= 2:
        slope = (Z_orig[1, :] - Z_orig[0, :]) / (gas_mults[1] - gas_mults[0])
        Z[0, 1:] = Z_orig[0, :] - slope * gas_mults[0]
    if len(el_mults) >= 2:
        slope = (Z_orig[:, 1] - Z_orig[:, 0]) / (el_mults[1] - el_mults[0])
        Z[1:, 0] = Z_orig[:, 0] - slope * el_mults[0]
    if len(gas_mults) >= 2 and len(el_mults) >= 2:
        Z[0, 0] = (Z[0, 1] + Z[1, 0]) / 2

    return X, Y, Z


def create_combined_contour_all_types(df_booster, df_ht, df_lt_combined):
    fig, ax = plt.subplots(figsize=(12, 9))

    datasets_2d = {
        "Booster": (df_booster, "black", "solid"),
        "Unrenovated (HT)": (df_ht, "red", "solid"),
    }
    legend_elements = []

    # Booster and HT
    for label, (df, color, ls) in datasets_2d.items():
        if df.empty:
            continue
        X, Y, Z = _build_extended_grid(df)
        if Z is None:
            continue
        if np.nanmin(Z) < 0 < np.nanmax(Z):
            ax.contour(X, Y, Z, levels=[0], colors=[color], linestyles=[ls], linewidths=2)
            legend_elements.append(
                plt.Line2D([0], [0], color=color, linestyle=ls, label=label, linewidth=2)
            )

    # LT+Reno family
    lt_linestyles = [
        "-", "--", ":", "-.",
        (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)),
        (0, (3, 5, 1, 5)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (1, 10)),
    ]
    unique_reno = sorted(df_lt_combined["renovation_cost_multiplier"].unique())

    for i, reno_mult in enumerate(unique_reno):
        df_sub = df_lt_combined[
            df_lt_combined["renovation_cost_multiplier"] == reno_mult
        ]
        if df_sub.empty:
            continue

        X, Y, Z = _build_extended_grid(df_sub)
        if Z is None:
            continue

        ls = lt_linestyles[i % len(lt_linestyles)]
        if np.nanmin(Z) < 0 < np.nanmax(Z):
            ax.contour(X, Y, Z, levels=[0], colors=["blue"], linestyles=[ls], linewidths=2)
            legend_elements.append(
                plt.Line2D(
                    [0], [0], color="blue", linestyle=ls,
                    label=f"Renovated (LT) - Reno Cost x{reno_mult:.2f}", linewidth=2,
                )
            )

    ax.set_xlabel("Electricity Price Multiplier", fontsize=20)
    ax.set_ylabel("Gas Price Multiplier", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.legend(
        handles=legend_elements, fontsize=12,
        title="Scenarios", title_fontsize=14, loc="lower right",
    )

    # Save
    save_filename = "combined_price_reno_sensitivity_contour_v2_all_types.png"
    paper_fig = Path(
        "/Users/lucacasamassima/Library/CloudStorage/GoogleDrive-lucasamassima@gmail.com/"
        "Other computers/My laptop/Documents/phd thesis/Possible papers/"
        "District Heating Comparison/paper_git/4th-Gen-Paper/figure"
    )

    for dest in [SENSITIVITY_DIR / save_filename, paper_fig / save_filename]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dest, bbox_inches="tight", dpi=1000)
        print(f"Saved: {dest}")

    plt.close(fig)


create_combined_contour_all_types(df_booster, df_ht, df_lt_combined)
print("Done.")
