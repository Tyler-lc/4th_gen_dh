import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

### first let's import all the data from the csv files

base_path = Path("sensitivity_analysis")
analysis_type_lt = "combined_electicity_gas_renovation_costs"
analysis_type_other = "combined_electicity_gas"

path_booster_agg = (
    base_path / "booster" / analysis_type_other / "data" / "mfh_savings_analysis.csv"
)
path_ht_agg = (
    base_path
    / "unrenovated"
    / analysis_type_other
    / "data"
    / "mfh_savings_analysis.csv"
)

df_booster = pd.read_csv(path_booster_agg)
df_ht = pd.read_csv(path_ht_agg)

print(f"Booster DF shape: {df_booster.shape}")
print(f"HT DF shape: {df_ht.shape}")
print(f"Booster NaNs in savings: {df_booster['average_savings'].isna().sum()}")
print(f"HT NaNs in savings: {df_ht['average_savings'].isna().sum()}")

lt_data_path = base_path / "renovated" / analysis_type_lt / "data"
all_lt_files = glob.glob(str(lt_data_path / f"{analysis_type_lt}_gas*_el*_reno*.csv"))

if not all_lt_files:
    raise FileNotFoundError(
        f"No detailed LT data files found in {lt_data_path}. Did 10d script run and save correctly?"
    )

lt_data_list = []
for f in all_lt_files:
    try:
        parts = Path(f).stem.split("_")
        gas_mult = float(parts[-3].replace("gas", ""))
        el_mult = float(parts[-2].replace("el", ""))
        reno_mult = float(parts[-1].replace("reno", ""))

        df_temp = pd.read_csv(f)
        mfh_savings = df_temp[df_temp["building_usage"] == "mfh"][
            "savings_npv_25years_ir_0.05"
        ].mean()

        if not pd.isna(mfh_savings):
            lt_data_list.append(
                {
                    "electricity_multiplier": el_mult,
                    "gas_multiplier": gas_mult,
                    "renovation_cost_multiplier": reno_mult,
                    "average_savings": mfh_savings,
                }
            )
    except Exception as e:
        print(f"Warning: Could not process file {f}: {e}")

df_lt_combined = pd.DataFrame(lt_data_list)

print(f"Combined LT DF shape: {df_lt_combined.shape}")
print(f"Combined LT NaNs in savings: {df_lt_combined['average_savings'].isna().sum()}")
print("LT DF Head:\n", df_lt_combined.head())
print("LT DF Tail:\n", df_lt_combined.tail())

if df_lt_combined.empty:
    raise ValueError("Failed to load or process any LT scenario data.")

print(f"Loaded {len(df_lt_combined)} data points for LT scenario.")
print(
    "Unique renovation cost multipliers found:",
    sorted(df_lt_combined["renovation_cost_multiplier"].unique()),
)

### now let's plot the data:


def create_combined_contour_v2(df_booster, df_ht, df_lt_combined):
    fig, ax = plt.subplots(figsize=(12, 9))
    # rounding_decimals = 3 # Rounding less critical now, but can keep if desired

    datasets_2d = {
        "Booster": (df_booster, "black", "solid"),
        "Unrenovated (HT)": (df_ht, "red", "solid"),
    }
    legend_elements = []

    # --- Plot 2D datasets ---
    for label, (df, color, linestyle) in datasets_2d.items():
        if df.empty:
            print(f"Warning: DataFrame for {label} is empty. Skipping.")
            continue

        # --- Create grid specific to *this* dataset ---
        try:
            df_el_mults = sorted(df["electricity_multiplier"].unique())
            df_gas_mults = sorted(df["gas_multiplier"].unique())

            if (
                not df_el_mults
                or not df_gas_mults
                or len(df_el_mults) < 2
                or len(df_gas_mults) < 2
            ):
                print(
                    f"Warning: Not enough unique multipliers in {label} for contour plot. Need at least 2x2 grid. Skipping."
                )
                continue

            X, Y = np.meshgrid(df_el_mults, df_gas_mults)
            # Pivot table is a robust way to create the Z matrix
            pivot = df.pivot_table(
                index="gas_multiplier",
                columns="electricity_multiplier",
                values="average_savings",
            )
            # Reindex to match the sorted lists used for meshgrid
            Z = pivot.reindex(index=df_gas_mults, columns=df_el_mults).values

        except Exception as e:
            print(f"Error preparing grid for {label}: {e}. Skipping.")
            continue

        print(
            f"NaN count in Z matrix for {label}: {np.isnan(Z).sum()} out of {Z.size}"
        )  # Should be low/zero now

        # Check if Z contains both positive and negative values for contouring
        if np.nanmin(Z) < 0 < np.nanmax(Z):
            cs = ax.contour(
                X,
                Y,
                Z,  # Use this dataset's specific grid
                levels=[0],
                colors=[color],
                linestyles=linestyle,
                linewidths=2,
            )
            legend_elements.append(
                plt.Line2D(
                    [0], [0], color=color, linestyle=linestyle, label=label, linewidth=2
                )
            )
        else:
            print(
                f"Warning: Cannot plot level 0 contour for {label} - data does not cross zero or is all NaN."
            )

    # --- Process LT scenario (multiple lines) ---
    unique_reno_mults = sorted(df_lt_combined["renovation_cost_multiplier"].unique())
    lt_base_color = "blue"
    lt_linestyles = [
        "-",
        "--",
        ":",
        "-.",
        (0, (3, 1, 1, 1)),
        (0, (5, 1)),
        (0, (1, 1)),
        (0, (3, 5, 1, 5)),
        (0, (5, 5)),
        (0, (3, 10, 1, 10)),
        (0, (1, 10)),
    ]
    if len(lt_linestyles) < len(unique_reno_mults):
        print("Warning: Not enough unique linestyles for all renovation multipliers.")
        lt_linestyles.extend(["-"] * (len(unique_reno_mults) - len(lt_linestyles)))

    for i, reno_mult in enumerate(unique_reno_mults):
        df_subset = df_lt_combined[
            df_lt_combined["renovation_cost_multiplier"] == reno_mult
        ].copy()  # Use copy to avoid potential warnings

        if df_subset.empty:
            print(f"Warning: No data for LT with reno_mult={reno_mult:.2f}. Skipping.")
            continue

        # --- Create grid specific to *this* subset ---
        try:
            df_el_mults_lt = sorted(df_subset["electricity_multiplier"].unique())
            df_gas_mults_lt = sorted(df_subset["gas_multiplier"].unique())

            if (
                not df_el_mults_lt
                or not df_gas_mults_lt
                or len(df_el_mults_lt) < 2
                or len(df_gas_mults_lt) < 2
            ):
                print(
                    f"Warning: Not enough unique multipliers in LT subset (reno={reno_mult:.2f}) for contour plot. Skipping."
                )
                continue

            X_lt, Y_lt = np.meshgrid(df_el_mults_lt, df_gas_mults_lt)
            # Pivot table for Z_lt
            pivot_lt = df_subset.pivot_table(
                index="gas_multiplier",
                columns="electricity_multiplier",
                values="average_savings",
            )
            Z_lt = pivot_lt.reindex(
                index=df_gas_mults_lt, columns=df_el_mults_lt
            ).values

        except Exception as e:
            print(f"Error preparing grid for LT (reno={reno_mult:.2f}): {e}. Skipping.")
            continue

        print(
            f"NaN count in Z_lt matrix for reno_mult={reno_mult:.2f}: {np.isnan(Z_lt).sum()} out of {Z_lt.size}"
        )  # Should be low/zero

        # Check if Z_lt contains both positive and negative values
        if np.nanmin(Z_lt) < 0 < np.nanmax(Z_lt):
            # --- Get the specific linestyle for this iteration ---
            linestyle_for_this_contour = lt_linestyles[i % len(lt_linestyles)]

            # --- Modified contour call ---
            cs_lt = ax.contour(
                X_lt,
                Y_lt,
                Z_lt,
                levels=[0],  # Still level 0
                colors=[lt_base_color],  # Keep color as a list element
                linestyles=linestyle_for_this_contour,  # Pass the *specific* style string/tuple
                linewidths=2,
            )
            lt_label = f"Renovated (LT) - Reno Cost x{reno_mult:.2f}"
            # --- Modified legend element creation ---
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=lt_base_color,
                    linestyle=linestyle_for_this_contour,  # Use the *specific* style
                    label=lt_label,
                    linewidth=2,
                )
            )
        else:
            print(
                f"Warning: Cannot plot level 0 contour for LT (reno={reno_mult:.2f}) - data does not cross zero or is all NaN."
            )

    # --- Final Plot Customization ---
    ax.set_xlabel("Electricity Price Multiplier", fontsize=16)
    ax.set_ylabel("Gas Price Multiplier", fontsize=16)
    ax.set_title("Break-even Lines Comparison (MFH Savings)", fontsize=18)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Determine plot limits based on all data points to ensure consistency
    all_el = (
        df_booster["electricity_multiplier"].tolist()
        + df_ht["electricity_multiplier"].tolist()
        + df_lt_combined["electricity_multiplier"].tolist()
    )
    all_gas = (
        df_booster["gas_multiplier"].tolist()
        + df_ht["gas_multiplier"].tolist()
        + df_lt_combined["gas_multiplier"].tolist()
    )
    if all_el and all_gas:  # Set limits only if data exists
        ax.set_xlim(min(all_el), max(all_el))
        ax.set_ylim(min(all_gas), max(all_gas))

    # --- Place Legend ---
    ax.legend(
        handles=legend_elements,
        fontsize=9,
        title="Scenarios",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    # --- Adjust subplot to make room ---
    fig.subplots_adjust(right=0.75)

    # --- Prepare Save Path ---
    save_filename = "combined_price_reno_sensitivity_contour_v2.png"
    save_path = base_path / save_filename
    absolute_save_path = save_path.resolve()
    print(f"Attempting to save figure to: {absolute_save_path}")

    # --- Ensure Directory Exists ---
    try:
        base_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {base_path}")
    except Exception as e:
        print(f"Error creating directory {base_path}: {e}")
        plt.close(fig)
        return

    # --- Save Figure ---
    try:
        plt.savefig(
            absolute_save_path,
            bbox_inches="tight",  # Add back bbox_inches='tight'
            dpi=300,
        )
        print(f"plt.savefig command executed for {absolute_save_path}")
        if absolute_save_path.is_file():
            print(f"Successfully saved figure: {absolute_save_path}")
        else:
            print(
                f"!!! Failed to save figure: File not found after save command at {absolute_save_path}"
            )
    except Exception as e:
        print(f"!!! Error during plt.savefig: {e}")

    plt.close(fig)


# Call the function with your imported dataframes
try:
    create_combined_contour_v2(
        df_booster, df_ht, df_lt_combined
    )  # Call the new function
    print("Combined contour plot created successfully.")
except Exception as e:
    print(f"Error during plotting: {e}")

print("Script finished.")
