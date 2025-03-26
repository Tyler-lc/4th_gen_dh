import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

### first let's import all the data from the csv files

base_path = "sensitivity_analysis"
path_booster = (
    f"{base_path}/booster/combined_electicity_gas/data/mfh_savings_analysis.csv"
)
path_ht = (
    f"{base_path}/unrenovated/combined_electicity_gas/data/mfh_savings_analysis.csv"
)
path_lt = f"{base_path}/renovated/combined_electicity_gas/data/mfh_savings_analysis.csv"

### now let's import the data:

df_booster = pd.read_csv(path_booster)
df_ht = pd.read_csv(path_ht)
df_lt = pd.read_csv(path_lt)

### now let's plot the data:


def create_combined_contour(df_booster, df_ht, df_lt):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Process each dataset
    datasets = {
        "Booster": df_booster,
        "Unrenovated (HT)": df_ht,
        "Renovated (LT)": df_lt,
    }

    # Define colors for each case
    colors = {"Booster": "black", "Unrenovated (HT)": "red", "Renovated (LT)": "blue"}

    for label, df in datasets.items():
        # Get unique multipliers
        el_mults = sorted(df["electricity_multiplier"].unique())
        gas_mults = sorted(df["gas_multiplier"].unique())

        # Create meshgrid for contour plot
        X, Y = np.meshgrid(el_mults, gas_mults)

        # Create data matrix for contour
        Z = np.zeros((len(gas_mults), len(el_mults)))

        # Reshape the data into the grid
        for i, gas_mult in enumerate(gas_mults):
            for j, el_mult in enumerate(el_mults):
                mask = (df["electricity_multiplier"] == el_mult) & (
                    df["gas_multiplier"] == gas_mult
                )
                Z[i, j] = df.loc[mask, "average_savings"].values[0]

        # Plot the break-even line
        cs = ax.contour(X, Y, Z, levels=[0], colors=[colors[label]], linestyles="solid")
        ax.clabel(cs, inline=True, fmt=label)

    ax.set_xlabel("Electricity Price Multiplier", fontsize=18)
    ax.set_ylabel("Gas Price Multiplier", fontsize=18)
    ax.set_title("Break-even Lines Comparison", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, label=label)
        for label, color in colors.items()
    ]
    ax.legend(handles=legend_elements, fontsize=14, loc="upper left")

    plt.tight_layout()
    plt.savefig(
        "sensitivity_analysis/combined_price_sensitivity_contour.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


# Call the function with your imported dataframes
create_combined_contour(df_booster, df_ht, df_lt)
print("done")
