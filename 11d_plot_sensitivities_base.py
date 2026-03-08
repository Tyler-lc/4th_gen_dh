"""
11d_plot_sensitivities_base.py

This script creates combined sensitivity analysis plots in a 2x2 layout showing all three district heating scenarios:
- HT Scenario (unrenovated buildings with high-temperature DH) - Top left
- LT+Reno Scenario (renovated buildings with low-temperature DH) - Top right
- Booster Scenario (unrenovated buildings with booster heat pumps) - Bottom left
- Legend (shared for all scenarios) - Bottom right

The script uses the processed data from the 'multitple_graphs' subdirectories created by the
sensitivity analysis scripts (09b_HT_Sens_Analysis.py, 09c_LT_Sens_Analysis.py, 09d_HT_Booster_Sens_Analysis.py).

Usage:
    python 11d_plot_sensitivities_base.py [analysis_type]

Arguments:
    analysis_type (optional): Type of sensitivity analysis to plot
                             Default: first available analysis type
                             Currently available: reduction_factor

Example:
    python 11d_plot_sensitivities_base.py reduction_factor

The resulting plot is saved to: sensitivity_analysis/combined_{analysis_type}_base_sensitivities.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.plotting import nfa_savings_operator_comparison
import glob

from config import SENSITIVITY_DIR, sensitivity_results_dir

output_dpi = 1000


def flatten_list(list_of_lists):
    flat_list = []
    for value in list_of_lists:
        flat_list.append(value[0])
    return flat_list


def import_data(analysis_type, simulation):
    main_path = (
        sensitivity_results_dir(simulation, analysis_type) / "data" / "multitple_graphs"
    )

    all_npv_data = {}
    all_npv_files = glob.glob(str(main_path / "all_npv_data_*.csv"))
    all_npv_files.sort()
    keys = []
    for files in all_npv_files:
        name = files.split("_")[-1]
        name = name.replace(".csv", "")
        keys.append(name)

    for key in keys:
        all_npv_data[key] = pd.read_csv(
            main_path / f"all_npv_data_{key}.csv", index_col=0
        )

    avg_savings_data_nfa = pd.read_csv(
        main_path / "avg_savings_data_nfa.csv", index_col=0
    )
    npv_operator_df = pd.read_csv(main_path / "npv_operator.csv", index_col=0)
    npv_operator = npv_operator_df.values.tolist()
    npv_operator = flatten_list(npv_operator)
    values_df = pd.read_csv(main_path / "values.csv", index_col=0)
    values = values_df.values.tolist()
    values = flatten_list(values)

    return all_npv_data, avg_savings_data_nfa, npv_operator, values


def create_combined_base_sensitivities_plot(analysis_type="reduction_factor"):
    """
    Create a combined 2x2 plot showing all three scenarios (HT, LT, Booster) for a specific analysis type.
    The fourth subplot (bottom-right) contains the shared legend for all scenarios.
    """

    # Define scenarios
    scenarios = [
        ("unrenovated", "HT Scenario"),
        ("renovated", "LT+Reno Scenario"),
        ("booster", "Booster Scenario"),
    ]

    # Create figure with subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    # Store legend data from first scenario
    legend_handles = []
    legend_labels = []

    for idx, (simulation, scenario_title) in enumerate(scenarios):
        try:
            print(f"Loading data for {scenario_title}...")

            # Load data for this scenario
            all_npv_data, avg_savings_data_nfa, npv_operator, values = import_data(
                analysis_type, simulation
            )

            print(f"  Loaded {len(values)} data points")

            # Create subplot for this scenario using the existing function
            ax = axes_flat[idx]

            # Use the existing nfa_savings_operator_comparison function but modify for subplot
            # We need to pass the specific axis to plot on

            # Create the plot directly on the subplot
            analysis_type_title = analysis_type.replace("_", " ").title()

            # Plot average customer savings on primary axis (left)
            ax.set_xlabel(f"{analysis_type_title}", fontsize=20)
            ax.set_ylabel(
                "Average Customer Savings (€/m²NFA)", color="tab:blue", fontsize=20
            )

            # Plot each building type's savings
            colors_palette = sns.color_palette(
                "colorblind", n_colors=len(avg_savings_data_nfa.columns)
            )
            markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

            for building_type, color_palette, marker in zip(
                avg_savings_data_nfa.columns, colors_palette, markers
            ):
                line = ax.plot(
                    values,
                    avg_savings_data_nfa[building_type],
                    marker=marker,
                    markersize=8,
                    label=building_type,
                    color=color_palette,
                    linestyle="-",
                    linewidth=2,
                    markerfacecolor=color_palette,
                    markeredgecolor="black",
                )

                # Store legend data from first scenario
                if idx == 0:
                    legend_handles.append(line[0])
                    legend_labels.append(building_type)

            ax.tick_params(axis="y", labelcolor="tab:blue")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.tick_params(axis="both", which="major", labelsize=18)

            # Create secondary axis (right) for DH operator NPV
            ax2 = ax.twinx()
            ax2.set_ylabel("DH Operator NPV (M€)", color="tab:red", fontsize=20)

            # Convert operator NPV to M€
            npv_operator_millions = [x / 1000000 for x in npv_operator]

            npv_line = ax2.plot(
                values,
                npv_operator_millions,
                "r-",
                linewidth=3,
                label="DH Operator NPV",
            )

            # Store NPV legend data from first scenario
            if idx == 0:
                legend_handles.append(npv_line[0])
                legend_labels.append("DH Operator NPV")

            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax2.tick_params(axis="both", which="major", labelsize=18)

            # Add title for this subplot
            ax.set_title(f"{scenario_title}", fontsize=22, pad=20)

        except Exception as e:
            print(f"Error processing {scenario_title}: {e}")
            axes_flat[idx].set_visible(False)
            continue

    # Use the fourth subplot (bottom-right) for the legend
    legend_ax = axes_flat[3]
    legend_ax.axis("off")  # Turn off the axes

    # Create the legend in the fourth subplot
    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        fontsize=20,
        title="Legend",
        title_fontsize=22,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Add main title
    analysis_type_title = analysis_type.replace("_", " ").title()
    # plt.suptitle(
    #     f"DH Operator NPV and Customer Savings\nSensitivity to {analysis_type_title}",
    #     fontsize=26,
    #     y=0.98,
    # )

    # Adjust layout
    plt.tight_layout()

    # Save the combined plot
    output_file = (
        SENSITIVITY_DIR / f"combined_{analysis_type}_base_sensitivities.png"
    )
    plt.savefig(output_file, bbox_inches="tight", dpi=output_dpi)
    print(f"Combined plot saved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    import sys
    import os

    print("Creating combined base sensitivities plot...")

    # Check what analysis types have multitple_graphs data available
    available_analyses = []
    for scenario in ["unrenovated", "renovated", "booster"]:
        scenario_path = SENSITIVITY_DIR / scenario
        if scenario_path.exists():
            for analysis_dir in os.listdir(scenario_path):
                multitple_graphs_path = (
                    scenario_path / analysis_dir / "data" / "multitple_graphs"
                )
                if multitple_graphs_path.exists():
                    if analysis_dir not in available_analyses:
                        available_analyses.append(analysis_dir)

    print(f"Available analysis types with processed data: {sorted(available_analyses)}")

    # Get analysis type from command line argument or use default
    analysis_type = None

    # Filter out Jupyter-specific arguments
    valid_args = []
    for arg in sys.argv[1:]:
        if (
            not arg.startswith("--f=")
            and not arg.startswith("-f=")
            and not ".json" in arg
        ):
            valid_args.append(arg)

    if valid_args:
        analysis_type = valid_args[0]
        if analysis_type not in available_analyses:
            print(
                f"Warning: '{analysis_type}' may not have processed data for all scenarios."
            )
            print(f"Available options: {sorted(available_analyses)}")
    else:
        if available_analyses:
            analysis_type = available_analyses[0]  # Use first available
        else:
            print(
                "No processed data found! Make sure to run the sensitivity analysis scripts first."
            )
            sys.exit(1)

    print(f"Creating plot for analysis type: {analysis_type}")
    create_combined_base_sensitivities_plot(analysis_type)
