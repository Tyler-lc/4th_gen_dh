import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def lcoh_operator_NPV(
    values, lcoh_dhg, lcoh_hp, npv_operator, analysis_type, simulation
):
    """
    Creates a plots with two subplots one of top of the other.
    The top plot shows the LCOH of the heat produced, the bottom shows the NPV of the DH Operator.
    ### Inputs:
    - values: range of values from the sensitivity analysis (e.g. range of supply temperatures 50, 55, 60..)
    - lcoh_dhg: list of LCOH values for the DHG
    - lcoh_hp: list of LCOH values for the HP
    - npv_operator: list of NPV values for the DH Operator
    - analysis_type: name of the sensitivity analysis type (e.g. "supply_temperature", "ir", "max_cop")
    - simulation: name of the simulation (e.g. "renovated")
    ### Output:
    - Saves a plot in the sensitivity_analysis/{simulation}/{analysis_type}/plots/ directory
    """
    # Create a figure with multiple subplots for different analyses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # First subplot for LCOH
    ax1.plot(values, lcoh_dhg, label="LCOH DHG")
    ax1.plot(values, lcoh_hp, label="LCOH HP")
    ax1.set_xlabel(f"{analysis_type}")
    ax1.set_ylabel("LCOH (€/kWh)")
    ax1.set_title(f"Sensitivity Analysis - LCOH vs {analysis_type}")
    ax1.legend()

    # Second subplot for Operator NPV
    ax2.plot(values, npv_operator, label="DH Operator NPV", color="green")
    ax2.set_xlabel(f"{analysis_type}")
    ax2.set_ylabel("NPV (€)")
    ax2.set_title(f"Sensitivity Analysis - DH Operator NPV vs {analysis_type}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_sensitivity_analysis.png"
    )
    plt.close()


def calculate_average_savings(all_npv_data, values, analysis_type):
    building_types = all_npv_data[values[0]]["building_usage"].unique()
    avg_savings_data = pd.DataFrame(columns=building_types, index=values)
    avg_savings_data_nfa = pd.DataFrame(columns=building_types, index=values)
    for value in values:
        if analysis_type == "ir":
            ir = value
        else:
            ir = 0.05
        # Calculate average savings for each building type at this value
        avg_savings = (
            all_npv_data[value]
            .groupby("building_usage")[f"savings_npv_25years_ir_{ir}"]
            .mean()
        )
        all_npv_data[value]["savings/NFA [€/m2]"] = (
            all_npv_data[value][f"savings_npv_25years_ir_{ir}"]
            / all_npv_data[value]["NFA"]
        )
        avg_savings_nfa = (
            all_npv_data[value].groupby("building_usage")["savings/NFA [€/m2]"].mean()
        )

        # Fill in the row for this value
        avg_savings_data.loc[value] = avg_savings
        avg_savings_data_nfa.loc[value] = avg_savings_nfa
    return avg_savings_data_nfa, avg_savings_data


def plot_savings_distribution(all_npv_data, values, analysis_type, simulation):
    for value in values:
        if analysis_type == "ir":
            ir = value
        else:
            ir = 0.05
        n_columns = 3
        # Plot histogram of savings distribution by building type
        plt.figure(figsize=(20, 15))
        building_types = all_npv_data[value]["building_usage"].unique()
        num_types = len(building_types)
        rows = (
            num_types + 2
        ) // n_columns  # Calculate number of rows needed for n_columns

        for i, building_type in enumerate(building_types, 1):
            plt.subplot(rows, n_columns, i)
            data = (
                all_npv_data[value][
                    all_npv_data[value]["building_usage"] == building_type
                ][f"savings_npv_25years_ir_{ir}"]
                / 1000
            )  # Convert to k€
            scatter = sns.histplot(data, kde=True)
            scatter.set_title(
                f"Savings Distribution - {building_type}\n{analysis_type}: {value}",
                fontsize=14,
            )
            scatter.set_xlabel("NPV Savings (k€)", fontsize=12)
            scatter.set_ylabel("Frequency", fontsize=12)
            scatter.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(
            f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_savings_distribution_{value}.png"
        )
        plt.close()

    # Now we can plot distributions using the stored data
    n_columns = 3
    for value in values:
        if analysis_type == "ir":
            ir = value
        else:
            ir = 0.05
        plt.figure(figsize=(20, 15))
        building_types = all_npv_data[value]["building_usage"].unique()
        num_types = len(building_types)
        rows = (num_types + 2) // n_columns

        for i, building_type in enumerate(building_types, 1):
            plt.subplot(rows, n_columns, i)
            data = (
                all_npv_data[value][
                    all_npv_data[value]["building_usage"] == building_type
                ][f"savings_npv_25years_ir_{ir}"]
                / 1000
            )  # Convert to k€
            scatter = sns.histplot(data, kde=True)
            scatter.set_title(
                f"Savings Distribution - {building_type}\n{analysis_type}: {value}",
                fontsize=14,
            )
            scatter.set_xlabel("NPV Savings (k€)", fontsize=12)
            scatter.set_ylabel("Frequency", fontsize=12)
            scatter.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(
            f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_savings_distribution_{value}.png"
        )
        plt.close()


def plot_savings_operator_comparison(
    avg_savings_data, npv_operator, all_npv_data, values, analysis_type, simulation
):

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    analysis_type_title = analysis_type.replace("_", " ").title()
    # Plot average customer savings on primary axis (left) - convert to k€
    ax1.set_xlabel(f"{analysis_type_title}", fontsize=12)
    ax1.set_ylabel("Average Customer Savings (k€)", color="tab:blue", fontsize=12)

    # Plot each building type's savings (converting to k€)
    colors = sns.color_palette("colorblind", n_colors=len(avg_savings_data.columns))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

    for building_type, color, marker in zip(avg_savings_data.columns, colors, markers):
        ax1.plot(
            values,
            avg_savings_data[building_type] / 1000,  # Convert to k€
            marker=marker,
            markersize=8,
            label=building_type,
            color=color,
            linestyle="-",
            linewidth=2,
            markerfacecolor=color,
            markeredgecolor="black",
        )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Create secondary axis (right) for DH operator NPV - convert to M€
    ax2 = ax1.twinx()
    ax2.set_ylabel("DH Operator NPV (M€)", color="tab:red", fontsize=12)

    # Convert operator NPV to M€
    npv_operator_millions = np.array(npv_operator) / 1000000

    ax2.plot(values, npv_operator_millions, "r-", linewidth=3, label="DH Operator NPV")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Add title
    plt.title(
        f"DH Operator NPV vs Building Type Savings\nSensitivity to {analysis_type_title}",
        fontsize=14,
    )

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        fontsize=10,
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_operator_vs_building_savings.png",
        bbox_inches="tight",
    )

    plt.close()


def nfa_savings_operator_comparison(
    avg_savings_data_nfa,
    npv_operator,
    all_npv_data,
    values,
    analysis_type,
    simulation,
):

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    analysis_type_title = analysis_type.replace("_", " ").title()
    # Plot average customer savings on primary axis (left) - convert to k€
    ax1.set_xlabel(f"{analysis_type_title}", fontsize=12)
    ax1.set_ylabel("Average Customer Savings (€/m2NFA)", color="tab:blue", fontsize=12)

    # Plot each building type's savings (converting to k€)
    colors = sns.color_palette("colorblind", n_colors=len(avg_savings_data_nfa.columns))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

    for building_type, color, marker in zip(
        avg_savings_data_nfa.columns, colors, markers
    ):
        ax1.plot(
            values,
            avg_savings_data_nfa[building_type],  # Convert to k€
            marker=marker,
            markersize=8,
            label=building_type,
            color=color,
            linestyle="-",
            linewidth=2,
            markerfacecolor=color,
            markeredgecolor="black",
        )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Create secondary axis (right) for DH operator NPV - convert to M€
    ax2 = ax1.twinx()
    ax2.set_ylabel("DH Operator NPV (M€)", color="tab:red", fontsize=12)

    # Convert operator NPV to M€
    npv_operator_millions = np.array(npv_operator) / 1000000

    ax2.plot(values, npv_operator_millions, "r-", linewidth=3, label="DH Operator NPV")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Add title
    plt.title(
        f"DH Operator NPV vs Building Type Savings\nSensitivity to {analysis_type_title}",
        fontsize=14,
    )

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        fontsize=10,
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        f"sensitivity_analysis/{simulation}/{analysis_type}/plots/{analysis_type}_operator_vs_building_savings_nfa.png",
        bbox_inches="tight",
    )

    plt.close()
