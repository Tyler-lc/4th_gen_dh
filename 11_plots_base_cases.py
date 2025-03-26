import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import all the npv data from the csv files
unrenovated_savings = pd.read_csv(
    "plots/HighTemperature/data_exports_1_dhg_lifetime_50/npv_data.csv"
)
renovated_savings = pd.read_csv(
    "plots/LowTemperature/data_exports_1_dhg_lifetime_50/npv_data.csv"
)
booster_savings = pd.read_csv(
    "plots/booster/data_exports_1_dhg_lifetime_50/npv_data.csv"
)

# Add scenario column to each dataframe
unrenovated_savings["scenario"] = "High Temperature"
renovated_savings["scenario"] = "Low Temperature"
booster_savings["scenario"] = "Booster"

# Combine the dataframes
combined_data = pd.concat([unrenovated_savings, renovated_savings, booster_savings])


def plot_npv_savings(combined_data, title, scenario):
    # Create the plot
    plt.figure(figsize=(20, 15))  # Made figure a bit larger
    sns.boxplot(
        data=combined_data,
        x="building_usage",
        y="npv_per_nfa",  # or your savings column name
        hue="scenario",
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 8,
        },
        medianprops={"color": "red", "linewidth": 1},
        boxprops={"alpha": 0.5},
        width=0.8,  # Reduced from default 0.8 to make boxes slimmer
        dodge=1,  # Ensures boxes are properly spaced
        gap=1.8,
    )

    plt.xticks(rotation=45, fontsize=22)
    plt.yticks(fontsize=22)
    plt.title(title, fontsize=30)
    plt.xlabel("Building Type", fontsize=25)
    plt.ylabel("NPV per Net Floor Area (€/m²)", fontsize=25)
    plt.legend(fontsize=20)

    plt.tight_layout()
    plt.savefig(f"plots/comparison_all_scenarios_{scenario}.png")
    plt.close()


plot_npv_savings(
    combined_data,
    "NPV Savings Distribution by Building Type and Scenario",
    "all_scenarios",
)


#### now we want also to plot for the case in which the DH operator is not allowed to make a profit
# first we need to import the correct data:

# unrenovated_savings_no_profit = pd.read_csv(
#     "plots/HighTemperature/data_exports_0.9509/npv_data.csv"
# )
# renovated_savings_no_profit = pd.read_csv(
#     "plots/LowTemperature/data_exports_0.6515/npv_data.csv"
# )
# booster_savings_no_profit = pd.read_csv(
#     "plots/booster/data_exports_0.8894/npv_data.csv"
# )

# # Add scenario column to each dataframe
# unrenovated_savings_no_profit["scenario"] = "High Temperature"
# renovated_savings_no_profit["scenario"] = "Low Temperature"
# booster_savings_no_profit["scenario"] = "Booster"

# # and now combine the dataframes
# combined_data_no_profit = pd.concat(
#     [
#         unrenovated_savings_no_profit,
#         renovated_savings_no_profit,
#         booster_savings_no_profit,
#     ]
# )

# plot_npv_savings(
#     combined_data_no_profit,
#     "NPV Savings Distribution by Building Type and Scenario - No Profit Allowed",
#     "all_scenarios_no_profit",
# )

print("done")
