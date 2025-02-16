import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import all the npv data from the csv files
unrenovated_savings = pd.read_csv("plots/HighTemperature/data_exports_1/npv_data.csv")
renovated_savings = pd.read_csv("plots/LowTemperature/data_exports_1/npv_data.csv")
booster_savings = pd.read_csv("plots/booster/data_exports_1/npv_data.csv")

# Add scenario column to each dataframe
unrenovated_savings["scenario"] = "High Temperature"
renovated_savings["scenario"] = "Low Temperature"
booster_savings["scenario"] = "Booster"

# Combine the dataframes
combined_data = pd.concat([unrenovated_savings, renovated_savings, booster_savings])

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
    width=0.7,  # Reduced from default 0.8 to make boxes slimmer
    dodge=1.2,  # Ensures boxes are properly spaced
    gap=1.8,
)

plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.title("NPV Savings Distribution by Building Type and Scenario", fontsize=30)
plt.xlabel("Building Type", fontsize=25)
plt.ylabel("NPV per Net Floor Area (€/m²)", fontsize=25)
plt.legend(fontsize=18)

plt.tight_layout()
plt.savefig("plots/comparison_all_scenarios.png")
plt.close()
