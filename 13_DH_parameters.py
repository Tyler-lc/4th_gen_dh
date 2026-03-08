import pandas as pd
import numpy as np

from config import GRID_CALCULATION_DIR, grid_results_parquet

scenarios = ["unrenovated", "renovated", "booster"]


unrenovated_result_df = pd.read_parquet(grid_results_parquet("unrenovated"))
renovated_result_df = pd.read_parquet(grid_results_parquet("renovated"))
booster_result_df = pd.read_parquet(grid_results_parquet("booster"))

unrenovated_cost = unrenovated_result_df["cost_total"].sum() / 1000000  # M€
unrenovated_length = unrenovated_result_df["Length"].sum() / 1000  # km
unrenovated_losses = unrenovated_result_df["Losses [W]"].sum() * 8760 / 1000000  # MWh

renovated_cost = renovated_result_df["cost_total"].sum() / 1000000  # M€
renovated_length = renovated_result_df["Length"].sum() / 1000  # km
renovated_losses = renovated_result_df["Losses [W]"].sum() * 8760 / 1000000  # MWh

booster_cost = booster_result_df["cost_total"].sum() / 1000000  # M€
booster_length = booster_result_df["Length"].sum() / 1000  # km
booster_losses = booster_result_df["Losses [W]"].sum() * 8760 / 1000000  # MWh

export_df = pd.DataFrame(
    index=scenarios, columns=["Cost [M€]", "Length [km]", "Losses [MWh]"]
)
export_df.loc["unrenovated"] = [
    unrenovated_cost,
    unrenovated_length,
    unrenovated_losses,
]
export_df.loc["renovated"] = [renovated_cost, renovated_length, renovated_losses]
export_df.loc["booster"] = [booster_cost, booster_length, booster_losses]


export_df.to_csv(GRID_CALCULATION_DIR / "dh_parameters.csv")
