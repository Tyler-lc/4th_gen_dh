import pandas as pd
import numpy as np
import time

# Sample data
occupancy_df = pd.DataFrame(
    {"occupancy": np.random.randint(0, 2, size=8760)},
    index=pd.date_range("2021-01-01", periods=8760, freq="H"),
)

# Add a column for the date
occupancy_df["date"] = occupancy_df.index.date

# Method 1: Using groupby.apply
start_time = time.time()
occupied_hours_by_day_apply = (
    occupancy_df[occupancy_df["occupancy"] == 1]
    .groupby("date")
    .apply(lambda x: x.index.hour.tolist())
)
method_1_time = time.time() - start_time

print(f"Method 1 (groupby.apply) execution time: {method_1_time:.4f} seconds")
print(occupied_hours_by_day_apply.head())


# Sample data (reusing the same occupancy_df)

# Add a column for the hour
occupied_hours_df = occupancy_df[occupancy_df["occupancy"] == 1].copy()
occupied_hours_df["hour"] = occupied_hours_df.index.hour

# Method 2: Using pivot_table and melt
start_time = time.time()
pivoted = occupied_hours_df.pivot_table(
    index="date", columns="hour", values="occupancy", aggfunc="size", fill_value=0
)
melted = pivoted.melt(ignore_index=False).reset_index()
occupied_hours_by_day_pivot = (
    melted[melted["value"] > 0].drop(columns="value").reset_index(drop=True)
)
occupied_hours_by_day_pivot = occupied_hours_by_day_pivot.groupby("date")["hour"].apply(
    list
)
method_2_time = time.time() - start_time

print(f"Method 2 (pivot_table and melt) execution time: {method_2_time:.4f} seconds")
print(occupied_hours_by_day_pivot.head())
