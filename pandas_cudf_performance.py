import cudf
import pandas as pd
import numpy as np
import time

# Create a pandas DataFrame as an example
dates = pd.date_range(start='2021-01-01', periods=8760, freq='H')
occupancy = np.random.randint(0, 2, size=(8760,))
occupancy_df = pd.DataFrame({'occupancy': occupancy}, index=dates)

# Convert pandas DataFrame to cuDF DataFrame
cudf_df = cudf.DataFrame.from_pandas(occupancy_df)

# Performance testing with pandas
start_time = time.time()
daily_awake_hours_pd = occupancy_df['occupancy'].resample('D').sum()
print(f"Pandas execution time: {time.time() - start_time} seconds")

# Performance testing with cuDF
start_time = time.time()
daily_awake_hours_cudf = cudf_df['occupancy'].resample('D').sum()
print(f"cuDF execution time: {time.time() - start_time} seconds")
