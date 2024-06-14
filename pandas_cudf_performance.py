# import cudf
import pandas as pd
import numpy as np
import time

# this whole thing was a bit of a waste of time. There is not a lot of improvement
# when running the code with GPU multi processing. On 876000 rows, regular CPU Pandas
# is still twice as fast as GPU cuDF.

# Create a larger pandas DataFrame as an example
dates = pd.date_range(start="2021-01-01", periods=876000, freq="H")
occupancy = np.random.randint(0, 2, size=(876000,))
occupancy_df = pd.DataFrame({"occupancy": occupancy}, index=dates)

# Convert pandas DataFrame to cuDF DataFrame
# cudf_df = cudf.DataFrame.from_pandas(occupancy_df)


# Function to measure execution time
def measure_time(df, func):
    start_time = time.time()
    result = func(df)
    end_time = time.time()
    return end_time - start_time


# Define the resampling function
def resample_sum(df):
    return df["occupancy"].resample("D").sum()


# Warm-up operations
# _ = cudf_df.head()
# _ = resample_sum(cudf_df)

# Measure pandas execution time
pandas_times = [measure_time(occupancy_df, resample_sum) for _ in range(10)]
print(f"Pandas execution times: {pandas_times}")
print(f"Pandas average execution time: {np.mean(pandas_times)} seconds")

# Measure cuDF execution time
# cudf_times = [measure_time(cudf_df, resample_sum) for _ in range(10)]
# print(f"cuDF execution times: {cudf_times}")
# print(f"cuDF average execution time: {np.mean(cudf_times)} seconds")
