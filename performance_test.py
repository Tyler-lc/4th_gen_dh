import pandas as pd
import numpy as np
import timeit

# Create a DataFrame with a single row for testing
data = {
    "n_floors": [3],
    "component": ["Roof"]
}
df = pd.DataFrame(data)

# Define the functions for different methods
def direct_lookup_iloc():
    return df.iloc[0]["n_floors"]

def direct_lookup_loc():
    return df.loc[0, "n_floors"]

def boolean_indexing():
    n_floors = df.loc[0, "n_floors"]
    return df[df["n_floors"] == n_floors]

def query_method():
    n_floors = df.loc[0, "n_floors"]
    return df.query('n_floors == @n_floors')

def eval_method():
    n_floors = df.loc[0, "n_floors"]
    return df[df.eval('n_floors == @n_floors')]

# Number of times to execute each method
number_of_executions = 10000

# Measure the performance of each method
time_direct_lookup_iloc = timeit.timeit(direct_lookup_iloc, number=number_of_executions)
time_direct_lookup_loc = timeit.timeit(direct_lookup_loc, number=number_of_executions)
time_boolean_indexing = timeit.timeit(boolean_indexing, number=number_of_executions)
time_query_method = timeit.timeit(query_method, number=number_of_executions)
time_eval_method = timeit.timeit(eval_method, number=number_of_executions)

print(f"Direct Lookup with iloc: {time_direct_lookup_iloc:.6f} seconds")
print(f"Direct Lookup with loc: {time_direct_lookup_loc:.6f} seconds")
print(f"Boolean Indexing: {time_boolean_indexing:.6f} seconds")
print(f"Query Method: {time_query_method:.6f} seconds")
print(f"Eval Method: {time_eval_method:.6f} seconds")
