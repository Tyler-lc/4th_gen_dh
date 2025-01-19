import pandas as pd
import numpy as np
import geopandas as gpd
import os
import sys
from building_analysis.building_generator import (
    iterator_generate_buildings,
    people_in_building,
    assign_people_id,
)
from utils.building_utilities import process_data

# as there is some randomness built-in to the generation of the buildings, we set the seed here for consistency
np.random.seed(42)

# first we need to process the QGIS data
# we need to set up the path to the QGIS data, the age distribution of the buildings, and the ceiling heights distribution
# as well as a list of the acronyms used to identify the residential buildings
path_qgis_data = "building_analysis/building_generator_data/frankfurt_v3.parquet"
age_distr_path = "building_analysis/building_generator_data/buildings_age.csv"
ceilings_heights_path = "building_analysis/building_generator_data/ceiling_heights.csv"
age_distr_df = pd.read_csv(age_distr_path)
ceiling_heights_df = pd.read_csv(ceilings_heights_path)
res_types = ["mfh", "th", "ab", "sfh"]
file_type = "parquet"

# now we process the data to get a useful input to be used in the building_generator function
geometric_data = process_data(
    path_qgis_data,
    file_type,
    age_distr_df,
    ceiling_heights_df,
    res_types,
)

# finally we use the iterator_generate_buildings function to generate the buildings with all the information
# needed to create the buildingstock. This gdf will be used in the Building class to calculate the useful energy demand
# First we need to set the path where the u-values for all the archetypes are stored. This should be a csv
# then we use the iterator_generate_buildings function to generate the buildings.
u_values_path = "building_analysis/building_generator_data/archetype_u_values.csv"
buildingstock = iterator_generate_buildings(
    geometric_data, u_values_path, convert_wkb=True, randomization_factor=0.01
)

# once we have generated the buildingstock we can add people to the building, since this requires us to know the total
# GFA of the area we analyze. So we need to first generate  the buildingstock. Now we can append information about people
# We need to first calculate the number of people per Gross Floor Area. We assume there are 9500 people in the area
buildingstock["n_people"] = people_in_building(buildingstock, res_types, 9500)
buildingstock["people_id"] = assign_people_id(buildingstock, res_types)

# making a dataframe is actually for data exploration with datawrangler in VS code. It is actually not necessary
df_buildingstock = pd.DataFrame(buildingstock)


# out of convention we put the geometry column at the end. Right now it is not. So we want to move it to the end
# Get the list of columns
columns = list(buildingstock.columns)

# Remove 'geometry' from the list and append it to the end
columns.remove("geometry")
columns.append("geometry")

# Reorder the DataFrame
buildingstock = buildingstock[columns]

# Verify the new order


# now we can save the buildingstock data to a file. We will save it as a parquet file
buildingstock.to_parquet("building_analysis/buildingstock/buildingstock.parquet")

# now we can open the file to check that everything is fine
buildingstock_from_parquet = gpd.read_parquet(
    "building_analysis/buildingstock/buildingstock.parquet"
)
are_equal = buildingstock_from_parquet.equals(buildingstock)
print(f" is the crs the same? {buildingstock.crs == buildingstock_from_parquet.crs}")
print(
    f"do we have the same columns? {buildingstock.columns == buildingstock_from_parquet.columns}"
)
print(
    f"do we have the same datatypes? {buildingstock.dtypes == buildingstock_from_parquet.dtypes}"
)
print(
    f"are the two dataframes the same? {buildingstock.equals(buildingstock_from_parquet)}"
)
print(
    f"do the two dataframes have the same shape? {buildingstock_from_parquet.shape == buildingstock.shape}"
)


# TODO we could actually turn this whole section into a utility function that checks the dataframe "integrity"

different_cols = []
if (buildingstock.equals(buildingstock_from_parquet)) == False:
    print(
        "the two dataframes seem to not be exactly the same. Checking for differences column by column"
    )
    for col in buildingstock.columns:
        if buildingstock[col].equals(buildingstock_from_parquet[col]) == False:
            print(f"column {col} is NOOOT the same")
            different_cols.append(col)
            # print(f"buildingstock: {buildingstock[col].unique()}")
            # print(f"buildingstock_from_parquet: {buildingstock_from_parquet[col].unique()}")
        else:
            print(f"column {col} are the same")

print(f"the columns that are different are: {different_cols}")
print("trying to get more information about the differences")

for col in different_cols:
    for idx, row in buildingstock.iterrows():
        if not np.array_equal(row[col], buildingstock_from_parquet.loc[idx, col]):
            print(f"row {idx} is different")
            print(f"buildingstock: {row[col]}")
            print(
                f"buildingstock_from_parquet: {buildingstock_from_parquet.loc[idx, col]}"
            )
            print("")
            break
    print(f"all data in {col} are the same")


# we would also like to check the data "integrity" now. Meaning we want to check whether there are 0s
# or nan in the data.

# we can check for nan values in the data

nan_values = buildingstock.isna().sum()
print(nan_values)
