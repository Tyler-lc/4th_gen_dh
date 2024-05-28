import pandas as pd
import numpy as np
import json
import timeit


class Building:
    def __init__(self, components, outside_temperature):
        self.components = components
        self.outside_temperature = outside_temperature

        # Example data setup
        self.opaque_surfaces = pd.DataFrame(
            {
                "surface_name": ["roof", "wall", "door"],
                "total_surface": [
                    components["roof_area"],
                    components["walls_area"],
                    components["door_area"],
                ],
                "u_value": [
                    components["roof_u_value"],
                    components["walls_u_value"],
                    components["door_u_value"],
                ],
            }
        )

    def is_summer(self, index):
        return index.month.isin([6, 7, 8])

    def compute_losses_original(self, surfaces_df, outside_temp, inside_temp):
        losses_df = pd.DataFrame(index=self.outside_temperature.index)
        for _, row in surfaces_df.iterrows():
            u_value = row["u_value"]
            area = row["total_surface"]
            surface_name = row["surface_name"]
            losses = u_value * area * (inside_temp - outside_temp) / 1000  # kWh
            losses[losses < 0] = 0
            losses[self.is_summer(losses.index)] = 0
            losses_df[surface_name] = losses
        return losses_df

    def compute_losses_first_vectorized(self, surfaces_df, outside_temp, inside_temp):
        u_values = surfaces_df["u_value"].values
        areas = surfaces_df["total_surface"].values
        surface_names = surfaces_df["surface_name"].values
        losses_df = pd.DataFrame(index=self.outside_temperature.index)
        for i, surface_name in enumerate(surface_names):
            losses = u_values[i] * areas[i] * (inside_temp - outside_temp) / 1000  # kWh
            losses[losses < 0] = 0
            losses[self.is_summer(losses.index)] = 0
            losses_df[surface_name] = losses
        return losses_df

    def compute_losses_fully_vectorized(self, surfaces_df, outside_temp, inside_temp):
        u_values = surfaces_df["u_value"].values
        areas = surfaces_df["total_surface"].values
        surface_names = surfaces_df["surface_name"].values
        u_matrix = np.repeat(u_values[:, np.newaxis], len(outside_temp), axis=1)
        area_matrix = np.repeat(areas[:, np.newaxis], len(outside_temp), axis=1)
        inside_temp_matrix = np.tile(inside_temp.values, (len(u_values), 1))
        outside_temp_matrix = np.tile(outside_temp.values, (len(u_values), 1))
        losses_matrix = (
            u_matrix * area_matrix * (inside_temp_matrix - outside_temp_matrix) / 1000
        ).T
        losses_df = pd.DataFrame(
            losses_matrix, index=outside_temp.index, columns=surface_names
        )
        losses_df[losses_df < 0] = 0
        losses_df.loc[self.is_summer(losses_df.index)] = 0
        return losses_df


# Test setup
def create_test_building():
    components = {
        "roof_area": 150,
        "walls_area": 300,
        "door_area": 10,
        "roof_u_value": 0.2,
        "walls_u_value": 0.3,
        "door_u_value": 0.4,
    }
    outside_temp = pd.Series(
        np.random.normal(5, 10, 8760),
        index=pd.date_range("2023-01-01", periods=8760, freq="H"),
    )
    inside_temp = pd.Series(
        np.full(8760, 20), index=pd.date_range("2023-01-01", periods=8760, freq="H")
    )
    building = Building(components, outside_temp)
    return building, outside_temp, inside_temp


# Performance test functions
def test_original_method(building, outside_temp, inside_temp):
    return building.compute_losses_original(
        building.opaque_surfaces, outside_temp, inside_temp
    )


def test_first_vectorized_method(building, outside_temp, inside_temp):
    return building.compute_losses_first_vectorized(
        building.opaque_surfaces, outside_temp, inside_temp
    )


def test_fully_vectorized_method(building, outside_temp, inside_temp):
    return building.compute_losses_fully_vectorized(
        building.opaque_surfaces, outside_temp, inside_temp
    )


# Running performance tests
if __name__ == "__main__":
    building, outside_temp, inside_temp = create_test_building()

    time_original = timeit.timeit(
        lambda: test_original_method(building, outside_temp, inside_temp), number=1000
    )
    time_first_vectorized = timeit.timeit(
        lambda: test_first_vectorized_method(building, outside_temp, inside_temp),
        number=1000,
    )
    time_fully_vectorized = timeit.timeit(
        lambda: test_fully_vectorized_method(building, outside_temp, inside_temp),
        number=1000,
    )

    print(f"Original Method: {time_original:.6f} seconds")
    print(f"First Vectorized Method: {time_first_vectorized:.6f} seconds")
    print(f"Fully Vectorized Method: {time_fully_vectorized:.6f} seconds")
