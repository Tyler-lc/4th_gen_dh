import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point, LineString
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np
import haversine as hs
from shapely.ops import transform
import pyproj
from tqdm import tqdm


def load_street_layout(file_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(file_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 25832:
        gdf = gdf.to_crs(epsg=25832)
    return gdf


def buildings_to_centroids(
    gdf: gpd.GeoDataFrame, crs_origin: str, crs_target: str
) -> gpd.GeoDataFrame:
    gdf_crs = gdf.crs
    gdf["centroid"] = gdf.geometry.centroid
    source_crs = pyproj.CRS(crs_origin)
    target_crs = pyproj.CRS(crs_target)
    project = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    ).transform
    gdf["centroid"] = gdf["centroid"].apply(lambda x: transform(project, x))

    # Convert the GeoDataFrame into the required format
    building_data = []
    for idx, row in gdf.iterrows():
        building_data.append(
            {
                "id": row["osm_id"],
                "coords": [row.centroid.y, row.centroid.x],
                "cap": row["capacity"],
            }
        )

    return building_data


def buildings_capacity(
    gdf: gpd.GeoDataFrame, column_name: str, rel_path: bool = True
) -> gpd.GeoDataFrame:
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        space_heating_path = row[column_name]
        if rel_path:
            space_heating_path = f"../{space_heating_path}"

        space_heating = pd.read_csv(space_heating_path, index_col=0)
        capacity = space_heating.max().values[0]
        gdf.loc[idx, "capacity"] = capacity
    return gdf


def load_supply_point(file_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(file_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 25832:
        gdf = gdf.to_crs(epsg=25832)

    # Ensure we have only one point
    if len(gdf) != 1:
        raise ValueError("The supply point file should contain exactly one point")

    return gdf


def create_street_network(street_layout):
    G = nx.Graph()
    for _, row in street_layout.iterrows():
        line = row.geometry
        start, end = line.coords[0], line.coords[-1]
        G.add_edge(start, end, length=line.length, geometry=line)
    return G


def nearest_point_on_street(G, point):
    nodes = np.array(G.nodes())
    tree = cKDTree(nodes)
    _, nearest_idx = tree.query(point.coords[0])
    return tuple(nodes[nearest_idx])


def add_buildings_to_network(G, building_data):
    for idx, building in building_data.iterrows():
        nearest_point = nearest_point_on_street(G, building.geometry.centroid)
        G.add_node(
            f"building_{idx}",
            pos=building.geometry.centroid.coords[0],
            demand=building["yearly_space_heating"],
        )  # Use the correct column name
        G.add_edge(
            f"building_{idx}",
            nearest_point,
            length=building.geometry.centroid.distance(Point(nearest_point)),
        )
    return G


def add_supply_point_to_network(G, supply_point):
    supply_geom = supply_point.geometry.iloc[0]
    nearest_point = nearest_point_on_street(G, supply_geom)
    G.add_node("supply", pos=supply_geom.coords[0], supply=True)
    G.add_edge(
        "supply", nearest_point, length=supply_geom.distance(Point(nearest_point))
    )
    return G


def calculate_mst(G):
    return nx.minimum_spanning_tree(G, weight="length")


def optimize_pipe_diameters(mst: nx.Graph, building_data: gpd.GeoDataFrame) -> nx.Graph:
    # Optimize pipe diameters based on demand and flow
    # This is a placeholder - you'll need to implement the actual optimization logic
    return mst


import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point


def visualize_grid(
    grid: nx.Graph,
    building_data: gpd.GeoDataFrame,
    supply_point: gpd.GeoDataFrame,
    street_layout: gpd.GeoDataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot street layout
    street_layout.plot(ax=ax, color="gray", linewidth=0.5)

    # Plot buildings
    building_data.plot(ax=ax, color="blue", alpha=0.5)

    # Plot grid
    pos = {
        node: (data["x"], data["y"])
        for node, data in grid.nodes(data=True)
        if "x" in data and "y" in data
    }
    nx.draw_networkx_edges(grid, pos, ax=ax, edge_color="orange", width=1)
    nx.draw_networkx_nodes(grid, pos, ax=ax, node_size=10, node_color="green")

    # Plot supply point
    supply_geom = supply_point.geometry.iloc[0]
    ax.scatter(supply_geom.x, supply_geom.y, color="red", s=100, zorder=5)

    # Set plot limits to focus on the area of interest
    x_coords = [data["x"] for _, data in grid.nodes(data=True) if "x" in data]
    y_coords = [data["y"] for _, data in grid.nodes(data=True) if "y" in data]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add a small buffer around the area of interest
    buffer = 0.1  # 10% buffer
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - buffer * x_range, x_max + buffer * x_range)
    ax.set_ylim(y_min - buffer * y_range, y_max + buffer * y_range)

    plt.title("District Heating Grid Layout")
    plt.axis("on")  # Turn axis on to see the scale
    plt.tight_layout()
    plt.show()
