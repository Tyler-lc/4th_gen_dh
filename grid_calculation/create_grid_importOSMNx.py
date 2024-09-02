import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import haversine as hs
import numpy as np
import pandas as pd
import osmnx as ox
import momepy
import pyproj
from shapely.ops import transform
from shapely.geometry import Point, LineString, Polygon


from grid_utils import buildings_to_centroids

####### Define supply nodes
n_supply_list = [
    {"id": 2, "coords": [50.096, 8.593], "cap": 60},
    # {"id": 6, "coords": [38.75246, -9.23775], "cap": 35},
]

### define sinks in the area
buildingstock_path = "../building_analysis/results/renovated_whole_buildingstock/buildingstock_renovated_results.parquet"
buildingstock = gpd.read_parquet(buildingstock_path)
n_demand_list = buildings_to_centroids(
    buildingstock, crs_origin="EPSG:25832", crs_target="EPSG:4326"
)


### define resolution to download the road network
high_cf = '["highway"~"trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential|living_street|service|pedestrian|unclassified|track|road|path"]'
medium_cf = '["highway"~"trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential"]'
medium_low_cf = '["highway"~"trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|living_street|service"]'
low_cf = '["highway"~"primary|primary_link|secondary|secondary_link"]'


coords_list = [
    [470125, 5549653],
    [470645, 5549330],
    [471081, 5549201],
    [472504, 5549209],
    [472151, 5549837],
    [471028, 5549848],
    [470291, 5549815],
]
polygon = Polygon(coords_list).convex_hull

# Plot the original polygon
x, y = polygon.exterior.xy
plt.figure()
plt.plot(x, y)
plt.fill(x, y, alpha=0.5, fc="r", ec="black")
plt.title("Original Polygon in EPSG:25832")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Define the source and target CRS
source_crs = pyproj.CRS("EPSG:25832")
target_crs = pyproj.CRS("EPSG:4326")

# Define the transformation function
project = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform

# Reproject the polygon
polygon_wgs84 = transform(project, polygon)

# Plot the reprojected polygon
x, y = polygon_wgs84.exterior.xy
plt.figure()
plt.plot(x, y)
plt.fill(x, y, alpha=0.5, fc="r", ec="black")
plt.title("Reprojected Polygon in EPSG:4326")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Now use the reprojected polygon with OSMnx
road_nw = ox.graph_from_polygon(polygon_wgs84, simplify=False, custom_filter=high_cf)

road_simplified = ox.simplify_graph(road_nw)

# Plot the original road network
fig, ax = plt.subplots(figsize=(10, 10))
ox.plot_graph(
    road_nw,
    ax=ax,
    node_size=0,
    edge_color="blue",
    edge_linewidth=0.5,
    show=False,
    close=False,
)
plt.title("Original Road Network")
plt.show()

# Plot the simplified road network
fig, ax = plt.subplots(figsize=(10, 10))
ox.plot_graph(
    road_simplified,
    ax=ax,
    node_size=0,
    edge_color="red",
    edge_linewidth=0.5,
    show=False,
    close=False,
)
plt.title("Simplified Road Network")
plt.show()


######## Pass osmid to nodes
for node in road_simplified.nodes:
    road_simplified.nodes[node]["osmid"] = node


###########REMOVE LONGER EDGES BETWEEN POINTS IF MUTIPLE EXIST#######################
##AS ONLY ONE (u,v - v,u) EDGE BETWEEN TWO POINTS CAN BE CONSIDERED FOR OPTIMIZATION#

nodes, edges = ox.graph_to_gdfs(road_simplified)

network_edges_crs = edges.crs
network_nodes_crs = nodes.crs

edges_to_drop = []
edges = edges.reset_index(level=[0, 1, 2])
edges_double = pd.DataFrame(edges)
edges_double["id"] = edges_double["u"].astype(str) + "-" + edges_double["v"].astype(str)

for i in edges_double["id"].unique():
    double_edges = edges_double[edges_double["id"] == i]
    if len(double_edges) > 1:
        mx_ind = double_edges["length"].idxmin()
        mx = double_edges.drop(mx_ind)
        edges_to_drop.append(mx)
    else:
        None

try:
    edges_to_drop = pd.concat(edges_to_drop)
    for i in zip(edges_to_drop["u"], edges_to_drop["v"], edges_to_drop["key"]):
        road_nw.remove_edge(u=i[0], v=i[1], key=i[2])
except:
    None


#########################REMOVE LOOPS FOR ONE WAYS#######################
######HAPPENS IF THE TWO EDGES BETWEEN TWO POINTS DO NOT HAVE THE########
###########################SAME LENGTH###################################

nodes, edges = ox.graph_to_gdfs(road_nw)
edges = edges.reset_index(level=[0, 1, 2])

edges_one_way = pd.DataFrame(edges[edges["oneway"] == True])
edges_one_way["id"] = list(zip(edges_one_way["u"], edges_one_way["v"]))

edges_to_drop = []


for i in edges_one_way["id"]:
    edges_u_v = edges_one_way[
        (edges_one_way["u"] == i[0]) & (edges_one_way["v"] == i[1])
    ]
    edges_v_u = edges_one_way[
        (edges_one_way["u"] == i[1]) & (edges_one_way["v"] == i[0])
    ]
    edges_all = pd.concat([edges_u_v, edges_v_u])
    if len(edges_all) > 1:
        mx_ind = edges_all["length"].idxmin()
        mx = edges_all.drop(mx_ind)
        edges_to_drop.append(mx)
    else:
        None

try:
    edges_to_drop = pd.concat(edges_to_drop).drop("id", axis=1)
    edges_to_drop = edges_to_drop[~edges_to_drop.index.duplicated(keep="first")]
    edges_to_drop = edges_to_drop.drop_duplicates(subset=["length"], keep="last")
    for i in zip(edges_to_drop["u"], edges_to_drop["v"], edges_to_drop["key"]):
        road_nw.remove_edge(u=i[0], v=i[1], key=i[2])
except:
    None

nx.set_edge_attributes(road_simplified, "street", "surface_type")
nx.set_edge_attributes(road_simplified, 0, "restriction")
nx.set_edge_attributes(road_simplified, 0, "surface_pipe")
nx.set_edge_attributes(road_simplified, 0, "existing_grid_element")
nx.set_edge_attributes(road_simplified, 0, "inner_diameter_existing_grid_element")
nx.set_edge_attributes(road_simplified, 0, "costs_existing_grid_element")

################################################################################
######CONNECT SOURCES AND SINKS TO OSM GRAPH####################################
# Connect supply nodes to the road network
for supply in n_supply_list:
    supply_id = supply["id"]
    supply_coords = supply["coords"]
    nearest_node = ox.distance.nearest_nodes(
        road_simplified, supply_coords[1], supply_coords[0]
    )

    # Add the supply node to the graph
    road_simplified.add_node(
        supply_id, y=supply_coords[0], x=supply_coords[1], osmid=supply_id
    )

    # Add an edge from the supply node to the nearest node on the road network
    road_simplified.add_edge(
        supply_id,
        nearest_node,
        length=hs.haversine(
            (supply_coords[0], supply_coords[1]),
            (
                road_simplified.nodes[nearest_node]["y"],
                road_simplified.nodes[nearest_node]["x"],
            ),
        )
        * 1000,
        surface_type="street",
        restriction=0,
        surface_pipe=0,
        existing_grid_element=0,
        inner_diameter_existing_grid_element=0,
        costs_existing_grid_element=0,
    )


# Connect demand nodes to the road network
for demand in n_demand_list:
    demand_id = demand["id"]
    demand_coords = demand["coords"]
    nearest_node = ox.distance.nearest_nodes(
        road_simplified, demand_coords[1], demand_coords[0]
    )

    # Add the demand node to the graph
    road_simplified.add_node(
        demand_id, y=demand_coords[0], x=demand_coords[1], osmid=demand_id
    )

    # Add an edge from the demand node to the nearest node on the road network
    road_simplified.add_edge(
        demand_id,
        nearest_node,
        length=hs.haversine(
            (demand_coords[0], demand_coords[1]),
            (
                road_simplified.nodes[nearest_node]["y"],
                road_simplified.nodes[nearest_node]["x"],
            ),
        )
        * 1000,
        surface_type="street",
        restriction=0,
        surface_pipe=0,
        existing_grid_element=0,
        inner_diameter_existing_grid_element=0,
        costs_existing_grid_element=0,
    )


road_simplified_undirected = ox.convert.to_undirected(road_simplified)
# road_simplified_undirected = ox.project_graph(road_simplified_undirected)

# Plot the final road network
fig, ax = plt.subplots(figsize=(10, 10))
ox.plot_graph(
    road_simplified_undirected,
    ax=ax,
    node_size=0,
    edge_color="green",
    edge_linewidth=0.5,
    show=False,
    close=False,
)
plt.title("Final Road Network")

# Plot supply nodes
for supply in n_supply_list:
    plt.plot(
        supply["coords"][1],
        supply["coords"][0],
        "bo",
        markersize=10,
        label="Supply Node" if supply == n_supply_list[0] else "",
    )

# Plot demand nodes
for demand in n_demand_list:
    plt.plot(
        demand["coords"][1],
        demand["coords"][0],
        "ro",
        markersize=1,
        label="Demand Node" if demand == n_demand_list[0] else "",
    )

# Add legend
plt.legend()
plt.show()
