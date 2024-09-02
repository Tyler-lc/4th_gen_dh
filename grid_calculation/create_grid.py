import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import haversine as hs
import numpy as np
import pandas as pd
import osmnx as ox
import momepy
from shapely.geometry import Point, LineString

# Load the street data
street_layout = gpd.read_file("streets_only_clean_length_25832.gpkg")
street_layout.crs = "EPSG:25832"
street_layout.index = street_layout["osm_id"]

high_cf = '["highway"~"trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential|living_street|service|pedestrian|unclassified|track|road|path"]'
medium_cf = '["highway"~"trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|residential"]'
medium_low_cf = '["highway"~"trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|living_street|service"]'
low_cf = '["highway"~"primary|primary_link|secondary|secondary_link"]'
G = momepy.gdf_to_nx(
    street_layout, approach="primal", length="length", multigraph=True, directed=True
)
G = nx.MultiDiGraph(G, bidirectional=True, custom_filter=high_cf)

G_simplified = ox.simplify_graph(G)


G_df = momepy.nx_to_gdf(G_simplified)

G.remove_nodes_from(nx.selfloop_edges(G))

# Simplify the graph using momepy


G_simplified.remove_nodes_from(nx.selfloop_edges(G_simplified))

# Remove isolated nodes
isolated_nodes = list(nx.isolates(G_simplified))
# G_simplified.remove_nodes_from(isolated_nodes)


# Merge close nodes
def merge_close_nodes(G, threshold=10):
    pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
    nodes_to_merge = []
    for u, v in G.edges():
        if np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) < threshold:
            nodes_to_merge.append((u, v))
    for u, v in nodes_to_merge:
        if u in G and v in G:  # Check again before contracting
            G = nx.contracted_nodes(G, u, v, self_loops=True)
    return G


# G_simplified = merge_close_nodes(G_simplified, threshold=1)
G_simplified.remove_nodes_from(nx.selfloop_edges(G_simplified))


# Print node attributes before
print("Node attributes before:")
for node in list(G_simplified.nodes)[:5]:  # Print first 5 nodes for brevity
    print(node, G_simplified.nodes[node])

# Assign "osmid" attribute to each node
for node in G_simplified.nodes:
    if node in G.nodes:
        G_simplified.nodes[node]["osmid"] = G.nodes[node].get("osmid", node)

# Print node attributes after
print("\nNode attributes after:")
for node in list(G_simplified.nodes)[:5]:  # Print first 5 nodes for brevity
    print(node, G_simplified.nodes[node])

# Remove longer edges between points if multiple exist
nodes, edges = ox.graph_to_gdfs(G_simplified)
G_df = momepy.nx_to_gdf(G_simplified)

edges_to_drop = []
edges = edges.reset_index(level=[0, 1, 2])
edges["id"] = edges["u"].astype(str) + "-" + edges["v"].astype(str)

for i in edges["id"].unique():
    double_edges = edges[edges["id"] == i]
    if len(double_edges) > 1:
        mx_ind = double_edges["length"].idxmin()
        mx = double_edges.drop(mx_ind)
        edges_to_drop.append(mx)

try:
    edges_to_drop = pd.concat(edges_to_drop)
    for i in zip(edges_to_drop["u"], edges_to_drop["v"], edges_to_drop["key"]):
        G_simplified.remove_edge(u=i[0], v=i[1], key=i[2])
except:
    pass

# Remove short edges
min_length = 2  # Set a minimum length threshold
edges_to_remove = [
    (u, v) for u, v, d in G_simplified.edges(data=True) if d["length"] < min_length
]
G_simplified.remove_edges_from(edges_to_remove)

# Explicitly remove self-loops again
G_simplified.remove_edges_from(nx.selfloop_edges(G_simplified))

# Plot the original and simplified graphs
fig, axes = plt.subplots(1, 2, figsize=(20, 7))  # Increase figure size

# Plot original graph
ax = axes[0]
pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
nx.draw(
    G,
    pos,
    ax=ax,
    node_size=5,
    node_color="blue",
    edge_color="gray",
    width=0.5,
    with_labels=False,
)
ax.set_title("Original Graph")

# Plot simplified graph
ax = axes[1]
pos_simplified = {n: [n[0], n[1]] for n in list(G_simplified.nodes)}
nx.draw(
    G_simplified,
    pos_simplified,
    ax=ax,
    node_size=5,
    node_color="blue",
    edge_color="gray",
    width=0.5,
    with_labels=False,
)
ax.set_title("Simplified Graph")

plt.show()

# Load the building data
buildingstock = gpd.read_parquet(
    "../building_analysis/results/renovated_whole_buildingstock/buildingstock_renovated_results.parquet"
)
building_polygons = buildingstock[["geometry"]]
building_polygons.crs = 25832
building_centroids = building_polygons.centroid

# Load the supply point
supply_point = gpd.read_file("supply_point.gpkg")
supply_point.crs = "EPSG:25832"
positions = {n: [n[0], n[1]] for n in list(G.nodes)}
# Plot the data
fig, axes = plt.subplots(
    1, 2, figsize=(30, 15)
)  # Create subplots for side-by-side plots

# Plot street layout with building centroids and supply point on the first subplot
ax = axes[0]
street_layout.plot(ax=ax, color="gray", linewidth=0.5, label="Streets")
supply_point.plot(ax=ax, color="red", markersize=100, label="Supply Point")
nx.draw(G, positions, ax=ax, node_size=1, node_color="blue")
ax.set_title("Street Layout with Building Centroids and Supply Point")
ax.legend(loc="upper left")

# Plot simplified graph on the second subplot
ax = axes[1]
street_layout.plot(ax=ax, color="gray", linewidth=0.5, label="Streets")
supply_point.plot(ax=ax, color="red", markersize=100, label="Supply Point")
nx.draw(G_simplified, positions, ax=ax, node_size=1, node_color="red")
ax.set_title("Simplified Graph")
ax.legend()

plt.show()
