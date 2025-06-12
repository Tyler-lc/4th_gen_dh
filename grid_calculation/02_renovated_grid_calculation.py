import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import haversine as hs
import numpy as np
import pandas as pd
import osmnx as ox
import pyproj
from shapely.ops import transform
from shapely.geometry import Point, LineString, Polygon
import folium
from folium.plugins import MarkerCluster


from grid_utils import buildings_to_centroids, buildings_capacity

####### Define supply nodes
n_supply_list = [
    {"id": 2, "coords": [50.096, 8.593], "cap": 100000},
    # {"id": 6, "coords": [38.75246, -9.23775], "cap": 35},
]

### define sinks in the area
buildingstock_path = "../building_analysis/results/renovated_whole_buildingstock/buildingstock_results_renovated.parquet"
buildingstock = gpd.read_parquet(buildingstock_path)

### replace the path backslash with forward slash
buildingstock["space_heating_path"] = buildingstock["space_heating_path"].str.replace(
    "\\", "/"
)
buildingstock["dhw_energy_path"] = buildingstock["dhw_energy_path"].str.replace(
    "\\", "/"
)

# filter out buildings with NFA < 30
buildingstock = buildingstock[buildingstock["NFA"] >= 30]


print("calculating space heating max \n")
buildingstock["space_heating_cap"] = buildings_capacity(
    buildingstock, "space_heating_path"
)

print("calculating dhw max \n")
buildingstock["dhw_cap"] = buildings_capacity(buildingstock, "dhw_energy_path")

print("calculating total capacity \n")
buildingstock["capacity"] = (
    buildingstock["space_heating_cap"] + buildingstock["dhw_cap"]
)

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


###############################################################################
########################### OPTIMIZE NETWORK ##################################
###############################################################################
water_den = 1000  # kg/m3
factor_street_terrain = 0.10
factor_street_overland = 0.4
heat_capacity = 4.18  # kJ/kg/K
flow_temp = 50  # C
return_temp = 25  # C
surface_losses_json = [
    {"dn": 0.02, "overland_losses": 0.115994719393908},
    {"dn": 0.025, "overland_losses": 0.138092834981244},
    {"dn": 0.032, "overland_losses": 0.15109757219986},
    {"dn": 0.04, "overland_losses": 0.171799705290563},
    {"dn": 0.05, "overland_losses": 0.193944276611768},
    {"dn": 0.065, "overland_losses": 0.219829984514374},
    {"dn": 0.08, "overland_losses": 0.231572190233268},
    {"dn": 0.1, "overland_losses": 0.241204678239951},
    {"dn": 0.125, "overland_losses": 0.280707496411117},
    {"dn": 0.15, "overland_losses": 0.320919871727017},
    {"dn": 0.2, "overland_losses": 0.338510752592325},
    {"dn": 0.25, "overland_losses": 0.326870584772369},
    {"dn": 0.3, "overland_losses": 0.376259860034531},
    {"dn": 0.35, "overland_losses": 0.359725182960969},
    {"dn": 0.4, "overland_losses": 0.372648018718974},
    {"dn": 0.45, "overland_losses": 0.427474040273953},
    {"dn": 0.5, "overland_losses": 0.359725658523504},
    {"dn": 0.6, "overland_losses": 0.420023799255459},
    {"dn": 0.7, "overland_losses": 0.478951907501331},
    {"dn": 0.8, "overland_losses": 0.540336445060049},
    {"dn": 0.9, "overland_losses": 0.600053256925217},
    {"dn": 1.0, "overland_losses": 0.662751592458654},
]

ground_temp = 8  # C
ambient_temp = 10  # C
###########COSTS DIGGING STREET#################
### ORIGINAL VALUES
# fc_dig_st = 350
# vc_dig_st = 700


### FlexyNets curve fit VALUES
fc_dig_st = 2.0583023861573324e-13
vc_dig_st = 73942.24960797853


###########COSTS DIGGING TERRAIN#################
fc_dig_tr = 200
vc_dig_tr = 500


###########COSTS PIPES###########################
# fc_pip = 50
# vc_pip = 700

### SERIES 1 VALUES
# fc_pip= 26.512730133354303
# vc_pip= 187.6969006863942

### SERIES 2 VALUES
fc_pip = 42.52334125532351
vc_pip = 147.1245773285605

###########COST FORMULAS EXPONENTS###############
### ORIGINAL VALUE
# vc_dig_st_ex = 1.1

### Flexynets curve fit Value
vc_dig_st_ex = 0.6445274602417989

vc_dig_tr_ex = 1.1

### Original Value
# vc_pip_ex = 1.3

### SERIES 1 VALUE
# vc_pip_ex = 1.45171040129113

### SERIES 2 VALUE
vc_pip_ex = 1.5669355282782995
###########INVESTMENT COSTS PUMPS###############

invest_pumps = 10000


################ Surface Losses #################
surface_losses_df = pd.DataFrame(surface_losses_json)


road_nw_solution = road_simplified_undirected.copy()

################################################################################
# CONVERT GRAPH TO DICT FOR PYOMO INPUT AND ONLY RETRIEVE DATA NEEDED ###########

road_nw_data = nx.to_dict_of_dicts(road_simplified_undirected)
road_nw_data_points = [(x, y) for x in road_nw_data for y in road_nw_data[x]]

road_nw_data_edges = []
for k, v in road_nw_data.items():
    for k1, v1 in v.items():
        road_nw_data_edges.append(
            [
                v1[0]["length"],
                v1[0]["surface_type"],
                v1[0]["restriction"],
                v1[0]["surface_pipe"],
                v1[0]["existing_grid_element"],
                v1[0]["inner_diameter_existing_grid_element"],
                v1[0]["costs_existing_grid_element"],
            ]
        )
        road_nw_data_names = list(v1.keys())

road_nw_data = dict(zip(road_nw_data_points, road_nw_data_edges))


# In our case we are not splittin the code in two parts. We are directly
# feeding the results of the Network Creation to the optimization module.
# In the original code we had to feed the Nodes and Edges into the optimisation
# function. So now we need to extract the nodes and edges from the road_nw_solution
# because that is what the original code expects.
nodes, edges = ox.graph_to_gdfs(road_nw_solution)

# If you need edge data as well, you can use:
edges_with_data = list(road_nw_solution.edges(data=True))
# road_simplified_undirected

################################################################################
###########ADD EDGE ATTRIBUTES THAT SHOULD BE PART OF GRAPH ALREADY#############


################################################################################
#########INVERT ALL KEYS TO BE SURE EACH EDGE EXISTS IN BOTH DIRECTIONS#########

for i, j in road_nw_data.keys():
    road_nw_data[(j, i)] = road_nw_data[(i, j)]

################################################################################
#########DEFINITION OF SOURCES AND SINKS FOR PYOMO##############################

# Create n_supply_dict directly from n_supply_list
n_supply_dict = {
    v["id"]: {"coords": tuple(v["coords"]), "cap": v["cap"]} for v in n_supply_list
}

n_demand_dict = {
    v["id"]: {"coords": tuple(v["coords"]), "cap": v["cap"]} for v in n_demand_list
}

N = list(nodes.index)
N_all_so_si = list(n_supply_dict.keys()) + list(n_demand_dict.keys())
N_supply = list(n_supply_dict.keys())
N_demand = list(n_demand_dict.keys())

##################################################################################
###################### PREPARE NODE PARAMETERS FOR PYOMO #########################

###################### TURN DICTS INTO DFS #######################################
supply_data_py = pd.DataFrame.from_dict(n_supply_dict, orient="index")
demand_data_py = pd.DataFrame.from_dict(n_demand_dict, orient="index")

###########DECIDE FOR THE CORRECT EXCHANGE CAPACITIES IF THE SUM OF CAPACITY SINKS
###### DOES NOT MATCH SUM OF CAPACITIES SOURCES (APPLIES FOR CASES THE GIS HAS NOT
###### ITERATED WITH THE TEO YET ##################################################

# if the supply capacity is higher than the demand, we take the capacity of the supply
# and redistribute it over the different supply nodes. (equally)
new_cap = []
if supply_data_py.cap.sum() > demand_data_py.cap.sum():
    for i in supply_data_py.cap:
        z = i
        diff = supply_data_py.cap.sum() - demand_data_py.cap.sum()
        share = z / supply_data_py.cap.sum()
        z = z - (diff * share)
        new_cap.append(z)

# if the demand capacity is higher than the supply, we take the demand capacity of each
# demand node and decrease it proportionally. So we decrease the total demand to match the supply
elif supply_data_py.cap.sum() < demand_data_py.cap.sum():
    for i in demand_data_py.cap:
        z = i
        diff = demand_data_py.cap.sum() - supply_data_py.cap.sum()
        share = z / demand_data_py.cap.sum()
        z = z - (diff * share)
        new_cap.append(z)
else:
    print("Supply capacity == demand capacity")

##################WRITE CORRECT VALUES TO DFS#################

if supply_data_py.cap.sum() > demand_data_py.cap.sum():
    supply_data_py.cap = new_cap
elif supply_data_py.cap.sum() < demand_data_py.cap.sum():
    demand_data_py.cap = new_cap
else:
    print("No adapted exchange capacities")


##################CONCAT DFS WITH ADAPTED EX CAPACITIES########

data_py = pd.concat([supply_data_py, demand_data_py], axis=0)
list_of_nodes = pd.DataFrame(N)


supply_data_py = list_of_nodes.merge(
    supply_data_py["cap"],
    left_on=list_of_nodes[0],
    right_on=supply_data_py.index,
    how="left",
).drop("key_0", axis=1)
demand_data_py = list_of_nodes.merge(
    demand_data_py["cap"],
    left_on=list_of_nodes[0],
    right_on=demand_data_py.index,
    how="left",
).drop("key_0", axis=1)
data_py = pd.concat([supply_data_py, demand_data_py["cap"]], axis=1)
data_py.columns = ["Nodes", "cap_sup", "cap_dem"]


###########ADD INPUTS FOR CONNECTIVITY RULE##########################
list_con_dem = []
for n in N:
    if n in N_all_so_si[1 : len(N_all_so_si)]:
        list_con_dem.append(1)
    else:
        list_con_dem.append(0)

data_py["con_dem"] = list_con_dem

list_con_sup = []
for n in N:
    if n in [N_all_so_si[0]]:
        list_con_sup.append(len(N_all_so_si[1 : len(N_all_so_si)]))
    else:
        list_con_sup.append(0)

data_py["con_sup"] = list_con_sup
data_py = data_py.fillna(0)
data_py.index = data_py.Nodes
del (supply_data_py, demand_data_py)


################################################################################
#################GET COPY OF ROAD_NW_DATA TO PRESERVE CORRECT LENGTH###########

road_nw_data_copy = road_nw_data.copy()

################################################################################
######################PREPARE EDGE PARAMETERS FOR PYOMO + WEIGTH LENGTH#########

road_nw_data = pd.DataFrame.from_dict(road_nw_data, orient="index")
road_nw_data.loc[road_nw_data[1] == "terrain", 0] = road_nw_data[0] * (
    1 - factor_street_terrain
)
road_nw_data.loc[road_nw_data[1] == "None", 0] = road_nw_data[0] * (
    1 - factor_street_overland
)
# road_nw_data = road_nw_data.sort_values(by=[1])
road_nw_data = road_nw_data.to_dict("index")


##################################################################################
########################### OPTIMIZE NETWORK ###################################
##################################################################################

from pyomo.environ import *
from pyomo.opt import *

opt = solvers.SolverFactory("gurobi_direct")
model = ConcreteModel()

###################################################################
######################SETS#########################################

model.node_set = Set(initialize=N)  # N is the IDs of all nodes
model.edge_set = Set(
    initialize=road_nw_data.keys()
)  # edges: ID pairs of nodes connected e.g. (17300064866, 1), (1, 17300064866) --> two way
model.flow_var_set = Set(
    initialize=np.arange(0, len(N_supply + N_demand) + 1, 1)
)  # 0, 1, ..., (total number of sources and sinks)-1

###################################################################
######################PARAMETERS EDGES#############################

model.edge_length = Param(
    model.edge_set,
    initialize={key: value[0] for (key, value) in road_nw_data.items()},
)
model.edge_restriction = Param(
    model.edge_set,
    initialize={key: value[2] for (key, value) in road_nw_data.items()},
)

model.edge_existing_grid = Param(
    model.edge_set,
    initialize={key: value[4] for (key, value) in road_nw_data.items()},
)

###################################################################
######################PARAMETERS NODES#############################

#####con_dem and con_sup are connection demand and connection supply
model.node_demand = Param(model.node_set, initialize=data_py["con_dem"].to_dict())
model.node_supply = Param(model.node_set, initialize=data_py["con_sup"].to_dict())

###################################################################
######################VARIABLES####################################

model.flow = Var(model.edge_set, within=model.flow_var_set)
# model.flow = Var(model.edge_set, within=np.arange(0, 10000, 1))
model.bool = Var(model.edge_set, domain=Binary)

########### RECODE FLOW VAR INTO BOOL VAR ###########################

Domain_points = [0.0, 0.0, 0.001, len(model.flow_var_set) + 1]
Range_points = [0.0, 0.0, 1.0, 1.0]

# here, we are creating a piecewise function to determine the value of "bool" variable
# based on the "flow" variable --> if "flow" == 0 then "bool" = 0, "bool" = 1 otherwise
model.con = Piecewise(
    model.edge_set,  # index
    model.bool,  # y value
    model.flow,  # x value
    pw_pts=Domain_points,  # domain of the piecewiese function
    pw_constr_type="EQ",  # means y=f(x)
    f_rule=Range_points,  # range of the piecewise function
    pw_repn="INC",  # indicates the type of piecewise representation to use
)

###################################################################
######################CONSTRAINT###################################

###########CONNECTIVITY CONSTRAINT#################################


def flow_rule(model, n):
    InFlow = sum(model.flow[i, j] for (i, j) in model.edge_set if j == n)
    OutFlow = sum(model.flow[i, j] for (i, j) in model.edge_set if i == n)

    input = InFlow + model.node_supply[n]
    output = OutFlow + model.node_demand[n]
    return input == output


model.flow_constraint = Constraint(N, rule=flow_rule)

###################################################################
######################OBJECTIVE####################################

###########TARGET FUNCTION#########################################

model.result = Objective(
    expr=sum(model.bool[a] * model.edge_length[a] * 2 for a in model.edge_set),
    sense=minimize,
)

###########CONSIDER EDGE RESTRICTION IN RESULT#####################

for i in model.edge_set:
    if model.edge_restriction[i] == -1:
        model.flow[i].setub(0)
    else:
        model.flow[i].setlb(0)

for i in model.edge_set:
    if model.edge_restriction[i] == 1:
        model.flow[i].setlb(1)
    else:
        model.flow[i].setlb(0)

###########SET EXISTING GRIDS AS PART OF SOLUTION#################

for i in model.edge_set:
    if model.edge_existing_grid[i] == 1:
        model.flow[i].setlb(1)
    else:
        model.flow[i]

###########SOLVE MODEL############################################
# opt.options[
#'timelimit'] = 60 * 12  ###max solver solution time, if exceeded the solver stops and takes the best found solution at that point

## Error handling
results = opt.solve(model, tee=True)
if results.solver.termination_condition == TerminationCondition.infeasible:
    raise RuntimeError("Routing is infeasible!")
# model.result.expr()

###########GET RESULTS############################################

result_data = model.bool.get_values()

result_graph = {k: v for k, v in result_data.items() if v > 0.1}

for i, j in list(result_graph):
    result_graph[(j, i)] = result_graph[(i, j)]

############GET THE PIPE DIAMETER FOR EVERY EDGE OF EXISTING GRID####
######FILTER ALL ELEMENTS FROM ROAD NETWORK THAT ARE IN SOLUTION OF##
##################### RESULT GRAPH###################################

road_nw_ex_grid = road_nw_data_copy.copy()
diff_edges_result_graph = set(road_nw_ex_grid) - set(result_graph)
for diff_edges_result_graph in diff_edges_result_graph:
    del road_nw_ex_grid[diff_edges_result_graph]

####Transform dict of solution edges to df#####

road_nw_ex_grid = pd.DataFrame.from_dict(road_nw_ex_grid, orient="index")

####in order to set a capacity limit on an existing grid edge we need to translate the
###pipe diameter into a thermal capacity before. We create a dataframe for conversion####

###################################################################
#######SET UP LOOK UP DF FOR POWER/DIAMETER CONVERSION#############

MW_dia_con = pd.DataFrame(columns=["Diameter", "v", "A", "MW"])
MW_dia_con["Diameter"] = np.arange(0.01, 1.001, 0.001)
MW_dia_con["A"] = ((MW_dia_con["Diameter"] / 2) ** 2) * 3.14159265359
MW_dia_con["v"] = 4.7617 * (MW_dia_con["Diameter"]) ** 0.3701 - 0.4834
MW_dia_con["MW"] = (
    MW_dia_con.A
    * MW_dia_con.v
    * water_den
    * heat_capacity
    * (abs(flow_temp - return_temp))
) / 1000
MW_dia_con["MW"] = round(MW_dia_con["MW"], 2)

###################################################################
#######FIND CORRESPONDING POWER VALUE FOR DIAMETER#################

MW_list = []
for i in road_nw_ex_grid[5]:
    index_dia = MW_dia_con["Diameter"].sub(i).abs().idxmin()
    MW_list.append(MW_dia_con["MW"][index_dia])

road_nw_ex_grid["MW"] = MW_list
road_nw_ex_grid["MW"] = road_nw_ex_grid["MW"].replace(0, 99999)

################################################
###Pyomo model flows V1 WITHOUT TEO#############
################################################

# we do not have existing capacity. So i set ex_cap as an empty list
ex_cap = []
if len(ex_cap) == 0:

    ###########DATA PREPARATION############################################

    data_py = data_py[
        data_py["Nodes"].isin(list({k[0] for k, v in result_graph.items()}))
    ]
    N = list(data_py.index)  # list of nodes existing in the solution

    opt = solvers.SolverFactory("gurobi_direct")
    model_nw = ConcreteModel()

    ###################################################################
    ######################SETS#########################################

    model_nw.node_set = Set(initialize=list({k[0] for k, v in result_graph.items()}))
    model_nw.edge_set = Set(initialize=result_graph.keys())

    ###################################################################
    ######################VARS#########################################

    model_nw.flow = Var(
        # NOTE: bounds changed for testing
        # model_nw.edge_set, bounds=(0, 9e+15)
        model_nw.edge_set,
        bounds=(0, 500000),
    )  ###real thermal flow on an edge, max thermal capacity set to 500 MW
    model_nw.cap_add = Var(
        # NOTE: bounds changed for testing
        # model_nw.edge_set, bounds=(0, 9e+15)
        model_nw.edge_set,
        bounds=(0, 500000),
    )  ###additional capacity required if bottleneck

    ###################################################################
    ######################PARAMETERS###################################

    model_nw.node_demand = Param(
        model_nw.node_set, initialize=data_py["cap_dem"].to_dict()
    )  # demand capacities of nodes
    model_nw.node_supply = Param(
        model_nw.node_set, initialize=data_py["cap_sup"].to_dict()
    )  # supply capacities of nodes
    model_nw.edge_capacities = Param(
        model_nw.edge_set, initialize=road_nw_ex_grid["MW"].to_dict()
    )  # thermal capacity of edge
    model_nw.edge_length = Param(
        model_nw.edge_set, initialize=road_nw_ex_grid[0].to_dict()
    )  # length of edge

    ###################################################################
    ######################CONSTRAINTS##################################

    def flow_rule_nw(model_nw, n):
        InFlow = sum(
            model_nw.flow[i, j] + model_nw.cap_add[i, j]
            for (i, j) in model_nw.edge_set
            if j == n
        )
        OutFlow = sum(
            model_nw.flow[i, j] + model_nw.cap_add[i, j]
            for (i, j) in model_nw.edge_set
            if i == n
        )

        input = InFlow + model_nw.node_supply[n]
        output = OutFlow + model_nw.node_demand[n]
        return input == output

    model_nw.flow_constraint = Constraint(N, rule=flow_rule_nw)

    def add_cap_rule(model_nw, i, j):
        return (
            model_nw.flow[i, j]
            <= model_nw.edge_capacities[i, j] + model_nw.cap_add[i, j]
        )

    model_nw.cap_constraint = Constraint(model_nw.edge_set, rule=add_cap_rule)

    ###################################################################
    ######################OBJECTIVE####################################

    model_nw.result_nw = Objective(
        expr=sum(
            model_nw.flow[a] * model_nw.edge_length[a] * 2
            + model_nw.cap_add[a] * 1000000000
            for a in model_nw.edge_set
        ),
        sense=minimize,
    )

    ## Error handling
    result_nw = opt.solve(model_nw, tee=False)
    if result_nw.solver.termination_condition == TerminationCondition.infeasible:
        raise RuntimeError("Routing is infeasible!")

    ###################################################################
    ######################GET RESULTS##################################

    result_data_flow = model_nw.flow.get_values()
    result_data_cap_add = model_nw.cap_add.get_values()
    keys_list = list(result_data_flow.keys())

    ###################################################################
    ######SAVE RESULTS FROM FLOW AND ADD CAP INTO ONE OBJECT###########

    result_data = {}

    for i in keys_list:
        result_data[i] = result_data_flow[i] + result_data_cap_add[i]

################################################
###Pyomo model flows WITH TEO###################
################################################
else:

    ###########DATA PREPARATION############################################
    #######FIGURE OUT EX CAPACITIES SEPERATED BY SOURCES AND SINKS#########

    Ex_cap_sinks = ex_cap[ex_cap["classification_type"] == "sink"]
    Ex_cap_sinks = Ex_cap_sinks.iloc[:, 2 : len(ex_cap.columns)]
    Ex_cap_sinks.index = Ex_cap_sinks["number"]
    Ex_cap_sinks = Ex_cap_sinks.drop("number", axis=1)

    Ex_cap_sources = ex_cap[ex_cap["classification_type"] == "source"]
    Ex_cap_sources = Ex_cap_sources.iloc[:, 2 : len(ex_cap.columns)]
    Ex_cap_sources.index = Ex_cap_sources["number"]
    Ex_cap_sources = Ex_cap_sources.drop("number", axis=1)

    ########GET ALL TIME STEPS FROM THE TEO ###############
    TS = ex_cap.iloc[:, 3 : len(ex_cap.columns)].columns
    # CREATE LIST WHERE ALL RESULTS FROM ITERATIONS ARE SAVED ##

    result_data_all_TS = []

    ########LOOP THROUGH EACH TIME STEP AND ###############
    #############PUT PARAMS TOGETHER IN DF#################

    for i in TS:
        data_py = data_py[
            data_py["Nodes"].isin(list({k[0] for k, v in result_graph.items()}))
        ]
        data_py = pd.DataFrame(data_py["Nodes"])

        N = list(data_py.index)

        data_py = data_py.merge(
            Ex_cap_sources[i],
            left_on=data_py["Nodes"],
            right_on=Ex_cap_sources.index,
            how="left",
        ).drop("key_0", axis=1)
        data_py = data_py.merge(
            Ex_cap_sinks[i],
            left_on=data_py["Nodes"],
            right_on=Ex_cap_sinks.index,
            how="left",
        ).drop("key_0", axis=1)
        data_py.index = data_py["Nodes"]
        data_py = data_py.fillna(0)

        #############SET UP MODEL###########################
        opt = solvers.SolverFactory("gurobi_direct")
        model_nw = ConcreteModel()

        ###################################################################
        ######################SETS#########################################

        model_nw.node_set = Set(
            initialize=list({k[0] for k, v in result_graph.items()})
        )
        model_nw.edge_set = Set(initialize=result_graph.keys())

        ###################################################################
        ######################VARS#########################################

        model_nw.flow = Var(
            model_nw.edge_set, bounds=(0, 500000)
        )  ###max set to thermal capacity of 500 MW
        model_nw.cap_add = Var(
            model_nw.edge_set, bounds=(0, 500000)
        )  ###additional capacity required if bottleneck

        ###################################################################
        ######################PARAMETERS###################################

        model_nw.node_demand = Param(
            model_nw.node_set, initialize=data_py.iloc[:, 2].to_dict()
        )
        model_nw.node_supply = Param(
            model_nw.node_set, initialize=data_py.iloc[:, 1].to_dict()
        )
        model_nw.edge_capacities = Param(
            model_nw.edge_set, initialize=road_nw_ex_grid["MW"].to_dict()
        )
        model_nw.edge_length = Param(
            model_nw.edge_set, initialize=road_nw_ex_grid[0].to_dict()
        )

        ###################################################################
        ######################CONSTRAINTS##################################

        def flow_rule_nw(model_nw, n):
            InFlow = sum(
                model_nw.flow[i, j] + model_nw.cap_add[i, j]
                for (i, j) in model_nw.edge_set
                if j == n
            )
            OutFlow = sum(
                model_nw.flow[i, j] + model_nw.cap_add[i, j]
                for (i, j) in model_nw.edge_set
                if i == n
            )

            input = InFlow + model_nw.node_supply[n]
            output = OutFlow + model_nw.node_demand[n]
            return input == output

        model_nw.flow_constraint = Constraint(N, rule=flow_rule_nw)

        def add_cap_rule(model_nw, i, j):
            return (
                model_nw.flow[i, j]
                <= model_nw.edge_capacities[i, j] + model_nw.cap_add[i, j]
            )

        model_nw.cap_constraint = Constraint(model_nw.edge_set, rule=add_cap_rule)

        ###################################################################
        ######################OBJECTIVE####################################

        model_nw.result_nw = Objective(
            expr=sum(
                model_nw.flow[a] * model_nw.edge_length[a] * 2
                + model_nw.cap_add[a] * 1000000000
                for a in model_nw.edge_set
            ),
            sense=minimize,
        )

        ## Error handling
        result_nw = opt.solve(model_nw, tee=False)
        if result_nw.solver.termination_condition == TerminationCondition.infeasible:
            raise ModuleRuntimeException(
                code=2.7,
                msg="Thermal capacity optimization with TEO is infeasible!",
            )

        ###################################################################
        ######################GET RESULTS##################################

        result_data_flow = model_nw.flow.get_values()
        result_data_cap_add = model_nw.cap_add.get_values()
        keys_list = list(result_data_flow.keys())

        ###################################################################
        ######SAVE RESULTS FROM FLOW AND ADD CAP INTO ONE OBJECT###########

        result_data = {}

        for i in keys_list:
            result_data[i] = (result_data_flow[i] or 0) + (result_data_cap_add[i] or 0)

        result_data_all_TS.append(pd.DataFrame.from_dict(result_data, orient="index"))

        del model_nw

    result_data = pd.concat(result_data_all_TS, axis=1)
    result_data["max"] = result_data.max(axis=1)
    # result_data["max"] = round(result_data["max"], 2)
    result_data = result_data["max"].to_dict()

###################################################################
##################CHECK IF EVERY EDGE HAS A FLOW###################

###########GET EDGES THAT HAVE A FLOW > 0 ON THEM##################

graph_test_data = result_data.copy()
graph_test_data = {k: v for k, v in graph_test_data.items() if v != 0}

###########CREATE GRAPH OUT OF ONLY EDGES WITH FLOW ON IT##########

graph_test = nx.Graph()
graph_test.add_edges_from(graph_test_data.keys())
# nx.draw(graph_test)

# Set up the plot
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

# Use spring layout for node positioning
pos = nx.spring_layout(graph_test)

# Draw nodes
nx.draw_networkx_nodes(graph_test, pos, node_size=50, node_color="lightblue")

# Draw edges
nx.draw_networkx_edges(graph_test, pos, edge_color="gray", alpha=0.5)

# Draw labels
# nx.draw_networkx_labels(graph_test, pos, font_size=8, font_family='sans-serif')

# Remove axis
plt.axis("off")

# Add a title
plt.title("Network Graph Visualization", fontsize=16)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()

###########GET NUMBER OF SUBGRAPHS IN GRAPH########################

subgraphs = len(
    list((graph_test.subgraph(c) for c in nx.connected_components(graph_test)))
)

if subgraphs > 1:
    ############################################################################
    #######ASSIGN VALUES TO EDGES WITH NO FLOW IF MULTIPLE SUBGRAPHS EXIST######
    ############################################################################

    ###################################################################
    #############FIND EDGES WITH NO FLOW ON IT#########################

    for i, j in result_data.keys():
        if (result_data[(i, j)] + result_data[(j, i)]) == 0:
            result_data[(i, j)] = 99999
        else:
            result_data[(i, j)] = result_data[(i, j)]

    zero_values = {k: v for k, v in result_data.items() if v == 99999}
    zero_values = pd.DataFrame.from_dict(zero_values.keys())
    zero_values = zero_values.stack()
    zero_values = (
        zero_values.value_counts()
    )  ###maybe cases occur where there is exactly the same start and end point
    zero_values = zero_values[zero_values == 1]
    conn_points = zero_values.index
    del zero_values

    graph_data = {k: v for k, v in result_data.items() if v > 0}
    graph = nx.Graph()
    for k, v in graph_data.items():
        graph.add_edge(k[0], k[1], weight=v)

    adj_list = []
    for i in conn_points:
        adj_df = pd.DataFrame.from_dict(graph[i].values())
        adj_df.columns.values[0] = i
        adj_list.append(adj_df)

    adj_df = pd.concat(adj_list, axis=1)
    adj_df = adj_df.replace(to_replace=99999, value=np.NaN)
    adj_df = adj_df.mean()
    adj_df_help = adj_df.copy()

    graph_data = {k: v for k, v in result_data.items() if v == 99999}
    graph = nx.Graph()
    for k, v in graph_data.items():
        graph.add_edge(k[0], k[1], weight=v)

    con_list = []
    for i in range(0, len(adj_df)):
        for z in range(0, len(adj_df_help)):
            con = nx.has_path(graph, adj_df.index[i], adj_df_help.index[z])
            con_list.append(
                [
                    adj_df.index[i],
                    adj_df.values[i],
                    adj_df_help.index[z],
                    adj_df_help.values[z],
                    con,
                ]
            )

    con_df = pd.DataFrame(con_list)
    con_df = con_df[con_df[0] != con_df[2]]
    con_df = con_df[con_df[4] == True].reset_index()
    con_df["sum_nodes"] = con_df[0].astype(int) + con_df[2].astype(int)
    index_to_keep = con_df["sum_nodes"].drop_duplicates().index
    con_df["ex_capacity"] = (con_df[1] + con_df[3]) / 2
    con_df = con_df.iloc[index_to_keep, :]

    graph = graph.to_directed()

    for pois in range(0, len(con_df)):
        node = con_df[0].iloc[pois]
        cap = con_df["ex_capacity"].iloc[pois]
        graph.nodes[node]["demand"] = cap

    for pois in range(0, len(con_df)):
        node = con_df[2].iloc[pois]
        cap = con_df["ex_capacity"].iloc[pois]
        graph.nodes[node]["demand"] = cap * -1

    flow_dict = nx.min_cost_flow(graph, demand="demand")

    all_tuples = []
    for k, v in flow_dict.items():
        for k1, v1 in v.items():
            all_tuples.append([tuple([k, k1]), v1])

    all_edges_dict = dict(zip([n[0] for n in all_tuples], [n[1] for n in all_tuples]))
    all_edges_dict = {k: v for k, v in all_edges_dict.items() if v != 0}
    all_edges_dict.update(
        {k: v for k, v in result_data.items() if v != 0 and v != 99999}
    )
else:
    result_data = {k: v for k, v in result_data.items() if v != 0}
    all_edges_dict = result_data

###################################################################
#####GET RESULTS, CONVERT IT TO DF AND MERGE IT WITH ALL EDGE ATTR.
#######FROM GRAPH##################################################

result_df = pd.DataFrame.from_dict(all_edges_dict, orient="index", columns=["MW"])
graph_df = pd.DataFrame.from_dict(road_nw_data_copy, orient="index")
result_df = result_df.merge(
    graph_df, left_on=result_df.index, right_on=graph_df.index, how="left"
)
result_df.index = result_df["key_0"]
result_df = result_df.drop(["key_0", 2, 4], axis=1)

# NOTE: converting the pipe capacities into MW (capacities received from CF are in kW)
result_df["MW"] = result_df["MW"] / 1000

#####Merge information about the capacity limits for existing grid elements######
result_df = result_df.merge(
    road_nw_ex_grid["MW"],
    left_on=result_df.index,
    right_on=road_nw_ex_grid.index,
    how="left",
)
result_df.index = result_df["key_0"]
result_df = result_df.drop(["key_0"], axis=1)

result_df.columns = [
    "MW",
    "Length",
    "Surface_type",
    "Surface_pipe",
    "Diameter_ex_grid",
    "Costs_ex_grid",
    "Capacity_limit",
]
result_df.loc[result_df["Capacity_limit"] == 99999, "Capacity_limit"] = np.nan
del graph_df

###############################################################################
########DETERMINE CONNECTING ELEMENTS BETWEEN ALL SOURCES AND SINKS############
###############################################################################

all_edges = list(result_df.index)
u = [n[0] for n in all_edges]
v = [n[1] for n in all_edges]
result_df["u"] = u
result_df["v"] = v
graph_solution = nx.from_pandas_edgelist(
    result_df, source="u", target="v", edge_attr="Length"
)

###########################################################################
#######FIND SHORTEST PATH BETWEEN ALL SOURCE/SINK PAIRS IN ONE DIRECTION###

shortest_paths = {}
for i in N_supply:
    for x in N_demand:
        shortest_paths[i, x] = nx.shortest_path(
            G=graph_solution, source=i, target=x, weight="Length"
        )

###################################################################
#######CONVERT ALL SOURCE/SINK PAIR DICT TO DF#####################

shortest_paths = pd.DataFrame.from_dict(shortest_paths, orient="index").transpose()

###################################################################
#######CREATE EDGE STRUCTURE FROM SHORTEST PATH DF#################

shortest_paths_copy = pd.DataFrame(
    columns=shortest_paths.columns, index=range(len(shortest_paths) - 1)
)

for i in range(0, len(shortest_paths) - 1):
    for z in range(0, len(shortest_paths.columns)):
        if shortest_paths.iloc[i, z] and shortest_paths.iloc[i + 1, z] > 0:
            shortest_paths_copy.iloc[i, z] = tuple(
                [shortest_paths.iloc[i, z], shortest_paths.iloc[i + 1, z]]
            )
        else:
            shortest_paths_copy.iloc[i, z] = 0

shortest_paths_ij = shortest_paths_copy.copy()
del shortest_paths_copy

###########################################################################
#######FIND SHORTEST PATH BETWEEN ALL SOURCE/SINK PAIRS IN OTHER DIRECTION#

shortest_paths_ji = shortest_paths_ij.copy()

for i in range(0, len(shortest_paths_ij)):
    for z in range(0, len(shortest_paths_ij.columns)):
        if shortest_paths_ji.iloc[i, z] != 0:
            shortest_paths_ji.iloc[i, z] = tuple(
                [shortest_paths_ij.iloc[i, z][1], shortest_paths_ij.iloc[i, z][0]]
            )
        else:
            shortest_paths_ji.iloc[i, z] = 0

###################################################################
####COMBINE TWO DFS WITH SHORTEST PATH INFORMATION#################

###########ASSIGN SAME COLUMN NAMES TO BOTH DFS TO CONCAT THEM#############

shortest_paths = (
    pd.concat([shortest_paths_ij, shortest_paths_ji], axis=0)
    .reset_index()
    .drop("index", axis=1)
)
shortest_paths.columns = [str(n) for n in list(shortest_paths.columns)]

###########MERGE EDGES SOLUTION AND SHORTEST PATH EACH SOURCE/SINK PAIR###

rows_list = []
for i in range(0, len(shortest_paths.columns)):
    dict1 = {}
    merge_df = result_df.merge(
        shortest_paths.iloc[:, i],
        left_on=result_df.index,
        right_on=shortest_paths.iloc[:, i],
        how="left",
    )
    dict1.update(merge_df)
    rows_list.append(pd.DataFrame(dict1))

###########COMBINE INFORMATION INTO SINGLE DF###########################

result_df = pd.concat(rows_list, axis=1)
result_df = result_df.loc[:, ~result_df.columns.duplicated()]

################################################
################Calculate losses################
################################################

###################################################################
#######SET UP LOOK UP DF FOR POWER/DIAMETER CONVERSION#############

MW_dia_con = pd.DataFrame(columns=["Diameter", "v", "A", "MW"])
MW_dia_con["Diameter"] = np.arange(0.01, 1.001, 0.001)
MW_dia_con["A"] = ((MW_dia_con["Diameter"] / 2) ** 2) * 3.14159265359
MW_dia_con["v"] = 4.7617 * (MW_dia_con["Diameter"]) ** 0.3701 - 0.4834
MW_dia_con["MW"] = (
    MW_dia_con.A
    * MW_dia_con.v
    * water_den
    * heat_capacity
    * (abs(flow_temp - return_temp))
) / 1000
MW_dia_con["MW"] = round(MW_dia_con["MW"], 3)
###################################################################
#######FIND CORRESPONDING DIAMETER FOR POWER VALUE#################

diameter_list = []
for i in result_df["MW"]:
    index_dia = MW_dia_con["MW"].sub(i).abs().idxmin()
    diameter_list.append(MW_dia_con["Diameter"][index_dia])

result_df["Diameter"] = diameter_list

###################################################################
#######OVERRULE DIAMETER BY EXISTING DIAMETER IF AVAILABLE#########

for i in range(0, len(result_df)):
    if result_df["Diameter_ex_grid"][i] != 0:
        result_df.loc[i, "Diameter"] = result_df.loc[i, "Diameter_ex_grid"]
    else:
        pass

result_df["Diameter"] = round(result_df["Diameter"], 3)

###################################################################
#########################LOSS CALCULATION##########################

loss_list = []
for i in range(0, len(result_df)):
    if result_df["Surface_pipe"][i] == 1:
        index_dia = surface_losses_df["dn"].sub(result_df["Diameter"][i]).abs().idxmin()
        loss_list.append(
            surface_losses_df["overland_losses"][index_dia]
            * abs((((flow_temp + return_temp) / 2) - ambient_temp))
        )
    else:  # if pipe is buried, than it uses the formulation from THERMOS
        loss_list.append(
            (
                abs((flow_temp + return_temp) / 2 - ground_temp)
                * (0.1685 * np.log(result_df["Diameter"][i]) + 0.85684)
            )
        )

result_df["Losses [W/m]"] = loss_list
result_df["Losses [W/m]"] = round(result_df["Losses [W/m]"], 3)
result_df["Length"] = round(result_df["Length"] * 2, 2)
result_df["Losses [W]"] = result_df["Losses [W/m]"] * result_df["Length"]
result_df["Losses [W]"] = round(result_df["Losses [W]"], 3)

################################################
################Calculate costs#################
################################################

result_df.loc[result_df["Surface_type"] == "street", "costs_digging"] = (
    fc_dig_st + (vc_dig_st * result_df["Diameter"]) ** vc_dig_st_ex
) * result_df["Length"]
result_df.loc[result_df["Surface_type"] == "terrain", "costs_digging"] = (
    fc_dig_tr + (vc_dig_tr * result_df["Diameter"]) ** vc_dig_tr_ex
) * result_df["Length"]
result_df.loc[result_df["Surface_type"] == "None", "costs_digging"] = 0
result_df["costs_piping"] = (
    fc_pip + (vc_pip * result_df["Diameter"]) ** vc_pip_ex
) * result_df["Length"]
result_df["cost_total"] = round(
    result_df["costs_piping"] + result_df["costs_digging"], 2
)
result_df.loc[result_df["Diameter_ex_grid"] > 0, "cost_total"] = result_df[
    "Costs_ex_grid"
]

###########EXTRACT SOLUTION BY EACH SOURCE/SINK PAIR####################

so_sin_cols = [n for n in list(result_df.columns) if "(" in n]
res_sources_sinks = result_df.copy()
res_sources_sinks = res_sources_sinks.fillna(0)

rows_list = []
for i in so_sin_cols:
    df = res_sources_sinks.loc[:, [i, "cost_total", "Losses [W]", "Length", "MW"]]
    df = df[df[i] != 0]
    df_sum = [
        i,
        df["Losses [W]"].sum(),
        df["MW"].sum(),
        df["Length"].sum(),
        df["cost_total"].sum(),
    ]
    rows_list.append(df_sum)

res_sources_sinks = pd.DataFrame(
    rows_list,
    columns=[
        "From/to",
        "Losses total [W]",
        "Installed capacity [MW]",
        "Length [m]",
        "Total_costs [EUR]",
    ],
)

sums = {
    "losses_total": result_df["Losses [W]"].sum(),
    "installed_capacity": result_df["MW"].sum(),
    "length": result_df["Length"].sum(),
    "total_costs": (result_df["cost_total"].sum() + invest_pumps),
}

# make the calculations for TEO
losses_in_kw = res_sources_sinks["Losses total [W]"].mean() / 1000
cost_in_kw = sums["total_costs"] / (sums["installed_capacity"] * 1000)
# the output for TEO
losses_cost_kw = {"losses_in_kw": losses_in_kw, "cost_in_kw": cost_in_kw}

#######VISUAL RESULTS##################

result_df_ij = result_df.copy()
result_df_ji = result_df.copy()
result_df_ji["u"] = result_df_ij["v"]
result_df_ji["v"] = result_df_ij["u"]
result_df_ji["key_0"] = (
    "(" + result_df_ji["u"].astype(str) + ", " + result_df_ji["v"].astype(str) + ")"
)

result_df_result = pd.concat([result_df_ji, result_df_ij], axis=0)

############DATA WILL BE DELETED LATER, ONLY FOR CHECK IF GIS WORKS CORRECTLY#################
##############################################################################################
##############################################################################################
##############################################################################################
edges = edges.reset_index(level=[0, 1, 2])
edges["from_to"] = (
    "(" + edges["u"].astype(str) + ", " + edges["v"].astype(str) + ")"
)  ####create new variable in edges where solution can be retrieved from

edges = edges.merge(
    result_df_result[
        [
            "MW",
            "Length",
            "Surface_type",
            "cost_total",
            "Diameter",
            "Losses [W/m]",
            "Losses [W]",
            "Capacity_limit",
        ]
    ],
    left_on=edges["from_to"],
    right_on=result_df_result["key_0"].astype(str),
    how="left",
)
edges["MW"] = edges["MW"].fillna(0)

edges_solution = edges[edges["MW"] != 0]

nodes_to_filter = pd.DataFrame()
nodes_to_filter = pd.concat(
    [edges_solution["u"], edges_solution["v"]], axis=0
)  ###create a vector out of all solution points (source and/or target)
nodes_to_filter = (
    nodes_to_filter.unique()
)  ###create a vector with only unique points (no double counts) - all point that are part of solution regardless if source or target points

nodes_solution = nodes[
    nodes["osmid"].isin(nodes_to_filter)
]  ###filter solution points from input graph

edges_solution  ###edges of solution network
nodes_solution  ###nodes of solution network

m = folium.Map(
    location=[
        list(n_demand_dict.values())[0]["coords"][0],
        list(n_demand_dict.values())[0]["coords"][1],
    ],
    zoom_start=11,
    control_scale=True,
)  #####create basemap
#####layer for whole routing area####
style = {
    "fillColor": "#00FFFFFF",
    "color": "#00FFFFFF",
}  ####create colour for layer
whole_area = folium.features.GeoJson(
    edges, name="area", style_function=lambda x: style, overlay=True
)  ###create layer with whole road network
path = folium.features.GeoJson(
    edges_solution.to_json(), name="path", overlay=True
)  ###create layer with solution edges

nodes_to_map = nodes_solution[nodes_solution["osmid"].isin(N_supply)].reset_index(
    drop=True
)

sources_cluster = MarkerCluster(name="Sources").add_to(m)

for idx, row in nodes_to_map.iterrows():
    # Use the y (latitude) and x (longitude) columns
    sources_cluster.add_child(
        folium.Marker(
            location=[row["y"], row["x"]],  # [latitude, longitude]
            icon=folium.Icon(color="red", icon="tint"),
            popup=f"Source: {idx}",  # You can customize this popup as needed
        )
    )
# this is a test
# Alternatively, if you prefer to use the geometry column:
# for idx, row in nodes_to_map.iterrows():
#     point = row['geometry']
#     sources_cluster.add_child(
#         folium.Marker(
#             location=[point.y, point.x],  # [latitude, longitude]
#             icon=folium.Icon(color="red", icon="tint"),
#             popup=f"Source: {idx}"
#         )
#     )

# print(nodes_to_map.columns)


sinks_cluster = MarkerCluster(name="Sinks").add_to(m)
edges_to_map = nodes_solution[nodes_solution["osmid"].isin(N_demand)].reset_index(
    drop=True
)
for i in range(0, len(edges_to_map)):
    sinks_cluster.add_child(
        folium.Marker(
            location=[edges_to_map.loc[i, "y"], edges_to_map.loc[i, "x"]],
            icon=folium.Icon(color="blue", icon="tint"),
            popup="Sink",
        )
    )

# sinks = folium.features.GeoJson(nodes_solution[nodes_solution["osmid"].isin(N_demand)], name="sinks",
#                                 overlay=True, tooltip="Sink")

path.add_to(m)  ###add layer to map
whole_area.add_to(m)  ###add layer to map
# sinks.add_to(m)
folium.LayerControl().add_to(m)  ###add layer control to map

####add labels
folium.features.GeoJsonPopup(
    fields=[
        "from_to",
        "MW",
        "Diameter",
        "Length",
        "Surface_type",
        "cost_total",
        "Losses [W/m]",
        "Losses [W]",
        "Capacity_limit",
    ],
    labels=True,
).add_to(path)

# folium.features.GeoJsonPopup(fields=["osmid"], labels=True).add_to(points)
####save map as html#####
# m.save("TEST.html")

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

###create graph object with all solution edges
potential_grid_area = road_nw_solution.copy()

nodes, edges = ox.graph_to_gdfs(road_nw_solution)

edges = edges.reset_index(level=[0, 1, 2])

edges["from_to"] = (
    "(" + edges["u"].astype(str) + ", " + edges["v"].astype(str) + ")"
)  ####create new variable in nodes where solution can be retrieved from

edges = edges.merge(
    result_df_result[
        [
            "MW",
            "Length",
            "Surface_type",
            "cost_total",
            "Diameter",
            "Losses [W/m]",
            "Losses [W]",
            "Capacity_limit",
        ]
    ],
    left_on=edges["from_to"],
    right_on=result_df_result["key_0"].astype(str),
    how="left",
)
edges["MW"] = edges["MW"].fillna(0)

edges.rename(
    columns={
        "Length": "Pipe length [m]",
        "Diameter": "Diameter [m]",
        "cost_total": "Total costs [EUR]",
        "Capacity_limit": "Capacity limit [MW]",
    },
    inplace=True,
    errors="ignore",
)

edges.set_index(["u", "v", "key"], inplace=True)
road_nw_solution = ox.graph_from_gdfs(nodes, edges)

edges_without_flow = [
    (u, v) for u, v, e in road_nw_solution.edges(data=True) if e["MW"] == 0
]

for i in edges_without_flow:
    road_nw_solution.remove_edge(i[0], i[1])

nodes_not_connected = list(nx.isolates(road_nw_solution))

road_nw_solution.remove_nodes_from(nodes_not_connected)

network_solution = road_nw_solution

selected_agents = (
    N_supply + N_demand
)  # output to MM: list of sources and sinks exist in the solution
# Create a new map centered on the first demand point


m = folium.Map(
    location=[
        list(n_demand_dict.values())[0]["coords"][0],
        list(n_demand_dict.values())[0]["coords"][1],
    ],
    zoom_start=11,
    control_scale=True,
)

# Add the network to the map with power information
folium.GeoJson(
    edges_solution.to_json(),
    name="Network",
    style_function=lambda x: {"color": "red", "weight": 5},
    tooltip=folium.GeoJsonTooltip(fields=["MW"], aliases=["Power (MW):"]),
).add_to(m)

# Add markers for supply points with capacity information
for node_id, node_data in n_supply_dict.items():
    folium.Marker(
        location=node_data["coords"],
        popup=folium.Popup(
            f"Supply Node {node_id}<br>Capacity: {node_data['cap']} MW", max_width=300
        ),
        tooltip=f"Supply Node {node_id}",
        icon=folium.Icon(color="green", icon="info-sign"),
    ).add_to(m)

# Add markers for demand points with capacity information
for node_id, node_data in n_demand_dict.items():
    folium.Marker(
        location=node_data["coords"],
        popup=folium.Popup(
            f"Demand Node {node_id}<br>Capacity: {node_data['cap']} MW", max_width=300
        ),
        tooltip=f"Demand Node {node_id}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display the map
m

from IPython.display import IFrame

# Save the map
m.save("renovated_map.html")
result_df.to_parquet("renovated_result_df.parquet")
print("map and result_df saved")
