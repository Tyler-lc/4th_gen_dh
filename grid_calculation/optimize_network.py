import networkx as nx
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory


def optimize_network(road_nw, n_supply_dict, n_demand_dict, params):
    model = ConcreteModel()

    # Define sets
    model.nodes = Set(initialize=road_nw.nodes())
    model.edges = Set(initialize=road_nw.edges())
    model.supply_nodes = Set(initialize=n_supply_dict.keys())
    model.demand_nodes = Set(initialize=n_demand_dict.keys())

    # Define variables
    model.flow = Var(model.edges, domain=NonNegativeReals)
    model.pipe_installed = Var(model.edges, domain=Binary)

    # Define constraints
    def flow_conservation_rule(model, node):
        if node in model.supply_nodes:
            return (
                sum(model.flow[i, j] for i, j in model.edges if j == node)
                - sum(model.flow[i, j] for i, j in model.edges if i == node)
                == -n_supply_dict[node]["cap"]
            )
        elif node in model.demand_nodes:
            return (
                sum(model.flow[i, j] for i, j in model.edges if j == node)
                - sum(model.flow[i, j] for i, j in model.edges if i == node)
                == n_demand_dict[node]["cap"]
            )
        else:
            return sum(model.flow[i, j] for i, j in model.edges if j == node) == sum(
                model.flow[i, j] for i, j in model.edges if i == node
            )

    model.flow_conservation = Constraint(model.nodes, rule=flow_conservation_rule)

    def pipe_capacity_rule(model, i, j):
        return model.flow[i, j] <= params["max_flow"] * model.pipe_installed[i, j]

    model.pipe_capacity = Constraint(model.edges, rule=pipe_capacity_rule)

    # Define objective
    def objective_rule(model):
        return sum(
            road_nw[i][j]["length"] * model.pipe_installed[i, j] for i, j in model.edges
        )

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # Solve the model
    solver = SolverFactory("glpk")
    results = solver.solve(model)

    # Extract results
    optimal_network = nx.Graph()
    for i, j in model.edges:
        if value(model.pipe_installed[i, j]) > 0.5:
            optimal_network.add_edge(
                i, j, flow=value(model.flow[i, j]), length=road_nw[i][j]["length"]
            )

    return optimal_network


def prepare_optimization_input(road_nw, n_supply_dict, n_demand_dict):
    params = {"max_flow": 100}  # Example parameter, adjust as needed
    return params


def run_optimize_network(road_nw, n_supply_dict, n_demand_dict):
    params = prepare_optimization_input(road_nw, n_supply_dict, n_demand_dict)
    optimal_network = optimize_network(road_nw, n_supply_dict, n_demand_dict, params)
    return optimal_network
