"""Estimate pumping energy for each district heating scenario.

Reads the grid-optimisation parquets (pipe diameters, lengths, thermal power)
and computes pressure drop and pumping power using Darcy-Weisbach friction
and minor losses (one long-radius bend per segment).

Two pumping-power figures are reported per scenario:
    1. Design pump power  — critical-path dp * total volume flow / eta.
       This is what a centralised pump must be sized for.
    2. Dissipated power   — sum(dp_i * V_dot_i) / eta.
       Actual hydraulic power consumed across the network; lower bound for
       pump power (the gap is throttling loss at balancing valves).

Results are saved to ``pumping_losses/<material>/`` as per-edge CSVs and a
cross-scenario summary.
"""

import networkx as nx
import numpy as np
import pandas as pd

from config import (
    ANNUAL_HOURS,
    DEFAULT_PIPE_MATERIAL,
    K_BEND_90,
    PIPE_ROUGHNESS,
    PUMP_EFFICIENCY_ELECTRIC,
    PUMP_EFFICIENCY_HYDRAULIC,
    PUMPING_LOSSES_DIR,
    SCENARIO_TEMPERATURES,
    grid_results_parquet,
)
from utils.hydraulics import (
    compute_edge_hydraulics,
    mass_flow_rate,
    pumping_power,
    water_density,
)

SCENARIOS = ["unrenovated", "renovated", "booster"]
MATERIALS = ["steel_new", "pvc_pe"]
CP_KJ_KGK = 4.18  # matches grid calculation scripts


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def _build_graph(df):
    """Build a networkx Graph from the grid result DataFrame.

    Returns the graph and the identified supply node (node with the highest
    total MW on its incident edges — the trunk of the tree).
    """
    G = nx.Graph()
    for idx, row in df.iterrows():
        u, v = int(row["u"]), int(row["v"])
        G.add_edge(u, v, index=idx, MW=row["MW"], dp=0.0)
    return G


def _find_supply_node(G):
    """Identify the supply node as the node whose incident edges carry the
    most total thermal power."""
    best_node = None
    best_mw = -1.0
    for node in G.nodes():
        total_mw = sum(G[node][nbr]["MW"] for nbr in G.neighbors(node))
        if total_mw > best_mw:
            best_mw = total_mw
            best_node = node
    return best_node


def _critical_path_dp(G, supply_node, edge_dp_map):
    """Find the maximum cumulative pressure drop from supply to any leaf.

    *edge_dp_map* maps (u, v) edge tuples (both orderings) to dp [Pa].
    Since the graph is a tree, shortest-path == only-path.
    """
    # Assign dp as edge weight
    for u, v, data in G.edges(data=True):
        idx = data["index"]
        dp = edge_dp_map.get(idx, 0.0)
        data["dp"] = dp

    max_dp = 0.0
    for node in G.nodes():
        if G.degree(node) == 1 and node != supply_node:
            path = nx.shortest_path(G, supply_node, node)
            dp_sum = 0.0
            for i in range(len(path) - 1):
                dp_sum += G[path[i]][path[i + 1]]["dp"]
            if dp_sum > max_dp:
                max_dp = dp_sum
    return max_dp


# ---------------------------------------------------------------------------
# Per-scenario computation
# ---------------------------------------------------------------------------

def compute_scenario(scenario, material):
    """Compute pumping losses for one scenario and one pipe material.

    Returns (per_edge_df, summary_dict).
    """
    temps = SCENARIO_TEMPERATURES[scenario]
    T_supply = temps["supply"]
    T_return = temps["return"]
    delta_T = T_supply - T_return
    roughness = PIPE_ROUGHNESS[material]

    df = pd.read_parquet(grid_results_parquet(scenario))

    # Filter to active edges (MW > 0)
    active = df[df["MW"] > 0].copy()

    # Compute hydraulics for every active edge
    hyd = compute_edge_hydraulics(
        Q_MW_array=active["MW"].values,
        D_array=active["Diameter"].values,
        L_total_array=active["Length"].values,
        T_supply_C=T_supply,
        T_return_C=T_return,
        roughness_m=roughness,
        n_bends_per_pipe=1,
        K_bend=K_BEND_90,
        cp_kJ_kgK=CP_KJ_KGK,
    )

    # Attach results to DataFrame
    for key in hyd:
        active[key] = hyd[key]

    # Volume flow per edge (at mean temperature for density)
    T_mean = (T_supply + T_return) / 2.0
    rho_mean = water_density(T_mean)
    active["V_dot_m3_s"] = active["mass_flow_kg_s"] / rho_mean

    # --- Network analysis ---
    G = _build_graph(active)
    supply_node = _find_supply_node(G)

    # Map edge index -> dp for critical path calculation
    edge_dp_map = dict(zip(active.index, active["dp_total_Pa"].values))
    dp_critical = _critical_path_dp(G, supply_node, edge_dp_map)

    # Total volume flow at supply = sum of all leaf demands
    # In a tree, every edge's flow is unique; the supply edge carries total flow.
    # We identify total flow as the mass flow on the edge incident to supply_node
    # with the highest MW.
    supply_edges = [(u, v) for u, v in G.edges(supply_node)]
    max_mw_edge = max(supply_edges, key=lambda e: G[e[0]][e[1]]["MW"])
    supply_edge_idx = G[max_mw_edge[0]][max_mw_edge[1]]["index"]
    m_dot_total = active.loc[supply_edge_idx, "mass_flow_kg_s"]
    V_dot_total = m_dot_total / rho_mean

    # Design pump power (critical path)
    P_design_W = pumping_power(
        dp_critical, V_dot_total,
        PUMP_EFFICIENCY_ELECTRIC, PUMP_EFFICIENCY_HYDRAULIC,
    )

    # Dissipated power (sum over all edges)
    P_dissipated_W = (
        (active["dp_total_Pa"] * active["V_dot_m3_s"]).sum()
        / (PUMP_EFFICIENCY_ELECTRIC * PUMP_EFFICIENCY_HYDRAULIC)
    )

    # Thermal losses and heat delivered
    # Heat delivered = flow at supply point, NOT sum of all edges (which
    # double-counts because upstream edges carry cumulative downstream flow).
    thermal_losses_W = df["Losses [W]"].sum()
    heat_delivered_W = m_dot_total * CP_KJ_KGK * 1e3 * delta_T

    # Energy fractions
    total_design = P_design_W + thermal_losses_W + heat_delivered_W
    total_dissip = P_dissipated_W + thermal_losses_W + heat_delivered_W

    # Annual energy values (power * 8760 h) — consistent with scenario scripts
    E_pump_design_MWh = P_design_W / 1e6 * ANNUAL_HOURS
    E_pump_dissip_MWh = P_dissipated_W / 1e6 * ANNUAL_HOURS
    E_thermal_loss_MWh = thermal_losses_W / 1e6 * ANNUAL_HOURS
    E_delivered_MWh = heat_delivered_W / 1e6 * ANNUAL_HOURS

    summary = {
        "scenario": scenario,
        "material": material,
        "T_supply_C": T_supply,
        "T_return_C": T_return,
        "roughness_mm": roughness * 1e3,
        "n_active_edges": len(active),
        "total_pipe_length_km": active["Length"].sum() / 1e3,
        "dp_critical_path_kPa": dp_critical / 1e3,
        "V_dot_total_m3_s": V_dot_total,
        # Power values
        "P_pump_design_kW": P_design_W / 1e3,
        "P_pump_dissipated_kW": P_dissipated_W / 1e3,
        "thermal_losses_kW": thermal_losses_W / 1e3,
        "heat_delivered_MW": heat_delivered_W / 1e6,
        # Annual energy values
        "E_pump_design_MWh": E_pump_design_MWh,
        "E_pump_dissipated_MWh": E_pump_dissip_MWh,
        "E_thermal_loss_MWh": E_thermal_loss_MWh,
        "E_delivered_MWh": E_delivered_MWh,
        "E_total_MWh": E_pump_design_MWh + E_thermal_loss_MWh + E_delivered_MWh,
        # Fractions (identical whether computed from power or energy)
        "fraction_pump_design_pct": P_design_W / total_design * 100,
        "fraction_pump_dissipated_pct": P_dissipated_W / total_dissip * 100,
        "fraction_thermal_loss_pct": thermal_losses_W / total_design * 100,
        "fraction_delivered_pct": heat_delivered_W / total_design * 100,
        # Flow diagnostics
        "max_velocity_m_s": active["velocity_m_s"].max(),
        "mean_velocity_m_s": active["velocity_m_s"].mean(),
        "pct_turbulent": (
            (active["flow_regime_supply"].astype(str) == "turbulent").sum()
            / len(active) * 100
        ),
        "pct_laminar": (
            (active["flow_regime_supply"].astype(str) == "laminar").sum()
            / len(active) * 100
        ),
    }

    return active, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_summaries = []

    for material in MATERIALS:
        out_dir = PUMPING_LOSSES_DIR / material
        out_dir.mkdir(parents=True, exist_ok=True)

        for scenario in SCENARIOS:
            print(f"\n{'=' * 70}")
            print(f"  {scenario.upper()} — {material} (roughness {PIPE_ROUGHNESS[material]*1e3:.3f} mm)")
            print(f"{'=' * 70}")

            edges_df, summary = compute_scenario(scenario, material)
            all_summaries.append(summary)

            # Save per-edge results
            export_cols = [
                "u", "v", "MW", "Diameter", "Length",
                "Losses [W]", "mass_flow_kg_s", "velocity_m_s",
                "Re_supply", "Re_return", "f_supply", "f_return",
                "dp_supply_Pa", "dp_return_Pa", "dp_total_Pa",
                "dp_minor_supply_Pa", "dp_minor_return_Pa",
                "flow_regime_supply", "flow_regime_return",
                "V_dot_m3_s",
            ]
            cols_present = [c for c in export_cols if c in edges_df.columns]
            edges_df[cols_present].to_csv(
                out_dir / f"{scenario}_edge_hydraulics.csv", index=False,
            )

            # Print summary
            s = summary
            print(f"  Supply/Return:       {s['T_supply_C']}/{s['T_return_C']} C")
            print(f"  Active edges:        {s['n_active_edges']}")
            print(f"  Total pipe length:   {s['total_pipe_length_km']:.1f} km")
            print(f"  Critical path dp:    {s['dp_critical_path_kPa']:.1f} kPa")
            print(f"  Volume flow (total): {s['V_dot_total_m3_s']:.4f} m3/s")
            print(f"  --- Annual energy ({ANNUAL_HOURS} h) ---")
            print(f"  Heat delivered:      {s['E_delivered_MWh']:>10,.0f} MWh/yr  ({s['fraction_delivered_pct']:.2f}%)")
            print(f"  Thermal losses:      {s['E_thermal_loss_MWh']:>10,.0f} MWh/yr  ({s['fraction_thermal_loss_pct']:.2f}%)")
            print(f"  Pump (design):       {s['E_pump_design_MWh']:>10,.0f} MWh/yr  ({s['fraction_pump_design_pct']:.3f}%)")
            print(f"  Pump (dissipated):   {s['E_pump_dissipated_MWh']:>10,.0f} MWh/yr  ({s['fraction_pump_dissipated_pct']:.3f}%)")
            print(f"  Total (design):      {s['E_total_MWh']:>10,.0f} MWh/yr")
            print(f"  --- Flow diagnostics ---")
            print(f"  Max velocity:        {s['max_velocity_m_s']:.2f} m/s")
            print(f"  Mean velocity:       {s['mean_velocity_m_s']:.2f} m/s")
            print(f"  Turbulent edges:     {s['pct_turbulent']:.1f}%")
            print(f"  Laminar edges:       {s['pct_laminar']:.1f}%")

    # Cross-scenario summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(PUMPING_LOSSES_DIR / "pumping_losses_summary.csv", index=False)

    print(f"\n\n{'=' * 90}")
    print("  CROSS-SCENARIO SUMMARY")
    print(f"{'=' * 90}")
    display_cols = [
        "scenario", "material",
        "E_delivered_MWh", "E_thermal_loss_MWh",
        "E_pump_design_MWh", "E_pump_dissipated_MWh", "E_total_MWh",
        "fraction_pump_design_pct", "fraction_thermal_loss_pct",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nResults saved to {PUMPING_LOSSES_DIR}/")


if __name__ == "__main__":
    main()