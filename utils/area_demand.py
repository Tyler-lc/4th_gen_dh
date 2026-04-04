"""Compute area-level hourly demand by summing building-level CSVs.

Replaces the pre-computed area_results CSVs with on-the-fly aggregation
from the buildingstock parquet and per-building hourly CSV files.  This
ensures the NFA filter is applied consistently and avoids stale-cache
bugs (e.g. 03_renovate_buildingstock.py only summing renovated buildings).

The returned DataFrames match the column layout of the original CSVs
so that downstream derivation code (total_useful_demand, delivered_energy,
hourly heat generated, etc.) can remain unchanged.
"""

import numpy as np
import pandas as pd
import geopandas as gpd

from config import (
    buildingstock_results_path,
    booster_buildingstock_results_path,
    results_dir,
)


def compute_area_demand(sim: str, nfa_min: float = 30) -> pd.DataFrame:
    """Aggregate hourly demand for unrenovated or renovated scenarios.

    Drop-in replacement for ``pd.read_csv(area_results_path(sim))``.

    Parameters
    ----------
    sim : str
        ``"unrenovated"`` or ``"renovated"``.
    nfa_min : float
        Minimum Net Floor Area threshold (m²).  Buildings below this
        are excluded.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (8760 rows), columns:
        ``dhw_energy``, ``space_heating``.
    """
    bs = gpd.read_parquet(buildingstock_results_path(sim))
    bs = bs[bs["NFA"] >= nfa_min]

    base = results_dir(sim)
    sh_dir = base / "space_heating"
    dhw_e_dir = base / "dhw_energy"

    total_sh = None
    total_dhw_e = None

    for _, row in bs.iterrows():
        fid = row["full_id"]

        sh = pd.read_csv(sh_dir / f"space_heating_{fid}.csv", index_col=0).iloc[:, 0].values
        dhw_e = pd.read_csv(dhw_e_dir / f"dhw_energy_{fid}.csv", index_col=0).iloc[:, 0].values

        if total_sh is None:
            total_sh = sh.astype(float)
            total_dhw_e = dhw_e.astype(float)
        else:
            total_sh += sh
            total_dhw_e += dhw_e

    # Build timestamp index from first CSV
    sample = pd.read_csv(
        sh_dir / f"space_heating_{bs.iloc[0]['full_id']}.csv", index_col=0
    )
    index = pd.to_datetime(sample.index)

    df = pd.DataFrame(index=index)
    df["dhw_energy"] = total_dhw_e
    df["space_heating"] = total_sh

    return df


def compute_booster_area_demand(nfa_min: float = 30) -> pd.DataFrame:
    """Aggregate hourly demand for the booster scenario.

    Drop-in replacement for ``pd.read_csv(booster_area_results_path())``.

    Parameters
    ----------
    nfa_min : float
        Minimum Net Floor Area threshold (m²).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (8760 rows), columns:
        ``area grid demand [kWh]``,
        ``area total boosters demand [kWh]``,
        ``area space heating demand [kWh]``,
        ``area dhw energy demand [kWh]``.
    """
    bs = gpd.read_parquet(booster_buildingstock_results_path())
    bs = bs[bs["NFA"] >= nfa_min]

    base = results_dir("booster")
    sh_booster_dir = base / "space_heating_booster"
    # Space heating and DHW are the same as unrenovated (booster doesn't renovate)
    unr_base = results_dir("unrenovated")
    sh_dir = unr_base / "space_heating"
    dhw_e_dir = unr_base / "dhw_energy"

    total_grid_demand = None
    total_el_demand = None
    total_sh = None
    total_dhw_e = None

    for _, row in bs.iterrows():
        fid = row["full_id"]

        # Booster per-building CSVs
        bdf = pd.read_csv(sh_booster_dir / f"{fid}_booster.csv", index_col=0)
        grid_demand = bdf["demand_on_dh_grid [kWh]"].values
        el_demand = bdf["el_demand [kWh]"].values

        # Space heating (unrenovated)
        sh = pd.read_csv(sh_dir / f"space_heating_{fid}.csv", index_col=0).iloc[:, 0].values

        # DHW
        dhw_e = pd.read_csv(dhw_e_dir / f"dhw_energy_{fid}.csv", index_col=0).iloc[:, 0].values

        if total_grid_demand is None:
            total_grid_demand = grid_demand.astype(float)
            total_el_demand = el_demand.astype(float)
            total_sh = sh.astype(float)
            total_dhw_e = dhw_e.astype(float)
        else:
            total_grid_demand += grid_demand
            total_el_demand += el_demand
            total_sh += sh
            total_dhw_e += dhw_e

    # Build timestamp index
    sample = pd.read_csv(
        sh_booster_dir / f"{bs.iloc[0]['full_id']}_booster.csv", index_col=0
    )
    index = pd.to_datetime(sample.index)

    df = pd.DataFrame(index=index)
    df["area grid demand [kWh]"] = total_grid_demand
    df["area total boosters demand [kWh]"] = total_el_demand
    df["area space heating demand [kWh]"] = total_sh
    df["area dhw energy demand [kWh]"] = total_dhw_e

    return df
