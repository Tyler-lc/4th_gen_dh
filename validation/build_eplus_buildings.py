"""Build synthetic buildings matching the EnergyPlus study constructions.

Uses exact geometry from Table A1 and construction layers from Tables A2-A5
of Casamassima & Kranzl (2024) to create building component DataFrames
that can be fed directly into the Building class.

U-values are calculated from the layer data (U = 1 / (Rsi + Σ(d/λ) + Rse)).
"""

import numpy as np
import pandas as pd
import math


# --- Surface resistances (ISO 6946) -----------------------------------------
RSI_WALL = 0.13    # m²K/W — internal, vertical
RSE_WALL = 0.04    # m²K/W — external, vertical
RSI_ROOF = 0.10    # m²K/W — internal, upward heat flow
RSE_ROOF = 0.04    # m²K/W — external
RSI_FLOOR = 0.17   # m²K/W — internal, downward heat flow
RSE_FLOOR = 0.04   # m²K/W — external (or ground)


def calc_u_value(layers: list[tuple[float, float]], rsi: float, rse: float) -> float:
    """Calculate U-value from layers [(thickness_m, lambda_W_mK), ...].

    For air gaps, pass lambda as None and thickness as thermal resistance directly.
    """
    R = rsi + rse
    for d, lam in layers:
        if lam is None:
            R += d  # d is already a thermal resistance
        else:
            R += d / lam
    return 1.0 / R


# =============================================================================
# AB 1945-1957 — Table A2
# Geometry: L=22, D=12, H_below_ground=2, H_above_ground=15.5, H_total=17.5
# No pitched roof. WWR = 0.231
# =============================================================================
ab3_walls = [
    (0.01, 1.163),   # Plaster
    (0.02, 0.033),   # Mineral Wool (ADDED insulation)
    (0.2, 0.68),     # Hollow Bricks
    (0.01, 1.163),   # Plaster
]
ab3_roof = [
    (0.125, 2.3),    # Reinforced Concrete
    (0.006, 0.16),   # PVC Membrane
    (0.02, 0.033),   # Mineral Wool (ADDED insulation)
    (0.01, 1.163),   # Plaster
]
ab3_floor = [
    (0.125, 2.3),    # Reinforced Concrete
    (0.006, 0.16),   # PVC Membrane
    (0.02, 0.033),   # Mineral Wool
    (0.01, 1.163),   # Plaster
]
ab3_window_u = 1 / (0.13 + 0.003/0.9 + 0.012/0.025 + 0.003/0.9 + 0.04)
# air gap thermal resistance ~0.012/0.025 is approximate; for 12mm air gap
# standard value is ~0.15 m²K/W (ISO 10077-1)
# Let's use the standard double-glazing value
ab3_window_u = 1 / (0.13 + 0.003/0.9 + 0.15 + 0.003/0.9 + 0.04)

# AB geometry
ab3_L, ab3_D = 22, 12
ab3_H_above = 15.5  # above ground
ab3_H_below = 2.0
ab3_H_total = 17.5
ab3_n_floors_above = 6  # ground floor + 5 upper floors (confirmed from E+ model)
ab3_ceiling_h = ab3_H_above / ab3_n_floors_above  # ~2.58m
ab3_GFA = ab3_L * ab3_D * ab3_n_floors_above  # only above-ground floors count
ab3_NFA = ab3_GFA * 0.8  # standard 80% net-to-gross ratio

ab3_wall_area = 2 * (ab3_L + ab3_D) * ab3_H_above  # above-ground walls only
ab3_roof_area = ab3_L * ab3_D  # flat roof
ab3_floor_area = ab3_L * ab3_D  # ground contact
ab3_WWR = 0.231
ab3_window_area = ab3_wall_area * ab3_WWR
ab3_door_area = 2.0

# =============================================================================
# MFH 1945-1957 — Table A3
# Geometry: L=23, D=15, H_below_ground=1.7, H_above_ground=5.8, H_total=7.5
# No pitched roof. WWR = 0.225
# =============================================================================
mfh3_walls = [
    (0.01, 1.163),   # Plaster
    (0.35, 0.68),    # Hollow Bricks (NO added insulation)
    (0.01, 1.163),   # Plaster
]
mfh3_roof = [
    (0.125, 2.3),    # Reinforced Concrete
    (0.006, 0.16),   # PVC Membrane
    (0.02, 0.033),   # Mineral Wool
    (0.01, 1.163),   # Plaster
]
mfh3_ceiling = [
    (0.01, 1.6),     # Ceramic Tiles
    (0.005, 0.03),   # Polyurethane
    (0.02, 0.033),   # Mineral Wool
    (0.125, 2.3),    # Reinforced Concrete
    (0.01, 1.163),   # Plaster
]
mfh3_floor = [
    (0.01, 1.163),   # Plaster
    (0.125, 2.3),    # Reinforced Concrete
    (0.02, 0.033),   # Mineral Wool
    (0.005, 0.03),   # Polyurethane
    (0.1, 1.6),      # Ceramic Tiles
]

mfh3_L, mfh3_D = 23, 15
mfh3_H_above = 5.8
mfh3_H_below = 1.7
mfh3_H_total = 7.5
mfh3_n_floors_above = round(mfh3_H_above / 3.0)  # ~2 floors
mfh3_ceiling_h = 3.0
mfh3_GFA = mfh3_L * mfh3_D * mfh3_n_floors_above
mfh3_NFA = mfh3_GFA * 0.8

mfh3_wall_area = 2 * (mfh3_L + mfh3_D) * mfh3_H_above
mfh3_roof_area = mfh3_L * mfh3_D
mfh3_floor_area = mfh3_L * mfh3_D
mfh3_WWR = 0.225
mfh3_window_area = mfh3_wall_area * mfh3_WWR

# =============================================================================
# MFH 1969-1978 — Table A4
# Geometry: L=17, D=10, H_below_ground=0, H_pitched_roof=3, H_above=12.55,
#           H_total=15.55. Pitched roof.  WWR = 0.151
# =============================================================================
mfh5_walls = [
    (0.01, 1.163),   # Plaster
    (0.015, 0.02),   # Phenolic Foam (ADDED insulation)
    (0.4, 0.68),     # Hollow bricks
    (0.01, 1.163),   # Plaster
]
mfh5_roof = [
    (0.125, 2.3),    # Reinforced Concrete
    (0.05, 0.02),    # Phenolic Foam (ADDED — 50mm!)
    (0.01, 1.163),   # Plaster
]
mfh5_floor = [
    (0.01, 1.163),   # Plaster
    (0.125, 2.3),    # Reinforced Concrete
    (0.02, 0.02),    # Phenolic Foam
    (0.005, 0.03),   # Polyurethane
    (0.1, 1.6),      # Floor Tiles
]

mfh5_L, mfh5_D = 17, 10
mfh5_H_wall = 12.55  # wall height (excluding pitched roof)
mfh5_H_pitched = 3.0
mfh5_H_total = 15.55
mfh5_n_floors = round(mfh5_H_wall / 3.0)  # ~4 floors
mfh5_ceiling_h = 3.0
mfh5_GFA = mfh5_L * mfh5_D * mfh5_n_floors
mfh5_NFA = mfh5_GFA * 0.8

mfh5_wall_area = 2 * (mfh5_L + mfh5_D) * mfh5_H_wall
# pitched roof: approximate as hypotenuse * length for both sides
mfh5_roof_slope_len = math.sqrt((mfh5_D / 2) ** 2 + mfh5_H_pitched ** 2)
mfh5_roof_area = 2 * mfh5_roof_slope_len * mfh5_L
mfh5_floor_area = mfh5_L * mfh5_D
mfh5_WWR = 0.151
mfh5_window_area = mfh5_wall_area * mfh5_WWR

# =============================================================================
# TH 1969-1978 — Table A5
# Geometry: L=12, D=9, H_below_ground=2.2, H_pitched_roof=2, H_above=6.4,
#           H_total=10.6. Pitched roof.  WWR = 0.098
# =============================================================================
th5_walls = [
    (0.01, 1.163),   # Plaster
    (0.01, 0.02),    # Phenolic Foam (ADDED — 10mm)
    (0.4, 0.68),     # Hollow Bricks
    (0.01, 1.163),   # Plaster
]
th5_roof = [
    (0.125, 2.3),    # Reinforced Concrete
    (0.05, 0.02),    # Phenolic Foam (ADDED — 50mm!)
    (0.01, 1.163),   # Plaster
]
th5_floor = [
    (0.01, 1.6),     # Ceramic Tiles
    (0.005, 0.03),   # Polyurethane
    (0.02, 0.02),    # Phenolic Foam
    (0.125, 2.3),    # Reinforced Concrete
    (0.01, 1.163),   # Plaster
]

th5_L, th5_D = 12, 9
th5_H_wall = 6.4  # above ground wall height
th5_H_below = 2.2
th5_H_pitched = 2.0
th5_H_total = 10.6
th5_n_floors_above = round(th5_H_wall / 3.0)  # ~2 floors
th5_ceiling_h = 3.0
th5_GFA = th5_L * th5_D * th5_n_floors_above
th5_NFA = th5_GFA * 0.8

th5_wall_area = 2 * (th5_L + th5_D) * th5_H_wall
th5_roof_slope_len = math.sqrt((th5_D / 2) ** 2 + th5_H_pitched ** 2)
th5_roof_area = 2 * th5_roof_slope_len * th5_L
th5_floor_area = th5_L * th5_D
th5_WWR = 0.098
th5_window_area = th5_wall_area * th5_WWR

# =============================================================================
# Calculate all U-values and print summary
# =============================================================================
buildings = {
    "AB 1945-1957 (ab3)": {
        "walls": (ab3_walls, RSI_WALL, RSE_WALL),
        "roof": (ab3_roof, RSI_ROOF, RSE_ROOF),
        "floor": (ab3_floor, RSI_FLOOR, RSE_FLOOR),
        "geometry": {
            "L": ab3_L, "D": ab3_D,
            "wall_area": ab3_wall_area, "roof_area": ab3_roof_area,
            "floor_area": ab3_floor_area, "window_area": ab3_window_area,
            "NFA": ab3_NFA, "GFA": ab3_GFA, "n_floors": ab3_n_floors_above,
            "ceiling_h": ab3_ceiling_h, "WWR": ab3_WWR,
            "H_above": ab3_H_above, "volume": ab3_L * ab3_D * ab3_H_above,
        },
    },
    "MFH 1945-1957 (mfh3)": {
        "walls": (mfh3_walls, RSI_WALL, RSE_WALL),
        "roof": (mfh3_roof, RSI_ROOF, RSE_ROOF),
        "floor": (mfh3_floor, RSI_FLOOR, RSE_FLOOR),
        "geometry": {
            "L": mfh3_L, "D": mfh3_D,
            "wall_area": mfh3_wall_area, "roof_area": mfh3_roof_area,
            "floor_area": mfh3_floor_area,
            "window_area": mfh3_wall_area * mfh3_WWR,
            "NFA": mfh3_NFA, "GFA": mfh3_GFA, "n_floors": mfh3_n_floors_above,
            "ceiling_h": mfh3_ceiling_h, "WWR": mfh3_WWR,
            "H_above": mfh3_H_above, "volume": mfh3_L * mfh3_D * mfh3_H_above,
        },
    },
    "MFH 1969-1978 (mfh5)": {
        "walls": (mfh5_walls, RSI_WALL, RSE_WALL),
        "roof": (mfh5_roof, RSI_ROOF, RSE_ROOF),
        "floor": (mfh5_floor, RSI_FLOOR, RSE_FLOOR),
        "geometry": {
            "L": mfh5_L, "D": mfh5_D,
            "wall_area": mfh5_wall_area, "roof_area": mfh5_roof_area,
            "floor_area": mfh5_floor_area,
            "window_area": mfh5_wall_area * mfh5_WWR,
            "NFA": mfh5_NFA, "GFA": mfh5_GFA, "n_floors": mfh5_n_floors,
            "ceiling_h": mfh5_ceiling_h, "WWR": mfh5_WWR,
            "H_above": mfh5_H_wall, "volume": mfh5_L * mfh5_D * mfh5_H_wall,
        },
    },
    "TH 1969-1978 (th5)": {
        "walls": (th5_walls, RSI_WALL, RSE_WALL),
        "roof": (th5_roof, RSI_ROOF, RSE_ROOF),
        "floor": (th5_floor, RSI_FLOOR, RSE_FLOOR),
        "geometry": {
            "L": th5_L, "D": th5_D,
            "wall_area": th5_wall_area, "roof_area": th5_roof_area,
            "floor_area": th5_floor_area,
            "window_area": th5_wall_area * th5_WWR,
            "NFA": th5_NFA, "GFA": th5_GFA, "n_floors": th5_n_floors_above,
            "ceiling_h": th5_ceiling_h, "WWR": th5_WWR,
            "H_above": th5_H_wall, "volume": th5_L * th5_D * th5_H_wall,
        },
    },
}

# TABULA U-values from archetype_u_values.csv for comparison
TABULA_UVALUES = {
    "AB 1945-1957 (ab3)":  {"roof": 0.645, "walls": 1.400, "floor": 0.770},
    "MFH 1945-1957 (mfh3)": {"roof": 1.276, "walls": 1.700, "floor": 0.770},
    "MFH 1969-1978 (mfh5)": {"roof": 0.508, "walls": 1.200, "floor": 1.080},
    "TH 1969-1978 (th5)":  {"roof": 0.508, "walls": 1.200, "floor": 1.080},
}

print("=" * 70)
print("U-VALUE COMPARISON: E+ constructions vs TABULA archetypes")
print("=" * 70)

for name, bldg in buildings.items():
    print(f"\n--- {name} ---")
    tabula = TABULA_UVALUES[name]
    for component in ["walls", "roof", "floor"]:
        layers, rsi, rse = bldg[component]
        u_eplus = calc_u_value(layers, rsi, rse)
        u_tabula = tabula[component]
        diff = (u_eplus - u_tabula) / u_tabula * 100
        print(f"  {component:6s}: E+={u_eplus:.3f}  TABULA={u_tabula:.3f}  "
              f"diff={diff:+.1f}%")

    g = bldg["geometry"]
    print(f"  Geometry: {g['L']}x{g['D']}m, {g['n_floors']} floors, "
          f"NFA={g['NFA']:.0f} m², wall={g['wall_area']:.0f} m², "
          f"roof={g['roof_area']:.0f} m², WWR={g['WWR']}")


# --- Build component DataFrames for the Building class -----------------------
def build_component_row(name, bldg, window_u=2.7, window_shgc=0.75):
    """Create a one-row DataFrame matching the building_generator output format.

    The Building class expects `windows` as a JSON string with per-cardinal
    area, u_value, and shgc.  We distribute window area evenly across 4 facades.
    """
    import json as _json

    g = bldg["geometry"]
    layers_w, rsi_w, rse_w = bldg["walls"]
    layers_r, rsi_r, rse_r = bldg["roof"]
    layers_f, rsi_f, rse_f = bldg["floor"]

    u_walls = calc_u_value(layers_w, rsi_w, rse_w)
    u_roof = calc_u_value(layers_r, rsi_r, rse_r)
    u_floor = calc_u_value(layers_f, rsi_f, rse_f)

    # distribute windows evenly across 4 facades (simplification)
    win_per_face = g["window_area"] / 4
    net_wall_area = g["wall_area"] - g["window_area"]

    windows_dict = {
        "south": {"area": win_per_face, "u_value": window_u, "shgc": window_shgc},
        "east":  {"area": win_per_face, "u_value": window_u, "shgc": window_shgc},
        "north": {"area": win_per_face, "u_value": window_u, "shgc": window_shgc},
        "west":  {"area": win_per_face, "u_value": window_u, "shgc": window_shgc},
    }

    row = {
        "full_id": f"eplus_{name.split('(')[1].strip(')')}",
        "building_usage": name.split("(")[1].strip(")").rstrip("0123456789"),
        "age_code": int(name.split("(")[1].strip(")").lstrip("abcdefghijklmnopqrstuvwxyz")),
        "roof_area": g["roof_area"],
        "roof_u_value": u_roof,
        "roof_slope": 0,  # flat approximation for QSS
        "walls_area": net_wall_area,
        "walls_u_value": u_walls,
        "ground_contact_area": g["floor_area"],
        "ground_contact_u_value": u_floor,
        "door_area": 2.0,
        "door_u_value": 3.0,
        "windows_area": g["window_area"],
        "windows_shgc": window_shgc,
        "window_u_value": window_u,
        "windows": _json.dumps(windows_dict),  # Building class expects JSON string
        "available_angles": [90, 90, 90, 90],
        "available_cardinals": ["south", "east", "north", "west"],
        "volume": g["volume"],
        "building_height": g["H_above"],
        "ceiling_height": g["ceiling_h"],
        "n_floors": g["n_floors"],
        "GFA": g["GFA"],
        "NFA": g["NFA"],
        "n_people": max(1, int(g["NFA"] / 35)),  # ~35 m² per person
        "people_id": [],
    }
    return pd.DataFrame([row])


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SYNTHETIC BUILDING COMPONENTS")
    print("=" * 70)
    for name, bldg in buildings.items():
        df = build_component_row(name, bldg)
        print(f"\n--- {name} ---")
        for col in ["roof_u_value", "walls_u_value", "ground_contact_u_value",
                     "walls_area", "roof_area", "ground_contact_area",
                     "windows_area", "NFA", "n_floors", "volume"]:
            print(f"  {col}: {df[col].values[0]}")
