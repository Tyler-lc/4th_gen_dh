"""Hydraulic calculations for district heating pipe networks.

Pure functions — no file I/O, no imports from config.  All parameters are
passed explicitly so the module remains testable and reusable.

Physics chain:
    thermal power → mass flow → velocity → Reynolds → friction factor
    → pressure drop (Darcy-Weisbach + minor losses) → pumping power

References:
    - Swamee & Jain (1976), J. Hydraul. Div. ASCE, 102(5), 657-664.
    - IAPWS-IF97 for water properties (simplified polynomial fits).
    - Vogel equation for dynamic viscosity.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Water properties (temperature-dependent)
# ---------------------------------------------------------------------------

def water_density(T_celsius):
    """Density of liquid water [kg/m^3] at temperature *T_celsius* [C].

    Tanaka et al. (2001), Metrologia 38(4), 301-309.
    Valid 0-100 C, error < 0.001 kg/m^3.
    """
    T = float(T_celsius)
    a1 = -3.983035
    a2 = 301.797
    a3 = 522528.9
    a4 = 69.34881
    a5 = 999.97495
    return a5 * (1.0 - (T + a1) ** 2 * (T + a2) / (a3 * (T + a4)))


def water_viscosity(T_celsius):
    """Dynamic viscosity of liquid water [Pa.s] at temperature *T_celsius* [C].

    Vogel equation: mu = A * 10^(B / (T_K - C)).
    Coefficients from Al-Shemmeri (2012).  Valid 0-100 C, error < 1%.
    """
    T_K = float(T_celsius) + 273.15
    A = 2.414e-5   # Pa.s
    B = 247.8       # K
    C = 140.0       # K
    return A * 10.0 ** (B / (T_K - C))


_CP_TABLE_T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_CP_TABLE_V = np.array(
    [4.2176, 4.1921, 4.1818, 4.1784, 4.1785, 4.1806,
     4.1843, 4.1895, 4.1963, 4.2053, 4.2157], dtype=float,
)


def water_specific_heat(T_celsius):
    """Specific heat of liquid water [kJ/(kg.K)] at *T_celsius* [C].

    Linear interpolation of NIST reference data.  Valid 0-100 C.
    """
    return float(np.interp(T_celsius, _CP_TABLE_T, _CP_TABLE_V))


# ---------------------------------------------------------------------------
# Flow quantities
# ---------------------------------------------------------------------------

def mass_flow_rate(Q_thermal_MW, cp_kJ_kgK, delta_T_K):
    """Mass flow rate [kg/s] from thermal power.

    m_dot = Q [W] / (cp [J/(kg·K)] * DT [K])
    """
    Q_W = Q_thermal_MW * 1.0e6
    cp_J = cp_kJ_kgK * 1.0e3
    return Q_W / (cp_J * delta_T_K)


def flow_velocity(m_dot_kg_s, rho_kg_m3, D_m):
    """Flow velocity [m/s] from mass flow rate, density and pipe diameter."""
    A = math.pi * (D_m / 2.0) ** 2
    return m_dot_kg_s / (rho_kg_m3 * A)


def reynolds_number(rho, v, D, mu):
    """Reynolds number [-]."""
    return rho * v * D / mu


# ---------------------------------------------------------------------------
# Friction factor
# ---------------------------------------------------------------------------

def friction_factor_swamee_jain(Re, epsilon, D):
    """Darcy friction factor via the Swamee-Jain (1976) explicit approximation.

    Handles three flow regimes:
        - Laminar  (Re < 2300):  f = 64 / Re
        - Transitional (2300 <= Re <= 4000): linear interpolation
        - Turbulent (Re > 4000):  Swamee-Jain formula

    Swamee-Jain validity: 5 x 10^3 <= Re <= 10^8,  10^-6 <= epsilon/D <= 10^-2.
    Accuracy: within 1% of the iterative Colebrook-White solution.
    """
    Re = float(Re)
    if Re <= 0:
        return 0.0
    if Re < 2300.0:
        return 64.0 / Re

    rel_rough = epsilon / D

    def _turbulent(Re_val):
        log_arg = rel_rough / 3.7 + 5.74 / Re_val**0.9
        return 0.25 / (math.log10(log_arg)) ** 2

    if Re > 4000.0:
        return _turbulent(Re)

    # Transitional: linear interpolation between laminar at 2300 and turbulent at 4000
    f_lam = 64.0 / 2300.0
    f_turb = _turbulent(4000.0)
    alpha = (Re - 2300.0) / (4000.0 - 2300.0)
    return (1.0 - alpha) * f_lam + alpha * f_turb


def friction_factor_colebrook(Re, epsilon, D, tol=1e-8, max_iter=50):
    """Darcy friction factor via iterative Colebrook-White solution.

    Fixed-point iteration on: 1/sqrt(f) = -2 log10(epsilon/(3.7D) + 2.51/(Re sqrt(f))).
    Falls back to 64/Re for laminar flow.
    """
    Re = float(Re)
    if Re <= 0:
        return 0.0
    if Re < 2300.0:
        return 64.0 / Re

    rel_rough = epsilon / D

    if Re <= 4000.0:
        f_lam = 64.0 / 2300.0
        f_turb_guess = friction_factor_colebrook(4000.0, epsilon, D, tol, max_iter)
        alpha = (Re - 2300.0) / (4000.0 - 2300.0)
        return (1.0 - alpha) * f_lam + alpha * f_turb_guess

    # Initial guess from Swamee-Jain
    f = friction_factor_swamee_jain(Re, epsilon, D)

    for _ in range(max_iter):
        rhs = -2.0 * math.log10(rel_rough / 3.7 + 2.51 / (Re * math.sqrt(f)))
        f_new = 1.0 / rhs**2
        if abs(f_new - f) < tol:
            return f_new
        f = f_new

    return f


# ---------------------------------------------------------------------------
# Pressure drop
# ---------------------------------------------------------------------------

def pressure_drop_darcy_weisbach(f, L, D, rho, v):
    """Pressure drop [Pa] from Darcy-Weisbach: dp = f * (L/D) * (rho * v^2 / 2)."""
    return f * (L / D) * (rho * v**2 / 2.0)


def pressure_drop_minor(K, rho, v):
    """Minor-loss pressure drop [Pa]: dp = K * (rho * v^2 / 2)."""
    return K * (rho * v**2 / 2.0)


# ---------------------------------------------------------------------------
# Pumping power
# ---------------------------------------------------------------------------

def pumping_power(dp_total_Pa, V_dot_m3_s, eta_electric, eta_hydraulic):
    """Electrical pumping power [W].

    P_elec = (dp * V_dot) / (eta_elec * eta_hyd)
    """
    return dp_total_Pa * V_dot_m3_s / (eta_electric * eta_hydraulic)


# ---------------------------------------------------------------------------
# Convenience: all hydraulics for a single pipe
# ---------------------------------------------------------------------------

def pipe_hydraulics(
    Q_thermal_MW,
    D_m,
    L_oneway_m,
    T_water_C,
    roughness_m,
    n_bends=1,
    K_bend=0.3,
    cp_kJ_kgK=4.18,
    delta_T_K=25.0,
    friction_method="swamee_jain",
):
    """Compute all hydraulic quantities for a single pipe (supply OR return).

    Parameters
    ----------
    Q_thermal_MW : float
        Thermal power transported [MW].
    D_m : float
        Internal pipe diameter [m].
    L_oneway_m : float
        One-way pipe length [m] (not the doubled trench length).
    T_water_C : float
        Water temperature in this pipe [C] (supply_temp or return_temp).
    roughness_m : float
        Absolute pipe roughness [m].
    n_bends : int
        Number of 90-degree bends in this pipe segment.
    K_bend : float
        Loss coefficient per bend.
    cp_kJ_kgK : float
        Specific heat capacity used for mass-flow back-calculation [kJ/(kg.K)].
        Default 4.18 matches the grid calculation scripts.
    delta_T_K : float
        Temperature difference supply - return [K].
    friction_method : str
        ``"swamee_jain"`` (default) or ``"colebrook"``.

    Returns
    -------
    dict with keys: mass_flow_kg_s, velocity_m_s, Re, f, dp_friction_Pa,
    dp_minor_Pa, dp_total_Pa, flow_regime
    """
    rho = water_density(T_water_C)
    mu = water_viscosity(T_water_C)

    m_dot = mass_flow_rate(Q_thermal_MW, cp_kJ_kgK, delta_T_K)
    v = flow_velocity(m_dot, rho, D_m)
    Re = reynolds_number(rho, v, D_m, mu)

    if friction_method == "colebrook":
        f = friction_factor_colebrook(Re, roughness_m, D_m)
    else:
        f = friction_factor_swamee_jain(Re, roughness_m, D_m)

    dp_friction = pressure_drop_darcy_weisbach(f, L_oneway_m, D_m, rho, v)
    dp_minor = pressure_drop_minor(n_bends * K_bend, rho, v)
    dp_total = dp_friction + dp_minor

    if Re < 2300:
        regime = "laminar"
    elif Re <= 4000:
        regime = "transitional"
    else:
        regime = "turbulent"

    return {
        "mass_flow_kg_s": m_dot,
        "velocity_m_s": v,
        "rho_kg_m3": rho,
        "mu_Pa_s": mu,
        "Re": Re,
        "f": f,
        "dp_friction_Pa": dp_friction,
        "dp_minor_Pa": dp_minor,
        "dp_total_Pa": dp_total,
        "flow_regime": regime,
    }


# ---------------------------------------------------------------------------
# Vectorised helpers for DataFrame operations
# ---------------------------------------------------------------------------

def compute_edge_hydraulics(
    Q_MW_array,
    D_array,
    L_total_array,
    T_supply_C,
    T_return_C,
    roughness_m,
    n_bends_per_pipe=1,
    K_bend=0.3,
    cp_kJ_kgK=4.18,
):
    """Compute hydraulics for arrays of pipe edges (both supply and return).

    *L_total_array* is the doubled length stored in the grid parquet (supply +
    return trench).  Internally split into L/2 for each pipe.

    Returns a dict of numpy arrays, one entry per edge.
    """
    n = len(Q_MW_array)
    L_oneway = np.asarray(L_total_array, dtype=float) / 2.0
    Q = np.asarray(Q_MW_array, dtype=float)
    D = np.asarray(D_array, dtype=float)
    delta_T = T_supply_C - T_return_C

    keys = [
        "mass_flow_kg_s", "velocity_m_s",
        "Re_supply", "Re_return", "f_supply", "f_return",
        "dp_supply_Pa", "dp_return_Pa", "dp_total_Pa",
        "dp_minor_supply_Pa", "dp_minor_return_Pa",
        "flow_regime_supply", "flow_regime_return",
    ]
    result = {k: np.empty(n) if "regime" not in k else [""] * n for k in keys}

    for i in range(n):
        if Q[i] <= 0 or D[i] <= 0:
            for k in keys:
                if "regime" in k:
                    result[k][i] = "none"
                else:
                    result[k][i] = 0.0
            continue

        supply = pipe_hydraulics(
            Q[i], D[i], L_oneway[i], T_supply_C, roughness_m,
            n_bends=n_bends_per_pipe, K_bend=K_bend,
            cp_kJ_kgK=cp_kJ_kgK, delta_T_K=delta_T,
        )
        ret = pipe_hydraulics(
            Q[i], D[i], L_oneway[i], T_return_C, roughness_m,
            n_bends=n_bends_per_pipe, K_bend=K_bend,
            cp_kJ_kgK=cp_kJ_kgK, delta_T_K=delta_T,
        )

        result["mass_flow_kg_s"][i] = supply["mass_flow_kg_s"]
        result["velocity_m_s"][i] = supply["velocity_m_s"]
        result["Re_supply"][i] = supply["Re"]
        result["Re_return"][i] = ret["Re"]
        result["f_supply"][i] = supply["f"]
        result["f_return"][i] = ret["f"]
        result["dp_supply_Pa"][i] = supply["dp_total_Pa"]
        result["dp_return_Pa"][i] = ret["dp_total_Pa"]
        result["dp_total_Pa"][i] = supply["dp_total_Pa"] + ret["dp_total_Pa"]
        result["dp_minor_supply_Pa"][i] = supply["dp_minor_Pa"]
        result["dp_minor_return_Pa"][i] = ret["dp_minor_Pa"]
        result["flow_regime_supply"][i] = supply["flow_regime"]
        result["flow_regime_return"][i] = ret["flow_regime"]

    return result