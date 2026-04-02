"""Unit tests for utils.hydraulics — water properties, friction factors, pressure drop."""

import math

import numpy as np
import pytest

from utils.hydraulics import (
    compute_edge_hydraulics,
    flow_velocity,
    friction_factor_colebrook,
    friction_factor_swamee_jain,
    mass_flow_rate,
    pipe_hydraulics,
    pressure_drop_darcy_weisbach,
    pressure_drop_minor,
    pumping_power,
    reynolds_number,
    water_density,
    water_specific_heat,
    water_viscosity,
)


# ---------------------------------------------------------------------------
# Water properties against NIST reference values
# ---------------------------------------------------------------------------

class TestWaterDensity:
    def test_at_4C_maximum(self):
        assert water_density(4.0) == pytest.approx(1000.0, rel=1e-3)

    def test_at_20C(self):
        assert water_density(20.0) == pytest.approx(998.2, rel=1e-3)

    def test_at_50C(self):
        assert water_density(50.0) == pytest.approx(988.0, rel=2e-3)

    def test_at_90C(self):
        assert water_density(90.0) == pytest.approx(965.3, rel=2e-3)

    def test_monotonic_decrease_above_4C(self):
        temps = [10, 20, 40, 60, 80, 100]
        densities = [water_density(T) for T in temps]
        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1]


class TestWaterViscosity:
    def test_at_20C(self):
        assert water_viscosity(20.0) == pytest.approx(1.002e-3, rel=0.05)

    def test_at_50C(self):
        assert water_viscosity(50.0) == pytest.approx(0.547e-3, rel=0.05)

    def test_at_90C(self):
        assert water_viscosity(90.0) == pytest.approx(0.315e-3, rel=0.05)

    def test_monotonic_decrease(self):
        temps = [10, 30, 50, 70, 90]
        viscosities = [water_viscosity(T) for T in temps]
        for i in range(len(viscosities) - 1):
            assert viscosities[i] > viscosities[i + 1]


class TestWaterSpecificHeat:
    def test_near_constant(self):
        for T in [20, 50, 90]:
            cp = water_specific_heat(T)
            assert 4.15 < cp < 4.25


# ---------------------------------------------------------------------------
# Flow quantities
# ---------------------------------------------------------------------------

class TestMassFlowRate:
    def test_1MW_25K(self):
        # 1 MW, cp=4.18, DT=25 => m_dot = 1e6 / (4180 * 25) = 9.569 kg/s
        m = mass_flow_rate(1.0, 4.18, 25.0)
        assert m == pytest.approx(9.569, rel=1e-3)

    def test_zero_power(self):
        assert mass_flow_rate(0.0, 4.18, 25.0) == 0.0


class TestFlowVelocity:
    def test_known_case(self):
        # m_dot=10 kg/s, rho=1000, D=0.1m => A = pi*0.05^2 = 7.854e-3
        # v = 10 / (1000 * 7.854e-3) = 1.273 m/s
        v = flow_velocity(10.0, 1000.0, 0.1)
        assert v == pytest.approx(1.273, rel=1e-3)


class TestReynoldsNumber:
    def test_known_case(self):
        # rho=1000, v=1.0, D=0.1, mu=1e-3 => Re = 100000
        Re = reynolds_number(1000.0, 1.0, 0.1, 1e-3)
        assert Re == pytest.approx(1e5, rel=1e-6)


# ---------------------------------------------------------------------------
# Friction factor
# ---------------------------------------------------------------------------

class TestFrictionFactorLaminar:
    def test_re_1000(self):
        f = friction_factor_swamee_jain(1000.0, 0.045e-3, 0.1)
        assert f == pytest.approx(0.064, rel=1e-6)

    def test_re_500(self):
        f = friction_factor_swamee_jain(500.0, 0.045e-3, 0.1)
        assert f == pytest.approx(0.128, rel=1e-6)

    def test_colebrook_agrees_laminar(self):
        f_sj = friction_factor_swamee_jain(1500.0, 0.045e-3, 0.1)
        f_cb = friction_factor_colebrook(1500.0, 0.045e-3, 0.1)
        assert f_sj == pytest.approx(f_cb, rel=1e-6)


class TestFrictionFactorTurbulent:
    def test_smooth_pipe_re_1e5(self):
        # Smooth pipe (epsilon -> 0): Blasius gives f ~ 0.0185 at Re=1e5
        f = friction_factor_swamee_jain(1e5, 1e-7, 0.1)
        assert 0.017 < f < 0.020

    def test_rough_pipe_re_1e5(self):
        # epsilon/D = 0.001: Moody chart gives f ~ 0.0222
        f = friction_factor_swamee_jain(1e5, 0.0001, 0.1)
        assert f == pytest.approx(0.0222, rel=0.05)

    def test_swamee_jain_vs_colebrook_agreement(self):
        test_cases = [
            (1e4, 0.045e-3, 0.1),
            (1e5, 0.045e-3, 0.2),
            (1e6, 0.5e-3, 0.5),
            (5e4, 0.007e-3, 0.05),
        ]
        for Re, eps, D in test_cases:
            f_sj = friction_factor_swamee_jain(Re, eps, D)
            f_cb = friction_factor_colebrook(Re, eps, D)
            assert f_sj == pytest.approx(f_cb, rel=0.02), (
                f"Re={Re}, eps={eps}, D={D}: SJ={f_sj:.6f}, CB={f_cb:.6f}"
            )


class TestFrictionFactorTransitional:
    def test_continuity_at_2300(self):
        f_below = friction_factor_swamee_jain(2299.0, 0.045e-3, 0.1)
        f_above = friction_factor_swamee_jain(2301.0, 0.045e-3, 0.1)
        assert abs(f_below - f_above) < 0.002

    def test_continuity_at_4000(self):
        f_below = friction_factor_swamee_jain(3999.0, 0.045e-3, 0.1)
        f_above = friction_factor_swamee_jain(4001.0, 0.045e-3, 0.1)
        assert abs(f_below - f_above) < 0.002


class TestFrictionFactorEdgeCases:
    def test_zero_Re(self):
        assert friction_factor_swamee_jain(0.0, 0.045e-3, 0.1) == 0.0

    def test_negative_Re(self):
        assert friction_factor_swamee_jain(-100.0, 0.045e-3, 0.1) == 0.0


# ---------------------------------------------------------------------------
# Pressure drop
# ---------------------------------------------------------------------------

class TestPressureDropDarcyWeisbach:
    def test_known_case(self):
        # f=0.02, L=100m, D=0.1m, rho=1000, v=1.0
        # dp = 0.02 * (100/0.1) * (1000*1^2/2) = 0.02 * 1000 * 500 = 10000 Pa
        dp = pressure_drop_darcy_weisbach(0.02, 100.0, 0.1, 1000.0, 1.0)
        assert dp == pytest.approx(10000.0, rel=1e-6)


class TestPressureDropMinor:
    def test_single_bend(self):
        # K=0.3, rho=1000, v=2.0 => dp = 0.3 * 1000 * 4 / 2 = 600 Pa
        dp = pressure_drop_minor(0.3, 1000.0, 2.0)
        assert dp == pytest.approx(600.0, rel=1e-6)


class TestPumpingPower:
    def test_known_case(self):
        # dp=100000 Pa, V_dot=0.01 m3/s, eta=0.9*0.8=0.72
        # P = 100000 * 0.01 / 0.72 = 1388.9 W
        P = pumping_power(100000.0, 0.01, 0.90, 0.80)
        assert P == pytest.approx(1388.9, rel=1e-3)


# ---------------------------------------------------------------------------
# Convenience function: pipe_hydraulics
# ---------------------------------------------------------------------------

class TestPipeHydraulics:
    def test_returns_all_keys(self):
        result = pipe_hydraulics(1.0, 0.1, 100.0, 50.0, 0.045e-3)
        expected_keys = {
            "mass_flow_kg_s", "velocity_m_s", "rho_kg_m3", "mu_Pa_s",
            "Re", "f", "dp_friction_Pa", "dp_minor_Pa", "dp_total_Pa",
            "flow_regime",
        }
        assert set(result.keys()) == expected_keys

    def test_turbulent_regime_for_main_pipe(self):
        result = pipe_hydraulics(5.0, 0.3, 200.0, 90.0, 0.045e-3)
        assert result["flow_regime"] == "turbulent"

    def test_velocity_sanity(self):
        # Typical DH: velocities should be 0.1 - 5 m/s
        result = pipe_hydraulics(1.0, 0.1, 100.0, 50.0, 0.045e-3)
        assert 0.1 < result["velocity_m_s"] < 5.0

    def test_dp_total_is_sum(self):
        result = pipe_hydraulics(1.0, 0.1, 100.0, 50.0, 0.045e-3)
        assert result["dp_total_Pa"] == pytest.approx(
            result["dp_friction_Pa"] + result["dp_minor_Pa"], rel=1e-10
        )

    def test_colebrook_method(self):
        r_sj = pipe_hydraulics(1.0, 0.1, 100.0, 50.0, 0.045e-3, friction_method="swamee_jain")
        r_cb = pipe_hydraulics(1.0, 0.1, 100.0, 50.0, 0.045e-3, friction_method="colebrook")
        assert r_sj["dp_total_Pa"] == pytest.approx(r_cb["dp_total_Pa"], rel=0.02)


# ---------------------------------------------------------------------------
# Vectorised convenience function
# ---------------------------------------------------------------------------

class TestComputeEdgeHydraulics:
    def test_basic_array(self):
        result = compute_edge_hydraulics(
            Q_MW_array=[1.0, 0.5],
            D_array=[0.1, 0.08],
            L_total_array=[200.0, 100.0],  # doubled lengths
            T_supply_C=90.0,
            T_return_C=65.0,
            roughness_m=0.045e-3,
        )
        assert len(result["dp_total_Pa"]) == 2
        assert all(result["dp_total_Pa"] > 0)

    def test_zero_flow_edge(self):
        result = compute_edge_hydraulics(
            Q_MW_array=[0.0],
            D_array=[0.1],
            L_total_array=[200.0],
            T_supply_C=50.0,
            T_return_C=25.0,
            roughness_m=0.045e-3,
        )
        assert result["dp_total_Pa"][0] == 0.0
        assert result["flow_regime_supply"][0] == "none"

    def test_supply_return_different_Re(self):
        result = compute_edge_hydraulics(
            Q_MW_array=[2.0],
            D_array=[0.15],
            L_total_array=[400.0],
            T_supply_C=90.0,
            T_return_C=65.0,
            roughness_m=0.045e-3,
        )
        # Higher temperature → lower viscosity → higher Re
        assert result["Re_supply"][0] > result["Re_return"][0]