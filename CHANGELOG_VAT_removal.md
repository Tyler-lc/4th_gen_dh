# CHANGELOG: Remove VAT from DH Operator Revenue

## Date: 2026-04-01

## Motivation
Reviewer R5-SC5 (ECMX-D-26-00514) flagged that VAT should not appear in an economic analysis.
Supervisor meeting (31 Mar 2026) decided: remove 7% VAT from DH operator revenue because
VAT is a pass-through to government, not real operator income. Residential customers still
pay VAT from their perspective (unchanged). The study adopts a financial analysis from the
perspective of individual agents (building owner, DH operator), not a societal perspective.

## What Changed
- New variable `price_heat_ex_vat` introduced: LCOH * (1 + margin) * reduction_factor (no taxation)
- `operator_selling_price` dict now uses `price_heat_ex_vat` for ALL customer tiers
- Previously, operator_selling_price included (1 + taxation) = 1.07 for all tiers

## What Did NOT Change
- `price_heat_eurokwh_residential`: still includes VAT (customer-facing price)
- `price_heat_eurokwh_non_residential`: still excludes VAT (non-residential don't pay VAT)
- `hp_energy_prices` / `customer_purchasing_price`: unchanged (customer NPV inputs)
- `taxation = 0.07` variable: still defined, still used in customer price formulas
- Customer NPV calculations: fully unchanged
- LCOH values: unchanged (cost-based, not revenue-based)

## Files Modified
1. 05b_HT_Scenario.py
2. 07_LT_Scenario2.py
3. 08_Booster_Scenario.py
4. 09b_HT_Sens_Analysis.py
5. 09c_LT_Sens_Analysis.py
6. 09d_HT_Booster_Sens_Analysis.py
7. 10a_HT_scenarios_gas_vs_electicity.py
8. 10b_LT_scenarios_gas_vs_electicity.py
9. 10c_HT_Booster_gas_vs_electricity.py
10. 10d_LT_scenarios_gas_vs_electicity_vs_rencosts.py

## Before Values (with VAT in operator revenue)

### HT Scenario
- npv_dh_operator: 9,088,273.42
- operator_selling_price: {r0: 0.1951, r1: 0.1590, r2: 0.1464, nr0: 0.1949, nr1: 0.1590, nr2: 0.1464}
- purchasing_price_heat: {r0: 0.1951, r1: 0.1590, r2: 0.1464, nr0: 0.1822, nr1: 0.1486, nr2: 0.1368}

### LT+Reno Scenario
- npv_dh_operator: 39,731,015.54
- operator_selling_price: {r0: 0.1641, r1: 0.1337, r2: 0.1231, nr0: 0.1640, nr1: 0.1337, nr2: 0.1231}
- purchasing_price_heat: {r0: 0.1641, r1: 0.1337, r2: 0.1231, nr0: 0.1532, nr1: 0.1250, nr2: 0.1151}

### Booster Scenario
- npv_dh_operator: 24,082,395.31
- operator_selling_price: {r0: 0.1685, r1: 0.1373, r2: 0.1264, nr0: 0.1684, nr1: 0.1373, nr2: 0.1264}
- purchasing_price_heat: {r0: 0.1685, r1: 0.1373, r2: 0.1264, nr0: 0.1574, nr1: 0.1284, nr2: 0.1182}

## Intermediate Values (VAT removed, old LCOH discounting)
_These values exposed a discounting bug — see below_

### HT Scenario
- npv_dh_operator: -8,721,957.83 (NEGATIVE — triggered investigation)
- operator_selling_price: {r0: 0.1823, r1: 0.1486, r2: 0.1368, nr0: 0.1822, nr1: 0.1486, nr2: 0.1368}
- purchasing_price_heat: {r0: 0.1951, r1: 0.1590, r2: 0.1464, nr0: 0.1822, nr1: 0.1486, nr2: 0.1368} (IDENTICAL to before ✓)

### LT+Reno Scenario
- npv_dh_operator: 31,913,867.26
- purchasing_price_heat: IDENTICAL to before ✓

### Booster Scenario
- npv_dh_operator: 8,697,619.28
- purchasing_price_heat: IDENTICAL to before ✓

---

## LCOH Discounting Fix (discovered 2026-04-01)

### Root Cause
The `calculate_lcoh` function in `costs/heat_supply.py` used `(1 + discount_rate) ** (t)` where
t starts at 0. This means year-0 operational costs and heat output are undiscounted (factor = 1).
But `npv_2` inserts the investment at position 0 and cash flows at positions 1-25, so `npf.npv`
discounts first-year revenue at `(1+r)^1`. This one-year shift means LCOH underestimates the
true break-even price by exactly `1/(1+r) = 4.76%` at r=5%.

### Fix
`costs/heat_supply.py` line 143: `(1 + discount_rate) ** (t)` → `(1 + discount_rate) ** (t + 1)`

### Effect
- All LCOH values increase by ~5%
- HP component of operator NPV: shortfall eliminated (aligned with NPV)
- DHG component: still has 50yr LCOH vs 25yr NPV horizon gap (controlled by residual value, deferred)

### File Modified
- costs/heat_supply.py (single line, propagates to all 10+ scripts via import)

---

## Outstanding Unrecovered Capital (OUC) Residual Value (2026-04-01)

### Motivation
With LCOH_dhg computed over 50 years but NPV evaluated over 25 years, the operator
under-recovers on the DHG investment. The previous fixed residual value (40%) was arbitrary
and did not match the LCOH economics. The OUC method computes the exact residual value
that makes the DHG component break even when selling at LCOH.

### Method
OUC = (1 - PV_heat_25yr / PV_heat_50yr) × (1+r)^25
At r=5%: OUC ≈ 77.2% of DHG investment (was 40%)

### What Changed
- New function `compute_ouc_residual()` added to `costs/heat_supply.py`
- `percent_residual_value = 0.4` replaced with `compute_ouc_residual(0.05, npv_years=25, lcoh_years=50)`
  in all 3 main scripts
- Default parameter `percent_residual_value: float = 0.4` updated to
  `compute_ouc_residual(0.05, npv_years=25, lcoh_years=50)` in all 7 sensitivity scripts

### Files Modified
- costs/heat_supply.py (new function)
- 05b_HT_Scenario.py, 07_LT_Scenario2.py, 08_Booster_Scenario.py (import + assignment)
- 09b, 09c, 09d (import + function parameter default)
- 10a, 10b, 10c, 10d (import + function parameter default)

### Note on sensitivity sweep
The 09* scripts sweep percent_residual_value as analysis type 8, passing explicit values
that override the default. The OUC default only applies to non-residual analysis types.

---

## LCOH Heat Quantity Consistency Fix (2026-04-01)

### Root Cause
The LCOH denominator used `areas_demand["delivered_energy"].sum()` (from hourly area-wide CSV),
which includes buildings with NFA < 30 m² (garages, sheds). But the revenue calculation uses
per-building data filtered to NFA >= 30. This mismatch meant the LCOH was computed on ~1% more
heat than was actually billed, making the selling price slightly too low.

### Fix (HT, LT, and all sensitivity scripts)
Moved the building stock load (with NFA >= 30 filter) before the LCOH section. `yearly_heat_supplied`
is now computed from `(buildingstock["yearly_dhw_energy"] + buildingstock["yearly_space_heating"]).sum() / efficiency_he`.
This matches exactly what `npv_data["yearly_demand_delivered_*_DH"]` uses for revenue.

### Fix (Booster — additional issues)
Two compensating errors were found in the Booster LCOH:

1. **LCOH denominator mismatch**: `lcoh_total_heat_generated_boosters` used useful demand (SH + DHW)
   but revenue was billed on delivered demand (useful / efficiency_he). The denominator was 25% too small.
   Fix: divide by `efficiency_he`.

2. **Grid heat cost mismatch**: `lcoh_heat_grid_boosters` multiplied `(LCOH_HP + LCOH_dhg)` by
   `total_heat_supplied_by_dhg` (grid demand at booster side of HE = 78.5 GWh). But LCOH_HP + LCOH_dhg
   was computed per kWh of HP-generated heat (103.7 GWh). The grid heat cost was underestimated by ~24%.
   Fix: use `yearly_heat_supplied_large_hp * 1000` (generated heat, matching the LCOH basis).

These two errors roughly cancelled in the old code (+25% over-billing vs -24% undercosting),
producing the misleading +€14.5M Booster NPV. With both fixed, the Booster NPV is now a small
positive (~€1.7M from tier markup), consistent with HT and LT.

### Files Modified
- 05b_HT_Scenario.py, 07_LT_Scenario2.py (building stock load moved before LCOH)
- 08_Booster_Scenario.py (building stock for LCOH + lcoh_total denominator + lcoh_heat_grid fix)
- 09b, 09c (building stock load moved before LCOH)
- 09d (LCOH heat supplied from filtered building stock + lcoh_total denominator + lcoh_heat_grid fix)
- 10a, 10b, 10d (building stock load added before LCOH)
- 10c (LCOH heat supplied from filtered building stock + lcoh_total denominator + lcoh_heat_grid fix)

---

## Verification

### LCOH/NPV consistency test
`validation/test_lcoh_npv_consistency.py` confirms NPV = 0.0000 for both:
- Case A: LCOH_HP (25yr) + LCOH_dhg (50yr) + OUC residual
- Case B: LCOH_HP (25yr) + LCOH_dhg (25yr) + no residual

### ETS2 carbon price calculation
`validation/ets2_carbon_price_equivalents.py` computes break-even gas multipliers
and carbon price equivalents from the sensitivity analysis outputs.

---

## Final Values (all fixes applied)

### HT Scenario
- npv_dh_operator: +2,136,890 EUR (+2.14 M)
- LCOH total: 140.0 EUR/MWh (HP: 121.2, DHG: 18.8)
- operator_selling_price: {r0: 0.1849, r1: 0.1507, r2: 0.1397, nr0: 0.1848, nr1: 0.1507, nr2: 0.1387}
- purchasing_price_heat: {r0: 0.1979, r1: 0.1612, r2: 0.1484, nr0: 0.1848, nr1: 0.1507, nr2: 0.1387}
- Total investment: 85.5 M EUR

### LT+Reno Scenario
- npv_dh_operator: +1,306,738 EUR (+1.31 M)
- LCOH total: 81.4 EUR/MWh (HP: 66.0, DHG: 15.4)
- operator_selling_price: {r0: 0.1063, r1: 0.0866, r2: 0.0814, nr0: 0.1085, nr1: 0.0884, nr2: 0.0814}
- purchasing_price_heat: {r0: 0.1138, r1: 0.0927, r2: 0.0871, nr0: 0.1085, nr1: 0.0884, nr2: 0.0814}
- Total investment: 34.4 M EUR

### Booster Scenario
- npv_dh_operator: +1,729,158 EUR (+1.73 M)
- LCOH total: 113.3 EUR/MWh (HP: 78.9, DHG: 10.2, booster composite: 113.3)
- operator_selling_price: {r0: 0.1511, r1: 0.1231, r2: 0.1133, nr0: 0.1510, nr1: 0.1231, nr2: 0.1133}
- purchasing_price_heat: {r0: 0.1617, r1: 0.1317, r2: 0.1212, nr0: 0.1510, nr1: 0.1231, nr2: 0.1133}
- Total investment: 58.6 M EUR

### Sensitivity Analysis (at electricity × 1.0)
- Booster break-even: gas × 1.15 (+15%), carbon price 74–105 EUR/tCO2
- HT break-even: gas × 1.39 (+39%), carbon price 192–274 EUR/tCO2
- LT+Reno (reno=0.0): gas × 0.83 (already profitable without gas increase)
- LT+Reno (reno=0.2): gas × 1.16 (+16%), comparable to Booster
- LT+Reno (reno=1.0): gas × 2.48 (+148%), not viable without renovation subsidies

### Operator breakeven RF (all scenarios)
- HT: ~0.992
- LT: ~0.981
- Booster: ~0.992

### Summary of changes vs original
| Metric | Original | Final | Cause |
|--------|----------|-------|-------|
| HT LCOH | 127 EUR/MWh | 140 EUR/MWh | LCOH discounting fix (+5%), heat quantity fix |
| LT LCOH | 105 EUR/MWh | 81 EUR/MWh | Heat quantity fix (renovated demand much lower) |
| Booster LCOH | 118 EUR/MWh | 113 EUR/MWh | Booster denominator + grid heat cost fixes |
| HT NPV | +9.09 M | +2.14 M | VAT removal + all fixes |
| LT NPV | +39.73 M | +1.31 M | VAT removal + all fixes |
| Booster NPV | +24.08 M | +1.73 M | VAT removal + all fixes |
| Booster gas break-even | +20% | +15% | Lower LCOH → lower customer prices |
| HT gas break-even | +36% | +39% | Higher LCOH → higher customer prices |
