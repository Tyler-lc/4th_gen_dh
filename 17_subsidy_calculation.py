"""Calculate the external subsidy required for customer break-even in each DH scenario.

At reduction factor (f_red) = 1.0, the operator breaks even and customers pay
full LCOH-based prices. Lowering f_red reduces customer heat costs but creates
an operator revenue shortfall that must be covered externally (e.g. government
subsidy). This script computes:

1. The break-even f_red where aggregate customer NPV savings = 0
2. The total subsidy NPV (25 yr, discounted) at that break-even point
3. The annualised subsidy (level annual payment over 25 years at 5%)

Results are saved to sensitivity_analysis/subsidy_results.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import SENSITIVITY_DIR

DISCOUNT_RATE = 0.05
ANALYSIS_PERIOD = 25
ANNUITY_FACTOR = (1 - (1 + DISCOUNT_RATE) ** -ANALYSIS_PERIOD) / DISCOUNT_RATE

scenarios = {
    "Booster": SENSITIVITY_DIR / "booster" / "reduction_factor" / "data",
    "HT": SENSITIVITY_DIR / "unrenovated" / "reduction_factor" / "data",
    "LT+Reno": SENSITIVITY_DIR / "renovated" / "reduction_factor" / "data",
}

results = []

for scenario, data_dir in scenarios.items():
    files = sorted(data_dir.glob("reduction_factor_*.csv"))

    records = []
    for f in files:
        rf_str = f.stem.replace("reduction_factor_", "")
        rf = float(rf_str)
        # Filter to clean grid values (0.1 step) to avoid interleaved data
        if not any(abs(rf - x) < 0.005 for x in np.arange(0.1, 2.05, 0.1)):
            continue

        df = pd.read_csv(f)
        records.append(
            {
                "f_red": round(rf, 2),
                "total_savings": df["savings_npv_25years_ir_0.05"].sum(),
                "total_dh_cost": df["npv_DH_25years_ir_0.05"].sum(),
                "mean_savings_per_building": df["savings_npv_25years_ir_0.05"].mean(),
                "n_buildings": len(df),
            }
        )

    rdf = pd.DataFrame(records).sort_values("f_red").reset_index(drop=True)

    # Total DH cost at f_red = 1.0 (operator revenue at full pricing)
    row_1 = rdf.loc[(rdf["f_red"] - 1.0).abs().idxmin()]
    total_dh_at_1 = abs(row_1["total_dh_cost"])

    # Find break-even f_red (aggregate savings crosses zero)
    sv = rdf["total_savings"].values
    fv = rdf["f_red"].values

    be_fred = None
    for i in range(len(sv) - 1):
        if sv[i] >= 0 and sv[i + 1] < 0:
            frac = sv[i] / (sv[i] - sv[i + 1])
            be_fred = fv[i] + frac * (fv[i + 1] - fv[i])
            break

    if be_fred is not None:
        price_reduction_pct = (1 - be_fred) * 100
        subsidy_npv = (1 - be_fred) * total_dh_at_1
        subsidy_annual = subsidy_npv / ANNUITY_FACTOR
    else:
        price_reduction_pct = None
        subsidy_npv = None
        subsidy_annual = None

    print(f"\n{'=' * 60}")
    print(f"  {scenario}")
    print(f"{'=' * 60}")
    print(f"  Total DH cost at f_red=1.0:  EUR {total_dh_at_1:>15,.0f}")
    print(f"  Total customer deficit:      EUR {row_1['total_savings']:>15,.0f}")

    if be_fred is not None:
        print(f"  Break-even f_red:            {be_fred:>15.3f}")
        print(f"  Price reduction needed:      {price_reduction_pct:>14.1f}%")
        print(f"  Subsidy NPV (25 yr):         EUR {subsidy_npv:>15,.0f}")
        print(f"  Subsidy NPV:                 EUR {subsidy_npv / 1e6:>14.1f}M")
        print(f"  Annualised subsidy:          EUR {subsidy_annual:>15,.0f}/yr")
        print(f"  Annualised subsidy:          EUR {subsidy_annual / 1e6:>14.2f}M/yr")
    else:
        if all(v < 0 for v in sv):
            print("  Customers never break even (negative at all f_red values)")
            print("  Even free heating cannot offset renovation costs")
        else:
            print("  Break-even not found in data range")

    results.append(
        {
            "scenario": scenario,
            "total_dh_cost_at_fred_1": total_dh_at_1,
            "total_customer_deficit_at_fred_1": row_1["total_savings"],
            "break_even_fred": be_fred,
            "price_reduction_pct": price_reduction_pct,
            "subsidy_npv_25yr": subsidy_npv,
            "subsidy_annual": subsidy_annual,
        }
    )

# Save results
out = pd.DataFrame(results)
out_path = SENSITIVITY_DIR / "subsidy_results.csv"
out.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")

# Summary table
print("\n" + "=" * 70)
print("  SUMMARY: External subsidy for customer break-even")
print("=" * 70)
print(f"  {'Scenario':<12s} | {'f_red':>8s} | {'Reduction':>10s} | {'Subsidy NPV':>14s} | {'Annual':>10s}")
print("-" * 70)
for r in results:
    if r["break_even_fred"] is not None:
        print(
            f"  {r['scenario']:<12s} | {r['break_even_fred']:>8.3f} | "
            f"{r['price_reduction_pct']:>9.1f}% | "
            f"EUR {r['subsidy_npv_25yr'] / 1e6:>8.1f}M | "
            f"EUR {r['subsidy_annual'] / 1e6:>5.2f}M/yr"
        )
    else:
        print(f"  {r['scenario']:<12s} | {'N/A':>8s} | {'N/A':>10s} | {'N/A':>14s} | {'N/A':>10s}")
