"""Compare HP CAPEX sensitivity results across all 3 DH scenarios.

Reads the sensitivity analysis outputs for inv_cost_multiplier sweeps
and produces:
  1. Summary table: LCOH, NPV operator, customer prices at each multiplier
  2. Figure: LCOH or customer price vs CAPEX multiplier, one line per scenario
  3. Check: does the scenario ranking change at any multiplier level?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SENSITIVITY_DIR = Path(__file__).resolve().parent.parent / "sensitivity_analysis"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "capex_sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "HT": {
        "dir": SENSITIVITY_DIR / "unrenovated" / "inv_cost_multiplier" / "data",
        "label": "HT (90°C)",
        "color": "C3",
    },
    "LT+Reno": {
        "dir": SENSITIVITY_DIR / "renovated" / "inv_cost_multiplier" / "data",
        "label": "LT+Reno (50°C)",
        "color": "C0",
    },
    "Booster": {
        "dir": SENSITIVITY_DIR / "booster" / "inv_cost_multiplier" / "data",
        "label": "Booster (50°C grid)",
        "color": "C2",
    },
}

def _discover_multipliers(scenario_dir: Path, min_val: float = 0.6) -> list:
    """Auto-discover available multiplier values from filenames, filtered to >= min_val."""
    import re
    mults = []
    for f in scenario_dir.glob("inv_cost_multiplier_*.csv"):
        m = re.search(r"inv_cost_multiplier_([\d.]+)\.csv$", f.name)
        if m:
            val = float(m.group(1))
            if val >= min_val:
                mults.append(val)
    return sorted(mults)


def load_scenario_data(scenario_dir: Path, multipliers: list = None) -> dict:
    """Load per-multiplier data for a scenario."""
    if multipliers is None:
        multipliers = _discover_multipliers(scenario_dir)
    results = []
    for mult in multipliers:
        fpath = scenario_dir / f"inv_cost_multiplier_{mult}.csv"
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found")
            continue
        df = pd.read_csv(fpath)
        # operator_selling_price is the customer price (€/kWh) — same for all at a given multiplier
        price = df["operator_selling_price"].iloc[0]

        # NPV savings vs gas (mean across all buildings)
        savings_col = [c for c in df.columns if "savings" in c][0]
        mean_savings = df[savings_col].mean()

        # Fraction of buildings where DH is cheaper than gas
        n_positive = (df[savings_col] > 0).sum()
        frac_positive = n_positive / len(df) * 100

        results.append(
            {
                "multiplier": mult,
                "capex_reduction_pct": round((1 - mult) * 100),
                "operator_selling_price": price,
                "mean_npv_savings": mean_savings,
                "pct_buildings_dh_cheaper": frac_positive,
                "n_buildings": len(df),
            }
        )
    return pd.DataFrame(results)


def load_operator_npv(scenario_dir: Path) -> pd.DataFrame:
    """Load operator NPV from the aggregated file."""
    mg_dir = scenario_dir / "multitple_graphs"
    npv_df = pd.read_csv(mg_dir / "npv_operator.csv", index_col=0)
    val_df = pd.read_csv(mg_dir / "values.csv", index_col=0)
    return pd.DataFrame(
        {
            "multiplier": val_df.iloc[:, 0].values,
            "npv_operator": npv_df.iloc[:, 0].values,
        }
    )


# --- Main --------------------------------------------------------------------
if __name__ == "__main__":
    all_data = {}
    all_npv = {}

    for name, info in SCENARIOS.items():
        print(f"\n=== {name} ===")
        data = load_scenario_data(info["dir"])
        npv = load_operator_npv(info["dir"])
        all_data[name] = data
        all_npv[name] = npv
        print(data.to_string(index=False))

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("RANKING CHECK: Does the scenario ordering change with CAPEX reduction?")
    print(f"{'='*80}")

    # Use multipliers common to all scenarios
    common_mults = None
    for data in all_data.values():
        s = set(data["multiplier"].round(10))
        common_mults = s if common_mults is None else common_mults & s
    for mult in sorted(common_mults):
        red = round((1 - mult) * 100)
        prices = {}
        for name, data in all_data.items():
            row = data[data["multiplier"] == mult]
            if len(row):
                prices[name] = row["operator_selling_price"].values[0]
        ranking = sorted(prices, key=lambda k: prices[k])
        prices_str = ", ".join(
            f"{k}={v:.4f}" for k, v in sorted(prices.items(), key=lambda x: x[1])
        )
        print(f"  CAPEX -{red:2d}%: {prices_str}  →  Ranking: {' < '.join(ranking)}")

    # --- Figure 1: Customer price vs CAPEX multiplier ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, info in SCENARIOS.items():
        data = all_data[name]
        ax.plot(
            data["capex_reduction_pct"],
            data["operator_selling_price"] * 100,
            marker="o",
            linewidth=2,
            color=info["color"],
            label=info["label"],
        )

    ax.set_xlabel("HP CAPEX Reduction [%]")
    ax.set_ylabel("Minimum Customer Price [ct/kWh]")
    ax.set_title("Impact of HP CAPEX Reduction on Customer Prices")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # 0% reduction on right, 40% on left
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "customer_price_vs_capex.png", dpi=200)
    paper_fig = Path(
        "/Users/lucacasamassima/Library/CloudStorage/GoogleDrive-lucasamassima@gmail.com/"
        "Other computers/My laptop/Documents/phd thesis/Possible papers/"
        "District Heating Comparison/paper_git/4th-Gen-Paper/figure"
    )
    fig.savefig(paper_fig / "hp_capex_sensitivity.png", dpi=200)
    print(f"\nSaved customer price plot")

    # --- Figure 2: Operator NPV vs CAPEX multiplier ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for name, info in SCENARIOS.items():
        npv = all_npv[name]
        ax2.plot(
            ((1 - npv["multiplier"]) * 100),
            npv["npv_operator"] / 1e6,
            marker="o",
            linewidth=2,
            color=info["color"],
            label=info["label"],
        )

    ax2.set_xlabel("HP CAPEX Reduction [%]")
    ax2.set_ylabel("Operator NPV [M€]")
    ax2.set_title("Impact of HP CAPEX Reduction on Operator NPV")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "operator_npv_vs_capex.png", dpi=200)
    print(f"Saved operator NPV plot")

    # --- Figure 3: Mean customer NPV savings vs gas ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for name, info in SCENARIOS.items():
        data = all_data[name]
        ax3.plot(
            data["capex_reduction_pct"],
            data["mean_npv_savings"],
            marker="o",
            linewidth=2,
            color=info["color"],
            label=info["label"],
        )

    ax3.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("HP CAPEX Reduction [%]")
    ax3.set_ylabel("Mean Customer NPV Savings vs Gas [€]")
    ax3.set_title("Customer Economics: DH vs Gas at Different HP CAPEX Levels")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / "customer_savings_vs_capex.png", dpi=200)
    print(f"Saved customer savings plot")

    # --- Save combined table ---
    combined = []
    for name, data in all_data.items():
        data_copy = data.copy()
        data_copy.insert(0, "scenario", name)
        npv = all_npv[name]
        data_copy = data_copy.merge(npv, on="multiplier", how="left")
        combined.append(data_copy)
    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv(RESULTS_DIR / "capex_sensitivity_summary.csv", index=False)
    print(f"\nSaved combined summary CSV")

    plt.show()
