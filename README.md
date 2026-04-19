# 4th Generation District Heating: Boosters or Building Renovations?

Techno-economic analysis of district heating strategies for Frankfurt-Griesheim-Mitte, comparing high-temperature DH, low-temperature DH with building renovation, and booster heat pump configurations.

This repository contains the simulation and analysis code accompanying the paper:

> **Boosters or Building Renovations? An evaluation of 4th Generation District Heating Strategies**
> Luca Casamassima et al. — *Energy Conversion and Management: X* (under review)

## Repository Structure

```
├── 01_create_people.py              # Generate occupant profiles
├── 01b_create_buildingstock.py      # Generate building stock from QGIS data
├── 02_calculate_energy_demand.py    # Space heating + DHW demand
├── 03_renovate_buildingstock.py     # Apply renovation scenarios
├── 04_calculate_NPV_renovation.py   # NPV of renovations
├── 05b_HT_Scenario.py              # High-temperature DH scenario
├── 07_LT_Scenario2.py              # Low-temperature DH scenario
├── 08_Booster_Scenario.py          # Booster heat pump scenario
├── 09b-09d_*_Sens_Analysis.py      # Sensitivity analyses
├── 10a-10d_*_gas_vs_electricity.py  # Gas vs electricity comparisons
├── 11-24_*.py                       # Plotting and supplementary analysis
├── run_all_scenarios.py             # Run full analysis pipeline
│
├── building_analysis/               # Building class, stock generation, results
│   ├── Building.py                  # Thermodynamic building simulation
│   └── results/                     # Per-scenario output (parquet, CSV)
├── Person/                          # Occupant profiles (occupancy, DHW)
├── costs/                           # LCOH, NPV, renovation cost models
├── grid_calculation/                # DHC network optimisation (Pyomo/Gurobi)
├── utils/                           # Plotting, hydraulics, building geometry
├── heat_supply/                     # Carnot/Lorentz COP calculations
├── irradiation_data/                # Weather and solar data for Frankfurt
├── sensitivity_analysis/            # Sensitivity analysis parameters and output
├── config.py                        # Central configuration (paths, constants)
└── tests/                           # Test suite with golden baseline
```

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/) (Miniconda or Anaconda). Conda-forge is the primary channel — binary builds of `geopandas`, `pyomo`, and friends are more reliable than pip on macOS/Linux.
- A MILP solver for the grid optimisation. The project defaults to [Gurobi](https://www.gurobi.com/) (commercial, free academic licenses available). `grid_calculation/` supports any Pyomo-compatible solver; swap at call site if Gurobi is unavailable.

### Recommended installation (conda)

```bash
git clone <repository-url>
cd 4th-gen-dh
conda env create -f environment.yml
conda activate dh_sim
```

`environment.yml` contains the versions used to produce the published results with loose pins, so the conda solver can substitute platform-appropriate builds. For a byte-exact snapshot of the environment that produced the paper's results, see `environment-lock.yml` (generated with `conda env export --no-builds` on macOS; platform-specific).

### Pip-only fallback

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Geospatial binaries (`geopandas`, `osmnx`) may require system-level build tools on this path. `requirements.txt` mirrors `pyproject.toml` for systems where `pip install -e .` is not available.

## Reproducing Results

### Full pipeline

Run all scenarios, sensitivity analyses, and generate all figures:

```bash
python run_all_scenarios.py
```

### Individual steps

The pipeline scripts are numbered sequentially. Each reads outputs from prior steps:

```bash
python 01_create_people.py              # Step 1: Generate occupant profiles
python 01b_create_buildingstock.py      # Step 2: Generate building stock
python 02_calculate_energy_demand.py    # Step 3: Calculate energy demand
python 03_renovate_buildingstock.py     # Step 4: Apply renovations
python 04_calculate_NPV_renovation.py   # Step 5: Customer NPV
python 05b_HT_Scenario.py              # Step 6: HT scenario (LCOH, NPV)
python 07_LT_Scenario2.py              # Step 7: LT+Reno scenario
python 08_Booster_Scenario.py          # Step 8: Booster scenario
```

Steps 09-11 run sensitivity analyses and generate comparison plots. Steps 12+ produce supplementary analysis for the paper.

### Stochastic elements

Person generation (Step 1) uses random occupancy profiles. To reproduce identical results, the pipeline uses deterministic seeding (`seed=42`). Results will vary across runs only if the seed is changed.

## Running Tests

```bash
# Run all tests
pytest

# Run only regression tests (checks results against golden baseline)
pytest -m regression

# Run only smoke tests (quick import and path checks)
pytest -m smoke
```

The golden baseline (`tests/golden_baseline.json`) captures shapes, column names, and summary statistics for all result files. To regenerate after a deliberate change:

```bash
python tests/capture_golden_baseline.py
```

## Data Flow

```
QGIS Building Data + Weather Data
  → Building Stock Generation (01b)
  → Person Generation (01)
  → Energy Demand Calculation (02)
  → Renovation Scenarios (03) → Customer NPV (04)
  → DHC Grid Optimisation (grid_calculation/)
  → Scenario LCOH Calculations (05-08)
  → Sensitivity Analysis (09) → Comparisons (10) → Plots (11+)
```

## Key Conventions

- **Building types**: `sfh` (single-family house), `mfh` (multi-family house), `ab` (apartment block), `th` (townhouse)
- **Economic model**: Dual-perspective (operator LCOH + customer NPV). LCOH uses component-specific lifetimes (25yr HP, 50yr grid). NPV uses 25yr horizon with outstanding unrecovered capital (OUC) for the grid residual value.
- **COP model**: Lorentz COP (preferred over Carnot for accuracy)
- **NFA filter**: Buildings with net floor area < 30 m² are excluded from economic calculations

## Citation

```bibtex
@article{casamassima2026boosters,
  title={Boosters or Building Renovations? An evaluation of 4th Generation District Heating Strategies},
  author={Casamassima, Luca},
  journal={Energy Conversion and Management: X},
  year={2026},
  note={Under review}
}
```
