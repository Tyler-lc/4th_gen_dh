# 4th Generation District Heating: Boosters or Building Renovations?

Techno-economic analysis of district heating strategies for Frankfurt-Griesheim-Mitte, comparing high-temperature DH, low-temperature DH with building renovation, and booster heat pump configurations.

This repository contains the simulation and analysis code accompanying the paper:

> **Boosters or Building Renovations? An evaluation of 4th Generation District Heating Strategies**
> Luca Casamassima et al. — *Energy Conversion and Management: X* (under review)

## About this snapshot

Branch `paper-ecmx-2026` is a frozen snapshot of the codebase aligned with the ECMX submission. The data files committed here (and the tar archive referenced in `docs/PAPER_DATA_ARCHIVE.md` for the gitignored result tree) are the exact numerical outputs analysed in the paper. The working golden baseline, `tests/golden_baseline.json`, fingerprints those paper-aligned outputs, so `pytest -m regression` validates the paper data rather than a freshly regenerated run.

**Honest note on reproducibility.** The code on this snapshot includes a hardened deterministic-seeding refactor introduced after submission (Phase 7a). Running the pipeline from scratch on this code produces a different sample of the same stochastic process — the per-entity RNG draws differ from the legacy un-seeded path that produced the paper data. Aggregate metrics drift in the directions documented in [`docs/baseline_comparison_2026-04-20.md`](docs/baseline_comparison_2026-04-20.md): door-area mean shifts -27 % through TABULA's fat-tailed lookups, age-code -2.4 %, yearly space heating +3.6 %, booster electricity up to +29 % in some sensitivity bins. None of these reflect a bug; they are the expected sample-draw difference between the legacy and hardened seeding regimes.

If you need to reproduce the paper's numerical results bit-for-bit, check out tag `paper-submission-v1` (commit `e156295`) and follow the restoration recipe in [`docs/PAPER_DATA_ARCHIVE.md`](docs/PAPER_DATA_ARCHIVE.md). That tag preserves the exact code state at submission. The current snapshot is the paper-citable artifact: improved code structure, paper data on disk, and a transparent record of where the post-submission cleanup diverges from the original draws.

## Repository Structure

```
├── 01_create_people.py              # Generate occupant profiles
├── 01b_create_buildingstock.py      # Generate building stock from QGIS data
├── 02_calculate_energy_demand.py    # Space heating + DHW demand
├── 02b_calculate_booster_demand.py  # Booster scenario demand (base case)
├── 02c_buildingstock_sensitivity_analysis.py  # Booster sensitivity (multiple t_grid)
├── 03_renovate_buildingstock.py     # Apply renovation scenarios
├── 05b_HT_Scenario.py               # High-temperature DH scenario
├── 07_LT_Scenario2.py               # Low-temperature DH scenario
├── 08_Booster_Scenario.py           # Booster heat pump scenario
├── 09b-09d_*_Sens_Analysis.py       # Sensitivity analyses
├── 10a-10d_*_gas_vs_electricity.py  # Gas vs electricity comparisons
├── 11-24_*.py                       # Plotting and supplementary analysis
├── run_all_scenarios.py             # Run full analysis pipeline
├── run_pipeline_regeneration.py     # Reproducible end-to-end pipeline runner
│
├── building_analysis/               # Building class, stock generation, results
│   ├── Building.py                  # Thermodynamic building simulation
│   ├── building_generator.py        # Stock generation from TABULA archetypes
│   └── results/                     # Per-scenario output (parquet, CSV)
├── Person/                          # Occupant profiles (occupancy, DHW)
├── costs/                           # LCOH and renovation cost models
├── grid_calculation/                # DHC network optimisation (Pyomo/Gurobi)
├── utils/                           # Plotting, hydraulics, building geometry
├── heat_supply/                     # Carnot COP calculations
├── irradiation_data/                # Weather and solar data for Frankfurt
├── sensitivity_analysis/            # Sensitivity analysis parameters and output
├── config.py                        # Central configuration (paths, constants)
├── tests/                           # Test suite + golden baseline
└── docs/                            # PAPER_DATA_ARCHIVE.md, baseline comparison, etc.
```

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/) (Miniconda or Anaconda). Conda-forge is the primary channel — binary builds of `geopandas`, `pyomo`, and friends are more reliable than pip on macOS/Linux.
- A MILP solver for the grid optimisation. The project defaults to [Gurobi](https://www.gurobi.com/) (commercial, free academic licenses available). `grid_calculation/` supports any Pyomo-compatible solver; swap at the call site if Gurobi is unavailable.

### Recommended installation (conda)

```bash
git clone <repository-url>
cd 4th-gen-dh
conda env create -f environment.yml
conda activate dh_sim
pip install -e .
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

For an end-to-end orchestrated run with progress logging and resumability, use `run_pipeline_regeneration.py`:

```bash
python run_pipeline_regeneration.py            # full pipeline
python run_pipeline_regeneration.py --list     # list stages without running
python run_pipeline_regeneration.py --skip-to <stage>
```

### Individual steps

The pipeline scripts are numbered sequentially. Each reads outputs from prior steps:

```bash
python 01b_create_buildingstock.py    # Step 1: Generate building stock from QGIS
python 01_create_people.py            # Step 2: Generate occupant profiles
python 02_calculate_energy_demand.py  # Step 3: Calculate energy demand
python 02b_calculate_booster_demand.py            # Step 3b: Booster base case
python 02c_buildingstock_sensitivity_analysis.py  # Step 3c: Booster sensitivity
python 03_renovate_buildingstock.py   # Step 4: Apply renovations
python 05b_HT_Scenario.py             # Step 5: HT scenario (LCOH, NPV)
python 07_LT_Scenario2.py             # Step 6: LT+Reno scenario
python 08_Booster_Scenario.py         # Step 7: Booster scenario
```

Steps 09-11 run sensitivity analyses and generate comparison plots. Steps 12+ produce supplementary analysis for the paper.

### Stochastic elements

Person and building generation use random draws seeded deterministically through `config.SEED` and `config.derive_seed`. Re-running the pipeline twice with the same seed produces bit-identical outputs. See [`docs/PAPER_DATA_ARCHIVE.md`](docs/PAPER_DATA_ARCHIVE.md) for how this seeding regime relates to the original paper-data draws.

## Running Tests

```bash
pytest                       # full suite (unit + behavioural + regression)
pytest -m regression         # regression against golden baseline only
pytest -m smoke              # quick import and path checks
pytest -m equivalence        # determinism/seeding tests
```

The golden baseline (`tests/golden_baseline.json`) captures shapes, column sets, summary statistics, and md5 hashes for all result files. On this snapshot it fingerprints the paper-data outputs. To regenerate after a deliberate change:

```bash
python tests/capture_golden_baseline.py
```

## Data Flow

```
QGIS Building Data + Weather Data
  → Building Stock Generation (01b)
  → Person Generation (01)
  → Energy Demand Calculation (02 / 02b / 02c)
  → Renovation Scenarios (03)
  → DHC Grid Optimisation (grid_calculation/)
  → Scenario LCOH + NPV (05-08)
  → Sensitivity Analysis (09) → Comparisons (10) → Plots (11+)
```

## Key Conventions

- **Building types**: `sfh` (single-family house), `mfh` (multi-family house), `ab` (apartment block), `th` (townhouse)
- **Economic model**: dual-perspective (operator LCOH + customer NPV). LCOH uses component-specific lifetimes (25 yr HP, 50 yr grid). NPV uses a 25 yr horizon with outstanding unrecovered capital (OUC) for the grid residual value.
- **COP model**: Carnot, evaluated at scenario-specific source/sink temperatures.
- **NFA filter**: buildings with net floor area < 30 m² are excluded from economic calculations.

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
