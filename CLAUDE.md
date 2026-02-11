# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research tool for analyzing **4th Generation District Heating** feasibility in Frankfurt-Griesheim-Mitte. Simulates building energy demand, evaluates renovation scenarios, calculates DHC network costs, and compares heating strategies (high-temperature DH, low-temperature DH, booster heat pumps) via economic metrics (LCOH, NPV).

## Environment

- **Python 3.9** with virtualenv at `.venv/`
- Activate: `source .venv/bin/activate`
- Key dependencies: pandas, geopandas, numpy, matplotlib, seaborn, networkx, osmnx, shapely, pyomo, gurobipy, numpy_financial

## Running the Pipeline

Scripts are numbered sequentially and form a data processing pipeline. Each script reads outputs from prior steps.

**Full pipeline (scenarios + analysis + plots):**
```bash
python run_all_scenarios.py
```

**Individual steps:**
```bash
python 01_create_people.py              # Generate person profiles
python 01b_create_buildingstock.py      # Generate building stock from QGIS data
python 02_calculate_energy_demand.py    # Calculate space heating + DHW demand
python 03_renovate_buildingstock.py     # Apply renovation scenarios
python 04_calculate_NPV_renovation.py   # NPV of renovations
python 05b_HT_Scenario.py              # High-temp DH scenario
python 07_LT_Scenario2.py              # Low-temp DH scenario
python 08_Booster_Scenario.py          # Booster heat pump scenario
```

Steps 09-11 run sensitivity analyses and generate plots. Steps 12+ do supplementary analysis.

**Multiprocessing variants** exist for energy demand calculation:
- `calculate_residential_energy_demand_multiprocessing.py`
- `calculate_non_residential_energy_demand_multiprocessing.py`
- `calculate_energy_demand_multiprocessing_whole_buildingstock.py`

## Architecture

### Core Classes

**`Person` (`Person/Person.py`)** — Simulates individual building occupants. Generates stochastic occupancy profiles (workday/free day) based on German sleep survey data, and DHW demand profiles tied to occupancy. Each person has randomized wake/sleep categories.

**`Building` (`building_analysis/Building.py`)** — Thermodynamic building simulation. Calculates heat losses (transmission, ventilation, infiltration), solar gains, and useful energy demand (UED). Contains a list of `Person` instances and aggregates their DHW profiles. Building properties (U-values, geometry) are determined by building type and age archetype.

Buildings contain Persons: `Building.people` is a list of `Person` objects; `Building.add_person()` adds occupants.

### Module Layout

| Directory | Purpose |
|---|---|
| `building_analysis/` | Building class, building stock generator, results storage |
| `Person/` | Person class for occupancy and DHW profiles |
| `costs/` | Heat supply economics (`heat_supply.py`: COP, CAPEX, LCOH) and renovation costs (`renovation_costs.py`: IWU cost models, NPV) |
| `grid_calculation/` | DHC network topology creation and Pyomo/Gurobi optimization |
| `utils/` | Plotting (`plotting.py`), building geometry utilities, energy calculations, QGIS integration |
| `dhw/` | Domestic hot water demand profile generation |
| `heat_supply/` | Carnot/Lorentz efficiency calculations |
| `irradiation_data/` | Weather, temperature, solar irradiation data for Frankfurt |
| `sensitivity_analysis/` | Output directories for scenario sensitivity results (unrenovated/renovated/booster) |
| `Databases/`, `p-gis/` | GIS data and EMB3Rs DHC network module |

### Data Flow

```
QGIS Building Data + Weather Data
  → Building Stock Generation (01b)
  → Person Generation (01)
  → Energy Demand Calculation (02)
  → Renovation Scenarios (03) → NPV (04)
  → DHC Grid Optimization (grid_calculation/)
  → Scenario LCOH Calculations (05-08)
  → Sensitivity Analysis (09) → Comparison Studies (10) → Plots (11)
```

### File Formats

- **Building stock**: Parquet (GeoDataFrames)
- **Energy results**: CSV in `building_analysis/results/`
- **DHW profiles**: CSV in `building_analysis/dhw_profiles/`
- **Grid cache**: Pickle files in `grid_calculation/cache/`
- **Plots**: PNG in `plots/`
- **Sensitivity parameters**: Excel (`sensitivity_analysis/sensitivity_analysis_parameters.xlsx`)

## Key Conventions

- Building types: `sfh` (single-family house), `mfh` (multi-family house), `ab` (apartment block), `th` (townhouse), plus non-residential
- Building age archetypes determine U-values and renovation potential
- Economic calculations use inflation-adjusted costs indexed to specific reference years
- Lorentz COP is preferred over Carnot for heat pump calculations (more accurate)
- Plots for publication have titles removed (recent commit convention)
- Stochastic elements: person generation uses random wake/sleep times and occupancy probabilities — results vary between runs
