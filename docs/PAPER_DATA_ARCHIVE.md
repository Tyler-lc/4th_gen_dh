# Paper submission data archive

This document records the state of the repository at the time of submission
to *Energy Conversion and Management: X* and the procedures for recovering
the exact code and results that underpin the paper.

## Two artefacts, two purposes

| Artefact | What it is | Use when |
|---|---|---|
| Branch `paper-ecmx-2026` | Cleaned-up codebase + paper data on disk + paper-aligned golden baseline. The commit at HEAD of this branch is the SHA cited from the paper. | You want to read, cite, or extend the paper-aligned codebase. Paper-data parquets are committed; gitignored result trees are restored from the tar archive below. |
| Tag `paper-submission-v1` | The exact commit submitted with the paper (`e156295`). Code state preserved as-is, including not-yet-cleaned modules and a `.venv` directory accidentally tracked at the time. | You need byte-for-byte reproducibility of the legacy un-seeded RNG path that produced the paper draws. |

The two are consistent: tracked paper-data parquets on `paper-ecmx-2026` are
identical (by content) to the same paths at `paper-submission-v1`. The
difference is structural — the snapshot ships an audited, deterministic
codebase whose source files no longer match the legacy submission code, while
the tag preserves the legacy code state.

## Restoring full paper data on `paper-ecmx-2026`

The branch already ships:

- All git-tracked paper-data parquets (`building_analysis/buildingstock/buildingstock.parquet`, `grid_calculation/*_result_df*.parquet`, `grid_calculation/sensitivity_analysis/.../booster_result_df_*.parquet`) at their paper-submission content.
- `tests/golden_baseline.json` aligned with paper outputs (this is `tests/golden_baseline_paper_submission.json` minus four entries pointing at `04_calculate_NPV_renovation.py` outputs — the producing script was removed during post-submission cleanup, ticket #134, so those four files have no current generator).
- `tests/golden_baseline_paper_submission.json` itself, the immutable fingerprint of the submission outputs.

To populate the gitignored result trees (`building_analysis/results/`,
`building_analysis/dhw_profiles/`, `sensitivity_analysis/`, `plots/`,
`grid_calculation/cache/`) on disk, restore the companion archive deposited
on Zenodo:

> **Zenodo:** <https://doi.org/10.5281/zenodo.19894657>
> **Archive:** `4th_gen_dh_paper_data_2026-04-29.tar.gz` (5.7 GB)
> **SHA256:** `18d00e3431460e08ba068ef47d9266abe6d8ec051915d519a18bdf2f0a78f01e`

### Easy path (recommended)

```bash
git checkout paper-ecmx-2026
conda env create -f environment.yml
conda activate dh_sim
pip install -e .
python scripts/restore_paper_data.py    # downloads, verifies SHA256, extracts
pytest -m regression                    # 29 tests pass against paper data
```

`scripts/restore_paper_data.py` finds the repository root automatically,
downloads the archive from Zenodo, verifies its SHA256, and extracts at the
correct location. Pass a local path as a positional argument if you have
the tar already (`python scripts/restore_paper_data.py /path/to/archive.tar.gz`).

### Manual path

```bash
git checkout paper-ecmx-2026
conda env create -f environment.yml
conda activate dh_sim
pip install -e .

# Download manually from https://doi.org/10.5281/zenodo.19894657 (or:)
curl -L -o 4th_gen_dh_paper_data_2026-04-29.tar.gz \
    "https://zenodo.org/records/19894657/files/4th_gen_dh_paper_data_2026-04-29.tar.gz"
shasum -a 256 4th_gen_dh_paper_data_2026-04-29.tar.gz
# expect: 18d00e3431460e08ba068ef47d9266abe6d8ec051915d519a18bdf2f0a78f01e

tar -xzf 4th_gen_dh_paper_data_2026-04-29.tar.gz
pytest -m regression
```

If the Zenodo deposit is unavailable, `paper-submission-v1` plus a fresh
pipeline run is the alternative reproducibility path; see "Restoration
procedure (legacy code state)" below.

## What is archived where

### 1. Code — git tag `paper-submission-v1`

The exact commit submitted with the paper is tagged `paper-submission-v1`
(commit `e156295`). The tag is pushed to the remote.

Recover the code state:

```bash
git fetch --tags
git checkout paper-submission-v1
```

### 2. Golden baseline — in-repo

`tests/golden_baseline_paper_submission.json` is a frozen copy of the golden
baseline captured against the paper-submission results. **Do not modify this
file.** It stays in the repository as the permanent fingerprint of the
paper's numerical results (row counts, column hashes, aggregate statistics,
MD5 of every result file).

On the snapshot branch `paper-ecmx-2026`, the working
`tests/golden_baseline.json` is also paper-aligned: it is the paper baseline
minus four entries (`costs/renovation_costs.csv`,
`costs/energy_savings_renovated.csv`, `costs/npv_data_renovated_gas.csv`,
`grid_calculation/booster_results.csv`) whose producer
(`04_calculate_NPV_renovation.py`) was removed during post-submission
cleanup (ticket #134). On other branches such as `publication-ready`, the
working baseline tracks the post-Phase-7b regeneration and is allowed to
drift from the paper baseline.

### 3. Result artefacts — Zenodo deposit

The following directories are **not** tracked in git (gitignored because of
size) and are archived on Zenodo as the companion deposit:

```text
building_analysis/results/
building_analysis/dhw_profiles/
sensitivity_analysis/
plots/
grid_calculation/cache/
```

Public deposit:

- DOI: <https://doi.org/10.5281/zenodo.19894657>
- File: `4th_gen_dh_paper_data_2026-04-29.tar.gz`
- Direct URL: <https://zenodo.org/records/19894657/files/4th_gen_dh_paper_data_2026-04-29.tar.gz>
- Compressed size: 5.7 GB
- Extracted size: ~20 GB
- Entries: 54,802
- SHA256: `18d00e3431460e08ba068ef47d9266abe6d8ec051915d519a18bdf2f0a78f01e`
- Created: 2026-04-29

### Rebuilding the archive

If you need to rebuild the archive from a populated working tree, the recipe
is:

```bash
tar --exclude='.DS_Store' --exclude='._*' --exclude='*.numbers' \
    --exclude='* Large.jpeg' --exclude='* Medium.jpeg' \
    --exclude='sensitivity_analysis/__init__.py' --exclude='__pycache__' \
    -czf paper_data.tar.gz \
    building_analysis/results \
    building_analysis/dhw_profiles \
    sensitivity_analysis \
    plots \
    grid_calculation/cache
```

## Restoration procedure (legacy code state)

To reproduce the paper's numerical results exactly:

1. Check out the tagged code:

   ```bash
   git checkout paper-submission-v1
   conda env create -f environment.yml
   conda activate dh_sim
   pip install -e .
   ```

2. Restore the result artefacts from the Zenodo deposit:

   ```bash
   curl -L -o paper_data.tar.gz \
       "https://zenodo.org/records/19894657/files/4th_gen_dh_paper_data_2026-04-29.tar.gz"
   tar -xzf paper_data.tar.gz
   ```

3. Verify integrity against the archived baseline:

   ```bash
   cp tests/golden_baseline_paper_submission.json tests/golden_baseline.json
   pytest -m regression
   ```

   All regression tests should pass against the restored artefacts.

## Why this archive exists

Phase 2 of the publication-readiness effort added a deterministic-seeding
mechanism to `Person` and `building_generator` but did not wire it through
the production entry points (`01_create_people.py`, `iterator_generate_buildings`).
Production Person generation therefore relied on uninitialised NumPy global
state, so re-running the pipeline from scratch does not reproduce the
paper's result files even though the *code* is identical.

Phase 7 closes this gap by threading a central seed through every
stochastic call. The cost is that results regenerated after Phase 7 will
differ in detail from the paper submission (aggregate metrics are expected
to agree within ~2%, which is the sample-draw noise). Phase 7b documents
the before/after comparison.

The archive — git tag, frozen baseline, external tar — guarantees that the
paper-submission state remains recoverable regardless of what the live
codebase looks like afterwards.
