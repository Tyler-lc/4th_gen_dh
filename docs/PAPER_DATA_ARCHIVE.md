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
`grid_calculation/cache/`) on disk, extract the tar archive recorded below:

```bash
git checkout paper-ecmx-2026
conda env create -f environment.yml
conda activate dh_sim
pip install -e .
tar -xzf <path-to-archive>/4th_gen_dh_paper_submission_2026-04-19.tar.gz
pytest -m regression                   # 29 tests, all pass against paper data
```

If the tar is unavailable in your environment, `paper-submission-v1` plus
fresh pipeline run is the alternative reproducibility path; see "Restoration
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

### 3. Result artefacts — external backup (user-managed)

The following directories are **not** tracked in git (they are gitignored
because of size). They were lost-and-regenerable under the old pipeline;
under the new pipeline they will be regenerated from scratch with different
seeds. To preserve the paper data, back them up externally before running
any Phase 7b regeneration:

```text
building_analysis/results/
building_analysis/dhw_profiles/
sensitivity_analysis/*/data/
sensitivity_analysis/*/renovated/data/
sensitivity_analysis/*/booster/data/
plots/
grid_calculation/cache/
```

### Recommended backup command

From the repository root:

```bash
tar --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
    -czf ~/backups/4th_gen_dh_paper_submission_2026-04-19.tar.gz \
    building_analysis/results \
    building_analysis/dhw_profiles \
    sensitivity_analysis \
    plots \
    grid_calculation/cache
```

Archive record:

- Primary location: `~/backups/4th_gen_dh_paper_submission_2026-04-19.tar.gz`
- Date: 2026-04-19
- Size: 6.07 GB (5.7 GB per `ls -lh`, macOS binary units)
- Entries: 54,608 (files + directories)
- Contents verified with `tar -tzf <archive> | wc -l`
- Secondary copy: local NAS (user-managed)
- SHA256: not computed; run `shasum -a 256 <archive>` and append here if integrity verification is later required.

## Restoration procedure (legacy code state)

To reproduce the paper's numerical results exactly:

1. Check out the tagged code:

   ```bash
   git checkout paper-submission-v1
   conda env create -f environment.yml
   conda activate dh_sim
   pip install -e .
   ```

2. Restore the result artefacts from the external backup:

   ```bash
   tar -xzf <path-to-archive> -C .
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
