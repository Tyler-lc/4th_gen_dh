"""Full pipeline regeneration under the hardened seeding regime (Phase 7b).

Runs every script whose output is captured by the golden baseline, in
dependency order. Hard-fails on the first error rather than continuing —
downstream scripts read earlier outputs, so silent continuation risks
stale-data bugs of the kind fixed in commit 78747d2.

Usage
-----
    conda activate dh_sim
    python run_pipeline_regeneration.py

Runtime: expect several hours end to end, dominated by
02_calculate_energy_demand.py (single-process per user preference) and
the sensitivity stages. tqdm progress bars from each script are forwarded
to this terminal.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _pipeline_env() -> dict:
    """Return a copy of os.environ with REPO_ROOT prepended to PYTHONPATH.

    Scripts in subdirectories (e.g. grid_calculation/) would otherwise
    fail to import ``config`` because Python only adds the script's own
    directory to sys.path on invocation. Prepending REPO_ROOT to
    PYTHONPATH makes top-level modules (``config``, top-level helpers)
    importable from every stage.
    """
    env = os.environ.copy()
    repo_root = str(REPO_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing}" if existing else repo_root
    )
    return env

# (stage label, script path relative to repo root)
PIPELINE = [
    # ── Data generation ──────────────────────────────────────────────────
    # 01b must precede 01: 01_create_people.py reads BUILDINGSTOCK_PATH to
    # learn which buildings need DHW profiles. Despite the numbering,
    # buildingstock is the prerequisite.
    ("Unrenovated buildingstock",
        "01b_create_buildingstock.py"),
    ("Persons (DHW + occupancy)",
        "01_create_people.py"),
    ("Unrenovated energy demand (res + non-res)",
        "02_calculate_energy_demand.py"),
    ("Renovate buildingstock + renovated energy demand",
        "03_renovate_buildingstock.py"),
    # 04_calculate_NPV_renovation.py is intentionally skipped. It has been
    # broken since Sept 2024 (consumer_size signature/caller drift) and its
    # outputs are not consumed by any downstream step in the paper pipeline.
    # Scheduled for removal together with the stale costs/*.csv outputs in
    # ticket #134.
    ("Booster energy demand (t_grid=50)",
        "02b_calculate_booster_demand.py"),

    # ── DHC grid optimisation ────────────────────────────────────────────
    # Each grid calc reads the upstream buildingstock+demand. The booster
    # grid must use the 02b output at matching t_grid to avoid the
    # stale-data bug fixed in 78747d2.
    ("Unrenovated DHC grid",
        "grid_calculation/01_unrenovated_grid_calculation.py"),
    ("Renovated DHC grid",
        "grid_calculation/02_renovated_grid_calculation.py"),
    ("Booster DHC grid",
        "grid_calculation/03_booster_grid_calculation.py"),

    # ── Base scenarios (LCOH + dual-perspective NPV) ─────────────────────
    ("HT scenario (05b)",
        "05b_HT_Scenario.py"),
    ("LT scenario (07)",
        "07_LT_Scenario2.py"),
    ("Booster scenario (08)",
        "08_Booster_Scenario.py"),

    # ── Sensitivity preparation ──────────────────────────────────────────
    # 02c reshuffles the buildingstock for sensitivity; depends on 03.
    ("Sensitivity buildingstock (02c)",
        "02c_buildingstoclk_sensitivity_analysis.py"),
    ("Booster sensitivity DHC grid",
        "grid_calculation/03_booster_grid_calculation_sensitivty_analysis.py"),

    # ── Sensitivity analyses ─────────────────────────────────────────────
    ("HT sensitivity (09b)",
        "09b_HT_Sens_Analysis.py"),
    ("LT sensitivity (09c, t_grid=50)",
        "09c_LT_Sens_Analysis.py"),
    ("Booster sensitivity (09d)",
        "09d_HT_Booster_Sens_Analysis.py"),
]


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def run_stage(index: int, total: int, label: str, script: str) -> float:
    banner = "=" * 90
    print(f"\n{banner}")
    print(f"[{index}/{total}] {label}")
    print(f"        script: {script}")
    print(banner, flush=True)

    script_path = REPO_ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing pipeline script: {script_path}")

    start = time.perf_counter()
    # cwd=REPO_ROOT because scripts assume the repo root as working dir
    # (several read relative paths via config.PROJECT_ROOT-derived absolute
    # paths, but a couple of the grid scripts still rely on cwd).
    # env=... prepends REPO_ROOT to PYTHONPATH so `from config import ...`
    # works in scripts that live in subdirectories (e.g. grid_calculation/).
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        env=_pipeline_env(),
        check=True,  # hard-fail: CalledProcessError on non-zero exit
    )
    return time.perf_counter() - start


def main() -> int:
    overall_start = time.perf_counter()
    durations: list[tuple[str, float]] = []

    for i, (label, script) in enumerate(PIPELINE, start=1):
        try:
            elapsed = run_stage(i, len(PIPELINE), label, script)
        except subprocess.CalledProcessError as exc:
            print(f"\n[FAIL] stage {i}/{len(PIPELINE)} ({label}) "
                  f"exited with status {exc.returncode}",
                  file=sys.stderr)
            print("Stopping to avoid stale-data propagation downstream.",
                  file=sys.stderr)
            return exc.returncode
        except FileNotFoundError as exc:
            print(f"\n[FAIL] {exc}", file=sys.stderr)
            return 1

        durations.append((label, elapsed))
        print(f"[OK] {label} — {format_duration(elapsed)}", flush=True)

    overall = time.perf_counter() - overall_start
    banner = "=" * 90
    print(f"\n{banner}")
    print(f"All {len(PIPELINE)} stages complete in {format_duration(overall)}.")
    print(banner)
    print(f"{'Stage':<65} {'Duration':>12}")
    print("-" * 90)
    for label, elapsed in durations:
        print(f"{label:<65} {format_duration(elapsed):>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
