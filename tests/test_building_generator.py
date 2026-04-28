"""Regression tests for ``building_analysis.building_generator``.

Pinned invariants:
- Per-building ±15% jitter is applied to every randomized U-value
  (roof, walls, floor, window, door). A duplicate template read after
  the jitter step previously overwrote door_u_value and reduced its
  fleet-wide distinct-value count to one per age bin.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from building_analysis.building_generator import generate_building
from config import U_VALUES_PATH


# Same building_type repeated across runs so any variance must come from
# the per-building jitter, not from the underlying template.
BUILDING_USAGE = "sfh"
AGE_CODE = 5
N_RUNS = 50


def _call_generate(seed: int):
    """Run generate_building once with a deterministic seed."""
    rng = np.random.RandomState(seed)
    geom = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return generate_building(
        building_usage=BUILDING_USAGE,
        age_code=AGE_CODE,
        building_id=f"test_{seed}",
        fid=seed,
        osm_id=seed,
        plot_area=100.0,
        roof_area=110.0,
        wall_area=200.0,
        volume=300.0,
        building_height=6.0,
        ceiling_height=2.7,
        roof_slope=20.0,
        angles_shared_borders=[],
        cardinal_directions_shared_borders=[],
        u_value_path=str(U_VALUES_PATH),
        geometry=geom,
        convert_wkb=False,
        rng=rng,
    )


@pytest.mark.regression
@pytest.mark.parametrize(
    "u_value_column",
    [
        "roof_u_value",
        "walls_u_value",
        "ground_contact_u_value",
        "door_u_value",
    ],
)
def test_per_building_jitter_applied_to_every_u_value(u_value_column):
    """50 runs with distinct seeds produce ≥30 distinct values per
    randomized opaque U-value. Anything below that floor implies the
    jitter step is being silently overwritten.
    """
    samples = [_call_generate(seed)[u_value_column].iloc[0] for seed in range(N_RUNS)]
    distinct = len(set(samples))
    assert distinct >= 30, (
        f"{u_value_column}: only {distinct} distinct values across {N_RUNS} runs; "
        "jitter is being overwritten downstream."
    )


@pytest.mark.regression
def test_window_u_value_jitter_applied():
    """Windows are stored as a JSON payload; check the four orientations
    each show variance across the 50 runs.
    """
    import json

    by_orient = {"north": [], "south": [], "east": [], "west": []}
    for seed in range(N_RUNS):
        windows = json.loads(_call_generate(seed)["windows"].iloc[0])
        for orient, payload in windows.items():
            by_orient[orient].append(payload["u_value"])

    for orient, values in by_orient.items():
        # Some orientations may have zero windows (no surface available),
        # which collapses to no entries — only assert variance for those
        # actually populated.
        if values:
            distinct = len(set(values))
            assert distinct >= 30, (
                f"window u_value ({orient}): only {distinct} distinct values "
                f"across {N_RUNS} runs; jitter is being overwritten downstream."
            )
