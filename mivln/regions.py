"""World-coordinate avoid regions and continuous segment checks (metres).

Regions describe forbidden floor space, not physical furniture. The swept agent
centre is tested against a closed XZ rectangle expanded by the agent radius.
This conservative square footprint includes the circular agent body; touching
the expanded boundary counts as a violation. Y identifies the affected floor.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping


def validate_region(value: Mapping) -> dict:
    """Return an owned, validated region; never retain caller-owned state."""
    if not isinstance(value, Mapping) or value.get("type") != "rectangle_xz":
        raise ValueError("Avoid region requires type='rectangle_xz'")
    if not isinstance(value.get("region_id"), str) or not value["region_id"].strip():
        raise ValueError("Avoid region requires a non-empty region_id")
    bounds = value.get("bounds_xz")
    try:
        valid = (len(bounds) == 2 and all(len(point) == 2 for point in bounds)
                 and all(type(v) in (int, float) and math.isfinite(v) for point in bounds for v in point)
                 and all(bounds[0][axis] < bounds[1][axis] for axis in (0, 1)))
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("bounds_xz requires finite [[min_x, min_z], [max_x, max_z]]")
    for key in ("floor_y", "floor_tolerance_m"):
        if type(value.get(key)) not in (int, float) or not math.isfinite(value[key]):
            raise ValueError(f"Avoid region requires finite {key}")
    if value["floor_tolerance_m"] <= 0:
        raise ValueError("floor_tolerance_m must be positive")
    return deepcopy(dict(value))


def segment_intersects_region(start, end, region: Mapping, radius_m: float = 0.0) -> bool:
    """Slab intersection includes crossings whose two endpoints are outside.

    Points are world XYZ agent-foot positions. Inflate only horizontally; Y is a
    floor-selection band rather than the height of an obstacle. A tiny numerical
    tolerance makes boundary contact consistently forbidden, including STOP.
    """
    if not math.isfinite(radius_m) or radius_m < 0:
        raise ValueError("radius_m must be finite and non-negative")
    if any(len(point) != 3 or not all(math.isfinite(v) for v in point) for point in (start, end)):
        raise ValueError("Segment requires finite XYZ endpoints")
    lo, hi = region["bounds_xz"]
    lower = (lo[0] - radius_m, region["floor_y"] - region["floor_tolerance_m"], lo[1] - radius_m)
    upper = (hi[0] + radius_m, region["floor_y"] + region["floor_tolerance_m"], hi[1] + radius_m)
    first, last = 0.0, 1.0
    for a, b, low, high in zip(start, end, lower, upper):
        low, high = low - 1e-6, high + 1e-6
        delta = b - a
        if abs(delta) < 1e-12:
            if a < low or a > high:
                return False
        else:
            enter, leave = sorted(((low - a) / delta, (high - a) / delta))
            first, last = max(first, enter), min(last, leave)
            if first > last:
                return False
    return True


def violating_regions(start, end, constraints, radius_m: float) -> list[str]:
    return [region["region_id"] for region in constraints
            if segment_intersects_region(start, end, region, radius_m)]


def trajectory_violations(trajectory: list, radius_m: float) -> list[dict]:
    """Use the destination row's truth: events apply BEFORE its action.

    The INITIALIZE row and actions preceding delivery carry no new constraint.
    Count violating actions, with all intersected region IDs on each action.
    """
    results = []
    for before, after in zip(trajectory, trajectory[1:]):
        regions = violating_regions(before["position"], after["position"],
                                    after["task_state"]["active_constraints"], radius_m)
        if regions:
            results.append({"step": after["step"], "region_ids": regions})
    return results
