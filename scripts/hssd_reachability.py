"""Inventory every source instance using an existing HSSD navmesh, without rendering.

Reachability is relative to one declared anchor and sampled same-floor stopping
points. Failure to find a point is not proof of geometric unreachability.
Repeated templates remain unmapped unless their instance identity is unique.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import sys

from scripts.benchmark_manifest import read_json, sha256
from scripts.hssd_targets import load_geometry
from scripts.hssd_visual_dataset import disjoint_output
from scripts.navigation_scene import PROJECT_ROOT, project_path
from scripts.public_inputs import write_json


STATUSES = ("reachable", "no_stopping_point_found", "mapping_ambiguous", "runtime_missing", "error")


def sample_stop(pathfinder, obj, anchor):
    """Search 36 points around the world AABB, at the anchor's floor height.

    World units are metres. Require 0.4–0.7 m horizontal AABB separation,
    snap displacement <= 0.15 m and vertical displacement <= 0.3 m. Unlike
    task selection, a reachable point may be less than one metre from anchor.
    This is finite sampling and does not certify visibility or object identity.
    """
    import habitat_sim
    import numpy as np

    low, high = (np.asarray(obj["world_aabb"][key], dtype=float) for key in ("min", "max"))
    anchor = np.asarray(anchor, dtype=np.float32)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or (low > high).any():
        raise ValueError("Invalid world AABB")
    choices = []
    for fraction in np.linspace(.1, .9, 9):
        x, z = (low + fraction*(high-low))[[0, 2]]
        for qx, qz in ((low[0]-.6, z), (high[0]+.6, z), (x, low[2]-.6), (x, high[2]+.6)):
            query = np.array([qx, anchor[1], qz], dtype=np.float32)
            point = np.asarray(pathfinder.snap_point(query), dtype=np.float32)
            if (not np.isfinite(point).all() or not pathfinder.is_navigable(point)
                    or abs(float(point[1]-anchor[1])) > .3
                    or float(np.linalg.norm((point-query)[[0, 2]])) > .15):
                continue
            gap = float(np.linalg.norm(np.maximum(np.maximum(low[[0, 2]]-point[[0, 2]],
                                                             point[[0, 2]]-high[[0, 2]]), 0)))
            if not .4 <= gap <= .7:
                continue
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = anchor, point
            if (not pathfinder.find_path(path) or not math.isfinite(path.geodesic_distance)
                    or path.geodesic_distance < 0):
                continue
            choices.append({"stop_position": point.tolist(),
                            "anchor_distance_m": float(path.geodesic_distance),
                            "xz_aabb_gap_m": gap, "navmesh_island": int(pathfinder.get_island(point))})
    return min(choices, key=lambda row: row["anchor_distance_m"]) if choices else None


def inventory(source_instances, runtime_objects, sampler):
    """Keep one row per canonical source instance, including every unknown case.

    Runtime IDs are not stable source identities. A template is matched only
    when it occurs exactly once in both lists; translations alone do not prove
    identity because object origins/COM and scene transforms can differ.
    """
    if not source_instances:
        raise ValueError("Source instance roster is empty")
    ids = [row["source_instance_index"] for row in source_instances]
    if any(type(identity) is not int or identity < 0 for identity in ids) or len(ids) != len(set(ids)):
        raise ValueError("Source instance indices must be unique nonnegative integers")
    source_counts = Counter(row["template_id"] for row in source_instances)
    runtime_by_template = defaultdict(list)
    for obj in runtime_objects:
        runtime_by_template[obj["template_id"]].append(obj)
    rows = []
    for obj in source_instances:
        row = {"source_instance_index": obj["source_instance_index"], "template_id": obj["template_id"],
               "category": obj.get("category", "unknown"),
               "category_labels_agree": obj.get("category_labels_agree"),
               "status": None, "runtime_object_id": None, "geometry": None, "error": None}
        matches = runtime_by_template[obj["template_id"]]
        if not matches:
            row["status"] = "runtime_missing"
        elif len(matches) != 1 or source_counts[obj["template_id"]] != 1:
            row["status"] = "mapping_ambiguous"
        else:
            row["runtime_object_id"] = matches[0]["object_id"]
            try:
                point = sampler(matches[0])
                row["geometry"] = point
                row["status"] = "reachable" if point is not None else "no_stopping_point_found"
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                row["status"], row["error"] = "error", str(error)
        rows.append(row)
    return rows


def coverage(rows):
    return {"planned_instances": len(rows),
            "by_status": {status: sum(row["status"] == status for row in rows) for status in STATUSES},
            "by_category": {category: {status: sum(row["category"] == category and row["status"] == status for row in rows)
                                       for status in STATUSES} for category in sorted({row["category"] for row in rows})},
            "reachable_labeled_instances": sum(row["status"] == "reachable" and row["category"] not in ("unknown", "") for row in rows),
            "human_approved_targets": "not_evaluated", "visibility_verified": False,
            "global_unreachable_count": None,
            "scope": "single_anchor_same_floor_finite_stop_sampling"}


def run(args):
    source = args.source_dir.resolve()
    paths = [source / name for name in ("assets.json", "metrics.json", "source_objects.json", "objects.json", "scene.navmesh")]
    paths.append(args.base_config.resolve())
    bindings = {str(path): sha256(path) for path in paths}
    config, assets, metrics = read_json(args.base_config), read_json(source / "assets.json"), read_json(source / "metrics.json")
    if assets.get("passed") is not True or metrics.get("success") is not True or config["scene_id"] != metrics["scene_id"]:
        raise ValueError("Require successful smoke evidence for the selected scene")
    source_instances = read_json(source / "source_objects.json")
    runtime_objects = read_json(source / "objects.json")
    if (len(source_instances) != assets["declared_object_instances"]
            or len(runtime_objects) != metrics["loaded_rigid_objects"]
            or len(runtime_objects) != len(source_instances)):
        raise ValueError("Source/runtime instance denominator differs from smoke evidence")
    pathfinder, provenance = load_geometry(config)
    if sha256(source / "scene.navmesh") != provenance["navmesh_sha256"]:
        raise ValueError("Smoke/config navmesh mismatch")
    for key in ("scene", "scene_dataset_config"):
        if assets["input_sha256"].get(provenance[key]) != provenance[key+"_sha256"]:
            raise ValueError(f"Smoke/config dependency mismatch: {key}")
    # Bind all raw dependencies declared by the smoke check, including semantic
    # metadata. The generated object table alone is not a live scene snapshot.
    for path, expected in assets["input_sha256"].items():
        resolved = project_path(path)
        if sha256(resolved) != expected:
            raise ValueError(f"Smoke dependency changed: {path}")
        bindings[str(resolved)] = expected
    for key in ("scene", "scene_dataset_config", "navmesh"):
        bindings[str(project_path(config[key]))] = sha256(project_path(config[key]))
    output = disjoint_output(args.output_dir, [source, *[Path(path) for path in bindings]])
    anchor = config["start_position"]
    if len(anchor) != 3 or not all(math.isfinite(value) for value in anchor) or not pathfinder.is_navigable(anchor):
        raise ValueError("Configured anchor must be a finite navigable world position")
    rows = inventory(source_instances, runtime_objects, lambda obj: sample_stop(pathfinder, obj, anchor))
    unchanged = all(sha256(Path(path)) == expected for path, expected in bindings.items())
    report = {**coverage(rows), "scene_id": config["scene_id"], "anchor_position": anchor,
              "source_files_unchanged": unchanged,
              "passed": unchanged and not any(row["status"] in ("error", "runtime_missing") for row in rows),
              "release_ready": False}
    output.mkdir(parents=True)
    write_json(output / "instances.json", {"kind": "hssd_instance_reachability_v1", "instances": rows})
    write_json(output / "summary.json", report)
    write_json(output / "source_bindings.json", {"files": bindings, "provenance": provenance,
               "code_bindings": {name: sha256(PROJECT_ROOT / name) for name in
                                 ("scripts/hssd_reachability.py", "scripts/hssd_targets.py", "scripts/run_navigation.py")}})
    with (output / "instances.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("source_instance_index", "template_id", "category", "status", "runtime_object_id", "anchor_distance_m", "error")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fields if key != "anchor_distance_m"},
                             "anchor_distance_m": (row["geometry"] or {}).get("anchor_distance_m")})
    print({key: report[key] for key in ("passed", "planned_instances", "by_status", "source_files_unchanged")})
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "outputs/hssd_smoke_001")
    parser.add_argument("--base-config", type=Path, default=PROJECT_ROOT / "configs/simulator/hssd_navigation_001.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    try:
        return run(parser.parse_args())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"Reachability inventory failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
