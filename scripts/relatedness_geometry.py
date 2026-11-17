"""Attach private, native-navmesh evidence to reviewed relatedness candidates.

Run in the Habitat environment. No Simulator, renderer, policy or model is
created: existing actions are replayed on the frozen navmesh up to delivery.
Every candidate stays in the report, including rejected/unclassified cases.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

from scripts import intervention_relatedness as related
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import PROJECT_ROOT, read_json, sha256
from scripts.export_visual_videos import verified_batch
from scripts.hssd_visual_dataset import disjoint_output
from scripts.public_inputs import write_json


def pose_matches(actual, expected, tolerance=1e-5):
    """Positions are world metres; rotations are unit xyzw, sign invariant."""
    def values(pose, key, length):
        value = pose.get(key)
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"Invalid pose {key}")
        if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
            raise ValueError(f"Non-finite pose {key}")
        return value
    a, b = values(actual, "position", 3), values(expected, "position", 3)
    qa, qb = values(actual, "rotation_xyzw", 4), values(expected, "rotation_xyzw", 4)
    if any(abs(sum(x*x for x in q)-1) > 1e-4 for q in (qa, qb)):
        raise ValueError("Non-unit pose quaternion")
    return (math.dist(a, b) <= tolerance
            and min(math.dist(qa, qb), math.dist(qa, [-x for x in qb])) <= 1e-6)


def measure_prefix(config, trajectory, events, replay):
    """Measure the state BEFORE action k, after exactly k prefix actions.

    trajectory[0] is initialization and trajectory[i+1] is after action i.
    Replay each pose so an off-by-one delivery cannot silently change evidence.
    """
    if len(events) != 1 or len(config["events"]) != 1:
        raise ValueError("Exactly one replacement event is required")
    event, definition = events[0], config["events"][0]
    step = event["delivered_step"]
    if (type(step) is not int or step < 0 or step >= len(trajectory)-1
            or definition["trigger"] != {"type": "step", "value": step}
            or event["event_id"] != definition["event_id"]
            or event["task_update"] != definition["task_update"]):
        raise ValueError("Delivery identity/step/update differs from frozen configuration")
    for index in range(step+1):
        if not pose_matches(replay.pose(), trajectory[index]):
            raise ValueError(f"Native replay differs from recorded pose before action {index}")
        if index < step:
            action = trajectory[index+1]["action"]
            if action not in ("move_forward", "turn_left", "turn_right"):
                raise ValueError("Unexpected terminal or semantic action in replacement prefix")
            replay.act(action)
    actual = replay.pose()
    if not pose_matches(actual, event["pose_before"]) or not pose_matches(actual, event["pose_after"]):
        raise ValueError("Event pose changed at delivery")
    new_goal = definition["task_update"]["new_goal"]
    distance = replay.distance(config["goal_positions"][new_goal])
    if type(distance) not in (float, int) or not math.isfinite(distance) or distance < 0:
        raise ValueError("Updated goal is unreachable or has invalid geodesic distance")
    # This is distance to the configured navigation stop point, not distance to
    # the object's surface. It conveys no instance visibility guarantee.
    return {"actual_intervention_pose": actual, "delivered_step": step,
            "updated_goal_geodesic_distance_m": distance,
            "distance_definition": "native_navmesh_to_configured_goal_stop_point",
            "prefix_actions_replayed": step, "geometry_evidence_status": "verified",
            "reference_route_overlap": None, "updated_goal_previously_seen": None}


def load_review(candidate_dir, review_dir):
    """Reconstruct candidates and validate the exact human declaration version."""
    path = candidate_dir / "manifest.json"
    manifest = read_json(path)
    if manifest.get("kind") != related.KIND or manifest.get("schema_version") != 1:
        raise ValueError("Unknown relatedness candidate package")
    origin = Path(manifest["source_preparation"])
    if sha256(origin / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source preparation changed")
    candidates = related.build_candidates(origin)
    if candidates != manifest["candidates"]:
        raise ValueError("Candidate roster differs from reviewed origins")
    record = read_json(review_dir / "review.json")
    if (record.get("kind") != "replacement_relatedness_review_v1"
            or Path(record["candidate_manifest"]).resolve() != path.resolve()
            or record["candidate_manifest_sha256"] != sha256(path)):
        raise ValueError("Relatedness review belongs to another candidate version")
    decisions = related.review_rows(candidates, record["decisions"])
    return origin, candidates, {row["case_id"]: row for row in decisions}


def run(args):
    origin, candidates, decisions = load_review(args.candidates_dir, args.review_dir)
    prepared, audited, roots, _ = verified_batch(args.run_dir, args.audit_dir)
    if Path(prepared["source_dir"]).resolve() != origin.resolve():
        raise ValueError("Navigation batch and candidates use different source preparations")
    roots.update(relatedness_candidates=args.candidates_dir.resolve(),
                 relatedness_review=args.review_dir.resolve(), navigation_audit=args.audit_dir.resolve())
    before = snapshot(roots)
    output = disjoint_output(args.output_dir, list(roots.values()))
    # Import lazily: help and regression tests work without Habitat installed.
    from scripts.hssd_visual_backend import NativeReplay
    rows = []
    for candidate in candidates:
        row = {"case_id": candidate["case_id"], "base_task_id": candidate["base_task_id"],
               "pair_id": candidate["pair_id"], "classification": decisions[candidate["case_id"]],
               "geometry_evidence_status": "failed", "error": None}
        replay = None
        try:
            slots = [slot for slot in audited["episodes"] if slot["case_id"] == candidate["case_id"]
                     and slot["task_type"] == "replacement" and slot["condition"] == "oracle"]
            if len(slots) != 1:
                raise ValueError("Expected exactly one audited Oracle source slot")
            slot = slots[0]
            folder = Path(slot["source_run"]) / "episodes" / slot["case_id"] / "oracle" / "private"
            config = read_json(folder / "config.json")
            original = read_json(Path(candidate["source_config"]))
            # Runtime adapters add fields; shared task semantics must be exact.
            for key in ("scene_id", "scene_sha256", "navmesh_sha256", "initial_goal", "goal_positions",
                        "events", "initial_instruction", "start_position", "start_rotation_xyzw", "seed", "max_steps"):
                if config[key] != original[key]:
                    raise ValueError(f"Runtime/source config mismatch: {key}")
            row["source_episode"] = str(folder.resolve())
            row["evidence_files"] = {name: sha256(folder / name)
                                     for name in ("config.json", "trajectory.json", "events.json")}
            replay = NativeReplay(config)
            row.update(measure_prefix(config, read_json(folder / "trajectory.json"),
                                      read_json(folder / "events.json"), replay))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        finally:
            if replay is not None:
                replay.close()
        rows.append(row)
    unchanged = before == snapshot(roots)
    passed = unchanged and bool(rows) and all(row["geometry_evidence_status"] == "verified" for row in rows)
    output.mkdir(parents=True)
    write_json(output / "geometry.json", {"kind": "relatedness_native_geometry_v1", "rows": rows})
    write_json(output / "source_bindings.json", {
        "roots": {key: str(path) for key, path in roots.items()}, "before": before,
        "source_files_unchanged": unchanged,
        "code_bindings": {name: sha256(PROJECT_ROOT / name) for name in
                          ("scripts/relatedness_geometry.py", "scripts/intervention_relatedness.py",
                           "scripts/export_visual_videos.py", "scripts/hssd_visual_backend.py")}})
    summary = {"passed": passed, "planned_candidates": len(candidates),
               "verified_candidates": sum(row["geometry_evidence_status"] == "verified" for row in rows),
               "source_files_unchanged": unchanged, "paired_runs_executed": 0,
               "route_overlap_status": "not_measured", "visibility_status": "not_measured",
               "release_ready": False}
    write_json(output / "summary.json", summary)
    print(summary)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates-dir", "review-dir", "run-dir", "audit-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"Relatedness geometry failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
