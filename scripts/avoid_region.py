"""Prepare (CPU), run (GPU), and audit one matched avoid-region experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from mivln import TaskState
from mivln.regions import segment_intersects_region, trajectory_violations
from scripts.avoid_planner import AvoidPlanningError, plan_route
from scripts.hssd_targets import sha256
from scripts.hssd_dataset import matching_frames, terminal_record, within
from scripts.intervention_timing import check_reference, load_json, write_json
from scripts.navigation_scene import check_scene_match, navmesh_path, project_path


class AvoidPreparationError(ValueError):
    """Retain candidate rejection evidence when a fixed task cannot be prepared."""

    def __init__(self, message, rejected_candidates):
        super().__init__(message)
        self.rejected_candidates = rejected_candidates


def prepare(args):
    """Freeze a new constraint against an existing successful no-event reference.

    The original start, goal, instance, sensors and action budget are inherited.
    Only a region is selected. Reject candidates that contain the trigger pose or
    goal, miss the future reference, or lack a route in the finite guard roadmap.
    No GPU, task resampling, reference overwrite, or implicit review is involved.
    """
    import habitat_sim
    import numpy as np
    from scripts.run_navigation import scene_provenance, validate_config

    reference = args.reference_dir.resolve()
    config = load_json(reference / "episode.json")
    validate_config(config)
    scene_provenance(config, project_path(config["scene"]))
    rows = check_reference(config, reference)
    if not config.get("scene_dataset_config") or not config.get("semantic_sensor"):
        raise ValueError("This entry requires an HSSD reference with semantic frames")
    for key in ("terminal_alignment", "terminal_visibility"):
        if not config.get(key, {}).get("enabled"):
            raise ValueError("Use the verified terminal-facing reference for this experiment")
    if args.output_dir.exists():
        raise FileExistsError("Use a new preparation output directory")
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh_path(config))):
        raise ValueError("Cannot load reference mesh")
    radius = config.get("agent_radius", .1)
    goal = config["goal_positions"][config["initial_goal"]]
    total = sum(row["distance_m"] for row in rows)
    progress = 0.0
    trigger = None
    for index, row in enumerate(rows[1:], 1):
        progress += row["distance_m"]
        if progress >= total * .25:
            trigger = index  # Row k is the pose BEFORE action k.
            break
    if trigger is None or trigger >= len(rows) - 2:
        raise ValueError("Reference has insufficient travel after the trigger")
    start = rows[trigger]["position"]
    rejected, selected = [], None
    for index in range(trigger + 3, len(rows) - 2):
        if rows[index]["distance_m"] < config["forward_step_m"] * .5:
            continue
        x, y, z = rows[index]["position"]
        for half_size in (.25, .4, .55):
            region = {"region_id": "avoid_001", "type": "rectangle_xz",
                      "bounds_xz": [[x-half_size, z-half_size], [x+half_size, z+half_size]],
                      "floor_y": y, "floor_tolerance_m": .5}
            candidate = {"reference_row": index, "half_size_m": half_size}
            if (segment_intersects_region(start, start, region, radius + .2)
                    or segment_intersects_region(goal, goal, region, radius + config["success_radius_m"])):
                rejected.append({**candidate, "reason": "trigger or goal too close"})
                continue
            try:
                route = plan_route(pathfinder, np.asarray(start, dtype=np.float32),
                                   np.asarray(goal, dtype=np.float32), region, radius)
            except AvoidPlanningError as error:
                rejected.append({**candidate, "reason": str(error)})
                continue
            selected = region, route, candidate
            break
        if selected:
            break
    if selected is None:
        raise AvoidPreparationError(
            f"No suitable region among {len(rejected)} candidates; do not change the reference", rejected)
    region, route, selected_candidate = selected
    # Strip runtime-only records; preserve all semantic and navigation inputs.
    for key in ("gpu_device_id", "navmesh_settings", "loaded_rigid_objects", "controller",
                "intervention_policy", "shortest_path_length_m", "versions", "expected_rigid_objects"):
        config.pop(key, None)
    config["episode_id"] = getattr(args, "episode_id", "hssd_avoid_region_001")
    config["avoid_planner"] = {"clearance_m": .35, "waypoint_radius_m": .2}
    config["events"] = [{"event_id": "add_avoid_001", "trigger": {"type": "step", "value": trigger},
        "utterance": "Keep going to the same destination, but avoid the marked floor region.",
        "task_update": {"operation": "add_avoid_region", "region": region}}]
    validate_config(config)
    # Verify the intended negative control using the actual old execution.
    hypothetical = deepcopy(rows)
    for row in hypothetical[trigger+1:]:
        row["task_state"]["active_constraints"] = [region]
    violations = trajectory_violations(hypothetical, radius)
    if not violations:
        raise ValueError("Selected region does not invalidate the reference after delivery")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "config.json", config)
    report = {"kind": "hssd_avoid_region_preparation_v1", "reference_dir": str(reference),
              "reference_hashes": {name: sha256(reference/name) for name in
                                   ("episode.json", "trajectory.json", "metrics.json")},
              "config_sha256": sha256(args.output_dir/"config.json"), "trigger_step": trigger,
              "trigger_pose": start, "selected_candidate": selected_candidate,
              "rejected_candidates": rejected, "planned_route": route,
              "reference_violation_count": len(violations),
              "reference_first_violation_step": violations[0]["step"],
              "purpose": "Privileged single-scene geometry smoke test; marked region is structured Oracle input"}
    write_json(args.output_dir / "preparation.json", report)
    print(json.dumps(report, indent=2))
    return 0


def validate_preparation(config_path):
    report = load_json(config_path.parent / "preparation.json")
    if report["kind"] != "hssd_avoid_region_preparation_v1" or sha256(config_path) != report["config_sha256"]:
        raise ValueError("Prepared configuration changed")
    if set(report["reference_hashes"]) != {"episode.json", "trajectory.json", "metrics.json"}:
        raise ValueError("Preparation lacks complete reference hashes")
    for name, digest in report["reference_hashes"].items():
        if sha256(Path(report["reference_dir"]) / name) != digest:
            raise ValueError(f"Prepared reference changed: {name}")
    return report


def run_experiment(args):
    from scripts.run_navigation import run, scene_provenance, validate_config

    prepared = validate_preparation(args.config.resolve())
    config = load_json(args.config)
    validate_config(config)
    scene = project_path(config["scene"])
    scene_provenance(config, scene)
    if args.output_dir.exists():
        raise FileExistsError("Use a new experiment output directory")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "experiment.json", {
        "config_sha256": sha256(args.output_dir/"config.json"), "preparation": prepared,
        "source_config": str(args.config.resolve()), "gpu_id": args.gpu_id})
    statuses = []
    for condition in ("control", "intervention", "ignore"):
        branch = deepcopy(config)
        branch["episode_id"] += "_" + condition
        if condition == "control":
            branch["events"] = []
        try:
            run(branch, scene, args.gpu_id, args.output_dir / condition,
                intervention_policy="ignore" if condition == "ignore" else "follow")
            statuses.append({"condition": condition, "completed": True, "error": None})
        except Exception as error:
            # Keep failed conditions and any partial frames in the denominator.
            statuses.append({"condition": condition, "completed": False,
                             "error": f"{type(error).__name__}: {error}"})
        write_json(args.output_dir / "run_status.json", statuses)
    report = summarize(args.output_dir)
    return 0 if report["passed"] else 1


def check_experiment(output):
    """Recompute all three planned conditions from saved poses and task truth.

    No Habitat import/rendering is needed. Point completion and instance evidence
    are checked separately from the conjunction with constraint satisfaction.
    """
    config = load_json(output / "config.json")
    experiment = load_json(output / "experiment.json")
    if sha256(output / "config.json") != experiment["config_sha256"]:
        raise ValueError("Experiment configuration changed")
    validate_preparation(Path(experiment["source_config"]))
    if sha256(Path(experiment["source_config"])) != experiment["config_sha256"]:
        raise ValueError("Experiment differs from prepared configuration")
    # Input integrity is checked without importing the GPU runner.
    for key in ("scene", "scene_dataset_config", "navmesh"):
        if sha256(project_path(config[key])) != config[key+"_sha256"]:
            raise ValueError(f"Experiment source changed: {key}")
    event = config["events"][0]
    trigger = event["trigger"]["value"]
    radius = config.get("agent_radius", .1)
    results, records = [], {}
    for condition in ("control", "intervention", "ignore"):
        item = {"condition": condition, "passed": False, "error": None}
        try:
            folder = output / condition
            episode, metrics, rows, logs = (load_json(folder / name) for name in
                ("episode.json", "metrics.json", "trajectory.json", "events.json"))
            check_scene_match(config, episode)
            for key, value in config.items():
                if key not in ("episode_id", "events", "scene", "scene_dataset_config", "navmesh") and episode.get(key) != value:
                    raise ValueError(f"Episode changed prepared input: {key}")
            expected_events = [] if condition == "control" else config["events"]
            if episode["events"] != expected_events or len(rows) < trigger+2:
                raise ValueError("Event configuration differs or trigger was not reached")
            truth, policy = TaskState(config["initial_goal"]), TaskState(config["initial_goal"])
            for index, row in enumerate(rows):
                delivered = index == trigger+1 and condition != "control"
                if delivered:
                    truth.apply_update(event["task_update"])
                    if condition == "intervention":
                        policy.apply_update(event["task_update"])
                if (row["step"] != (index-1 if index else None)
                        or row["task_state"] != truth.to_dict() or row["policy_task_state"] != policy.to_dict()
                        or row["active_goal"] != truth.active_goal or row["policy_goal"] != policy.active_goal
                        or row["delivered_event_ids"] != ([event["event_id"]] if delivered else [])):
                    raise ValueError("Task/policy timeline or action numbering differs")
                for kind in ("rgb", "depth", "semantic"):
                    frame = within(folder, row["frames"][kind])
                    if not frame.is_file() or frame.stat().st_size == 0:
                        raise ValueError(f"Missing or empty {kind} frame")
            if rows[0]["action"] != "INITIALIZE" or rows[-1]["action"] != "STOP":
                raise ValueError("Episode lacks initialization or STOP")
            # Quaternion storage uses float32; allow the same precision as poses.
            if (math.dist(rows[0]["position"], config["start_position"]) > 1e-5 or
                    math.dist(rows[0]["rotation_xyzw"], config["start_rotation_xyzw"]) > 1e-5):
                raise ValueError("Initial pose changed")
            if condition == "control":
                if logs:
                    raise ValueError("Control delivered an event")
            else:
                if (len(logs) != 1 or logs[0]["event_id"] != event["event_id"]
                        or logs[0]["delivered_step"] != trigger or logs[0]["task_update"] != event["task_update"]
                        or logs[0]["pose_before"] != {key: rows[trigger][key] for key in ("position", "rotation_xyzw")}
                        or logs[0]["pose_after"] != logs[0]["pose_before"]
                        or logs[0]["ignored_by_policy"] != (condition == "ignore")
                        or logs[0]["task_state_before"] != rows[trigger]["task_state"]
                        or logs[0]["task_state_after"] != rows[trigger+1]["task_state"]
                        or logs[0]["policy_task_state_before"] != rows[trigger]["policy_task_state"]
                        or logs[0]["policy_task_state_after"] != rows[trigger+1]["policy_task_state"]):
                    raise ValueError("Event was not delivered once before its action without reset")
            violations = trajectory_violations(rows, radius)
            point = bool(metrics["stopped"] and metrics["final_goal"] == config["initial_goal"]
                         and metrics["stopped_goal"] == config["initial_goal"]
                         and metrics["navigation_error_m"] is not None
                         and 0 <= metrics["navigation_error_m"] <= config["success_radius_m"]
                         and metrics["steps"] == len(rows)-1 <= config["max_steps"])
            length = sum(math.dist(a["position"], b["position"]) for a, b in zip(rows, rows[1:]))
            if (not point or not math.isclose(length, metrics["path_length_m"], abs_tol=1e-5)
                    or metrics["planner_error"] is not None or metrics["point_goal_success"] is not True
                    or metrics["constraint_violation_count"] != len(violations)
                    or metrics["constraint_compliance"] != (not violations)
                    or metrics["first_constraint_violation_step"] != (violations[0]["step"] if violations else None)
                    or metrics["dynamic_success"] != (point and not violations)
                    or metrics["success"] != metrics["dynamic_success"]
                    or metrics["max_steps"] != config["max_steps"]
                    or metrics["scheduled_event_count"] != len(expected_events)
                    or metrics["delivered_event_count"] != len(expected_events)
                    or metrics["intervention_step"] != (None if condition == "control" else trigger)
                    or metrics["ignored_event_count"] != (1 if condition == "ignore" else 0)
                    or metrics["intervention_policy"] != ("ignore" if condition == "ignore" else "follow")
                    or metrics["collision_count"] != sum(bool(row["collided"]) for row in rows)
                    or load_json(folder/"constraint_violations.json") != violations):
                raise ValueError("Point completion or independently measured constraint metrics disagree")
            # terminal_record checks point AND visibility. For this operation
            # dynamic_success additionally includes constraints; adapt only the
            # read-only view passed to that existing independent diagnostic.
            diagnostic = terminal_record(config, {**metrics, "dynamic_success": point}, rows, folder, "reference")
            if not diagnostic["passed"] or metrics["constrained_semantic_success"] != bool(point and not violations and diagnostic["visible"]):
                raise ValueError(f"Terminal evidence failed: {diagnostic}")
            replans = load_json(folder/"replans.json")
            if condition == "intervention":
                if violations or len(replans) != 1 or metrics["response_latency_steps"] != 0:
                    raise ValueError("Follow did not replan safely at delivery")
                route = replans[0]
                if (route["step"] != trigger or len(route["points"]) < 2 or
                        any(segment_intersects_region(a, b, event["task_update"]["region"], radius)
                            for a, b in zip(route["points"], route["points"][1:]))):
                    raise ValueError("Logged constrained route is not legal")
            elif replans:
                raise ValueError("Control/Ignore unexpectedly replanned")
            if condition == "ignore" and (not violations or metrics["response_latency_steps"] is not None
                                           or metrics["termination"] != "constraint_violation"):
                raise ValueError("Ignore did not fail specifically because of the added constraint")
            if load_json(folder/"final_task_state.json") != truth.to_dict() or load_json(folder/"final_policy_state.json") != policy.to_dict():
                raise ValueError("Final truth/policy snapshots differ")
            item.update({"passed": True, "point_goal_success": point,
                         "dynamic_success": metrics["dynamic_success"], "violation_count": len(violations),
                         "first_violation_step": metrics["first_constraint_violation_step"],
                         "path_length_m": length, "terminal_instance_pixels": metrics["terminal_instance_pixels"]})
            records[condition] = rows
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            item["error"] = str(error)
        results.append(item)
    # Equal physical execution for Ignore, and actual RGB-D/semantic payloads,
    # prevent an apparent negative control caused by a changed simulator run.
    pairing_errors = []
    try:
        control = records["control"]
        follow, ignore = records["intervention"], records["ignore"]
        reference_dir = Path(experiment["preparation"]["reference_dir"])
        reference = load_json(reference_dir/"trajectory.json")
        if control != reference or not matching_frames(reference_dir, reference, output/"control", control):
            raise ValueError("Control differs from the frozen preparation reference")
        if control[:trigger+1] != follow[:trigger+1] or not matching_frames(output/"control", control[:trigger+1], output/"intervention", follow[:trigger+1]):
            raise ValueError("Follow's pre-event trajectory or observations differ")
        keys = ("step", "action", "position", "rotation_xyzw", "collided", "distance_m")
        physical = lambda trajectory: [{key: row[key] for key in keys} for row in trajectory]
        if physical(control) != physical(ignore) or not matching_frames(output/"control", control, output/"ignore", ignore):
            raise ValueError("Ignore physical execution or observations differ from control")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        pairing_errors.append(str(error))
    statuses = load_json(output/"run_status.json")
    report = {"passed": all(item["passed"] for item in results) and not pairing_errors
              and {item["condition"] for item in statuses} == {"control", "intervention", "ignore"}
              and len(statuses) == 3 and all(item["completed"] for item in statuses),
              "expected_episodes": 3, "conditions": results, "pairing_errors": pairing_errors,
              "run_status": statuses, "protocol": "add_avoid_region_v1"}
    return report


def summarize(output):
    """Persist a fresh audit; batch callers can use check_experiment read-only."""
    report = check_experiment(output)
    write_json(output/"summary.json", report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="CPU only: freeze a region from an existing HSSD reference")
    preparing.add_argument("--reference-dir", type=Path, required=True)
    running = commands.add_parser("run", help="GPU: run control, follow, Ignore and audit all three")
    running.add_argument("--config", type=Path, required=True)
    running.add_argument("--gpu-id", type=int, default=0)
    commands.add_parser("summarize", help="CPU only: recheck saved states, poses and instance evidence")
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "run":
        if args.gpu_id < 0:
            parser.error("GPU index must be non-negative")
        return run_experiment(args)
    return 0 if summarize(args.output_dir)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
