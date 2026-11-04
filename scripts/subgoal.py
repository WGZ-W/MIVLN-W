"""CPU preparation/audit and GPU matched temporary-subgoal navigation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

from mivln import TaskState
from mivln.regions import trajectory_violations
from mivln.subgoals import COMPLETE_SUBGOAL, completion_is_valid, sequence_satisfied
from scripts.hssd_dataset import matching_frames, same_task, terminal_record, within
from scripts.hssd_targets import sha256, validate_catalog
from scripts.intervention_timing import check_reference, load_json, write_json
from scripts.navigation_scene import check_scene_match, navmesh_path, project_path


def pathfinder_for(config):
    """CPU mesh queries only; this never constructs a Simulator or renderer."""
    import habitat_sim

    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh_path(config))):
        raise ValueError("Cannot load frozen navigation mesh")
    return pathfinder


def distance_between(pathfinder, start, goal):
    import numpy as np
    from scripts.run_navigation import shortest_path

    path = shortest_path(pathfinder, np.asarray(start, dtype=np.float32), np.asarray(goal, dtype=np.float32))
    return float(path.geodesic_distance) if path is not None else None


def reviewed_centers(config, goals):
    """Bind both existing goal IDs to explicit reviewed instances and stopping points.

    The catalog is pre-existing evidence, not an automatic approval mechanism.
    Centres are for privileged heading checks; navigation still targets the
    reviewed stopping points, never the object centre itself.
    """
    source = config["catalog_source"]
    if sha256(Path(source["path"])) != source["sha256"]:
        raise ValueError("Reviewed catalog changed")
    targets = {target["target_id"]: target for target in validate_catalog(load_json(Path(source["path"])))}
    centers = {}
    for goal in goals:
        target, metadata = targets[goal], config["goal_metadata"][goal]
        if (target["template_id"] != metadata["template_id"]
                or target["source_instance_index"] != metadata["source_instance_index"]
                or target["stop_position"] != config["goal_positions"][goal]):
            raise ValueError(f"Goal differs from its reviewed instance: {goal}")
        centers[goal] = target["world_aabb"]["center"]
    return centers


def reference_trigger(rows, fraction=.25):
    """Freeze an event at a fraction of actual reference travel, before action k.

    Row k is the post-action-(k-1) pose, hence the pose before action k.
    Fractions refer to travelled metres, not action counts or remaining distance.
    A late trigger must still precede the reference's terminal action.
    """
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("Trigger progress must be finite and strictly between zero and one")
    distances = [row["distance_m"] for row in rows]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in distances):
        raise ValueError("Reference travel must contain finite nonnegative distances")
    total = sum(distances)
    if total <= 0:
        raise ValueError("Reference has no travelled distance")
    progress = 0.
    for index, row in enumerate(rows[1:], 1):
        progress += row["distance_m"]
        if progress >= total*fraction:
            if index >= len(rows)-2:
                break
            return index
    raise ValueError("Reference cannot support the fixed intervention trigger")


def prepare(args):
    """Derive a B-then-A task from an existing A/B configuration and A reference."""
    from scripts.run_navigation import scene_provenance, validate_config

    if args.output_dir.exists():
        raise FileExistsError("Use a new subgoal preparation directory")
    source_path, reference = args.source_config.resolve(), args.reference_dir.resolve()
    source = load_json(source_path)
    validate_config(source)
    provenance = scene_provenance(source, project_path(source["scene"]))
    rows = check_reference(source, reference)
    same_task(source, load_json(reference/"episode.json"))
    diagnostic = terminal_record(source, load_json(reference/"metrics.json"), rows, reference, "reference")
    if not diagnostic["passed"]:
        raise ValueError(f"Reference terminal evidence is invalid: {diagnostic}")
    root_goal = source["initial_goal"]
    candidates = [goal for goal in source["goal_positions"] if goal != root_goal]
    subgoal = getattr(args, "subgoal", None)
    if subgoal is None:
        if len(candidates) != 1:
            raise ValueError("Specify --subgoal from the existing reviewed goals")
        subgoal = candidates[0]
    if subgoal not in candidates:
        raise ValueError("Temporary subgoal must be a different existing goal")
    centers = reviewed_centers(source, (root_goal, subgoal))
    fraction = getattr(args, "progress_fraction", .25)
    trigger = reference_trigger(rows, fraction)
    pathfinder = pathfinder_for(source)
    legs = {"trigger_to_b_m": distance_between(pathfinder, rows[trigger]["position"], source["goal_positions"][subgoal]),
            "b_to_a_m": distance_between(pathfinder, source["goal_positions"][subgoal], source["goal_positions"][root_goal])}
    if any(distance is None or distance <= 2*source["success_radius_m"] for distance in legs.values()):
        raise ValueError("B and its trigger/restored A must be reachable and have separate success regions")
    config = deepcopy(source)
    config.update(provenance)
    config["episode_id"] = getattr(args, "episode_id", "hssd_subgoal_001")
    config["subgoal_protocol"] = {"enabled": True, "completion_action": COMPLETE_SUBGOAL}
    config["events"] = [{"event_id": "insert_001", "trigger": {"type": "step", "value": trigger},
        "utterance": f"First go to {source['goal_metadata'][subgoal].get('description', subgoal)}. "
                     f"Then resume going to {source['goal_metadata'][root_goal].get('description', root_goal)}.",
        "task_update": {"operation": "insert_subgoal", "subgoal": subgoal, "resume_goal": root_goal}}]
    validate_config(config)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/"config.json", config)
    report = {"kind": "hssd_subgoal_preparation_v1", "source_config": str(source_path),
              "source_config_sha256": sha256(source_path), "reference_dir": str(reference),
              "reference_hashes": {name: sha256(reference/name) for name in
                  ("episode.json", "trajectory.json", "metrics.json", "terminal_diagnostics.json")},
              "config_sha256": sha256(args.output_dir/"config.json"), "root_goal": root_goal,
              "subgoal": subgoal, "required_sequence": [subgoal, root_goal], "target_centers": centers,
              "trigger_step": trigger, "trigger_pose": rows[trigger]["position"], "geometry": legs,
              "progress_fraction": fraction,
              "trigger_protocol": "cumulative_reference_travel_before_action",
              "purpose": "Privileged B-then-A task, not language understanding"}
    write_json(args.output_dir/"preparation.json", report)
    print(json.dumps(report, indent=2))
    return 0


def validate_preparation(config_path):
    report = load_json(config_path.parent/"preparation.json")
    if report.get("kind") != "hssd_subgoal_preparation_v1" or sha256(config_path) != report["config_sha256"]:
        raise ValueError("Prepared subgoal configuration changed")
    source_path = Path(report["source_config"])
    if sha256(source_path) != report["source_config_sha256"]:
        raise ValueError("Source A/B configuration changed")
    required = {"episode.json", "trajectory.json", "metrics.json", "terminal_diagnostics.json"}
    if set(report["reference_hashes"]) != required:
        raise ValueError("Incomplete source reference hashes")
    for name, digest in report["reference_hashes"].items():
        if sha256(Path(report["reference_dir"])/name) != digest:
            raise ValueError(f"Reference changed: {name}")
    config, source = load_json(config_path), load_json(source_path)
    same_task(source, config)
    # Legacy single-case preparations used the implicit 25% protocol. Preserve
    # their hashes and results while checking new early/middle/late inputs too.
    rows = check_reference(source, Path(report["reference_dir"]))
    trigger = reference_trigger(rows, report.get("progress_fraction", .25))
    if (config["events"][0]["trigger"] != {"type": "step", "value": trigger}
            or report["trigger_step"] != trigger or report["trigger_pose"] != rows[trigger]["position"]
            or report["root_goal"] != config["initial_goal"]
            or report["required_sequence"] != [report["subgoal"], report["root_goal"]]
            or config["events"][0]["task_update"] != {"operation": "insert_subgoal",
                "subgoal": report["subgoal"], "resume_goal": report["root_goal"]}):
        raise ValueError("Prepared subgoal trigger or required sequence changed")
    if reviewed_centers(source, report["required_sequence"]) != report["target_centers"]:
        raise ValueError("Reviewed target centres changed")
    for key in ("scene", "scene_dataset_config", "navmesh"):
        if sha256(project_path(config[key])) != config[key+"_sha256"]:
            raise ValueError(f"Subgoal source changed: {key}")
    return report


def run_experiment(args):
    from scripts.run_navigation import run, scene_provenance, validate_config

    prepared = validate_preparation(args.config.resolve())
    config = load_json(args.config)
    validate_config(config)
    scene = project_path(config["scene"])
    scene_provenance(config, scene)
    if args.output_dir.exists():
        raise FileExistsError("Use a new subgoal experiment output directory")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/"config.json", config)
    write_json(args.output_dir/"experiment.json", {"source_config": str(args.config.resolve()),
        "config_sha256": sha256(args.output_dir/"config.json"), "preparation": prepared, "gpu_id": args.gpu_id})
    statuses = []
    for condition in ("control", "intervention", "ignore"):
        branch = deepcopy(config)
        branch["episode_id"] += "_"+condition
        if condition == "control":
            branch["events"] = []
        try:
            run(branch, scene, args.gpu_id, args.output_dir/condition,
                intervention_policy="ignore" if condition == "ignore" else "follow")
            statuses.append({"condition": condition, "completed": True, "error": None})
        except Exception as error:
            statuses.append({"condition": condition, "completed": False,
                             "error": f"{type(error).__name__}: {error}"})
        write_json(args.output_dir/"run_status.json", statuses)
    return 0 if summarize(args.output_dir)["passed"] else 1


def yaw_at(row, center):
    import numpy as np
    from habitat_sim.utils.common import quat_from_coeffs
    from scripts.terminal_view import yaw_action

    return yaw_action(SimpleNamespace(position=np.asarray(row["position"]),
        rotation=quat_from_coeffs(np.asarray(row["rotation_xyzw"]))), center, 7.5)[1]


def check_diagnostic(config, folder, rows, goal, distance, yaw, digest, negative=False):
    """Recount raw instance pixels and independently verify a stage's evidence."""
    import numpy as np
    from PIL import Image

    report = load_json(folder/"terminal_diagnostics.json")
    turns = sum(row.get("controller_phase") == "terminal_align" for row in rows)
    point = distance is not None and 0 <= distance <= config["success_radius_m"]
    view = {"final_goal": goal, "dynamic_success": point, "point_goal_success": point,
        "terminal_semantic_success": bool(point and report.get("instance_visible")),
        "terminal_instance_visible": report.get("instance_visible"), "terminal_instance_pixels": report.get("instance_pixels"),
        "terminal_diagnostics_sha256": digest, "terminal_alignment_complete": True,
        "terminal_yaw_error_degrees": yaw, "terminal_alignment_turns": turns}
    # Ignore aligns to policy A although its pending truth is B. Only its final
    # heading check uses A; stage B always uses B. All identities remain truth-side.
    checked = terminal_record(config, view, rows, folder, "ignore" if negative else "reference")
    if not checked["passed"]:
        raise ValueError(f"Invalid independent instance evidence: {checked}")
    mask = np.load(folder/"terminal_instance_mask.npy", allow_pickle=False)
    if (mask.dtype != np.bool_ or mask.shape != tuple(config["resolution"])
            or int(mask.sum()) != report["instance_pixels"]
            or not np.array_equal(mask, np.asarray(Image.open(folder/"terminal_instance_mask.png")) > 0)):
        raise ValueError("Raw instance mask disagrees with its diagnostic")
    return report


def check_action_displacement(pathfinder, config, previous, row):
    """Reject inconsistent motion; verify unusual projection with native replay.

    Habitat's navmesh movement filter can project a nominal 0.25 m forward
    endpoint beyond a simple Euclidean step bound near polygon boundaries.
    A larger displacement is accepted ONLY if the same no-sensor native Agent
    action with the frozen PathFinder reproduces the recorded endpoint. No
    Simulator, renderer, sensors or physics world is constructed for this check.
    Returns explicit evidence when the historical step bound needs this proof.
    """
    movement = math.dist(row["position"], previous["position"])
    if (not math.isclose(movement, row["distance_m"], abs_tol=1e-5)
            or row["action"] != "move_forward" and movement > 1e-5):
        raise ValueError("Recorded action displacement is inconsistent")
    if movement <= config["forward_step_m"]+.03:
        return None
    import habitat_sim
    import numpy as np
    from habitat_sim.utils.common import quat_from_coeffs

    graph = habitat_sim.SceneGraph()
    specification = habitat_sim.agent.AgentConfiguration()
    specification.sensor_specifications = []
    specification.height, specification.radius = config.get("agent_height", 1.5), config.get("agent_radius", .1)
    specification.action_space = {"move_forward": habitat_sim.agent.ActionSpec("move_forward",
        habitat_sim.agent.ActuationSpec(amount=config["forward_step_m"]))}
    agent = habitat_sim.Agent(graph.get_root_node().create_child(), specification,
        controls=habitat_sim.agent.ObjectControls(move_filter_fn=pathfinder.try_step))
    try:
        state = habitat_sim.AgentState()
        state.position = np.asarray(previous["position"], dtype=np.float32)
        state.rotation = quat_from_coeffs(np.asarray(previous["rotation_xyzw"]))
        agent.set_state(state)
        agent.act("move_forward")
        expected = agent.get_state().position.tolist()
        error = math.dist(expected, row["position"])
        if not math.isfinite(error) or error > 1e-5:
            raise ValueError("Recorded forward displacement differs from native navmesh action replay")
        return {"step": row["step"], "nominal_step_m": config["forward_step_m"],
                "measured_displacement_m": movement, "native_endpoint": expected,
                "endpoint_error_m": error, "simulator_created": False}
    finally:
        agent.close()


def check_experiment(output):
    """CPU-only independent replay: measure B, then restore truth and require A STOP.

    This does not use policy completion claims or cached summary flags as truth.
    It loads the frozen navmesh for geodesic checks; no Simulator is created.
    """
    import numpy as np
    from PIL import Image

    config, experiment = load_json(output/"config.json"), load_json(output/"experiment.json")
    if sha256(output/"config.json") != experiment["config_sha256"]:
        raise ValueError("Experiment configuration changed")
    prepared = validate_preparation(Path(experiment["source_config"]))
    if prepared != experiment["preparation"] or prepared["config_sha256"] != experiment["config_sha256"]:
        raise ValueError("Experiment differs from its frozen preparation")
    pathfinder = pathfinder_for(config)
    event = config["events"][0]
    root, subgoal, trigger = config["initial_goal"], event["task_update"]["subgoal"], event["trigger"]["value"]
    tolerance = config["terminal_alignment"].get("yaw_tolerance_degrees", 7.5)
    results, records = [], {}
    for condition in ("control", "intervention", "ignore"):
        item = {"condition": condition, "passed": False, "error": None}
        try:
            folder = output/condition
            episode, metrics, rows, logs, completions = (load_json(folder/name) for name in
                ("episode.json", "metrics.json", "trajectory.json", "events.json", "subgoal_events.json"))
            check_scene_match(config, episode)
            for key, value in config.items():
                if key not in ("episode_id", "events", "scene", "scene_dataset_config", "navmesh") and episode.get(key) != value:
                    raise ValueError(f"Episode changed frozen input: {key}")
            expected_events = [] if condition == "control" else [event]
            if episode["events"] != expected_events or len(rows) < trigger+2:
                raise ValueError("Wrong event configuration or trigger was never reached")
            truth, belief = TaskState(root), TaskState(root)
            validated, used, projection_checks = [], set(), []
            for index, row in enumerate(rows):
                action_step = index-1
                delivered = index > 0 and action_step == trigger and condition != "control"
                if delivered:
                    truth.apply_update(event["task_update"])
                    if condition == "intervention":
                        belief.apply_update(event["task_update"])
                if index and row["action"] == COMPLETE_SUBGOAL:
                    if condition != "intervention" or action_step <= trigger or not belief.suspended_goals:
                        raise ValueError("Unexpected or premature COMPLETE_SUBGOAL")
                    matches = [n for n, record in enumerate(completions) if record["step"] == action_step]
                    if len(matches) != 1 or matches[0] in used:
                        raise ValueError("Missing, duplicate or reused completion evidence")
                    used.add(matches[0])
                    receipt = completions[matches[0]]
                    before_truth, before_belief = truth.to_dict(), belief.to_dict()
                    pose = {key: row[key] for key in ("position", "rotation_xyzw")}
                    if (receipt["goal_id"] != subgoal or receipt["pose_before"] != pose or receipt["pose_after"] != pose
                            or any(row[key] != rows[index-1][key] for key in ("position", "rotation_xyzw"))):
                        raise ValueError("Completion changed pose or names the wrong goal")
                    distance = distance_between(pathfinder, row["position"], config["goal_positions"][subgoal])
                    yaw = yaw_at(row, prepared["target_centers"][subgoal])
                    evidence_folder = within(folder, receipt["diagnostic_dir"])
                    diagnostic = check_diagnostic(config, evidence_folder, rows[:index+1], subgoal, distance, yaw, receipt["diagnostic_sha256"])
                    if not np.array_equal(np.asarray(Image.open(evidence_folder/"terminal_rgb.png")),
                                          np.asarray(Image.open(within(folder, row["frames"]["rgb"])))):
                        raise ValueError("Completion RGB differs from its trajectory frame")
                    valid = completion_is_valid(distance, config["success_radius_m"], yaw, tolerance, diagnostic)
                    if (not valid or receipt["valid"] is not True or distance is None
                            or not math.isclose(receipt["distance_m"], distance, abs_tol=1e-5)
                            or not math.isclose(receipt["yaw_error_degrees"], yaw, abs_tol=1e-4)):
                        raise ValueError("Subgoal completion not supported by measured geometry and visibility")
                    truth.complete_subgoal(subgoal)
                    belief.complete_subgoal(subgoal)
                    if (receipt["task_state_before"] != before_truth or receipt["task_state_after"] != truth.to_dict()
                            or receipt["policy_task_state_before"] != before_belief
                            or receipt["policy_task_state_after"] != belief.to_dict()):
                        raise ValueError("Completion/restoration state snapshots differ")
                    validated.append(receipt)
                if (row["step"] != (None if not index else action_step)
                        or row["task_state"] != truth.to_dict() or row["policy_task_state"] != belief.to_dict()
                        or row["active_goal"] != truth.active_goal or row["policy_goal"] != belief.active_goal
                        or row["delivered_event_ids"] != ([event["event_id"]] if delivered else [])):
                    raise ValueError("Task/policy state or action timeline differs from independent replay")
                if row["action"] not in ("INITIALIZE", "move_forward", "turn_left", "turn_right", "STOP", COMPLETE_SUBGOAL):
                    raise ValueError("Unknown protocol action")
                if index and row["action"] in ("INITIALIZE", "STOP") and index != len(rows)-1:
                    raise ValueError("Initialization/STOP occurred inside the episode")
                if not pathfinder.is_navigable(np.asarray(row["position"], dtype=np.float32)):
                    raise ValueError("Recorded pose is not navigable")
                for kind in ("rgb", "depth", "semantic"):
                    frame = within(folder, row["frames"][kind])
                    if not frame.is_file() or frame.stat().st_size == 0:
                        raise ValueError("Missing or empty observation")
                if index:
                    checked_action = check_action_displacement(pathfinder, config, rows[index-1], row)
                    if checked_action is not None:
                        projection_checks.append(checked_action)
            if rows[0]["action"] != "INITIALIZE" or rows[-1]["action"] != "STOP" or len(used) != len(completions):
                raise ValueError("Missing final STOP or unused completion claims")
            if (math.dist(rows[0]["position"], config["start_position"]) > 1e-5
                    or math.dist(rows[0]["rotation_xyzw"], config["start_rotation_xyzw"]) > 1e-5):
                raise ValueError("Initial pose changed")
            if len(logs) != len(expected_events):
                raise ValueError("Human event count differs")
            if logs:
                logged = logs[0]
                if (logged["event_id"] != event["event_id"] or logged["task_update"] != event["task_update"]
                        or logged["delivered_step"] != trigger
                        or logged["pose_before"] != {key: rows[trigger][key] for key in ("position", "rotation_xyzw")}
                        or logged["pose_after"] != logged["pose_before"]
                        or logged["task_state_before"] != rows[trigger]["task_state"]
                        or logged["task_state_after"] != rows[trigger+1]["task_state"]
                        or logged["policy_task_state_before"] != rows[trigger]["policy_task_state"]
                        or logged["policy_task_state_after"] != rows[trigger+1]["policy_task_state"]
                        or logged["ignored_by_policy"] != (condition == "ignore")):
                    raise ValueError("Human event did not preserve pose and state transition")
            root_distance = distance_between(pathfinder, rows[-1]["position"], config["goal_positions"][root])
            active_distance = distance_between(pathfinder, rows[-1]["position"], config["goal_positions"][truth.active_goal])
            point = active_distance is not None and active_distance <= config["success_radius_m"]
            sequence = sequence_satisfied(root, subgoal, truth, validated) if condition != "control" else True
            root_success = root_distance is not None and root_distance <= config["success_radius_m"]
            joint = bool(root_success and sequence and not trajectory_violations(rows, config.get("agent_radius", .1)))
            length = sum(math.dist(a["position"], b["position"]) for a, b in zip(rows, rows[1:]))
            if (not root_success or metrics["root_goal_success"] != root_success
                    or metrics["point_goal_success"] != point or metrics["dynamic_success"] != joint or metrics["success"] != joint
                    or metrics["sequence_success"] != joint or metrics["root_goal"] != root
                    or metrics["required_sequence"] != ([root] if condition == "control" else [subgoal, root])
                    or metrics["subgoal_completed"] != bool(validated)
                    or metrics["resumed_original_goal"] != (condition == "intervention" and sequence)
                    or metrics["steps"] != len(rows)-1 or not len(rows)-1 <= config["max_steps"]
                    or not metrics["stopped"] or metrics["max_steps"] != config["max_steps"]
                    or metrics["planner_error"] is not None or not math.isclose(metrics["path_length_m"], length, abs_tol=1e-5)
                    or active_distance is None or not math.isclose(metrics["navigation_error_m"], active_distance, abs_tol=1e-5)
                    or metrics["final_goal"] != truth.active_goal or metrics["policy_goal"] != belief.active_goal
                    or metrics["stopped_goal"] != root or metrics["goal_update_compliance"] != point
                    or metrics["collision_count"] != sum(bool(row["collided"]) for row in rows)
                    or metrics["intervention_policy"] != ("ignore" if condition == "ignore" else "follow")
                    or metrics["intervention_received"] != (condition != "control")
                    or metrics["response_latency_steps"] != (0 if condition == "intervention" else None)
                    or metrics["intervention_step"] != (None if condition == "control" else trigger)
                    or metrics["scheduled_event_count"] != len(expected_events) or metrics["delivered_event_count"] != len(expected_events)
                    or metrics["ignored_event_count"] != (1 if condition == "ignore" else 0)):
                raise ValueError("Metrics disagree with measured ordered completion and final STOP")
            yaw = yaw_at(rows[-1], prepared["target_centers"][belief.active_goal])
            final_report = check_diagnostic(config, folder, rows, truth.active_goal, active_distance, yaw,
                metrics["terminal_diagnostics_sha256"], negative=condition == "ignore")
            if (not math.isclose(metrics["terminal_yaw_error_degrees"], yaw, abs_tol=1e-4)
                    or metrics["terminal_alignment_complete"] is not True
                    or metrics["terminal_alignment_turns"] != sum(row.get("controller_phase") == "terminal_align" for row in rows)
                    or metrics["terminal_visibility_evaluated"] is not True
                    or metrics["terminal_instance_visible"] != final_report["instance_visible"]
                    or metrics["terminal_instance_pixels"] != final_report["instance_pixels"]
                    or metrics["terminal_semantic_success"] != bool(point and metrics["terminal_instance_visible"])):
                raise ValueError("Final terminal metrics disagree")
            replans = load_json(folder/"replans.json")
            if condition == "intervention":
                if (not joint or len(replans) != 2 or [entry["goal_id"] for entry in replans] != [subgoal, root]
                        or [entry["step"] for entry in replans] != [trigger, validated[0]["step"]]
                        or replans[1].get("reason") != "resume_goal"):
                    raise ValueError("Follow did not perform one B stage then restore A")
            elif replans or completions:
                raise ValueError("Control/Ignore claimed a subgoal or replan")
            if condition == "ignore" and (joint or metrics["termination"] != "subgoal_sequence_incomplete"):
                raise ValueError("Ignore did not fail specifically for omitting the temporary task")
            if load_json(folder/"final_task_state.json") != truth.to_dict() or load_json(folder/"final_policy_state.json") != belief.to_dict():
                raise ValueError("Final truth/policy states disagree")
            item.update({"passed": True, "root_goal_success": root_success, "point_goal_success": point,
                         "subgoal_completed": bool(validated), "sequence_success": joint,
                         "dynamic_success": joint, "steps": metrics["steps"], "path_length_m": length,
                         "completion_step": validated[0]["step"] if validated else None,
                         "native_projection_checks": projection_checks})
            records[condition] = rows
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            item["error"] = str(error)
        results.append(item)
    pairing_errors = []
    try:
        control, follow, ignore = (records[name] for name in ("control", "intervention", "ignore"))
        reference_dir = Path(prepared["reference_dir"])
        reference = load_json(reference_dir/"trajectory.json")
        if control != reference or not matching_frames(reference_dir, reference, output/"control", control):
            raise ValueError("Control differs from the frozen A reference")
        if control[:trigger+1] != follow[:trigger+1] or not matching_frames(output/"control", control[:trigger+1], output/"intervention", follow[:trigger+1]):
            raise ValueError("Follow differs before insertion")
        keys = ("step", "action", "position", "rotation_xyzw", "distance_m", "collided")
        physical = lambda rows: [{key: row[key] for key in keys} for row in rows]
        if physical(control) != physical(ignore) or not matching_frames(output/"control", control, output/"ignore", ignore):
            raise ValueError("Ignore physical execution differs from Control")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        pairing_errors.append(str(error))
    statuses = load_json(output/"run_status.json")
    passed = (all(item["passed"] for item in results) and not pairing_errors and len(statuses) == 3
              and {item["condition"] for item in statuses} == {"control", "intervention", "ignore"}
              and all(item["completed"] for item in statuses))
    return {"passed": passed, "protocol": "insert_subgoal_v1",
            "audit_protocol": "native_projection_endpoint_v2", "expected_episodes": 3,
            "conditions": results, "pairing_errors": pairing_errors, "run_status": statuses}


def summarize(output):
    report = check_experiment(output)
    write_json(output/"summary.json", report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="CPU: derive one reviewed B-then-A task")
    preparing.add_argument("--source-config", type=Path, required=True)
    preparing.add_argument("--reference-dir", type=Path, required=True)
    preparing.add_argument("--subgoal")
    preparing.add_argument("--progress-fraction", type=float, default=.25,
                           help="Fraction of actual reference travel, frozen before an action (default .25)")
    running = commands.add_parser("run", help="GPU: matched Control/Follow/Ignore with ordered completion")
    running.add_argument("--config", type=Path, required=True)
    running.add_argument("--gpu-id", type=int, default=0)
    commands.add_parser("summarize", help="CPU Habitat mesh queries: audit stages, masks and order")
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
