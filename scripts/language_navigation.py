"""Prepare/run/audit one reviewed HSSD language goal-replacement experiment.

Three matched branches use the same navigator: labelled Oracle update, model
update, and Ignore. Preparation and auditing are CPU operations; only run starts
renderers and calls a model. Model inputs contain no geometry or reference label.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError

from mivln import EpisodeEvaluator, TaskState
from mivln.navigation_updates import (apply_navigation_prediction, navigation_request,
                                     validate_navigation_candidates)
from mivln.task_update_prompts import encode, fingerprint, system_prompt
from scripts import hssd_dataset, predict_task_updates as client, subgoal
from scripts import task_update as base
from scripts.export_video import export_video
from scripts.hssd_targets import sha256, validate_catalog
from scripts.intervention_timing import check_reference, select_triggers
from scripts.navigation_scene import check_scene_match, project_path


PROTOCOL = "hssd_language_replacement_v1"
CONDITIONS = ("oracle", "model", "ignore")
JOURNAL_FILES = ("model_inputs.jsonl", "model_responses.jsonl", "model_system_prompt.txt", "run_status.json")


def lines(path):
    return [base.strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def code_hashes():
    return {name: sha256(base.ROOT / name) for name in (
        "scripts/language_navigation.py", "scripts/run_navigation.py", "mivln/navigation_updates.py",
        "mivln/task_update.py", "mivln/task_state.py", "mivln/evaluator.py", "scripts/predict_task_updates.py",
        "scripts/subgoal.py", "scripts/terminal_view.py", "mivln/task_update_prompts.py", "mivln/event_engine.py",
        "scripts/task_update.py", "scripts/intervention_timing.py", "scripts/navigation_scene.py",
        "scripts/hssd_dataset.py", "scripts/hssd_targets.py")}


def reviewed_candidates(config):
    """Bind public names to existing reviewed instances, never to guessed points."""
    source = config["catalog_source"]
    if sha256(Path(source["path"])) != source["sha256"]:
        raise ValueError("Reviewed catalog changed")
    targets = {item["target_id"]: item for item in validate_catalog(base.read_json(Path(source["path"])))}
    result = []
    for identity in sorted(config["goal_positions"]):
        target, metadata = targets[identity], config["goal_metadata"][identity]
        description = target["review"]["description"]
        if (target["stop_position"] != config["goal_positions"][identity]
                or target["template_id"] != metadata["template_id"]
                or target["source_instance_index"] != metadata["source_instance_index"]
                or description != metadata["description"]):
            raise ValueError("Navigation target differs from reviewed instance/description")
        result.append({"goal_id": identity, "description": description, "aliases": [description]})
    validate_navigation_candidates(result, config["goal_positions"])
    return result


def derive_config(source, task_id, timing):
    config = deepcopy(source)
    if (len(config.get("events", [])) != 1 or config["events"][0]["task_update"].get("operation") != "replace_goal"
            or config["events"][0]["task_update"]["old_goal"] != config["initial_goal"]
            or config.get("subgoal_protocol") is not None
            or not config.get("terminal_alignment", {}).get("enabled")
            or not config.get("terminal_visibility", {}).get("enabled")):
        raise ValueError("Use one reviewed replacement with terminal facing and instance visibility")
    config["episode_id"] = f"language_{task_id}_{timing}"
    config["language_candidates"] = reviewed_candidates(config)
    return config


def prepare(args):
    """Reuse a verified timing from the fixed task collection; sample nothing."""
    source = args.source_dir.resolve()
    manifest = base.read_json(source / "manifest.json")
    hssd_dataset.validate_inputs(source, manifest)
    selected = [task for task in manifest["tasks"] if task["task_id"] == args.task_id]
    if len(selected) != 1:
        raise ValueError("Task ID is not in the frozen collection")
    task = selected[0]
    task_dir = hssd_dataset.within(source, task["output_subdir"])
    source_config = task_dir / f"timing/configs/{args.timing}.json"
    original = base.read_json(source_config)
    frozen = base.read_json(hssd_dataset.within(source, task["base_config"]))
    hssd_dataset.same_task(frozen, original)
    expected_event = deepcopy(frozen["events"][0])
    expected_event["event_id"] = original["events"][0]["event_id"]
    reference = task_dir / "reference"
    rows = check_reference(frozen, reference)
    trigger = next(case["trigger_step"] for case in select_triggers(rows) if case["name"] == args.timing)
    expected_event["trigger"]["value"] = trigger
    if original["events"] != [expected_event]:
        raise ValueError("Timing changed its task, message, reference update or measured trigger")
    config = derive_config(original, args.task_id, args.timing)
    # Preparation imports neither run_navigation nor Simulator. Validate hashes
    # of the scene/mesh too; runtime performs the full native configuration check.
    sources = [source / "manifest.json", source_config, hssd_dataset.within(source, task["base_config"]),
        hssd_dataset.within(source, task["reference_config"]), reference / "episode.json",
        reference / "metrics.json", reference / "trajectory.json", Path(config["catalog_source"]["path"])]
    for key in ("scene", "scene_dataset_config", "navmesh"):
        path = project_path(config[key])
        if sha256(path) != config[key + "_sha256"]:
            raise ValueError(f"Frozen scene input changed: {key}")
        sources.append(path)
    output = base.fresh_output(args.output_dir)
    base.write_json(output / "config.json", config)
    base.write_json(output / "manifest.json", {"schema_version": 1, "kind": PROTOCOL,
        "conditions": list(CONDITIONS), "task_id": args.task_id, "timing": args.timing,
        "source_config": str(source_config), "reference_dir": str(reference),
        "config_sha256": sha256(output / "config.json"),
        "sources": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in sources],
        "scope": "Single replacement; reviewed candidates and privileged Oracle navigation; no visual grounding claim"})
    print(f"Prepared {args.task_id}/{args.timing}: trigger {trigger}; {len(config['language_candidates'])} public candidates; no model or GPU calls")
    return 0


def load_preparation(folder):
    manifest = base.read_json(folder / "manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != PROTOCOL
            or manifest.get("conditions") != list(CONDITIONS)):
        raise ValueError("Unsupported language-navigation preparation")
    if sha256(folder / "config.json") != manifest["config_sha256"]:
        raise ValueError("Prepared configuration changed")
    for record in manifest["sources"]:
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Preparation source changed: " + record["path"])
    config = base.read_json(folder / "config.json")
    expected = derive_config(base.read_json(Path(manifest["source_config"])), manifest["task_id"], manifest["timing"])
    if config != expected:
        raise ValueError("Prepared task/public catalog differs from the source")
    return manifest, config


def model_payload(controls, system, request):
    return client.payload_for(controls["model"], system, request, controls["temperature"],
                              controls["max_tokens"], controls["thinking_mode"])


class ModelClient:
    """One stateless raw call per delivered event; append exact input/outcome."""

    def __init__(self, controls, system, output, api_key_env):
        self.controls, self.system, self.output = controls, system, output
        self.endpoint = client.endpoint_for(controls["base_url"])
        self.api_key = os.environ.get(api_key_env, "")

    def __call__(self, request):
        payload = model_payload(self.controls, self.system, request)
        record = {"request_id": request["request_id"], "request_sha256": fingerprint(request),
                  "payload": payload, "payload_sha256": fingerprint(payload)}
        with (self.output / "model_inputs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encode(record) + "\n")
        raw, metadata, error = None, {}, None
        try:
            raw, metadata = client.complete_payload(self.endpoint, payload, self.api_key, self.controls["timeout_seconds"])
        except HTTPError as failure:
            error = f"HTTP error {failure.code}"
            if failure.fp is not None:
                failure.close()
        except (URLError, OSError, ValueError, KeyError, IndexError, TypeError) as failure:
            error = f"Model call failed: {type(failure).__name__}"
        except KeyboardInterrupt:
            error = "Interrupted during model call"
            raise
        finally:
            with (self.output / "model_responses.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(encode({"request_id": request["request_id"], "raw_prediction": raw,
                                     "metadata": metadata, "error": error}) + "\n")
        return raw, metadata, error


def run_experiment(args):
    """Run all three planned conditions; task/model failures never remove slots."""
    prepared = args.prepared_dir.resolve()
    _, config = load_preparation(prepared)
    if (args.gpu_id < 0 or not args.model.strip() or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.temperature) or args.temperature < 0 or args.max_tokens < 1
            or args.thinking_mode not in ("off", "on", "default")):
        raise ValueError("Invalid GPU/model generation parameters")
    client.endpoint_for(args.base_url)
    from scripts.run_navigation import run, scene_provenance, validate_config
    validate_config(config)
    scene_provenance(config, project_path(config["scene"]))
    output = base.fresh_output(args.output_dir)
    controls = {"model": args.model, "base_url": args.base_url, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "timeout_seconds": args.timeout, "thinking_mode": args.thinking_mode}
    system = system_prompt(base.INSTRUCTIONS)
    (output / "model_system_prompt.txt").write_text(system, encoding="utf-8")
    for name in JOURNAL_FILES[:2]:
        (output / name).touch(exist_ok=False)
    updater = ModelClient(controls, system, output, args.api_key_env)
    statuses = [{"condition": name, "completed": False, "error": "Not run"} for name in CONDITIONS]
    run_record = {"kind": PROTOCOL, "prepared_dir": str(prepared),
        "prepared_manifest_sha256": sha256(prepared / "manifest.json"), "controls": controls,
        "gpu_id": args.gpu_id, "system_prompt_sha256": sha256(output / "model_system_prompt.txt"),
        "started_at": datetime.now(timezone.utc).isoformat(), "code_sha256": code_hashes()}
    base.write_json(output / "run.json", run_record)
    base.write_json(output / "run_status.json", statuses)
    for status in statuses:
        condition = status["condition"]
        branch = deepcopy(config)
        branch["episode_id"] += "_" + condition
        interrupted = False
        try:
            print(f"Running {condition}...", flush=True)
            run(branch, project_path(config["scene"]), args.gpu_id, output / condition,
                intervention_policy={"oracle": "follow", "model": "model", "ignore": "ignore"}[condition],
                task_updater=updater if condition == "model" else None)
            status.update(completed=True, error=None)
        except KeyboardInterrupt:
            status["error"], interrupted = "Interrupted", True
        except Exception as error:
            status["error"] = f"{type(error).__name__}: {error}"
        base.write_json(output / "run_status.json", statuses)
        if interrupted:
            break
    run_record.update(finished_at=datetime.now(timezone.utc).isoformat(),
                      files_sha256={name: sha256(output / name) for name in JOURNAL_FILES})
    base.write_json(output / "run.json", run_record)
    return 0 if summarize(output)["passed"] else 1


def replay_model(config, output, run):
    """Rebuild the exact model boundary without an HTTP call or reference repair."""
    records = base.read_json(output / "model/policy_updates.json")
    inputs, responses = [lines(output / name) for name in JOURNAL_FILES[:2]]
    if len(records) != 1 or len(inputs) != 1 or len(responses) != 1:
        raise ValueError("One delivered event must have exactly one input, response and policy update")
    event = config["events"][0]
    state = TaskState(config["initial_goal"])
    request = navigation_request(event["trigger"]["value"], event["utterance"], config["language_candidates"], state)
    payload = model_payload(run["controls"], system_prompt(base.INSTRUCTIONS), request)
    if inputs[0] != {"request_id": request["request_id"], "request_sha256": fingerprint(request),
                     "payload": payload, "payload_sha256": fingerprint(payload)}:
        raise ValueError("Actual model input differs from the public policy-only request")
    response = responses[0]
    if response["request_id"] != request["request_id"]:
        raise ValueError("Response belongs to another request")
    state, expected = apply_navigation_prediction(state, request, response["raw_prediction"], response["metadata"], response["error"])
    record = records[0]
    if ({key: record[key] for key in expected} != expected or record["event_id"] != event["event_id"]
            or record["delivered_step"] != event["trigger"]["value"]
            or not isinstance(record["callback_seconds"], (int, float))
            or not math.isfinite(record["callback_seconds"]) or record["callback_seconds"] < 0):
        raise ValueError("Model transition differs from raw response replay")
    correct = record["outcome"] == "applied" and record["prediction"] == {"status": "ok", "task_update": event["task_update"]}
    return state, record, correct


def audit_episode(config, directory, condition, pathfinder, centers, model_state=None, model_record=None):
    """Audit truth/policy timelines and measured STOP, including valid failures.

    Integrity and success are separate: a malformed prediction can produce an
    honestly failed navigation episode with perfectly valid observations/logs.
    """
    import numpy as np
    from PIL import Image
    episode, metrics, rows, events = [base.read_json(directory / name) for name in
        ("episode.json", "metrics.json", "trajectory.json", "events.json")]
    hssd_dataset.same_task(config, episode)
    event = config["events"][0]
    trigger = event["trigger"]["value"]
    policy_name = {"oracle": "follow", "model": "model", "ignore": "ignore"}[condition]
    if (episode["events"] != config["events"] or episode.get("language_candidates") != config["language_candidates"]
            or metrics["intervention_policy"] != policy_name or episode["intervention_policy"] != policy_name
            or not rows or rows[0]["action"] != "INITIALIZE" or len(rows) <= trigger + 1
            or metrics["steps"] != len(rows) - 1 or metrics["max_steps"] != config["max_steps"]
            or math.dist(rows[0]["position"], config["start_position"]) > 1e-5
            or min(math.dist(rows[0]["rotation_xyzw"], config["start_rotation_xyzw"]),
                   math.dist(rows[0]["rotation_xyzw"], [-v for v in config["start_rotation_xyzw"]])) > 1e-5):
        raise ValueError("Episode settings, initial pose or step roster changed")
    truth, policy = TaskState(config["initial_goal"]), TaskState(config["initial_goal"])
    projection_checks = []
    for index, row in enumerate(rows):
        if index == trigger + 1:
            truth.apply_update(event["task_update"])
            if condition == "oracle":
                policy.apply_update(event["task_update"])
            elif condition == "model":
                policy = deepcopy(model_state)
        if (row["step"] != (index - 1 if index else None) or row["task_state"] != truth.to_dict()
                or row["active_goal"] != truth.active_goal or row["policy_task_state"] != policy.to_dict()
                or row["policy_goal"] != policy.active_goal
                or row["delivered_event_ids"] != ([event["event_id"]] if index == trigger + 1 else [])):
            raise ValueError("Truth/policy state or action delivery differs from independent replay")
        if row["action"] not in ("INITIALIZE", "move_forward", "turn_left", "turn_right", "STOP"):
            raise ValueError("Unexpected navigation action")
        if index:
            checked = subgoal.check_action_displacement(pathfinder, config, rows[index - 1], row)
            if checked:
                projection_checks.append(checked)
        for key in ("rgb", "depth", "semantic"):
            path = hssd_dataset.within(directory, row["frames"][key])
            if not path.is_file():
                raise ValueError("Missing sensor frame")
            if key == "rgb":
                with Image.open(path) as image:
                    if image.size != (config["resolution"][1], config["resolution"][0]):
                        raise ValueError("RGB resolution changed")
                    image.load()
            else:
                array = np.load(path, allow_pickle=False)
                if array.shape != tuple(config["resolution"]) or not np.isfinite(array).all():
                    raise ValueError("Invalid depth/semantic observation")
                if key == "semantic" and not np.issubdtype(array.dtype, np.integer):
                    raise ValueError("Semantic frame is not an integer array")
    pose = lambda row: {key: row[key] for key in ("position", "rotation_xyzw")}
    if (len(events) != 1 or any(events[0][key] != value for key, value in event.items())
            or events[0]["delivered_step"] != trigger or events[0]["pose_before"] != pose(rows[trigger])
            or events[0]["pose_after"] != pose(rows[trigger])
            or events[0]["task_state_before"] != rows[trigger]["task_state"]
            or events[0]["task_state_after"] != rows[trigger + 1]["task_state"]
            or events[0]["policy_task_state_before"] != rows[trigger]["policy_task_state"]
            or events[0]["policy_task_state_after"] != rows[trigger + 1]["policy_task_state"]
            or events[0]["ignored_by_policy"] != (condition == "ignore")
            or metrics["scheduled_event_count"] != 1 or metrics["delivered_event_count"] != 1):
        raise ValueError("Event receipt, pose continuity or before-action state update changed")
    for filename, expected in (("final_task_state.json", truth.to_dict()), ("final_policy_state.json", policy.to_dict())):
        if base.read_json(directory / filename) != expected:
            raise ValueError("Final state snapshot changed")
    replans = base.read_json(directory / "replans.json")
    should_replan = condition == "oracle" or condition == "model" and model_record["outcome"] == "applied"
    if len(replans) != int(should_replan):
        raise ValueError("Planner reset does not match the actual policy update")
    if should_replan:
        planned = replans[0]
        distance = subgoal.distance_between(pathfinder, rows[trigger]["position"], config["goal_positions"][policy.active_goal])
        if (planned["goal_id"] != policy.active_goal or planned["step"] != trigger
                or math.dist(planned["goal"], config["goal_positions"][policy.active_goal]) > 1e-5
                or math.dist(planned["start"], rows[trigger]["position"]) > 1e-5
                or distance is None or not math.isclose(distance, planned["distance_m"], abs_tol=1e-4)):
            raise ValueError("Recorded replan uses a reference target instead of the actual policy target")
    length = sum(row["distance_m"] for row in rows[1:])
    if (not math.isclose(length, metrics["path_length_m"], abs_tol=1e-5)
            or metrics["collision_count"] != sum(bool(row["collided"]) for row in rows)
            or metrics["stopped"] != (rows[-1]["action"] == "STOP")
            or any(row["action"] == "STOP" for row in rows[:-1])):
        raise ValueError("Motion or STOP accounting differs from raw actions")
    distances = {goal: subgoal.distance_between(pathfinder, rows[-1]["position"], point)
                 for goal, point in config["goal_positions"].items()}
    for goal, distance in distances.items():
        cached = metrics["goal_distances_m"][goal]
        if (distance is None) != (cached is None) or distance is not None and not math.isclose(distance, cached, abs_tol=1e-4):
            raise ValueError("Final distances differ from independent navmesh queries")
    compliant = trigger if condition == "oracle" or condition == "model" and model_state.to_dict() == truth.to_dict() else None
    evaluated = EpisodeEvaluator().evaluate_navigation(task_state=truth, stopped=metrics["stopped"], goal_distances_m=distances,
        success_radius_m=config["success_radius_m"], intervention_step=trigger, first_compliant_action_step=compliant,
        path_length=length, steps=len(rows) - 1, max_steps=config["max_steps"])
    if any(metrics[key] != value for key, value in evaluated.to_dict().items()):
        raise ValueError("Saved success/compliance differs from independent navigation evaluation")
    diagnostic = base.read_json(directory / "terminal_diagnostics.json")
    if sha256(directory / "terminal_diagnostics.json") != metrics["terminal_diagnostics_sha256"] or not diagnostic["evaluated"]:
        raise ValueError("Terminal diagnostic missing or changed")
    target = config["goal_metadata"][truth.active_goal]
    if (diagnostic["goal_id"] != truth.active_goal or diagnostic["template_id"] != target["template_id"]
            or diagnostic["source_instance_index"] != target["source_instance_index"]
            or diagnostic["method"] != "isolated_instance_semantic_relabel_v1"
            or not diagnostic["labels_restored"] or not diagnostic["pose_unchanged"]
            or diagnostic["pose"] != pose(rows[-1])):
        raise ValueError("Terminal evidence does not identify the evaluator's target at STOP")
    filenames = {"terminal_instance_mask.npy", "terminal_instance_mask.png", "terminal_rgb.png"}
    if len(diagnostic["files"]) != 3 or {entry["path"] for entry in diagnostic["files"]} != filenames:
        raise ValueError("Incomplete terminal evidence")
    for record in diagnostic["files"]:
        if sha256(hssd_dataset.within(directory, record["path"])) != record["sha256"]:
            raise ValueError("Terminal evidence payload changed")
    mask = np.load(directory / "terminal_instance_mask.npy", allow_pickle=False)
    with Image.open(directory / "terminal_instance_mask.png") as image:
        png_mask = np.asarray(image) > 0
    pixels = int(mask.sum())
    visible = pixels >= config["terminal_visibility"]["min_pixels"]
    if (mask.dtype != np.bool_ or mask.shape != tuple(config["resolution"])
            or not np.array_equal(mask, png_mask)
            or pixels != diagnostic["instance_pixels"] or diagnostic["min_pixels"] != config["terminal_visibility"]["min_pixels"]
            or diagnostic["instance_visible"] != visible or metrics["terminal_instance_pixels"] != pixels
            or metrics["terminal_instance_visible"] != visible):
        raise ValueError("Instance pixels disagree with raw diagnostic masks")
    yaw = subgoal.yaw_at(rows[-1], centers[policy.active_goal])
    aligned = bool(metrics["stopped"] and abs(yaw) <= config["terminal_alignment"]["yaw_tolerance_degrees"] + 1e-5)
    if (metrics["terminal_alignment_complete"] != aligned
            or not math.isclose(yaw, metrics["terminal_yaw_error_degrees"], abs_tol=1e-4)
            or metrics["terminal_alignment_turns"] != sum(row.get("controller_phase") == "terminal_align" for row in rows)
            or metrics["success"] != evaluated.dynamic_success or metrics["point_goal_success"] != evaluated.dynamic_success
            or metrics["terminal_visibility_evaluated"] is not True or metrics["policy_goal"] != policy.active_goal
            or metrics["success_radius_m"] != config["success_radius_m"]
            or (metrics["navigation_error_m"] is None) != (distances[truth.active_goal] is None)
            or distances[truth.active_goal] is not None and not math.isclose(metrics["navigation_error_m"], distances[truth.active_goal], abs_tol=1e-4)
            or metrics["terminal_semantic_success"] != bool(evaluated.dynamic_success and visible)):
        raise ValueError("Terminal pose/visibility metrics differ from independent evidence")
    if condition == "model":
        if (events[0].get("model_update_index") != 0 or metrics["model_update_count"] != 1
                or metrics["model_update_applied_count"] != int(model_record["outcome"] == "applied")
                or metrics["model_call_failure_count"] != int(model_record["outcome"] == "call_failed")
                or metrics["model_callback_seconds"] != model_record["callback_seconds"]):
            raise ValueError("Model callback accounting changed")
    old = config["initial_goal"]
    negative = bool(condition == "ignore" and metrics["stopped"] and distances[old] is not None
        and distances[old] <= config["success_radius_m"] and not evaluated.dynamic_success
        and metrics["planner_error"] is None and metrics["termination"] == "stopped_at_cancelled_goal")
    return {"condition": condition, "integrity_passed": True, "dynamic_success": evaluated.dynamic_success,
        "joint_success": bool(evaluated.dynamic_success and aligned and visible), "ignore_expected_failure": negative,
        "steps": len(rows) - 1, "path_length_m": length, "navigation_error_m": distances[truth.active_goal],
        "native_projection_checks": projection_checks, "error": None}


def check_experiment(output):
    run = base.read_json(output / "run.json")
    prepared = Path(run["prepared_dir"])
    if run.get("kind") != PROTOCOL or sha256(prepared / "manifest.json") != run["prepared_manifest_sha256"]:
        raise ValueError("Experiment preparation changed")
    _, config = load_preparation(prepared)
    if run["code_sha256"] != code_hashes():
        raise ValueError("Execution/audit code changed; retain old evidence before re-auditing")
    for name in JOURNAL_FILES:
        if sha256(output / name) != run["files_sha256"][name]:
            raise ValueError("Run journal changed: " + name)
    system = system_prompt(base.INSTRUCTIONS)
    if (output / "model_system_prompt.txt").read_text(encoding="utf-8") != system or sha256(output / "model_system_prompt.txt") != run["system_prompt_sha256"]:
        raise ValueError("Actual system prompt changed")
    statuses = base.read_json(output / "run_status.json")
    if [row["condition"] for row in statuses] != list(CONDITIONS):
        raise ValueError("Planned condition roster changed")
    pathfinder, centers = subgoal.pathfinder_for(config), subgoal.reviewed_centers(config, config["goal_positions"])
    model_state, model_record, update_correct, model_error = None, None, False, None
    try:
        model_state, model_record, update_correct = replay_model(config, output, run)
    except (OSError, ValueError, KeyError, TypeError) as error:
        model_error = str(error)
    results = []
    for status in statuses:
        condition = status["condition"]
        try:
            if not status["completed"]:
                raise ValueError(status["error"] or "Condition did not complete")
            if condition == "model" and model_error:
                raise ValueError(model_error)
            results.append(audit_episode(config, output / condition, condition, pathfinder, centers, model_state, model_record))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            results.append({"condition": condition, "integrity_passed": False, "dynamic_success": False,
                "joint_success": False, "ignore_expected_failure": False, "error": str(error)})
    prefix_ok, prefix_error = False, None
    try:
        trigger = config["events"][0]["trigger"]["value"]
        reference = base.read_json(output / "oracle/trajectory.json")[:trigger + 1]
        physical = lambda rows: [{key: row[key] for key in ("step", "action", "position", "rotation_xyzw", "collided", "distance_m")} for row in rows]
        for condition in ("model", "ignore"):
            other = base.read_json(output / condition / "trajectory.json")[:trigger + 1]
            if physical(reference) != physical(other) or not hssd_dataset.matching_frames(output / "oracle", reference, output / condition, other):
                raise ValueError("Pre-intervention physical trajectory or sensor payload differs")
        prefix_ok = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        prefix_error = str(error)
    integrity = prefix_ok and all(row["integrity_passed"] for row in results)
    passed = integrity and results[0]["joint_success"] and results[2]["ignore_expected_failure"] and update_correct and results[1]["joint_success"]
    failure_stage = ("evidence" if not results[1]["integrity_passed"] else
                     "model_call" if model_record["outcome"] == "call_failed" else
                     "update" if not update_correct else "navigation" if not results[1]["joint_success"] else None)
    return {"kind": PROTOCOL, "planned_episodes": 3, "conditions": results,
        "integrity_passed": integrity, "prefix_matched": prefix_ok, "prefix_error": prefix_error,
        "model_update_correct": update_correct, "model_outcome": model_record["outcome"] if model_record else None,
        "model_callback_seconds": model_record["callback_seconds"] if model_record else None,
        "model_pipeline_success": bool(update_correct and results[1]["joint_success"] and results[1]["integrity_passed"]),
        "model_failure_stage": failure_stage,
        "passed": bool(passed), "audit_code_sha256": code_hashes(),
        "limitations": ["Single reviewed task, one replacement, closed candidate catalog",
            "Text model predicts update; navigator uses privileged coordinates/navmesh",
            "Model call blocks before the action; report wall time separately from action-step latency"]}


def summarize(output):
    report = check_experiment(output)
    base.write_json(output / "summary.json", report)
    print(encode(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="CPU: bind a reviewed fixed task and timing")
    preparing.add_argument("--source-dir", type=Path, required=True)
    preparing.add_argument("--task-id", default="task_001")
    preparing.add_argument("--timing", choices=("early", "middle", "late"), default="middle")
    running = commands.add_parser("run", help="GPU + model: Oracle/model/Ignore")
    running.add_argument("--prepared-dir", type=Path, required=True)
    running.add_argument("--gpu-id", type=int, default=0)
    running.add_argument("--model", required=True)
    running.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    running.add_argument("--temperature", type=float, default=0.0)
    running.add_argument("--max-tokens", type=int, default=512)
    running.add_argument("--timeout", type=float, default=120.0)
    running.add_argument("--thinking-mode", choices=("off", "on", "default"), default="off")
    running.add_argument("--api-key-env", default="TASK_UPDATE_API_KEY")
    commands.add_parser("summarize", help="CPU Habitat mesh audit, no renderer or model")
    videos = commands.add_parser("videos", help="CPU FFmpeg export of all three trajectories")
    videos.add_argument("--fps", type=float, default=10.0)
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("prepare", "run"):
        return {"prepare": prepare, "run": run_experiment}[args.command](args)
    if args.command == "summarize":
        return 0 if summarize(args.output_dir)["passed"] else 1
    for condition in CONDITIONS:
        print(export_video(args.output_dir / condition, args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
