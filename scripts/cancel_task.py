"""Prepare cancel-task cases, run a CPU fixture, or audit its saved evidence.

There is deliberately no Habitat/model run command yet. New preparation and
synthetic fixture outputs never overwrite legacy configs, metrics or videos.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from mivln.cancel_task import (PROTOCOL, CancellationTaskState, CancellationVisualSession,
    evaluate_cancellation, inspect_trajectory, pose, validate_config)
from mivln.event_engine import EventEngine, InterventionEvent
from mivln.visual_frames import RGBFrame
from mivln.visual_inputs import call_policy
from scripts.benchmark_manifest import PROJECT_ROOT, read_json, sha256, within
from scripts.public_inputs import fresh_output, write_json, write_lines


CONDITIONS = ("control", "follow", "ignore")


def prepare(args):
    """Freeze one cancel-only case at each cumulative-reference-travel timing.

    Original reference poses/distances are checked; no new trajectory is claimed.
    Every source base retains three slots even if its reference cannot be used.
    Cancellation language is a template awaiting human review, not approved data.
    """
    from scripts import hssd_dataset as source_checker
    from scripts.intervention_timing import check_reference
    from scripts.subgoal import reference_trigger

    source = args.source_dir.resolve()
    output = fresh_output(args.output_dir, source)
    manifest = read_json(source / "manifest.json")
    source_checker.validate_inputs(source, manifest)
    bindings = {}

    def bind(path):
        path = Path(path).resolve()
        digest = sha256(path)
        if str(path) in bindings and bindings[str(path)] != digest:
            raise ValueError("Source changed during preparation")
        bindings[str(path)] = digest
        return read_json(path)

    if bind(source / "manifest.json") != manifest:
        raise ValueError("Source manifest changed after validation")
    bind(Path(manifest["catalog_source"]["path"]))
    if bindings[str(Path(manifest["catalog_source"]["path"]).resolve())] != manifest["catalog_source"]["sha256"]:
        raise ValueError("Reviewed catalog changed after validation")
    cases, configs = [], {}
    for task in manifest["tasks"]:
        base, rows, error, triggers = None, None, None, {}
        reference = within(source, task["output_subdir"]) / "reference"
        try:
            base = bind(within(source, task["base_config"]))
            bind(within(source, task["reference_config"]))
            for key in ("base_config", "reference_config"):
                if bindings[str(within(source, task[key]))] != task[key+"_sha256"]:
                    raise ValueError("Frozen source config changed after validation")
            episode, metrics, rows = [bind(reference / name) for name in ("episode.json", "metrics.json", "trajectory.json")]
            bind(reference / "terminal_diagnostics.json")
            rows = check_reference(base, reference)
            source_checker.same_task(base, episode)
            source_checker.check_episode_records(base, base, "control", reference, metrics, rows)
            terminal = source_checker.terminal_record(base, metrics, rows, reference, "reference")
            if not terminal["passed"]:
                raise ValueError(terminal["error"] or "Reference instance evidence failed")
            for before, after in zip(rows, rows[1:]):
                if not math.isclose(math.dist(before["position"], after["position"]), after["distance_m"], abs_tol=1e-5):
                    raise ValueError("Reference travel is not measured pose displacement")
            triggers = {name: reference_trigger(rows, fraction) for name, fraction in (("early", .25), ("middle", .5), ("late", .75))}
            if len(set(triggers.values())) != 3:
                raise ValueError("Reference cannot support three distinct cancellation timings")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as failure:
            error = str(failure)
        for timing, fraction in (("early", .25), ("middle", .5), ("late", .75)):
            case_id = f"{task['task_id']}_{timing}_cancel"
            row = {"case_id": case_id, "source_task_id": task["task_id"], "timing": timing,
                "progress_fraction": fraction, "reference_dir": str(reference), "prepared": False,
                "error": error, "config": f"configs/{case_id}.json", "language_review_status": "not_recorded"}
            if error is None:
                try:
                    k = triggers[timing]
                    # Retain private geometry for future environment setup; no
                    # old replacement event/alternate goal is inherited.
                    keys = ("scene", "scene_id", "scene_dataset_config", "scene_sha256", "scene_dataset_config_sha256",
                        "navmesh", "navmesh_sha256", "seed", "max_steps", "resolution", "sensor_height",
                        "agent_radius", "agent_height", "forward_step_m", "turn_angle_degrees", "success_radius_m",
                        "start_position", "start_rotation_xyzw", "initial_goal", "initial_instruction", "catalog_source")
                    config = {key: deepcopy(base[key]) for key in keys if key in base}
                    root = base["initial_goal"]
                    config.update(cancel_protocol=PROTOCOL, task_type="cancel_task", episode_id=case_id,
                        goal_positions={root: deepcopy(base["goal_positions"][root])},
                        goal_metadata={root: deepcopy(base["goal_metadata"][root])},
                        events=[{"event_id": case_id, "trigger": {"type": "step", "value": k},
                            "utterance": "Cancel the task. Stop now.", "task_update": {"operation": "cancel_task"}}])
                    validate_config(config)
                    configs[row["config"]] = config
                    row.update(prepared=True, trigger_step=k, base_task_id=f"{base['scene_id']}@{base['scene_sha256']}/{task['task_id']}")
                except (ValueError, KeyError, TypeError, IndexError) as failure:
                    row["error"] = str(failure)
            cases.append(row)
    if any(not Path(path).is_file() or sha256(path) != digest for path, digest in bindings.items()):
        raise ValueError("Consumed source files changed")
    output.mkdir(parents=True)
    for relative, config in configs.items():
        write_json(output / relative, config)
    for case in cases:
        case["config_sha256"] = sha256(output / case["config"]) if case["prepared"] else None
    count = len(manifest["tasks"])
    report = {"schema_version": 1, "kind": "cancel_task_preparation_v1", "protocol": PROTOCOL,
        "source_dir": str(source), "source_manifest_sha256": bindings[str(source / 'manifest.json')],
        "base_tasks": count, "planned_cases": count*3, "prepared_cases": len(configs),
        "conditions": list(CONDITIONS), "expected_episodes": count*9,
        "trigger_protocol": "cumulative_reference_travel_before_action", "cases": cases,
        "passed": len(configs) == count*3, "navigation_executed": False,
        "language_review_status": "not_recorded", "habitat_run_entry_implemented": False}
    write_json(output / "manifest.json", report)
    write_json(output / "source_bindings.json", {"files": bindings, "unchanged": True})
    print(f"Cancel preparation: {len(configs)}/{count*3} cases; {count*9} planned episodes, none executed")
    return 0 if report["passed"] else 1


def fixture_config():
    return {"kind": "synthetic_cancel_corridor_v1", "cancel_protocol": PROTOCOL,
        "initial_goal": "a", "initial_instruction": "Go to the island.", "max_steps": 8,
        "start_position": [0., 0., 0.], "start_rotation_xyzw": [0., 0., 0., 1.],
        "goal_positions": {"a": [0., 0., -1.]}, "success_radius_m": .3,
        "events": [{"event_id": "private_cancel", "trigger": {"type": "step", "value": 2},
            "utterance": "Cancel the task. Stop now.", "task_update": {"operation": "cancel_task"}}]}


def policy(request, assets, condition):
    """Fixture only: recognize the exact message or deliberately ignore it."""
    messages = request["message_history"] + ([request["current_intervention"]] if request["current_intervention"] else [])
    if condition == "follow" and any("stop now" in item["message"].lower() for item in messages):
        return '{"action":"STOP"}'
    return json.dumps({"action": "move_forward" if len(request["history"]["actions"]) < 4 else "STOP"})


def run_fixture(config, directory, condition):
    """Deterministic, obstacle-free corridor; no simulator or language model."""
    truth = CancellationTaskState(config["initial_goal"])
    engine = EventEngine([] if condition == "control" else [InterventionEvent.from_dict(config["events"][0])])
    session = CancellationVisualSession(config["initial_instruction"],
        {"width": 8, "height": 8, "fx": 4, "fy": 4, "cx": 3.5, "cy": 3.5}, max_steps=config["max_steps"])
    position = deepcopy(config["start_position"])
    rotation = deepcopy(config["start_rotation_xyzw"])
    def frame():
        pixels = bytes([40, 80, int(100 + -position[2]*40)])*64
        return RGBFrame.collect(session.session_id, session.step if not session.initialized else session.step+1, 8, 8, pixels)
    session.initialize(frame())
    trajectory = [{"step": None, "action": "INITIALIZE", "position": deepcopy(position),
        "rotation_xyzw": rotation[:], "distance_m": 0., "collided": False, "delivered_event_ids": []}]
    events, records, assets = [], [], {}
    while not session.closed:
        k = session.step
        before = {"position": position[:], "rotation_xyzw": rotation[:]}
        delivered = engine.step(k, truth)
        for item in delivered:
            events.append({"event_id": item.event_id, "delivered_step": k, "task_update": dict(item.task_update),
                           "pose_before": deepcopy(before), "pose_after": deepcopy(before)})
            session.deliver(item.to_dict())
        request = session.request()
        assets.update(session.assets_for(request["request_id"]))
        record = call_policy(session, lambda req, data: policy(req, data, condition))
        records.append(record)
        if record["error"]:
            break
        action = record["action"]["action"]
        # This fixture supports the declared deterministic line policy only.
        # Other actions remain real failed trials, not coerced STOP/forward.
        if action not in ("move_forward", "STOP"):
            session.fail("unsupported_fixture_action")
            break
        if action == "move_forward":
            position[2] -= .25
        after_frame = None if action == "STOP" else frame()
        session.advance(record["action"], after_frame)
        trajectory.append({"step": k, "action": action, "position": position[:], "rotation_xyzw": rotation[:],
            "distance_m": math.dist(before["position"], position), "collided": False,
            "delivered_event_ids": [item.event_id for item in delivered]})
    directory.mkdir(parents=True)
    write_json(directory / "trajectory.json", trajectory)
    write_json(directory / "events.json", events)
    write_json(directory / "final_task_state.json", truth.to_dict())
    write_json(directory / "run_status.json", {"completed": session.closed and session.failure_reason is None,
                                              "error": session.failure_reason})
    write_lines(directory / "policy_records.jsonl", records)
    for identity, data in assets.items():
        path = directory / "assets" / f"{identity}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def check_fixture(root):
    """Read-only reconstruction; cached metrics/summary flags are not consulted."""
    config = read_json(root / "config.json")
    experiment = read_json(root / "experiment.json")
    if experiment.get("protocol") != PROTOCOL or experiment.get("config_sha256") != sha256(root / "config.json"):
        raise ValueError("Cancellation fixture configuration changed")
    validate_config(config)
    if config.get("kind") != "synthetic_cancel_corridor_v1":
        raise ValueError("This fixture checker is not a Habitat navigation evaluator")
    k = config["events"][0]["trigger"]["value"]
    rows, physical, request_images = [], {}, {}
    for condition in CONDITIONS:
        folder = root / condition
        row = {"condition": condition, "evidence_valid": False, "task_success": False, "passed": False, "error": None}
        try:
            trajectory, events = read_json(folder / "trajectory.json"), read_json(folder / "events.json")
            measured = inspect_trajectory(config, trajectory)
            status = read_json(folder / "run_status.json")
            root_distance = math.dist(trajectory[-1]["position"], config["goal_positions"][config["initial_goal"]])
            root_success = bool(measured["stopped_step"] is not None and measured["steps"] <= config["max_steps"]
                                and root_distance <= config["success_radius_m"])
            row["root_point_goal_success"] = root_success
            if condition == "control":
                if events or any(item["delivered_event_ids"] for item in trajectory) or any(item["action"] == "COMPLETE_SUBGOAL" for item in trajectory):
                    raise ValueError("Control has an intervention or unsupported completion")
                truth = CancellationTaskState(config["initial_goal"])
                row.update(measured, evidence_valid=True, task_success=root_success)
            else:
                metrics = evaluate_cancellation(config, trajectory, events)
                row.update(metrics, task_success=metrics["cancellation_success"])
                truth = CancellationTaskState(config["initial_goal"])
                if metrics["cancellation_received"]:
                    truth.apply_update(config["events"][0]["task_update"])
            if read_json(folder / "final_task_state.json") != truth.to_dict():
                raise ValueError("Final evaluator state differs from replayed cancellation")
            physical[condition] = [{key: item[key] for key in ("step", "action", "position", "rotation_xyzw", "distance_m", "collided")}
                                   for item in trajectory]
            records = [read_json_line(line) for line in (folder / "policy_records.jsonl").read_text().splitlines() if line.strip()]
            request_images[condition] = []
            from mivln.visual_frames import validate_png
            from mivln.visual_messages import parse_action, validate_action, validate_request
            session_id = None
            for index, record in enumerate(records):
                request = validate_request(record["request"])
                session_id = session_id or request["session_id"]
                if request["step"] != index or request["session_id"] != session_id:
                    raise ValueError("Noncontiguous actual policy calls")
                if record["error"] or record["action"] is None:
                    raise ValueError("Policy call/action contract failed")
                validate_action(record["action"], request)
                if parse_action(record["raw_response"], request) != record["action"]:
                    raise ValueError("Accepted action differs from the actual raw response")
                if trajectory[index+1]["action"] != record["action"]["action"]:
                    raise ValueError("Executed action differs from actual policy response")
                expected_message = {"delivered_step": k, "message": config["events"][0]["utterance"]}
                if (request["initial_instruction"] != config["initial_instruction"]
                        or request["current_intervention"] != (expected_message if condition != "control" and index == k else None)
                        or request["message_history"] != ([expected_message] if condition != "control" and index > k else [])):
                    raise ValueError("Actual public messages differ from host delivery")
                expected_actions = [{"step": item["step"], "action": item["action"], "collided": item["collided"]}
                                    for item in trajectory[1:index+1]]
                if request["history"]["actions"] != expected_actions:
                    raise ValueError("Public action history differs from actual execution")
                images = [(index, request["observation"]["rgb"]),
                          *[(entry["step"], entry["rgb"]) for entry in request["history"]["observations"]]]
                for image_step, resource in images:
                    path = folder / "assets" / f"{resource['asset_id']}.png"
                    if path.is_symlink() or sha256(path) != resource["sha256"]:
                        raise ValueError("Actual policy RGB bytes changed")
                    pixels = validate_png(path.read_bytes(), resource["width"], resource["height"])
                    expected_pixels = bytes([40, 80, int(100 + -trajectory[image_step]["position"][2]*40)])*64
                    if pixels != expected_pixels:
                        raise ValueError("RGB does not match the declared synthetic collector")
                request_images[condition].append(request["observation"]["rgb"]["sha256"])
            if len(records) != measured["steps"] or not status["completed"] or status["error"]:
                raise ValueError("Fixture call/action execution incomplete")
            row["passed"] = bool(row["evidence_valid"] and (row["task_success"] if condition != "ignore"
                else not row["task_success"] and row["cancellation_received"] and root_success))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            row.update(passed=False, evidence_valid=False, task_success=False, error=str(error))
            if "cancellation_success" in row:
                row["cancellation_success"] = False
        rows.append(row)
    pairing = []
    if physical.get("control") != physical.get("ignore") or request_images.get("control") != request_images.get("ignore"):
        pairing.append("Ignore physical execution/RGB does not reproduce Control")
    if (physical.get("control", [])[:k+1] != physical.get("follow", [])[:k+1]
            or request_images.get("control", [])[:k+1] != request_images.get("follow", [])[:k+1]):
        pairing.append("Pre-delivery physical/RGB prefixes differ")
    return {"passed": all(row["passed"] for row in rows) and not pairing, "protocol": PROTOCOL,
        "expected_episodes": 3, "conditions": rows, "pairing_errors": pairing,
        "real_model_called": False, "habitat_rendered": False, "kind": "cpu_cancel_task_fixture"}


def read_json_line(line):
    from mivln.task_update import strict_json
    return strict_json(line)


def demo(args):
    output = fresh_output(args.output_dir)
    config = fixture_config()
    validate_config(config)
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "experiment.json", {"protocol": PROTOCOL, "config_sha256": sha256(output / "config.json")})
    for condition in CONDITIONS:
        run_fixture(config, output / condition, condition)
    report = check_fixture(output)
    write_json(output / "summary.json", report)
    print(f"Cancellation CPU fixture: passed={report['passed']}; Control/Follow succeed, Ignore must fail cancellation")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "demo", "audit"):
        child = sub.add_parser(command)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "outputs/hssd_dataset_facing_001")
        elif command == "audit":
            child.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            return prepare(args)
        if args.command == "demo":
            return demo(args)
        from scripts.benchmark_audit import snapshot
        source = args.input_dir.resolve()
        output = fresh_output(args.output_dir, source)
        before = snapshot({"fixture": source})
        report = check_fixture(source)
        after = snapshot({"fixture": source})
        report["source_files_unchanged"] = before == after
        report["passed"] = report["passed"] and before == after
        output.mkdir(parents=True)
        write_json(output / "summary.json", report)
        write_json(output / "source_bindings.json", {"before": before, "after": after})
        print(f"Read-only cancellation audit: {report['passed']}, sources unchanged={before == after}")
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        print(f"Cancellation task failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
