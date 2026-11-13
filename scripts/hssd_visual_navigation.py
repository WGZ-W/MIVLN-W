"""Middle-only real-HSSD V preparation, six isolated runs and no-render audit.

The ordinary policy boundary stays the frozen V loop. Current drivers are
privileged Oracle/Ignore controls, not language/vision models. Preparation and
batch validation need no renderer; run/episode require the Habitat environment.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

from mivln import benchmark_navigation as navigation
from mivln.cancel_task import same_pose
from mivln.visual_frames import RGBFrame, validate_png
from mivln.visual_messages import parse_action
from mivln.cancel_task import CancellationVisualSession
from scripts import benchmark_navigation as source
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import PROJECT_ROOT, sha256, within
from scripts.language_annotations import read_json
from scripts.public_inputs import fresh_output, write_json
from scripts.hssd_visual_backend import SCOPE, HabitatBackend, NativeReplay, OraclePolicy, camera_calibration, native_modules


KIND = "hssd_middle_visual_preparation_v1"
CONDITIONS = ("oracle", "ignore")
CODE = ("scripts/hssd_visual_backend.py", "scripts/hssd_visual_navigation.py", "scripts/run_navigation.py",
        "scripts/terminal_view.py", "scripts/subgoal.py", "mivln/benchmark_navigation.py",
        "mivln/cancel_task.py", "mivln/visual_inputs.py", "mivln/visual_messages.py", "mivln/visual_frames.py")


def bindings():
    return {name: sha256(PROJECT_ROOT / name) for name in CODE}


def build(source_dir, base_task):
    """Select exactly three middle cases without changing geometry/text/budgets."""
    source_dir = source_dir.resolve()
    manifest = source.load_preparation(source_dir)
    cases = source.select_cases(manifest, base_task, "middle")
    if base_task == "all" or len(cases) != 3 or {row["task_type"] for row in cases} != set(navigation.TASK_TYPES):
        raise ValueError("First real-HSSD acceptance uses exactly one base task and all three middle types")
    configs, plan = {}, []
    for case in cases:
        if not case["prepared"]:
            raise ValueError("Prepare all three middle cases before running a renderer: " + str(case["error"]))
        config = deepcopy(read_json(within(source_dir, case["config"])))
        config["execution_scope"] = SCOPE
        config["source_prepared_config"] = {"path": str(within(source_dir, case["config"])), "sha256": case["config_sha256"]}
        navigation.validate_case(config)
        relative = "configs/" + case["case_id"] + ".json"
        configs[relative] = config
        for condition in CONDITIONS:
            plan.append({"case_id": case["case_id"], "task_type": case["task_type"], "condition": condition,
                "config": relative, "config_sha256": None, "directory": "episodes/" + case["case_id"] + "/" + condition})
    triggers = {config["events"][0]["trigger"]["value"] for config in configs.values()}
    if len(triggers) != 1:
        raise ValueError("The three types must share the same middle delivery slot")
    return {"kind": KIND, "execution_scope": SCOPE, "source_dir": str(source_dir),
        "source_manifest_sha256": sha256(source_dir / "manifest.json"), "base_task": base_task, "timing": "middle",
        "planned_cases": 3, "planned_episodes": plan, "trigger_step": triggers.pop(),
        "language_revision": next(iter(configs.values()))["language_binding"],
        "code_bindings": bindings(), "hssd_navigation_executed": False, "release_ready": False}, configs


def prepare(args):
    output = fresh_output(args.output_dir, args.input_dir)
    manifest, configs = build(args.input_dir, args.base_task)
    output.mkdir(parents=True)
    for name, config in configs.items():
        write_json(output / name, config)
    for slot in manifest["planned_episodes"]:
        slot["config_sha256"] = sha256(output / slot["config"])
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {manifest['base_task']} middle at pre-action step {manifest['trigger_step']}: 3 cases / 6 planned HSSD episodes; none executed")
    return 0


def load_prepared(directory):
    directory = directory.resolve()
    actual = read_json(directory / "manifest.json")
    if actual.get("kind") != KIND or actual.get("code_bindings") != bindings():
        raise ValueError("HSSD preparation/code binding changed; use a new prepared version")
    expected, configs = build(Path(actual["source_dir"]), actual["base_task"])
    compared = deepcopy(actual)
    for slot in compared["planned_episodes"]:
        path = within(directory, slot["config"])
        if sha256(path) != slot["config_sha256"] or read_json(path) != configs[slot["config"]]:
            raise ValueError("Prepared HSSD config differs from its reviewed source")
        slot["config_sha256"] = None
    if compared != expected:
        raise ValueError("HSSD preparation roster/source changed")
    return actual


def episode(args):
    """One child/Simulator per slot. A failed branch leaves its actual evidence."""
    prepared = load_prepared(args.input_dir)
    slot = next((row for row in prepared["planned_episodes"]
                 if row["case_id"] == args.case_id and row["condition"] == args.condition), None)
    if slot is None:
        raise ValueError("Unknown planned HSSD episode")
    output = fresh_output(args.output_dir, args.input_dir)
    config = read_json(within(args.input_dir, slot["config"]))
    output.mkdir(parents=True)
    write_json(output / "private/config.json", config)
    backend = None
    try:
        backend = HabitatBackend(config, args.gpu_id, output)
        policy = OraclePolicy(config, backend, args.condition)
        result = navigation.run_episode(config, backend, policy)
        write_result(output, config, result)
        source.write_lines(output / "private/policy_trace.jsonl", policy.trace)
        # A valid Ignore failure is an expected experimental outcome, not a child
        # crash. The independent batch audit decides whether it failed correctly.
        return 0 if result["interface_error"] is None else 1
    except Exception as error:
        write_json(output / "private/slot_error.json", {"error": f"{type(error).__name__}: {error}"})
        return 1
    finally:
        if backend is not None:
            backend.close()


def write_result(output, config, result):
    """Append loop artifacts to the already-created real diagnostic directory.

    The frozen CPU writer requires an entirely new root. Real measurements must
    save masks during execution, so their existing private directory is retained.
    The caller has already reserved this fresh episode root before initialization.
    """
    for name, value in (("config", config), ("trajectory", result["trajectory"]), ("events", result["events"]),
                        ("metrics", result["metrics"]), ("final_task_state", result["final_task_state"])):
        write_json(output / "private" / (name + ".json"), value)
    write_json(output / "private/run_status.json", {"interface_error": result["interface_error"]})
    source.write_lines(output / "private/policy_records.jsonl", result["policy_records"])
    source.write_lines(output / "public/requests.jsonl", [row["request"] for row in result["policy_records"]])
    source.write_lines(output / "public/actions.jsonl", [row["action"] for row in result["policy_records"] if row["action"] is not None])
    for identity, data in result["assets"].items():
        path = output / "public/assets" / (identity + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def check_diagnostic(folder, config, row, goal, measurement, expected_pixels=None, runtime_target=None):
    """Recount instance pixels and bind identity/pose/raw RGB without a renderer."""
    import numpy as np
    from PIL import Image

    evidence, report = measurement["evidence"], measurement["diagnostic"]
    diagnostic_dir = within(folder, evidence["directory"])
    path = diagnostic_dir / "terminal_diagnostics.json"
    if sha256(path) != evidence["sha256"] or read_json(path) != report:
        raise ValueError("Instance diagnostic content/hash changed")
    metadata = config["goal_metadata"][goal]
    if (report.get("evaluated") is not True or report.get("goal_id") != goal
            or any(report.get(key) != metadata[key] for key in ("template_id", "source_instance_index"))
            or report.get("method") != "isolated_instance_semantic_relabel_v1"
            or report.get("labels_restored") is not True or report.get("pose_unchanged") is not True
            or not same_pose(report["pose"], row)):
        raise ValueError("Instance diagnostic identity/restoration/pose mismatch")
    if runtime_target is not None and report.get("object_id") != runtime_target["object_id"]:
        raise ValueError("Diagnostic runtime object differs from the resolved source instance")
    expected_files = {"terminal_instance_mask.npy", "terminal_instance_mask.png", "terminal_rgb.png"}
    if len(report["files"]) != 3 or {entry["path"] for entry in report["files"]} != expected_files:
        raise ValueError("Incomplete instance evidence")
    for entry in report["files"]:
        if sha256(within(diagnostic_dir, entry["path"])) != entry["sha256"]:
            raise ValueError("Instance evidence payload changed")
    mask = np.load(diagnostic_dir / "terminal_instance_mask.npy", allow_pickle=False)
    if (mask.dtype != np.bool_ or mask.shape != tuple(config["resolution"])
            or not np.array_equal(mask, np.asarray(Image.open(diagnostic_dir / "terminal_instance_mask.png")) > 0)):
        raise ValueError("Raw instance masks disagree")
    threshold, count = config["terminal_visibility"]["min_pixels"], int(mask.sum())
    if (report["instance_pixels"] != count or report["min_pixels"] != threshold
            or report["instance_visible"] is not (count >= threshold)):
        raise ValueError("Instance pixel counts/threshold changed")
    if expected_pixels is not None:
        rgb = np.asarray(Image.open(diagnostic_dir / "terminal_rgb.png"))
        if rgb.shape != (*config["resolution"], 3) or rgb.dtype != np.uint8 or rgb.tobytes() != expected_pixels:
            raise ValueError("Private diagnostic RGB differs from the same-pose policy RGB")


def audit_episode(folder, expected):
    """Rebuild V history from actual PNGs, replay native actions, rescore truth.

    No policy or renderer is invoked. Scene correspondence of pixels relies on
    the trusted collector; this audit checks their bytes/causal chain and the
    saved instance evidence, not independent re-rendering or camera attestation.
    """
    config = read_json(folder / "private/config.json")
    if config != expected:
        raise ValueError("Executed config differs from prepared reviewed HSSD case")
    records = source.lines(folder / "private/policy_records.jsonl")
    trajectory, events = [read_json(folder / ("private/" + name + ".json")) for name in ("trajectory", "events")]
    if not records or len(trajectory) < 1:
        raise ValueError("Missing actual policy inputs/trajectory")
    backend = read_json(folder / "private/backend.json")
    _, _, _, legacy, _ = native_modules()
    scene = legacy.project_path(config["scene"])
    provenance = legacy.scene_provenance(config, scene)
    if any(backend.get(key) != value for key, value in provenance.items()):
        raise ValueError("Executed scene/mesh provenance differs from frozen inputs")
    camera = camera_calibration(legacy.make_configuration(config, scene, backend["gpu_id"]))
    if (backend["camera"] != camera or backend["execution_scope"] != SCOPE or backend["agent_initializations"] != 1
            or backend["loaded_rigid_objects"] != provenance["expected_rigid_objects"]):
        raise ValueError("Camera/scope/initialization provenance mismatch")
    from scripts.subgoal import reviewed_centers
    centres = reviewed_centers(config, config["goal_positions"])
    for goal, centre in centres.items():
        metadata, target = config["goal_metadata"][goal], backend["targets"][goal]
        if (math.dist(centre, target["world_center"]) > 1e-4
                or any(target[key] != metadata[key] for key in ("template_id", "source_instance_index"))):
            raise ValueError("Executed instance differs from its reviewed source")
    assets_dir, used_assets = folder / "public/assets", set()
    session = CancellationVisualSession(config["initial_instruction"], camera, max_steps=config["max_steps"],
        forward_step_m=config["forward_step_m"], turn_angle_degrees=config["turn_angle_degrees"],
        success_radius_m=config["success_radius_m"], max_history_frames=4, history_stride=1,
        session_id=records[0]["request"]["session_id"])

    def frame(index):
        identity = trajectory[index]["rgb_asset_id"]
        path = assets_dir / (identity + ".png")
        if path.is_symlink():
            raise ValueError("Linked image asset")
        pixels = validate_png(path.read_bytes(), camera["width"], camera["height"])
        return RGBFrame.collect(session.session_id, index, camera["width"], camera["height"], pixels, identity)

    replay = NativeReplay(config)
    current = frame(0)
    session.initialize(current)
    interface_error = None

    def verify_measurements(row, pixels):
        if not same_pose(replay.pose(), row):
            raise ValueError("Actual pose differs from no-render native action replay")
        if set(row["measurements"]) != set(config["goal_positions"]):
            raise ValueError("Missing/extra goal measurements")
        for goal, point in config["goal_positions"].items():
            value = row["measurements"][goal]
            distance = replay.distance(point)
            recorded = value["distance_m"]
            if not (distance is None and recorded is None or type(recorded) in (int, float) and distance is not None
                    and math.isclose(recorded, distance, abs_tol=1e-5, rel_tol=0)):
                raise ValueError("Recorded distance differs from native geodesic measurement")
            if not math.isclose(replay.yaw(backend["targets"][goal]["world_center"]), value["yaw_error_degrees"], abs_tol=1e-4):
                raise ValueError("Recorded heading differs from native pose/object measurement")
            if distance is not None and distance <= config["success_radius_m"]:
                check_diagnostic(folder, config, row, goal, value, pixels, backend["targets"][goal])
            elif value["diagnostic"] != {"evaluated": False, "reason": "outside_stopping_radius"} or value["evidence"] is not None:
                raise ValueError("Unexpected or falsely granted far-goal instance diagnostic")
    try:
        verify_measurements(trajectory[0], current.pixels)
        for step, record in enumerate(records):
            if step == config["events"][0]["trigger"]["value"]:
                session.deliver(config["events"][0])
            request = session.request()
            if request != record["request"]:
                raise ValueError("Actual V request differs from causal language/RGB/action history")
            for identity, data in session.assets_for(request["request_id"]).items():
                path = assets_dir / (identity + ".png")
                if path.is_symlink() or path.read_bytes() != data:
                    raise ValueError("Actual policy image bytes changed")
                used_assets.add(identity + ".png")
            if record.get("execution_error"):
                raise ValueError("Backend transition failed; incomplete physical evidence")
            try:
                action = parse_action(record["raw_response"], request)
            except (ValueError, TypeError):
                if not record["error"] or record["action"] is not None or len(trajectory) != step+1 or step != len(records)-1:
                    raise ValueError("Malformed response was not retained as a failed, unexecuted slot")
                interface_error = "policy_call_or_action_contract_failed"
                break
            if record["error"] is not None or action != record["action"]:
                raise ValueError("Accepted action differs from actual raw response")
            if step+1 >= len(trajectory):
                raise ValueError("Missing executed action")
            before, row = trajectory[step], trajectory[step+1]
            collided = replay.act(action["action"])
            if row["action"] != action["action"] or row["collided"] != collided:
                raise ValueError("Recorded action/collision differs from native execution")
            after_frame = None
            if action["action"] != "STOP":
                if step+1 < len(records):
                    after_frame = frame(step+1)
                else:
                    # The final post-budget capture was never sent to a policy.
                    # Rebuild only its private ID, without claiming pixel replay.
                    after_frame = RGBFrame.collect(session.session_id, step+1, camera["width"], camera["height"],
                                                   current.pixels, row["rgb_asset_id"])
            session.advance(action, after_frame, collided=collided)
            current = after_frame or current
            verify_measurements(row, current.pixels if action["action"] == "STOP" or step+1 < len(records) else None)
            if not math.isclose(math.dist(before["position"], row["position"]), row["distance_m"], abs_tol=1e-5):
                raise ValueError("Recorded displacement differs from actual poses")
        if len(trajectory) != len(records)+1-int(interface_error is not None):
            raise ValueError("Policy-call/executed-action count mismatch")
        if interface_error is None and not session.closed:
            raise ValueError("Truncated run without STOP/budget exhaustion")
        if {path.name for path in assets_dir.iterdir()} != used_assets:
            raise ValueError("Missing/extra/unissued policy assets")
        if source.lines(folder / "public/requests.jsonl") != [row["request"] for row in records]:
            raise ValueError("Published requests differ from actual policy calls")
        if source.lines(folder / "public/actions.jsonl") != [row["action"] for row in records if row["action"] is not None]:
            raise ValueError("Published actions differ from actual raw responses")
        if read_json(folder / "private/run_status.json") != {"interface_error": interface_error}:
            raise ValueError("Stored run status differs from preserved responses")
        metrics = navigation.evaluate_episode(config, trajectory, events, interface_error=interface_error)
        if read_json(folder / "private/final_task_state.json") != metrics.get("final_task_state"):
            raise ValueError("Final evaluator state differs from replay")
        return metrics, {"trajectory": trajectory, "policy_records": records}
    finally:
        replay.close()


def check_run(directory):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("kind") != "hssd_middle_visual_run_v1" or manifest.get("code_bindings") != bindings():
        raise ValueError("HSSD run kind/code bindings changed")
    prepared_dir = Path(manifest["prepared_dir"])
    if sha256(prepared_dir / "manifest.json") != manifest["prepared_sha256"]:
        raise ValueError("HSSD preparation manifest changed")
    prepared = load_prepared(prepared_dir)
    if manifest["planned_episodes"] != prepared["planned_episodes"]:
        raise ValueError("HSSD planned denominator changed")
    statuses = read_json(directory / "execution_status.json")
    if len(statuses) != 6 or any(status.get("case_id") != slot["case_id"] or status.get("condition") != slot["condition"]
                                 for status, slot in zip(statuses, prepared["planned_episodes"])):
        raise ValueError("Child status denominator/order changed")
    rows, results, pairing = [], {}, []
    for slot, status in zip(prepared["planned_episodes"], statuses):
        row = {**slot, "task_success": False, "evidence_valid": False, "passed": False}
        try:
            if status.get("state") != "completed" or status.get("returncode") != 0:
                raise ValueError("HSSD child did not complete: " + str(status))
            config = read_json(within(prepared_dir, slot["config"]))
            metrics, result = audit_episode(within(directory, slot["directory"]), config)
            row.update(metrics)
            if slot["condition"] == "oracle":
                row["passed"] = bool(metrics["evidence_valid"] and metrics["task_success"])
            else:
                row["passed"] = bool(metrics["evidence_valid"] and not metrics["task_success"]
                    and metrics["intervention_received"] and metrics.get("stopped_step") is not None
                    and not metrics["semantic_violations"] and not metrics["interface_error"]
                    and navigation.goal_complete(config, result["trajectory"][-1]["measurements"], config["initial_goal"]))
            results[(slot["task_type"], slot["condition"])] = result
        except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, OverflowError) as error:
            row.update(error=str(error), passed=False, task_success=False, evidence_valid=False)
        rows.append(row)
    def physical(result, count=None):
        return [{key: row[key] for key in ("step", "action", "position", "rotation_xyzw", "distance_m", "collided")}
                for row in result["trajectory"][:count]]
    def images(result, count=None):
        return [row["request"]["observation"]["rgb"]["sha256"] for row in result["policy_records"][:count]]
    reference = results.get(("replacement", "ignore"))
    for family in navigation.TASK_TYPES:
        ignored, oracle = results.get((family, "ignore")), results.get((family, "oracle"))
        if reference and ignored and (physical(reference) != physical(ignored) or images(reference) != images(ignored)):
            pairing.append(family + ": Ignore physical/RGB execution differs across task types")
        k = prepared["trigger_step"]
        if ignored and oracle and (physical(ignored, k+1) != physical(oracle, k+1) or images(ignored, k+1) != images(oracle, k+1)):
            pairing.append(family + ": pre-delivery physical/RGB prefix changed")
    return {"passed": len(rows) == 6 and all(row["passed"] for row in rows) and not pairing,
        "execution_scope": SCOPE, "planned_cases": 3, "planned_episodes": 6, "episodes": rows,
        "accepted_episode_slots": sum(row["passed"] for row in rows), "pairing_errors": pairing,
        "policy_privileges": ["correct_private_update", "geodesic_follower", "goal_positions", "instance_centres", "agent_pose"],
        "real_model_called": False, "pixel_scene_correspondence_independently_rerendered": False, "release_ready": False}


def geometry(args):
    """Replay the frozen A prefix to middle with a no-sensor native Agent.

    This tests the actual mesh/action/pose bridge, not visibility or the full
    B-then-A route. It does not instantiate HabitatBackend or a Simulator.
    """
    prepared = load_prepared(args.input_dir)
    output = fresh_output(args.output_dir, args.input_dir)
    config = read_json(within(args.input_dir, prepared["planned_episodes"][0]["config"]))
    path = Path(config["trigger_reference"]["path"])
    before = sha256(path)
    rows = read_json(path)
    replay = NativeReplay(config)
    report = {"passed": False, "scope": "native_reference_prefix_only", "base_task": prepared["base_task"],
        "timing": "middle", "trigger_step": prepared["trigger_step"], "simulator_created": False,
        "visibility_evaluated": False, "full_intervention_episode_executed": False,
        "prepared_manifest_sha256": sha256(args.input_dir / "manifest.json"), "reference_sha256": before,
        "code_bindings": bindings(), "error": None}
    try:
        if not same_pose(replay.pose(), rows[0]):
            raise ValueError("Native rehearsal changed the start pose")
        for step, row in enumerate(rows[1:prepared["trigger_step"]+1]):
            collided = replay.act(row["action"])
            if not same_pose(replay.pose(), row) or collided != row["collided"]:
                raise ValueError(f"Native reference prefix diverged at action {step}")
        report.update(passed=True, pose_before_intervention=replay.pose(),
            distances_m={goal: replay.distance(point) for goal, point in config["goal_positions"].items()})
    except (ValueError, RuntimeError, KeyError) as error:
        report["error"] = str(error)
    finally:
        replay.close()
    report["reference_unchanged"] = before == sha256(path)
    report["passed"] = report["passed"] and report["reference_unchanged"]
    output.mkdir(parents=True)
    write_json(output / "geometry.json", report)
    print(f"Native middle-prefix rehearsal: passed={report['passed']}; no renderer or instance visibility")
    return 0 if report["passed"] else 1


def run(args):
    """Run children sequentially; never drop/retry a failed or pending slot."""
    prepared = load_prepared(args.input_dir)
    output = fresh_output(args.output_dir, args.input_dir)
    output.mkdir(parents=True)
    write_json(output / "manifest.json", {"kind": "hssd_middle_visual_run_v1", "prepared_dir": str(args.input_dir.resolve()),
        "prepared_sha256": sha256(args.input_dir / "manifest.json"), "gpu_id": args.gpu_id,
        "planned_episodes": prepared["planned_episodes"], "code_bindings": bindings()})
    statuses = [{"case_id": slot["case_id"], "condition": slot["condition"], "state": "pending", "returncode": None}
                for slot in prepared["planned_episodes"]]
    write_json(output / "execution_status.json", statuses)
    for slot, status in zip(prepared["planned_episodes"], statuses):
        status["state"] = "running"
        write_json(output / "execution_status.json", statuses)
        command = [sys.executable, "-B", "-m", "scripts.hssd_visual_navigation", "episode", "--input-dir", str(args.input_dir.resolve()),
            "--case-id", slot["case_id"], "--condition", slot["condition"], "--gpu-id", str(args.gpu_id),
            "--output-dir", str(output / slot["directory"])]
        try:
            log = output / "logs" / (slot["case_id"] + "_" + slot["condition"] + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status.update(state="completed" if completed.returncode == 0 else "failed", returncode=completed.returncode)
        except KeyboardInterrupt:
            status["state"] = "interrupted"
            write_json(output / "execution_status.json", statuses)
            raise
        except OSError as error:
            status.update(state="failed", error=str(error))
        write_json(output / "execution_status.json", statuses)
    report = check_run(output)
    write_json(output / "summary.json", report)
    print(f"HSSD middle: accepted={report['accepted_episode_slots']}/6, passed={report['passed']}; inspect per-slot raw evidence")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "geometry", "run", "episode", "audit"):
        child = sub.add_parser(command)
        child.add_argument("--input-dir", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--base-task", default="task_001")
        if command in ("run", "episode"):
            child.add_argument("--gpu-id", type=int, default=0)
        if command == "episode":
            child.add_argument("--case-id", required=True)
            child.add_argument("--condition", choices=CONDITIONS, required=True)
    args = parser.parse_args()
    try:
        if getattr(args, "gpu_id", 0) < 0:
            raise ValueError("Use a nonnegative assigned GPU ID")
        if args.command in ("prepare", "geometry", "run", "episode"):
            return globals()[args.command](args)
        output = fresh_output(args.output_dir, args.input_dir)
        before = snapshot({"hssd_run": args.input_dir})
        report = check_run(args.input_dir)
        after = snapshot({"hssd_run": args.input_dir})
        report["source_files_unchanged"] = before == after
        report["passed"] = report["passed"] and before == after
        output.mkdir(parents=True)
        write_json(output / "audit.json", report)
        write_json(output / "source_bindings.json", {"before": before, "after": after})
        print(f"Read-only HSSD middle audit: passed={report['passed']}, unchanged={before == after}")
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, OverflowError, ImportError) as error:
        print(f"HSSD V navigation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
