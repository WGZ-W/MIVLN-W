"""Freeze and evaluate reviewed HSSD semantic tasks with three timing/condition branches."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import itertools
import math
from pathlib import Path
import random
import sys

from scripts.debug_dataset import distance_to_polyline, execute_stage
from scripts.hssd_targets import load_geometry, sha256, validate_catalog
from scripts.intervention_timing import TIMINGS, check_reference, load_json, select_triggers, write_json
from scripts.navigation_scene import check_scene_match
from scripts.run_ignore_intervention import check_negative
from scripts.run_intervention import check_pair


# Only these shared settings may be inherited. In particular, old goal_metadata,
# goal_source and initial language must never survive a newly sampled task.
NAVIGATION_KEYS = (
    "scene", "scene_id", "scene_dataset_config", "scene_sha256", "scene_dataset_config_sha256",
    "dataset_revision", "navmesh", "navmesh_sha256", "agent_height", "agent_radius",
    "semantic_sensor", "success_radius_m", "max_steps", "resolution", "sensor_height",
    "forward_step_m", "turn_angle_degrees",
)
TASK_KEYS = ("initial_instruction", "initial_goal", "start_position", "start_rotation_xyzw", "goal_positions",
             "goal_metadata", "catalog_source", "seed", "max_steps", "resolution", "sensor_height",
             "success_radius_m", "forward_step_m", "turn_angle_degrees")
CONDITIONS = ("control", "intervention", "ignore")


def within(root: Path, relative: str) -> Path:
    """Manifest-controlled writes must stay below this dataset, including symlinks."""
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"Dataset path escapes the output directory: {relative}") from None
    if result == root.resolve():
        raise ValueError("Task path cannot be the dataset root")
    return result


def make_task(template, target_a, target_b, start, yaw, seed, task_id, catalog_source):
    """Bind language, stable source instances, and fixed stopping points together."""
    goal_a, goal_b = target_a["target_id"], target_b["target_id"]
    config = {key: deepcopy(template[key]) for key in NAVIGATION_KEYS if key in template}
    config.update({
        "episode_id": task_id, "seed": seed, "initial_goal": goal_a,
        "initial_instruction": f"Go to {target_a['review']['description']} and stop beside it.",
        "start_position": list(start), "start_rotation_xyzw": [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)],
        "goal_positions": {goal_a: target_a["stop_position"], goal_b: target_b["stop_position"]},
        "goal_metadata": {target["target_id"]: {
            "template_id": target["template_id"], "source_instance_index": target["source_instance_index"],
            "category": target["category"], "description": target["review"]["description"],
        } for target in (target_a, target_b)},
        "catalog_source": catalog_source,
        # Placeholder only. run() ALWAYS executes an event-free reference first;
        # intervention_timing then replaces this step with actual reference triggers.
        "events": [{"event_id": f"{task_id}_replace", "trigger": {"type": "step", "value": 1},
                    "utterance": f"Cancel the previous destination. Go to {target_b['review']['description']} instead.",
                    "task_update": {"operation": "replace_goal", "old_goal": goal_a, "new_goal": goal_b}}],
    })
    return config


def sample_tasks(pathfinder, template, targets, count, seed, attempts, catalog_source):
    """Balance unordered pairs and their reverse directions before repeating pairs.

    Only starts/orientations are resampled. A/B always remain reviewed object
    instances; failure to sample a scheduled pair aborts preparation, not the pair.
    """
    import numpy as np
    from scripts.run_navigation import shortest_path, validate_config

    pairs = list(itertools.combinations(sorted(targets, key=lambda target: target["target_id"]), 2))
    random.Random(seed).shuffle(pairs)
    schedule = [direction for a, b in pairs for direction in ((a, b), (b, a))]
    prepared, starts = [], []
    for index in range(count):
        target_a, target_b = schedule[index % len(schedule)]
        a, b = (np.asarray(target["stop_position"], dtype=np.float32) for target in (target_a, target_b))
        between = shortest_path(pathfinder, a, b)
        if (between is None or not max(2.0, 2 * template["success_radius_m"]) < between.geodesic_distance <= 20.0):
            raise ValueError(f"Scheduled target pair has overlapping or disconnected stop regions: {target_a['target_id']}, {target_b['target_id']}")
        task_seed = seed + index
        pathfinder.seed(task_seed)
        for attempt in range(1, attempts + 1):
            start = np.asarray(pathfinder.get_random_navigable_point(island_index=pathfinder.get_island(a)), dtype=np.float32)
            if (not np.isfinite(start).all() or not pathfinder.is_navigable(start)
                    or abs(float(start[1] - a[1])) > 0.5
                    or any(math.dist(start, old) < 1.0 for old in starts)):
                continue
            initial, replacement = (shortest_path(pathfinder, start, goal) for goal in (a, b))
            if (initial is None or replacement is None or not 4.0 <= initial.geodesic_distance <= 15.0
                    or not 2.0 <= replacement.geodesic_distance <= 20.0):
                continue
            route_gap = distance_to_polyline(b, [np.asarray(point).tolist() for point in initial.points])
            if route_gap < 1.0:
                continue
            task_id = f"task_{index + 1:03d}"
            config = make_task(template, target_a, target_b, start.tolist(), random.Random(task_seed).uniform(-math.pi, math.pi),
                               task_seed, task_id, catalog_source)
            validate_config(config)
            record = {
                "task_id": task_id, "target_a": target_a["target_id"], "target_b": target_b["target_id"],
                "sampling_attempt": attempt, "start_to_a_m": float(initial.geodesic_distance),
                "start_to_b_m": float(replacement.geodesic_distance), "a_to_b_m": float(between.geodesic_distance),
                "b_to_reference_polyline_m": route_gap, "base_config": f"configs/{task_id}/base.json",
                "reference_config": f"configs/{task_id}/reference.json", "output_subdir": f"tasks/{task_id}",
            }
            prepared.append((config, record))
            starts.append(start.tolist())
            break
        else:
            raise ValueError(f"Cannot sample {target_a['target_id']} -> {target_b['target_id']} after {attempts} attempts; no task has been written")
    return prepared


def prepare(args) -> int:
    catalog = load_json(args.catalog)
    targets = validate_catalog(catalog)
    pathfinder, provenance = load_geometry(catalog["scene_config"])
    source = {"path": str(args.catalog.resolve()), "sha256": sha256(args.catalog)}
    prepared = sample_tasks(pathfinder, catalog["scene_config"], targets, args.count, args.seed, args.max_attempts, source)
    # No output files until every scheduled task has passed geometry filtering.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for config, record in prepared:
        reference = deepcopy(config)
        reference["episode_id"] += "_reference"
        reference["events"] = []
        for key, value in (("base_config", config), ("reference_config", reference)):
            write_json(args.output_dir / record[key], value)
            record[f"{key}_sha256"] = sha256(args.output_dir / record[key])
        records.append(record)
    directed = {(record["target_a"], record["target_b"]) for record in records}
    manifest = {
        "schema_version": 1, "kind": "hssd_semantic_dataset", "catalog_source": source,
        "preparation_provenance": provenance,
        "purpose": "Single-scene, instance-grounded Oracle debugging; not language understanding or scene generalization",
        "task_count": len(records), "expected_episodes": len(records) * 10,
        "expected_pairs": len(records) * 3, "expected_negative_controls": len(records) * 3,
        "seed": args.seed, "targets_snapshot": targets, "tasks": records,
        "coverage": {"directed_pairs": len(directed), "unordered_pairs": len({tuple(sorted(pair)) for pair in directed}),
                     "reverse_pairs_present": all((b, a) in directed for a, b in directed)},
        "filters": {"start_to_a_m": [4, 15], "start_to_b_m": [2, 20], "a_to_b_m": [2, 20],
                    "minimum_start_separation_m": 1.0, "b_to_reference_polyline_m": 1.0},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Prepared {len(records)} fixed tasks, {len(records) * 10} planned episodes; {manifest['coverage']}")
    return 0


def validate_inputs(output: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "hssd_semantic_dataset":
        raise ValueError("Unsupported semantic dataset manifest")
    tasks = manifest["tasks"]
    if not tasks or len(tasks) != manifest["task_count"] or len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Frozen task count or task identities changed")
    source = manifest["catalog_source"]
    if sha256(Path(source["path"])) != source["sha256"]:
        raise ValueError("Reviewed catalog changed after task preparation")
    validate_catalog(load_json(Path(source["path"])))
    for task in tasks:
        within(output, task["output_subdir"])
        for key in ("base_config", "reference_config"):
            if sha256(within(output, task[key])) != task[f"{key}_sha256"]:
                raise ValueError(f"Frozen task configuration changed: {task['task_id']} {key}")


def run(args) -> int:
    """Execute once, retain failures, and never replace tasks after GPU outcomes."""
    output = args.output_dir.resolve()
    manifest = load_json(output / "manifest.json")
    validate_inputs(output, manifest)
    # Validate all inputs and output ownership before launching the first child.
    from scripts.run_navigation import scene_provenance, validate_config
    from scripts.navigation_scene import project_path

    if (output / "run_status.json").exists():
        raise FileExistsError("Run already started; prepare a NEW dataset directory for a full rerun")
    for task in manifest["tasks"]:
        destination = within(output, task["output_subdir"])
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Task output is not empty: {destination}")
        config = load_json(within(output, task["base_config"]))
        validate_config(config)
        scene_provenance(config, project_path(config["scene"]))
    statuses = {}
    prefix = [sys.executable, "-u", "-m"]
    for index, task in enumerate(manifest["tasks"], 1):
        destination = within(output, task["output_subdir"])
        statuses[task["task_id"]] = []

        def stage(name, command):
            print(f"[{index}/{len(manifest['tasks'])}] {task['task_id']}: {name}", flush=True)
            log = destination / f"{name}.log"
            code = execute_stage(command, log)
            statuses[task["task_id"]].append({"stage": name, "returncode": code, "log": str(log.relative_to(output))})
            write_json(output / "run_status.json", statuses)
            return code

        common = ["--gpu-id", str(args.gpu_id), "--no-video"]
        if stage("reference", [*prefix, "scripts.run_navigation", "--config", str(within(output, task["reference_config"])),
                               "--output-dir", str(destination / "reference"), *common]):
            continue
        if stage("timing_prepare", [*prefix, "scripts.intervention_timing", "prepare", "--base-config",
                                    str(within(output, task["base_config"])), "--reference-dir", str(destination / "reference"),
                                    "--output-dir", str(destination / "timing"), "--keep-target"]):
            continue
        for timing, _ in TIMINGS:
            config = destination / f"timing/configs/{timing}.json"
            pair = destination / f"timing/{timing}"
            stage(f"{timing}_pair", [*prefix, "scripts.run_intervention", "--config", str(config),
                                      "--output-dir", str(pair), *common])
            try:
                control_ok = load_json(pair / "control/metrics.json")["dynamic_success"] is True
            except (OSError, ValueError, KeyError):
                control_ok = False
            if control_ok:
                # The Ignore CLI returns 0 for a correctly rejected navigation task.
                # A follow failure does not prevent testing Ignore against a valid control.
                stage(f"{timing}_ignore", [*prefix, "scripts.run_ignore_intervention", "--config", str(config),
                                           "--reference-dir", str(pair / "control"), "--output-dir", str(pair / "ignore"), *common])
            else:
                statuses[task["task_id"]].append({"stage": f"{timing}_ignore", "returncode": None,
                                                   "reason": "No successful matching control; condition not run"})
                write_json(output / "run_status.json", statuses)
    return 0 if summarize(output)["passed"] else 1


def same_task(base: dict, episode: dict) -> None:
    check_scene_match(base, episode)
    for key in TASK_KEYS:
        if episode.get(key) != base.get(key):
            raise ValueError(f"Episode changed the frozen semantic task: {key}")


def matching_frames(left_dir, left, right_dir, right) -> bool:
    """Compare actual payloads, not equal filenames; empty frames cannot pass."""
    if len(left) != len(right) or not left:
        return False
    for a, b in zip(left, right):
        for kind in ("rgb", "depth", "semantic"):
            if sha256(within(left_dir, a["frames"][kind])) != sha256(within(right_dir, b["frames"][kind])):
                return False
    return True


def check_episode_records(base, variant, condition, directory, metrics, trajectory) -> None:
    """Check complete sensor payloads and the truth/policy timeline independently.

    Row 0 is INITIALIZE; action k is row k+1. This offset matters for an event
    delivered BEFORE action k, and for Ignore's deliberately different policy state.
    """
    if (type(metrics["dynamic_success"]) is not bool or not trajectory or trajectory[0]["action"] != "INITIALIZE"
            or metrics["steps"] != len(trajectory) - 1 or metrics["max_steps"] != base["max_steps"]):
        raise ValueError("Episode step/frame accounting differs from the frozen budget")
    goal_a = base["initial_goal"]
    event = variant["events"][0]
    goal_b, trigger = event["task_update"]["new_goal"], event["trigger"]["value"]
    for index, row in enumerate(trajectory):
        if row["step"] != (index - 1 if index else None):
            raise ValueError("Non-contiguous action steps")
        updated = condition != "control" and index > trigger
        truth = goal_b if updated else goal_a
        if row["active_goal"] != truth or row["policy_goal"] != (goal_a if condition == "ignore" else truth):
            raise ValueError("Recorded truth or policy goal violates the event timeline")
        for kind in ("rgb", "depth", "semantic"):
            path = within(directory, row["frames"][kind])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty {kind} observation")
    actual_length = sum(math.dist(a["position"], b["position"]) for a, b in zip(trajectory, trajectory[1:]))
    if (not math.isclose(actual_length, metrics["path_length_m"], abs_tol=1e-5)
            or sum(bool(row["collided"]) for row in trajectory) != metrics["collision_count"]):
        raise ValueError("Path length or collision count disagrees with recorded poses/actions")


def condition_statistics(cases, total):
    """Missing runs remain in success denominators; means name their sample counts."""
    result = {"total": total, "available": sum(case["available"] for case in cases),
              "validation_passed": sum(case["passed"] for case in cases),
              "dynamic_successes": sum(case["dynamic_success"] is True for case in cases)}
    result["dynamic_success_rate"] = result["dynamic_successes"] / total if total else None
    result["metric_counts"] = {}
    for key in ("steps", "path_length_m", "navigation_error_m", "post_intervention_spl", "collision_count"):
        values = [case[key] for case in cases if type(case.get(key)) in (int, float) and math.isfinite(case[key])]
        result[f"mean_{key}"] = sum(values) / len(values) if values else None
        result["metric_counts"][key] = len(values)
    return result


def summarize(output: Path) -> dict:
    """Recompute acceptance and preserve every planned condition in denominators."""
    manifest = load_json(output / "manifest.json")
    statuses = load_json(output / "run_status.json") if (output / "run_status.json").is_file() else {}
    global_errors = []
    try:
        validate_inputs(output, manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        global_errors.append(str(error))
    rows, cases, observed = [], [], 0
    for task in manifest["tasks"]:
        destination = within(output, task["output_subdir"])
        row = {"task_id": task["task_id"], "passed": False, "reference_success": False, "errors": [], "timings": []}
        base, reference, triggers = None, None, {}
        try:
            base = load_json(within(output, task["base_config"]))
            reference = check_reference(base, destination / "reference")
            same_task(base, load_json(destination / "reference/episode.json"))
            check_episode_records(base, base, "control", destination / "reference",
                                  load_json(destination / "reference/metrics.json"), reference)
            triggers = {item["name"]: item for item in select_triggers(reference)}
            row["reference_success"] = True
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            row["errors"].append(f"reference: {error}")
        if (destination / "reference/metrics.json").is_file():
            observed += 1
        for timing, _ in TIMINGS:
            pair = destination / f"timing/{timing}"
            record = {"name": timing, "pair_passed": False, "negative_check_passed": False, "errors": []}
            for condition in CONDITIONS:
                directory = pair / condition
                item = {"task_id": task["task_id"], "timing": timing, "condition": condition,
                        "passed": False, "available": False, "dynamic_success": None, "error": None}
                if (directory / "metrics.json").is_file():
                    observed += 1
                try:
                    metrics = load_json(directory / "metrics.json")
                    item["available"] = True
                    item.update({key: metrics.get(key) for key in ("dynamic_success", "steps", "path_length_m",
                                "navigation_error_m", "post_intervention_spl", "collision_count", "termination")})
                    if base is None or timing not in triggers:
                        raise ValueError("Valid frozen reference and trigger unavailable")
                    variant = load_json(destination / f"timing/configs/{timing}.json")
                    same_task(base, variant)
                    event = variant["events"][0]
                    if (len(variant["events"]) != 1 or event["trigger"]["value"] != triggers[timing]["trigger_step"]
                            or event["task_update"] != base["events"][0]["task_update"]
                            or event["utterance"] != base["events"][0]["utterance"]):
                        raise ValueError("Timing event changed task meaning or reference-derived trigger")
                    episode = load_json(directory / "episode.json")
                    same_task(base, episode)
                    if episode["events"] != ([] if condition == "control" else variant["events"]):
                        raise ValueError("Recorded event configuration does not match this condition")
                    trajectory = load_json(directory / "trajectory.json")
                    check_episode_records(base, variant, condition, directory, metrics, trajectory)
                    if condition == "control":
                        item["passed"] = bool(trajectory == reference and metrics["dynamic_success"]
                            and trajectory[-1]["action"] == "STOP"
                            and matching_frames(destination / "reference", reference, directory, trajectory))
                    elif condition == "intervention":
                        result = check_pair(pair, event)
                        control = load_json(pair / "control/trajectory.json")
                        item["passed"] = bool(result["passed"] and matching_frames(
                            pair / "control", control[:event["trigger"]["value"] + 1], directory,
                            trajectory[:event["trigger"]["value"] + 1]))
                        record["pair_passed"] = item["passed"]
                    else:
                        result = check_negative(directory, variant, pair / "control")
                        control = load_json(pair / "control/trajectory.json")
                        item["passed"] = bool(result["negative_check_passed"] and matching_frames(
                            pair / "control", control, directory, trajectory))
                        record["negative_check_passed"] = item["passed"]
                    if not item["passed"]:
                        raise ValueError("Recomputed trajectory/event/observation acceptance failed")
                except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                    item["error"] = str(error)
                    record["errors"].append(f"{condition}: {error}")
                cases.append(item)
            record["passed"] = all(item["passed"] for item in cases[-3:])
            record["pair_passed"] = record["pair_passed"] and all(item["passed"] for item in cases[-3:-1])
            record["negative_check_passed"] = record["negative_check_passed"] and cases[-3]["passed"]
            row["timings"].append(record)
        row["failed_stages"] = [stage for stage in statuses.get(task["task_id"], []) if stage["returncode"] != 0]
        row["passed"] = bool(row["reference_success"] and all(item["passed"] for item in row["timings"])
                             and not row["failed_stages"] and not global_errors)
        rows.append(row)
    planned = manifest["task_count"]
    report = {"passed": bool(rows) and all(row["passed"] for row in rows) and not global_errors,
              "passed_tasks": sum(row["passed"] for row in rows), "total_tasks": planned,
              "passed_pairs": sum(case["pair_passed"] for row in rows for case in row["timings"]), "total_pairs": planned * 3,
              "passed_negative_controls": sum(case["negative_check_passed"] for row in rows for case in row["timings"]),
              "total_negative_controls": planned * 3, "expected_episodes": planned * 10,
              "observed_episode_metrics": observed, "input_errors": global_errors,
              "failed_tasks": [row["task_id"] for row in rows if not row["passed"]], "tasks": rows,
              "conditions": {condition: condition_statistics(
                  [case for case in cases if case["condition"] == condition], planned * 3) for condition in CONDITIONS},
              "by_timing": {timing: {condition: condition_statistics(
                  [case for case in cases if case["condition"] == condition and case["timing"] == timing], planned)
                  for condition in CONDITIONS} for timing, _ in TIMINGS}}
    write_json(output / "summary.json", report)
    columns = list(dict.fromkeys(key for item in cases for key in item))
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cases)
    print(f"Tasks {report['passed_tasks']}/{planned}; pairs {report['passed_pairs']}/{planned * 3}; "
          f"negative controls {report['passed_negative_controls']}/{planned * 3}; input errors: {global_errors}")
    return report


def videos(args) -> int:
    from scripts.export_video import export_video

    manifest = load_json(args.output_dir / "manifest.json")
    tasks = [task for task in manifest["tasks"] if args.task_id is None or task["task_id"] == args.task_id]
    if not tasks:
        raise ValueError("Unknown task ID")
    failures = []
    for task in tasks:
        folder = within(args.output_dir, task["output_subdir"])
        episodes = [folder / "reference", *[folder / f"timing/{name}/{condition}"
                    for name, _ in TIMINGS for condition in ("intervention", "ignore")]]
        for episode in episodes:
            if (episode / "trajectory.mp4").exists():
                continue
            try:
                print(export_video(episode))
            except (OSError, ValueError, RuntimeError) as error:
                failures.append({"episode": str(episode), "error": str(error)})
    write_json(args.output_dir / "video_status.json", {"failed_exports": failures})
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="CPU: sample fixed starts and REVIEWED semantic A/B pairs")
    preparation.add_argument("--catalog", type=Path, required=True)
    preparation.add_argument("--count", type=int, default=10)
    preparation.add_argument("--seed", type=int, default=20261026)
    preparation.add_argument("--max-attempts", type=int, default=5000)
    running = commands.add_parser("run", help="GPU children: reference + three control/follow/ignore sets")
    running.add_argument("--gpu-id", type=int, default=0)
    commands.add_parser("summarize", help="CPU only: recompute all conditions, including missing/failed cases")
    exporting = commands.add_parser("videos", help="CPU: export reference/follow/ignore videos with FFmpeg")
    exporting.add_argument("--task-id")
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.count < 6 or args.count % 2 or args.max_attempts < 1 or args.seed < 0:
            parser.error("Use an even --count >= 6 (three pairs with both directions), positive attempts and non-negative seed")
        return prepare(args)
    if args.command == "run":
        if args.gpu_id < 0:
            parser.error("--gpu-id must be non-negative")
        return run(args)
    if args.command == "videos":
        return videos(args)
    return 0 if summarize(args.output_dir.resolve())["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
