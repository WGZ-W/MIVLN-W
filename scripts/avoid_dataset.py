"""Freeze and run avoid-region cases against every task in a fixed HSSD set.

The planned denominator is three episodes per SOURCE task, including tasks that
cannot be prepared. Runs never resample tasks, retry failures, or overwrite prior
results. Only prepare/run need Habitat; summarize/videos do not create a renderer.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from scripts import avoid_region
from scripts.debug_dataset import execute_stage
from scripts.hssd_dataset import same_task, terminal_record, validate_inputs, within
from scripts.hssd_targets import sha256
from scripts.intervention_timing import check_reference, load_json, write_json
from scripts.navigation_scene import project_path


CONDITIONS = ("control", "intervention", "ignore")
REFERENCE_FILES = ("episode.json", "trajectory.json", "metrics.json", "terminal_diagnostics.json")


def source_tasks(source):
    """Read the complete source roster, not a successful-task subset.

    Each event-free reference must match its frozen base configuration and have
    valid point/instance evidence. Validate all references before creating output
    so input corruption is not confused with geometric preparation rejection.
    """
    manifest = load_json(source/"manifest.json")
    validate_inputs(source, manifest)
    records = []
    for task in manifest["tasks"]:
        reference = within(source, task["output_subdir"]+"/reference")
        base = load_json(within(source, task["base_config"]))
        rows = check_reference(base, reference)
        episode = load_json(reference/"episode.json")
        same_task(base, episode)
        if episode.get("events"):
            raise ValueError("Avoid preparation requires event-free references")
        if not all(episode.get(key, {}).get("enabled") for key in ("terminal_alignment", "terminal_visibility")):
            raise ValueError("Use the fixed terminal-facing collection")
        diagnostic = terminal_record(base, load_json(reference/"metrics.json"), rows, reference, "reference")
        if not diagnostic["passed"]:
            raise ValueError(f"Invalid reference instance evidence: {task['task_id']}: {diagnostic}")
        # IDs become output path components; reject traversal rather than infer
        # an alternative task ID and accidentally change the fixed roster.
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id or any(c not in
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in task_id):
            raise ValueError("Source task_id must be a simple identifier")
        records.append({"task_id": task_id, "reference_dir": str(reference),
                        "reference_hashes": {name: sha256(reference/name) for name in REFERENCE_FILES},
                        "target_id": base["initial_goal"],
                        "category": base["goal_metadata"][base["initial_goal"]]["category"]})
    return records


def prepare(args):
    """Attempt one region per original task; retain every rejection and its slot."""
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new batch preparation directory")
    records = source_tasks(source)
    source_digest = sha256(source/"manifest.json")
    output.mkdir(parents=True)
    (output/"preparation_logs").mkdir()
    manifest = {"schema_version": 1, "kind": "hssd_avoid_dataset",
                "source_dir": str(source), "source_manifest_sha256": source_digest,
                "task_count": len(records), "expected_episodes": len(records)*3,
                "trigger_protocol": "25_percent_reference_travel_before_action",
                "purpose": "Single-scene privileged Oracle geometry; not language understanding",
                "tasks": []}
    for index, source_record in enumerate(records, 1):
        task_id = source_record["task_id"]
        relative = f"preparations/{task_id}"
        record = {**source_record, "prepared": False, "preparation_error": None,
                  "preparation_dir": relative, "output_subdir": f"tasks/{task_id}"}
        print(f"[{index}/{len(records)}] prepare {task_id}", flush=True)
        with (output/f"preparation_logs/{task_id}.log").open("w", encoding="utf-8") as log:
            try:
                with redirect_stdout(log):
                    avoid_region.prepare(SimpleNamespace(reference_dir=Path(record["reference_dir"]),
                        output_dir=within(output, relative), episode_id=f"hssd_avoid_{task_id}"))
                record.update({"prepared": True,
                    "config_sha256": sha256(output/relative/"config.json"),
                    "preparation_sha256": sha256(output/relative/"preparation.json")})
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                record["preparation_error"] = f"{type(error).__name__}: {error}"
                record["rejected_candidates"] = getattr(error, "rejected_candidates", [])
                log.write(record["preparation_error"]+"\n")
                print(f"  preparation failed: {record['preparation_error']}", flush=True)
        manifest["tasks"].append(record)
        write_json(output/"manifest.json", manifest)
    # A zero exit indicates every original task got a valid fixed case. A
    # nonzero preparation still preserves a usable manifest for eligible tasks.
    manifest["prepared_tasks"] = sum(task["prepared"] for task in manifest["tasks"])
    manifest["preparation_complete"] = True
    write_json(output/"manifest.json", manifest)
    validate_batch(output, manifest)
    print(f"Prepared {manifest['prepared_tasks']}/{len(records)} tasks; planned episodes: {len(records)*3}")
    return 0 if manifest["prepared_tasks"] == len(records) else 1


def validate_batch(output, manifest):
    """Preflight all fixed inputs before any GPU child or summary overwrite."""
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "hssd_avoid_dataset"
            or manifest.get("preparation_complete") is not True):
        raise ValueError("Unsupported or incomplete avoid batch preparation")
    source = Path(manifest["source_dir"])
    if sha256(source/"manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    roster = source_tasks(source)
    tasks = manifest["tasks"]
    if ([item["task_id"] for item in roster] != [item["task_id"] for item in tasks]
            or len(tasks) != manifest["task_count"] or manifest["expected_episodes"] != len(tasks)*3
            or sum(item["prepared"] is True for item in tasks) != manifest["prepared_tasks"]):
        raise ValueError("Original task roster or planned denominator changed")
    for record, original in zip(tasks, roster):
        for key in original:
            if record.get(key) != original[key]:
                raise ValueError(f"Source reference changed: {record['task_id']} {key}")
        within(output, record["output_subdir"])
        preparation = within(output, record["preparation_dir"])
        if record["prepared"] is not True:
            if not record["preparation_error"]:
                raise ValueError("Unprepared task lacks a recorded reason")
            continue
        for name, key in (("config.json", "config_sha256"), ("preparation.json", "preparation_sha256")):
            if sha256(preparation/name) != record[key]:
                raise ValueError(f"Frozen avoid preparation changed: {record['task_id']} {name}")
        prepared = avoid_region.validate_preparation(preparation/"config.json")
        if prepared["reference_dir"] != original["reference_dir"]:
            raise ValueError("Preparation refers to a different task reference")
        config = load_json(preparation/"config.json")
        same_task(config, load_json(Path(original["reference_dir"])/"episode.json"))
        for key in ("scene", "scene_dataset_config", "navmesh"):
            if sha256(project_path(config[key])) != config[key+"_sha256"]:
                raise ValueError(f"Scene input changed: {record['task_id']} {key}")


def run(args):
    """Run each eligible three-condition experiment once in an isolated process."""
    output = args.output_dir.resolve()
    manifest = load_json(output/"manifest.json")
    validate_batch(output, manifest)
    for name in ("run_status.json", "tasks", "run_logs"):
        if (output/name).exists():
            raise FileExistsError("Batch has already started; prepare a new directory, do not resume/overwrite")
    # Also inspect all manifest destinations up front, before launching any job.
    if any(within(output, task["output_subdir"]).exists() for task in manifest["tasks"]):
        raise FileExistsError("A task output directory already exists")
    statuses = {}
    write_json(output/"run_status.json", statuses)
    for index, task in enumerate(manifest["tasks"], 1):
        task_id = task["task_id"]
        if task["prepared"] is not True:
            statuses[task_id] = {"attempted": False, "returncode": None,
                                 "reason": task["preparation_error"]}
        else:
            log = output/f"run_logs/{task_id}.log"
            command = [sys.executable, "-u", "-m", "scripts.avoid_region", "run", "--config",
                       str(within(output, task["preparation_dir"])/"config.json"), "--gpu-id", str(args.gpu_id),
                       "--output-dir", str(within(output, task["output_subdir"]))]
            print(f"[{index}/{len(manifest['tasks'])}] run {task_id}", flush=True)
            try:
                returncode = execute_stage(command, log)
                statuses[task_id] = {"attempted": True, "returncode": returncode,
                                     "log": str(log.relative_to(output))}
            except OSError as error:
                statuses[task_id] = {"attempted": True, "returncode": None, "reason": str(error)}
        write_json(output/"run_status.json", statuses)
    return 0 if summarize(output)["passed"] else 1


def statistics(cases):
    """Report denominators and valid paired samples explicitly; null stays null."""
    report = {"planned": len(cases), "available": sum(row["available"] for row in cases),
              "validation_passed": sum(row["validation_passed"] for row in cases),
              "reported_point_goal_successes": sum(row.get("point_goal_success") is True for row in cases),
              "reported_dynamic_successes": sum(row.get("dynamic_success") is True for row in cases),
              "point_goal_successes": sum(row["validation_passed"] and row.get("point_goal_success") is True for row in cases),
              "dynamic_successes": sum(row["validation_passed"] and row.get("dynamic_success") is True for row in cases)}
    report["dynamic_success_rate"] = report["dynamic_successes"]/len(cases) if cases else None
    for key in ("path_length_m", "violation_count"):
        values = [row[key] for row in cases if row["validation_passed"] and row.get(key) is not None]
        report[key+"_count"] = len(values)
        report["mean_"+key] = sum(values)/len(values) if values else None
    return report


def summarize(output):
    """Re-audit raw branch artifacts; never trust cached per-task summaries."""
    manifest = load_json(output/"manifest.json")
    errors = []
    try:
        validate_batch(output, manifest)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        errors.append(str(error))
    statuses = load_json(output/"run_status.json") if (output/"run_status.json").is_file() else {}
    cases, tasks, overheads = [], [], []
    for task in manifest["tasks"]:
        folder = within(output, task["output_subdir"])
        audit, audit_error = None, task.get("preparation_error")
        if task["prepared"] is True:
            try:
                audit = avoid_region.check_experiment(folder)
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                audit_error = str(error)
        task_status = statuses.get(task["task_id"], {})
        completed = task_status.get("attempted") is True and task_status.get("returncode") == 0
        local = []
        for condition in CONDITIONS:
            row = {"task_id": task["task_id"], "target_id": task["target_id"], "category": task["category"],
                   "condition": condition, "prepared": task["prepared"], "available": False,
                   "validation_passed": False, "point_goal_success": None, "dynamic_success": None,
                   "violation_count": None, "path_length_m": None, "error": audit_error}
            metrics_path = folder/condition/"metrics.json"
            try:
                metrics = load_json(metrics_path)
                if not isinstance(metrics, dict):
                    raise ValueError("Episode metrics must be an object")
                row.update({"available": True, "point_goal_success": metrics.get("point_goal_success"),
                            "dynamic_success": metrics.get("dynamic_success"),
                            "violation_count": metrics.get("constraint_violation_count"),
                            "path_length_m": metrics.get("path_length_m")})
            except (OSError, ValueError, TypeError) as error:
                row["error"] = row["error"] or str(error)
            if audit:
                checked = next(item for item in audit["conditions"] if item["condition"] == condition)
                row["validation_passed"] = bool(checked["passed"] and not audit["pairing_errors"] and completed and not errors)
                row["error"] = checked["error"] or ("; ".join(audit["pairing_errors"]) or row["error"])
                if not completed:
                    row["error"] = row["error"] or "Batch child did not complete successfully"
            if errors:
                row["error"] = row["error"] or "; ".join(errors)
            local.append(row)
        passed = all(row["validation_passed"] for row in local) and bool(audit and audit["passed"])
        tasks.append({"task_id": task["task_id"], "prepared": task["prepared"], "passed": passed,
                      "error": audit_error or "; ".join(f"{row['condition']}: {row['error']}" for row in local if row["error"]) or None,
                      "run_status": task_status})
        if passed:
            overheads.append({"task_id": task["task_id"],
                              "extra_path_length_m": local[1]["path_length_m"]-local[0]["path_length_m"]})
        cases.extend(local)
    count = manifest["task_count"]
    report = {"passed": len(tasks) == count and count > 0 and all(task["passed"] for task in tasks) and not errors,
              "total_tasks": count, "prepared_tasks": sum(task["prepared"] is True for task in tasks),
              "passed_tasks": sum(task["passed"] for task in tasks), "expected_episodes": count*3,
              "prepared_episode_budget": sum(task["prepared"] is True for task in tasks)*3,
              "observed_episode_metrics": sum(row["available"] for row in cases),
              "missing_episode_metrics": count*3-sum(row["available"] for row in cases),
              "preparation_failed_tasks": [task["task_id"] for task in tasks if task["prepared"] is not True],
              "failed_tasks": [task["task_id"] for task in tasks if not task["passed"]], "input_errors": errors,
              "conditions": {condition: statistics([row for row in cases if row["condition"] == condition]) for condition in CONDITIONS},
              "paired_overhead_count": len(overheads), "paired_overheads": overheads,
              "mean_extra_path_length_m": sum(row["extra_path_length_m"] for row in overheads)/len(overheads) if overheads else None,
              "tasks": tasks}
    write_json(output/"summary.json", report)
    with (output/"cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]) if cases else ["task_id", "condition"])
        writer.writeheader()
        writer.writerows(cases)
    print(f"Prepared {report['prepared_tasks']}/{count}; passed {report['passed_tasks']}/{count}; "
          f"episodes {report['observed_episode_metrics']}/{count*3}; input errors: {errors}")
    return report


def videos(args):
    from scripts.export_video import export_video

    output = args.output_dir.resolve()
    manifest = load_json(output/"manifest.json")
    failures, exported = [], 0
    for task in manifest["tasks"]:
        for condition in CONDITIONS:
            folder = within(output, task["output_subdir"])/condition
            if (folder/"trajectory.mp4").is_file() and (folder/"trajectory.mp4").stat().st_size > 0:
                continue
            try:
                export_video(folder)
                exported += 1
            except (OSError, ValueError, RuntimeError) as error:
                failures.append({"task_id": task["task_id"], "condition": condition, "error": str(error)})
    write_json(output/"video_status.json", {"new_exports": exported, "failed_exports": failures})
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="CPU: attempt a region for EVERY original HSSD task")
    preparing.add_argument("--source-dir", type=Path, required=True)
    running = commands.add_parser("run", help="GPU children: three branches per prepared task, no retries")
    running.add_argument("--gpu-id", type=int, default=0)
    commands.add_parser("summarize", help="CPU: recompute raw evidence and preserve all planned denominators")
    commands.add_parser("videos", help="CPU: export all three branches per task with FFmpeg")
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "run":
        if args.gpu_id < 0:
            parser.error("GPU index must be non-negative")
        return run(args)
    if args.command == "videos":
        return videos(args)
    return 0 if summarize(args.output_dir)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
