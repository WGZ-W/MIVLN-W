"""Fixed-roster, multi-timing HSSD subgoal insertion and resumption experiments.

Each original task retains three planned branches at every requested timing,
even if preparation or execution fails. CPU preparation reuses approved A/B
instances and existing A references; only ``run`` launches GPU subprocesses.
Summaries independently replay raw evidence, never cached child success flags.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

from scripts import subgoal
from scripts.avoid_dataset import source_tasks as checked_source_tasks
from scripts.debug_dataset import execute_stage
from scripts.hssd_dataset import within
from scripts.hssd_targets import sha256
from scripts.intervention_timing import load_json, write_json


CONDITIONS = ("control", "intervention", "ignore")
TIMINGS = {"early": .25, "middle": .5, "late": .75}
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError)


def timing_names(names):
    """Keep a declared timing order; reject duplicates instead of changing N."""
    if (not isinstance(names, (list, tuple)) or not names
            or any(name not in TIMINGS for name in names) or len(set(names)) != len(names)):
        raise ValueError("Select unique timings from early, middle, late")
    return list(names)


def source_tasks(source):
    """Validate the full frozen facing roster and retain each source base path."""
    records = checked_source_tasks(source)
    manifest = load_json(source/"manifest.json")
    if not records or len({item["task_id"] for item in records}) != len(records):
        raise ValueError("Source roster must be nonempty and have unique task identifiers")
    for record, task in zip(records, manifest["tasks"]):
        path = within(source, task["base_config"])
        record.update(source_config=str(path), source_config_sha256=sha256(path))
    return records


def prepare(args):
    """Attempt all source-task/timing slots, with no task/target resampling."""
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new subgoal batch directory")
    names = timing_names(getattr(args, "timings", list(TIMINGS)))
    originals = source_tasks(source)
    output.mkdir(parents=True)
    (output/"preparation_logs").mkdir()
    manifest = {"schema_version": 1, "kind": "hssd_subgoal_dataset",
        "source_dir": str(source), "source_manifest_sha256": sha256(source/"manifest.json"),
        "task_count": len(originals), "timings": names,
        "timing_progress": {name: TIMINGS[name] for name in names},
        "case_count": len(originals)*len(names), "expected_episodes": len(originals)*len(names)*3,
        "trigger_protocol": "cumulative_reference_travel_before_action",
        "purpose": "Single-scene privileged Oracle sequence evaluation, not language understanding",
        "tasks": originals, "cases": [], "preparation_complete": False}
    for task in originals:
        triggers = set()
        for name in names:
            case_id = f"{task['task_id']}_{name}"
            record = {"case_id": case_id, "task_id": task["task_id"], "timing": name,
                "progress_fraction": TIMINGS[name], "target_id": task["target_id"],
                "category": task["category"], "prepared": False, "preparation_error": None,
                "preparation_dir": f"preparations/{task['task_id']}/{name}",
                "output_subdir": f"tasks/{task['task_id']}/{name}"}
            print(f"[{len(manifest['cases'])+1}/{manifest['case_count']}] prepare {case_id}", flush=True)
            with (output/f"preparation_logs/{case_id}.log").open("w", encoding="utf-8") as log:
                try:
                    folder = within(output, record["preparation_dir"])
                    with redirect_stdout(log):
                        subgoal.prepare(SimpleNamespace(source_config=Path(task["source_config"]),
                            reference_dir=Path(task["reference_dir"]), output_dir=folder,
                            subgoal=None, progress_fraction=TIMINGS[name], episode_id=f"hssd_subgoal_{case_id}"))
                    prepared = subgoal.validate_preparation(folder/"config.json")
                    if prepared["trigger_step"] in triggers:
                        raise ValueError("Requested timing repeats an earlier trigger action")
                    triggers.add(prepared["trigger_step"])
                    record.update(prepared=True, subgoal=prepared["subgoal"],
                        trigger_step=prepared["trigger_step"],
                        config_sha256=sha256(folder/"config.json"),
                        preparation_sha256=sha256(folder/"preparation.json"))
                except ERRORS as error:
                    record["preparation_error"] = f"{type(error).__name__}: {error}"
                    log.write(record["preparation_error"]+"\n")
                    print(f"  preparation failed: {record['preparation_error']}", flush=True)
            manifest["cases"].append(record)
            write_json(output/"manifest.json", manifest)
    manifest.update(preparation_complete=True,
                    prepared_cases=sum(case["prepared"] is True for case in manifest["cases"]))
    write_json(output/"manifest.json", manifest)
    validate_batch(output, manifest)
    print(f"Prepared {manifest['prepared_cases']}/{manifest['case_count']} cases; "
          f"planned episodes: {manifest['expected_episodes']}")
    return 0 if manifest["prepared_cases"] == manifest["case_count"] else 1


def validate_batch(output, manifest):
    """Check EVERY frozen input and path before launching the first GPU child."""
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "hssd_subgoal_dataset"
            or manifest.get("preparation_complete") is not True):
        raise ValueError("Unsupported or incomplete subgoal batch preparation")
    names = timing_names(manifest["timings"])
    if manifest["timing_progress"] != {name: TIMINGS[name] for name in names}:
        raise ValueError("Timing fractions changed")
    source = Path(manifest["source_dir"])
    if sha256(source/"manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    originals = source_tasks(source)
    if originals != manifest["tasks"]:
        raise ValueError("Source roster, configuration or reference changed")
    cases = manifest["cases"]
    expected = [(task, name) for task in originals for name in names]
    if (manifest["task_count"] != len(originals) or manifest["case_count"] != len(expected)
            or manifest["expected_episodes"] != len(expected)*3 or len(cases) != len(expected)
            or manifest["prepared_cases"] != sum(case["prepared"] is True for case in cases)):
        raise ValueError("Fixed task/timing roster or planned denominator changed")
    for case, (task, name) in zip(cases, expected):
        required = {"case_id": f"{task['task_id']}_{name}", "task_id": task["task_id"],
            "timing": name, "progress_fraction": TIMINGS[name], "target_id": task["target_id"],
            "category": task["category"], "preparation_dir": f"preparations/{task['task_id']}/{name}",
            "output_subdir": f"tasks/{task['task_id']}/{name}"}
        if any(case.get(key) != value for key, value in required.items()):
            raise ValueError("Task/timing identity or output destination changed")
        within(output, case["output_subdir"])
        folder = within(output, case["preparation_dir"])
        if case["prepared"] is not True:
            if not case["preparation_error"]:
                raise ValueError("Unprepared case lacks a reason")
            continue
        for file, key in (("config.json", "config_sha256"), ("preparation.json", "preparation_sha256")):
            if sha256(folder/file) != case[key]:
                raise ValueError(f"Frozen subgoal preparation changed: {case['case_id']} {file}")
        report = subgoal.validate_preparation(folder/"config.json")
        if (report["reference_dir"] != task["reference_dir"]
                or report["reference_hashes"] != task["reference_hashes"]
                or report["source_config"] != task["source_config"]
                or report["source_config_sha256"] != task["source_config_sha256"]
                or report["progress_fraction"] != TIMINGS[name]
                or report["root_goal"] != task["target_id"] or report["subgoal"] != case["subgoal"]
                or report["trigger_step"] != case["trigger_step"]):
            raise ValueError(f"Prepared case differs from source or timing: {case['case_id']}")


def run(args):
    """Run each prepared case once; failed children cannot hide later cases."""
    output = args.output_dir.resolve()
    manifest = load_json(output/"manifest.json")
    validate_batch(output, manifest)
    if any((output/name).exists() for name in ("run_status.json", "tasks", "run_logs")):
        raise FileExistsError("Batch already started; no automatic retry, resume or overwrite")
    if any(within(output, case["output_subdir"]).exists() for case in manifest["cases"]):
        raise FileExistsError("Case output already exists")
    statuses = {}
    write_json(output/"run_status.json", statuses)
    for index, case in enumerate(manifest["cases"], 1):
        identity = case["case_id"]
        if case["prepared"] is not True:
            statuses[identity] = {"attempted": False, "returncode": None, "reason": case["preparation_error"]}
        else:
            log = output/f"run_logs/{identity}.log"
            command = [sys.executable, "-u", "-m", "scripts.subgoal", "run", "--config",
                str(within(output, case["preparation_dir"])/"config.json"), "--gpu-id", str(args.gpu_id),
                "--output-dir", str(within(output, case["output_subdir"]))]
            print(f"[{index}/{manifest['case_count']}] run {identity}", flush=True)
            try:
                statuses[identity] = {"attempted": True, "returncode": execute_stage(command, log),
                                      "log": str(log.relative_to(output))}
            except OSError as error:
                statuses[identity] = {"attempted": True, "returncode": None, "reason": str(error)}
        write_json(output/"run_status.json", statuses)
    return 0 if summarize(output)["passed"] else 1


def statistics(rows):
    """Success rates use every planned slot; means count only audited samples."""
    report = {"planned": len(rows), "available": sum(row["available"] for row in rows),
              "validation_passed": sum(row["validation_passed"] for row in rows)}
    for key in ("root_goal_success", "point_goal_success", "subgoal_completed", "sequence_success", "dynamic_success"):
        report["reported_"+key+"_count"] = sum(row.get(key) is True for row in rows)
        count = sum(row["validation_passed"] and row.get(key) is True for row in rows)
        report[key+"_count"] = count
        report[key+"_rate"] = count/len(rows) if rows else None
    for key in ("steps", "path_length_m", "completion_step"):
        values = [row[key] for row in rows if row["validation_passed"] and row.get(key) is not None]
        report[key+"_count"] = len(values)
        report["mean_"+key] = sum(values)/len(values) if values else None
    return report


def check_batch(output):
    """Read-only batch audit; original child summaries are never rewritten.

    The single-case checker remeasures B and A using the frozen mesh, recounts
    instance masks and replays evaluator/policy states. Cached success flags are
    insufficient. Missing, invalid or unprepared branches retain their CSV slot.
    """
    manifest = load_json(output/"manifest.json")
    errors = []
    try:
        validate_batch(output, manifest)
    except ERRORS as error:
        errors.append(str(error))
    try:
        statuses = load_json(output/"run_status.json")
        if not isinstance(statuses, dict):
            raise ValueError("Batch run status must be an object")
    except ERRORS as error:
        statuses = {}
        errors.append(str(error))
    rows, case_reports, overheads = [], [], []
    for case in manifest["cases"]:
        folder = within(output, case["output_subdir"])
        audit, audit_error = None, case.get("preparation_error")
        if case["prepared"] is True:
            try:
                audit = subgoal.check_experiment(folder)
                # Bind actual child input to THIS batch slot, not merely to any
                # valid preparation. This rejects copied/swapped task outputs.
                experiment = load_json(folder/"experiment.json")
                if (experiment["source_config"] != str(within(output, case["preparation_dir"])/"config.json")
                        or experiment["config_sha256"] != case["config_sha256"]):
                    raise ValueError("Child experiment belongs to a different batch case")
            except ERRORS as error:
                audit, audit_error = None, str(error)
        status = statuses.get(case["case_id"], {})
        # A historical child returned 1 solely because the old audit rejected a
        # native navmesh projection. Preserve that exit code. Recovery requires
        # ALL original branch executions completed, a fresh full raw audit, and
        # explicit native endpoint proof; arbitrary nonzero exits still fail.
        recovered = bool(isinstance(status, dict) and status.get("attempted") is True
            and status.get("returncode") == 1 and audit and audit["passed"]
            and any(item.get("native_projection_checks") for item in audit["conditions"]))
        completed = bool(isinstance(status, dict) and status.get("attempted") is True
                         and (status.get("returncode") == 0 or recovered))
        local = []
        for condition in CONDITIONS:
            row = {"case_id": case["case_id"], "task_id": case["task_id"], "timing": case["timing"],
                "target_id": case["target_id"], "category": case["category"], "subgoal": case.get("subgoal"),
                "condition": condition, "prepared": case["prepared"], "available": False,
                "validation_passed": False, "root_goal_success": None, "point_goal_success": None,
                "subgoal_completed": None, "sequence_success": None, "dynamic_success": None,
                "steps": None, "path_length_m": None, "completion_step": None,
                "collision_count": None, "planner_error": None, "error": audit_error}
            try:
                metrics = load_json(folder/condition/"metrics.json")
                if not isinstance(metrics, dict):
                    raise ValueError("Episode metrics must be an object")
                row["available"] = True
                for key in ("root_goal_success", "point_goal_success", "subgoal_completed", "sequence_success",
                            "dynamic_success", "steps", "path_length_m", "collision_count", "planner_error"):
                    row[key] = metrics.get(key)
            except ERRORS as error:
                row["error"] = row["error"] or str(error)
            if audit:
                checked = next(item for item in audit["conditions"] if item["condition"] == condition)
                row["completion_step"] = checked.get("completion_step")
                row["validation_passed"] = bool(checked["passed"] and audit["passed"] and completed and not errors)
                row["error"] = checked["error"] or "; ".join(audit["pairing_errors"]) or row["error"]
            if not completed:
                row["error"] = row["error"] or "Batch child did not complete successfully"
            if errors:
                row["error"] = row["error"] or "; ".join(errors)
            local.append(row)
        passed = all(row["validation_passed"] for row in local)
        case_reports.append({"case_id": case["case_id"], "task_id": case["task_id"], "timing": case["timing"],
                             "prepared": case["prepared"], "passed": passed, "run_status": status,
                             "audit_recovered": recovered,
                             "error": "; ".join(f"{row['condition']}: {row['error']}" for row in local if row["error"]) or None})
        if passed:
            overheads.append({"case_id": case["case_id"], "timing": case["timing"],
                              "extra_path_length_m": local[1]["path_length_m"]-local[0]["path_length_m"]})
        rows.extend(local)
    count = manifest["case_count"]
    names = manifest["timings"]
    report = {"passed": count > 0 and len(case_reports) == count and all(case["passed"] for case in case_reports) and not errors,
        "protocol": "insert_subgoal_v1", "audit_protocol": "native_projection_endpoint_v2",
        "total_tasks": manifest["task_count"], "total_cases": count,
        "timings": names, "prepared_cases": sum(case["prepared"] is True for case in case_reports),
        "passed_cases": sum(case["passed"] for case in case_reports), "expected_episodes": count*3,
        "observed_episode_metrics": sum(row["available"] for row in rows),
        "missing_episode_metrics": count*3-sum(row["available"] for row in rows),
        "preparation_failed_cases": [case["case_id"] for case in case_reports if not case["prepared"]],
        "failed_cases": [case["case_id"] for case in case_reports if not case["passed"]], "input_errors": errors,
        "recovered_audit_cases": [case["case_id"] for case in case_reports if case["audit_recovered"]],
        "conditions": {condition: statistics([row for row in rows if row["condition"] == condition]) for condition in CONDITIONS},
        "by_timing": {name: {condition: statistics([row for row in rows if row["timing"] == name and row["condition"] == condition])
                             for condition in CONDITIONS} for name in names},
        "paired_overhead_count": len(overheads), "paired_overheads": overheads,
        "mean_extra_path_length_m": sum(row["extra_path_length_m"] for row in overheads)/len(overheads) if overheads else None,
        "cases": case_reports}
    return report, rows


def summarize(output):
    """Write only the batch-level summary and full planned-denominator CSV."""
    report, rows = check_batch(output)
    write_json(output/"summary.json", report)
    with (output/"cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["case_id", "condition"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {report['prepared_cases']}/{report['total_cases']}; passed {report['passed_cases']}; "
          f"observed {report['observed_episode_metrics']}/{report['expected_episodes']}; errors: {report['input_errors']}")
    return report


def videos(args):
    """Encode existing observations only; preserve failed/missing exports too."""
    from scripts.export_video import export_video

    output = args.output_dir.resolve()
    manifest = load_json(output/"manifest.json")
    validate_batch(output, manifest)
    exported, existing, failures = 0, 0, []
    for case in manifest["cases"]:
        for condition in CONDITIONS:
            folder = within(output, case["output_subdir"])/condition
            video = folder/"trajectory.mp4"
            if video.is_file() and video.stat().st_size > 0:
                existing += 1
                continue
            try:
                export_video(folder)
                exported += 1
            except (OSError, ValueError, RuntimeError) as error:
                failures.append({"case_id": case["case_id"], "condition": condition, "error": str(error)})
    write_json(output/"video_status.json", {"expected_videos": manifest["expected_episodes"],
        "new_exports": exported, "existing_videos": existing, "failed_exports": failures})
    return 1 if failures else 0


def cpu_check(output):
    """Rehearse all eligible fixed cases; never promote geometry to GPU success."""
    from scripts.subgoal_cpu import check_case

    manifest = load_json(output/"manifest.json")
    validate_batch(output, manifest)
    records = []
    for case in manifest["cases"]:
        result = {"case_id": case["case_id"], "geometry_passed": False,
                  "error": case["preparation_error"]}
        if case["prepared"] is True:
            try:
                result.update(check_case(within(output, case["preparation_dir"])/"config.json"))
            except ERRORS as error:
                result["error"] = f"{type(error).__name__}: {error}"
        records.append(result)
        print(f"CPU geometry {case['case_id']}: {result['geometry_passed']}", flush=True)
    passed = sum(record["geometry_passed"] is True for record in records)
    report = {"simulator_created": False, "visibility_evaluated": False,
              "total_cases": len(records), "geometry_passed_cases": passed,
              "all_geometry_passed": passed == len(records), "cases": records}
    write_json(output/"cpu_navigation_check.json", report)
    return 0 if report["all_geometry_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="CPU: freeze all source tasks at requested travel fractions")
    preparing.add_argument("--source-dir", type=Path, required=True)
    preparing.add_argument("--timings", nargs="+", choices=list(TIMINGS), default=list(TIMINGS))
    running = commands.add_parser("run", help="GPU: sequential isolated three-branch children, no retry/overwrite")
    running.add_argument("--gpu-id", type=int, default=0)
    commands.add_parser("summarize", help="CPU Habitat mesh queries: independently audit every fixed case")
    commands.add_parser("videos", help="CPU/FFmpeg: encode every planned branch")
    commands.add_parser("cpu-check", help="Native Agent/PathFinder rehearsal; no Simulator or visibility")
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "run":
        if args.gpu_id < 0:
            parser.error("GPU index must be nonnegative")
        return run(args)
    if args.command == "videos":
        return videos(args)
    if args.command == "cpu-check":
        return cpu_check(args.output_dir.resolve())
    return 0 if summarize(args.output_dir)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
