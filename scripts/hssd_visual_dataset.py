"""Full-roster middle V navigation batches with read-only evidence reuse.

This additive coordinator leaves the frozen single-task executor/evaluator
unchanged. Preparation never renders; run launches one single-task child at a
time; summaries/audits reconstruct the raw six-branch evidence for every task.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from pathlib import Path
import subprocess
import sys

from mivln.benchmark_navigation import TASK_TYPES
from scripts import benchmark_navigation as source
from scripts import hssd_visual_navigation as single
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import PROJECT_ROOT, sha256, within
from scripts.language_annotations import read_json
from scripts.public_inputs import fresh_output, write_json


KIND = "hssd_middle_visual_batch_preparation_v1"
RUN_KIND = "hssd_middle_visual_batch_run_v1"
CODE = "scripts/hssd_visual_dataset.py"


def code_bindings():
    return {CODE: sha256(PROJECT_ROOT / CODE), **single.bindings()}


def require_disjoint(output, protected):
    """Check ownership even for an already-existing prepared batch directory."""
    output = output.resolve()
    for path in protected:
        path = path.resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError("Output overlaps protected source/preparation/evidence: " + str(path))


def disjoint_output(output, protected):
    """Reports/children must never overwrite or live inside reused evidence."""
    require_disjoint(output, protected)
    return fresh_output(output)


def roster(directory):
    """Use all validated source bases in canonical order, with no subset flag."""
    manifest = source.load_preparation(directory)
    rows = source.select_cases(manifest, "all", "middle")
    tasks = list(dict.fromkeys(row["source_task_id"] for row in rows))
    if not tasks or len(tasks) != manifest["base_tasks"] or len(rows) != len(tasks)*3:
        raise ValueError("The batch requires the full source roster and all three middle cases")
    return tasks


def planned_slots(tasks):
    return [{"task_id": task["task_id"], "case_id": f"{task['task_id']}_{family}_middle",
             "task_type": family, "condition": condition}
            for task in tasks for family in TASK_TYPES for condition in single.CONDITIONS]


def build_task(source_dir, identity):
    """Keep a failed preparation as six planned slots, never remove that base."""
    try:
        manifest, configs = single.build(source_dir, identity)
        return manifest, configs, None
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return None, {}, str(error)


def seal_prepared(directory, manifest, configs, *, write=False):
    """Match the frozen single-task format without changing its writer or hashes."""
    expected = deepcopy(manifest)
    if write:
        for name, config in configs.items():
            write_json(directory / name, config)
    for slot in expected["planned_episodes"]:
        path = within(directory, slot["config"])
        if read_json(path) != configs[slot["config"]]:
            raise ValueError("Batch preparation changed reviewed task content")
        slot["config_sha256"] = sha256(path)
    if write:
        write_json(directory / "manifest.json", expected)
    elif read_json(directory / "manifest.json") != expected:
        raise ValueError("Single-task preparation differs from its frozen source")
    return expected


def reuse_identity(directory):
    run = read_json(directory / "manifest.json")
    if run.get("kind") != "hssd_middle_visual_run_v1" or run.get("code_bindings") != single.bindings():
        raise ValueError("Reused run must use the same HSSD executor/evaluator code")
    prepared = single.load_prepared(Path(run["prepared_dir"]))
    if sha256(Path(run["prepared_dir"]) / "manifest.json") != run["prepared_sha256"]:
        raise ValueError("Reused preparation binding changed")
    return prepared["base_task"], run, prepared


def inspect_reuse(directory, expected, configs):
    """Audit complete evidence, including honest task failures, without rerunning.

    Whole-tree snapshots bind raw PNGs, responses, masks, logs and caches. A
    cached success flag is never used to decide eligibility. Adding files later
    changes this binding; optional videos should go in a separate export tree.
    """
    before = snapshot({"reuse": directory})["reuse"]
    identity, run, prepared = reuse_identity(directory)
    comparable = deepcopy(prepared)
    for slot in comparable["planned_episodes"]:
        path = within(Path(run["prepared_dir"]), slot["config"])
        if read_json(path) != configs[slot["config"]]:
            raise ValueError("Reused geometry, text, seed, timing or language revision differs")
        slot["config_sha256"] = None
    if comparable != expected:
        raise ValueError("Reused task/preparation differs from the full-roster source")
    report = single.check_run(directory)
    after = snapshot({"reuse": directory})["reuse"]
    if before != after:
        raise ValueError("Reused source changed during read-only validation")
    # The original acceptance result is preserved, not used as a success filter.
    # Invalid or missing evidence is ineligible; a complete valid Oracle failure
    # stays a failed task in the new batch rather than being selectively rerun.
    if (len(report["episodes"]) != 6 or report["pairing_errors"]
            or any(not row["evidence_valid"] for row in report["episodes"])):
        raise ValueError("Reused run has incomplete/invalid evidence or pairing")
    return {"run_dir": str(directory.resolve()), "prepared_dir": str(Path(run["prepared_dir"]).resolve()),
            "prepared_manifest_sha256": run["prepared_sha256"], "gpu_id": run["gpu_id"], "files": after,
            "source_acceptance_passed": report["passed"]}, report


def prepare(args):
    source_dir = args.source_dir.resolve()
    identities = roster(source_dir)
    candidates, reused_preparations = {}, []
    for value in args.reuse_run:
        directory = value.resolve()
        identity, run, _ = reuse_identity(directory)
        if identity not in identities or identity in candidates:
            raise ValueError("Duplicate or out-of-roster reused task: " + identity)
        candidates[identity] = directory
        reused_preparations.append(Path(run["prepared_dir"]))
    output = disjoint_output(args.output_dir, [source_dir, *candidates.values(), *reused_preparations])
    tasks, preparations, reports = [], {}, {}
    for identity in identities:
        expected, configs, error = build_task(source_dir, identity)
        task = {"task_id": identity, "prepared_dir": f"tasks/{identity}/prepared",
            "prepared_manifest_sha256": None, "preparation_error": error,
            "reuse": None, "reuse_audit": None, "reuse_audit_sha256": None}
        if identity in candidates:
            if error is not None:
                raise ValueError("Cannot reuse an unprepared source task: " + identity)
            task["reuse"], reports[identity] = inspect_reuse(candidates[identity], expected, configs)
        preparations[identity] = expected, configs
        tasks.append(task)
    # All source/reuse checks above complete before creating any destination.
    output.mkdir(parents=True)
    for task in tasks:
        identity = task["task_id"]
        expected, configs = preparations[identity]
        if expected is not None:
            folder = within(output, task["prepared_dir"])
            seal_prepared(folder, expected, configs, write=True)
            task["prepared_manifest_sha256"] = sha256(folder / "manifest.json")
        if task["reuse"]:
            task["reuse_audit"] = f"reuse_audits/{identity}.json"
            write_json(output / task["reuse_audit"], reports[identity])
            task["reuse_audit_sha256"] = sha256(output / task["reuse_audit"])
    reused = sum(task["reuse"] is not None for task in tasks)
    manifest = {"schema_version": 1, "kind": KIND, "timing": "middle",
        "source_dir": str(source_dir), "source_manifest_sha256": sha256(source_dir / "manifest.json"),
        "code_bindings": code_bindings(), "task_count": len(tasks), "planned_cases": len(tasks)*3,
        "planned_episodes": len(tasks)*6, "reused_tasks": reused, "reused_episodes": reused*6,
        "new_episodes": (len(tasks)-reused)*6, "tasks": tasks, "episode_plan": planned_slots(tasks)}
    write_json(output / "manifest.json", manifest)
    report = collect(manifest, {}, reports)
    write_json(output / "preparation_summary.json", report)
    print(f"Prepared {len(tasks)} middle bases / {len(tasks)*6} slots: {reused*6} reused, {(len(tasks)-reused)*6} new; no GPU execution")
    return 0 if all(task["preparation_error"] is None for task in tasks) else 1


def validate_prepared(directory):
    """Preflight the entire roster, every config and all reuse before any GPU."""
    directory = directory.resolve()
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != KIND
            or manifest.get("timing") != "middle" or manifest.get("code_bindings") != code_bindings()):
        raise ValueError("Batch kind/timing/code bindings changed")
    source_dir = Path(manifest["source_dir"])
    identities = roster(source_dir)
    if sha256(source_dir / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Full-roster source manifest changed")
    tasks = manifest["tasks"]
    if [task["task_id"] for task in tasks] != identities or manifest["episode_plan"] != planned_slots(tasks):
        raise ValueError("Batch must retain the entire canonical task/condition denominator")
    reused = sum(task["reuse"] is not None for task in tasks)
    counts = {"task_count": len(tasks), "planned_cases": len(tasks)*3, "planned_episodes": len(tasks)*6,
              "reused_tasks": reused, "reused_episodes": reused*6, "new_episodes": (len(tasks)-reused)*6}
    if any(type(manifest[key]) is not int or manifest[key] != value for key, value in counts.items()):
        raise ValueError("Frozen batch counts changed")
    require_disjoint(directory, [source_dir, *[Path(task["reuse"][key]) for task in tasks if task["reuse"]
                                              for key in ("run_dir", "prepared_dir")]])
    reports = {}
    for task in tasks:
        identity = task["task_id"]
        if task["prepared_dir"] != f"tasks/{identity}/prepared":
            raise ValueError("Task preparation ownership changed")
        expected, configs, error = build_task(source_dir, identity)
        if task["preparation_error"] != error:
            raise ValueError("Task preparation status changed")
        if expected is not None:
            folder = within(directory, task["prepared_dir"])
            seal_prepared(folder, expected, configs)
            if sha256(folder / "manifest.json") != task["prepared_manifest_sha256"]:
                raise ValueError("Prepared single-task seal changed")
        elif task["prepared_manifest_sha256"] is not None or task["reuse"] is not None:
            raise ValueError("Unprepared source cannot claim prepared/reused evidence")
        if task["reuse"]:
            reuse = task["reuse"]
            reuse_dir = Path(reuse["run_dir"])
            if snapshot({"reuse": reuse_dir})["reuse"] != reuse["files"]:
                raise ValueError("Reused source file roster/bytes changed: " + identity)
            actual, reports[identity] = inspect_reuse(reuse_dir, expected, configs)
            if actual != reuse:
                raise ValueError("Reused identity/configuration/audit binding changed")
            path = within(directory, task["reuse_audit"])
            if sha256(path) != task["reuse_audit_sha256"] or read_json(path) != reports[identity]:
                raise ValueError("Frozen reuse audit changed")
    return manifest, reports


def collect(manifest, statuses, reports):
    """Fixed-denominator coverage and scores, separated by task and condition.

    Pairing invalidates acceptance, not the recorded raw metric itself. Counts
    in success numerators require valid, paired raw evidence. Pending/missing
    slots contribute zero and remain explicitly provisional until audited.
    """
    rows, execution_errors = [], []
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    for slot in manifest["episode_plan"]:
        identity = slot["task_id"]
        task, state, report = tasks[identity], statuses.get(identity), reports.get(identity)
        row = {**slot, "origin": "reused" if task["reuse"] else "new", "integrity_passed": False,
               "task_success": False, "measured_task_success": None, "accepted": False,
               "audit_status": "pending", "source_run": task["reuse"]["run_dir"] if task["reuse"] else None}
        if report is not None:
            keys = [(r["case_id"], r["task_type"], r["condition"]) for r in report["episodes"]]
            expected = [(s["case_id"], s["task_type"], s["condition"]) for s in manifest["episode_plan"] if s["task_id"] == identity]
            if keys != expected:
                raise ValueError("Per-task audit returned a changed/duplicate six-slot roster")
            actual = next(r for r in report["episodes"] if r["case_id"] == slot["case_id"] and r["condition"] == slot["condition"])
            valid = bool(actual["evidence_valid"] and not report["pairing_errors"])
            row.update(integrity_passed=valid, measured_task_success=actual["task_success"],
                task_success=bool(valid and actual["task_success"]), accepted=bool(valid and actual["passed"]),
                audit_status="valid" if valid else "invalid", metrics=actual,
                pairing_errors=report["pairing_errors"])
        elif task["preparation_error"]:
            row.update(audit_status="preparation_failed", error=task["preparation_error"])
        elif state is not None:
            row.update(audit_status=state["state"], error=state.get("error"), source_run=state.get("run_dir"))
        if state is not None and state.get("run_dir"):
            row["source_run"] = state["run_dir"]
        rows.append(row)
    for identity, state in statuses.items():
        if state["state"] not in ("completed", "reused") or state["state"] == "completed" and state["returncode"] != 0:
            execution_errors.append({"task_id": identity, **state})
    groups = {}
    for family in TASK_TYPES:
        groups[family] = {}
        for condition in single.CONDITIONS:
            group = [row for row in rows if row["task_type"] == family and row["condition"] == condition]
            count = len(group)
            successes = sum(row["task_success"] for row in group)
            accepted = sum(row["accepted"] for row in group)
            groups[family][condition] = {"planned": count, "valid_evidence": sum(row["integrity_passed"] for row in group),
                "task_successes": successes, "task_success_rate": successes/count,
                "accepted": accepted, "acceptance_rate": accepted/count}
    audited = sum(row["integrity_passed"] for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    task_statistics = []
    for task in manifest["tasks"]:
        group = [row for row in rows if row["task_id"] == task["task_id"]]
        task_statistics.append({"task_id": task["task_id"], "origin": "reused" if task["reuse"] else "new",
            "planned_episodes": 6, "valid_evidence": sum(row["integrity_passed"] for row in group),
            "accepted": sum(row["accepted"] for row in group),
            "oracle_successes": sum(row["task_success"] for row in group if row["condition"] == "oracle"),
            "ignore_expected_failures": sum(row["accepted"] for row in group if row["condition"] == "ignore")})
    return {"passed": audited == accepted == manifest["planned_episodes"] and not execution_errors,
        "task_count": manifest["task_count"], "planned_cases": manifest["planned_cases"],
        "planned_episodes": manifest["planned_episodes"], "reused_episodes": manifest["reused_episodes"],
        "new_episodes": manifest["new_episodes"], "valid_evidence_episodes": audited,
        "accepted_episode_slots": accepted, "results_provisional": audited != manifest["planned_episodes"]
            or any(state["state"] in ("pending", "running", "interrupted") for state in statuses.values()),
        "oracle_macro_success_rate": sum(groups[family]["oracle"]["task_success_rate"] for family in TASK_TYPES)/len(TASK_TYPES),
        "by_task_type": groups, "tasks": task_statistics, "episodes": rows, "execution_errors": execution_errors,
        "real_model_called": False, "release_ready": False}


def write_report(output, report):
    write_json(output / "summary.json", report)
    source.write_lines(output / "cases.jsonl", report["episodes"])
    fields = ("task_id", "case_id", "task_type", "condition", "origin", "source_run", "audit_status",
              "integrity_passed", "task_success", "accepted")
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in report["episodes"])
    source.write_lines(output / "tasks.jsonl", report["tasks"])
    with (output / "tasks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["tasks"][0]))
        writer.writeheader()
        writer.writerows(report["tasks"])


def check_run(directory):
    run = read_json(directory / "manifest.json")
    if run.get("kind") != RUN_KIND or run.get("code_bindings") != code_bindings():
        raise ValueError("Batch run/code binding changed")
    if type(run.get("gpu_id")) is not int or run["gpu_id"] < 0:
        raise ValueError("Invalid assigned batch GPU")
    prepared_dir = Path(run["prepared_dir"])
    if sha256(prepared_dir / "manifest.json") != run["prepared_manifest_sha256"]:
        raise ValueError("Prepared batch manifest changed")
    manifest, reports = validate_prepared(prepared_dir)
    if run["episode_plan"] != manifest["episode_plan"]:
        raise ValueError("Batch run dropped or changed planned episode slots")
    states = read_json(directory / "execution_status.json")
    if [state["task_id"] for state in states] != [task["task_id"] for task in manifest["tasks"]]:
        raise ValueError("Task execution status denominator/order changed")
    statuses = {state["task_id"]: state for state in states}
    for task in manifest["tasks"]:
        identity, state = task["task_id"], statuses[task["task_id"]]
        expected_path = (Path(task["reuse"]["run_dir"]) if task["reuse"] else directory / "tasks" / identity / "run").resolve()
        if state["origin"] != ("reused" if task["reuse"] else "new") or Path(state["run_dir"]).resolve() != expected_path:
            raise ValueError("Task result ownership/reuse binding changed")
        if task["reuse"]:
            if state["state"] != "reused" or state["gpu_id"] != task["reuse"]["gpu_id"]:
                raise ValueError("Reused task state/GPU changed")
            continue
        if state["gpu_id"] != run["gpu_id"]:
            raise ValueError("New-task GPU differs from assigned batch GPU")
        if (expected_path / "manifest.json").exists():
            try:
                child = read_json(expected_path / "manifest.json")
                child_prepared = within(prepared_dir, task["prepared_dir"])
                if (Path(child["prepared_dir"]).resolve() != child_prepared.resolve()
                        or child["prepared_sha256"] != task["prepared_manifest_sha256"] or child["gpu_id"] != run["gpu_id"]):
                    raise ValueError("Child run differs from the frozen task preparation/GPU")
                reports[identity] = single.check_run(expected_path)
            except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, ImportError) as error:
                state["error"] = str(error)
        elif state["state"] == "completed":
            state.update(state="failed", error="Completed child is missing its run manifest")
    return collect(manifest, statuses, reports)


def run(args):
    if type(args.gpu_id) is not int or args.gpu_id < 0:
        raise ValueError("Use a nonnegative assigned GPU ID")
    manifest, reports = validate_prepared(args.input_dir)
    protected = [args.input_dir, Path(manifest["source_dir"]), *[Path(task["reuse"][key])
        for task in manifest["tasks"] if task["reuse"] for key in ("run_dir", "prepared_dir")]]
    output = disjoint_output(args.output_dir, protected)
    output.mkdir(parents=True)
    write_json(output / "manifest.json", {"kind": RUN_KIND, "prepared_dir": str(args.input_dir.resolve()),
        "prepared_manifest_sha256": sha256(args.input_dir / "manifest.json"), "gpu_id": args.gpu_id,
        "episode_plan": manifest["episode_plan"], "code_bindings": code_bindings()})
    states = [{"task_id": task["task_id"], "origin": "reused" if task["reuse"] else "new",
        "state": "reused" if task["reuse"] else "pending", "returncode": None, "error": None,
        "gpu_id": task["reuse"]["gpu_id"] if task["reuse"] else args.gpu_id,
        "run_dir": task["reuse"]["run_dir"] if task["reuse"] else str((output / "tasks" / task["task_id"] / "run").resolve())}
        for task in manifest["tasks"]]
    write_json(output / "execution_status.json", states)
    for task, state in zip(manifest["tasks"], states):
        if task["reuse"]:
            continue
        if task["preparation_error"]:
            state.update(state="failed", error=task["preparation_error"])
            write_json(output / "execution_status.json", states)
            continue
        state["state"] = "running"
        write_json(output / "execution_status.json", states)
        command = [sys.executable, "-B", "-m", "scripts.hssd_visual_navigation", "run", "--input-dir",
            str(within(args.input_dir, task["prepared_dir"])), "--gpu-id", str(args.gpu_id), "--output-dir", state["run_dir"]]
        interrupted = False
        try:
            print(f"Running {task['task_id']} middle: three types / six branches", flush=True)
            log = output / "logs" / (task["task_id"] + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as stream:
                child = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
            state.update(state="completed" if child.returncode == 0 else "failed", returncode=child.returncode)
        except KeyboardInterrupt:
            state.update(state="interrupted", error="Interrupted; later slots stay pending")
            interrupted = True
        except OSError as error:
            state.update(state="failed", error=str(error))
        write_json(output / "execution_status.json", states)
        if interrupted:
            break
    report = check_run(output)
    write_report(output, report)
    print(f"Middle batch: accepted {report['accepted_episode_slots']}/{report['planned_episodes']}; provisional={report['results_provisional']}")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "summarize", "audit"):
        child = sub.add_parser(command)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "outputs/benchmark_v_navigation_prepare_002")
            child.add_argument("--reuse-run", type=Path, action="append", default=[])
        else:
            child.add_argument("--input-dir", type=Path, required=True)
        if command == "run":
            child.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            return prepare(args)
        if args.command == "run":
            if args.gpu_id < 0:
                raise ValueError("Use a nonnegative assigned GPU ID")
            return run(args)
        run_manifest = read_json(args.input_dir / "manifest.json")
        prepared_dir = Path(run_manifest["prepared_dir"])
        prepared, _ = validate_prepared(prepared_dir)
        roots = {"batch_run": args.input_dir, "batch_preparation": prepared_dir,
                 "source_preparation": Path(prepared["source_dir"]),
                 **{key+"_"+task["task_id"]: Path(task["reuse"][key]) for task in prepared["tasks"]
                    if task["reuse"] for key in ("run_dir", "prepared_dir")}}
        output = disjoint_output(args.output_dir, list(roots.values()))
        before = snapshot(roots) if args.command == "audit" else None
        report = check_run(args.input_dir)
        if before is not None:
            after = snapshot(roots)
            report["source_files_unchanged"] = before == after
            report["passed"] = report["passed"] and before == after
        output.mkdir(parents=True)
        write_report(output, report)
        if before is not None:
            write_json(output / "audit.json", report)
            write_json(output / "source_bindings.json", {"before": before, "after": after})
        print(f"Read-only middle {args.command}: accepted={report['accepted_episode_slots']}/{report['planned_episodes']}; passed={report['passed']}")
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, ImportError) as error:
        print(f"HSSD middle batch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
