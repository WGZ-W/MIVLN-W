"""Fixed-collection language navigation, with read-only verified result reuse.

The denominator is the complete source roster times Oracle/model/Ignore. This
module only orchestrates the existing single-task protocol: it neither changes
model prompts nor supplies a reference answer after an unsuccessful prediction.
Preparation/auditing use CPU evidence; only ``run`` launches GPU child processes.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
from types import SimpleNamespace

from mivln.task_update_prompts import system_prompt
from scripts import hssd_dataset, language_navigation as single, task_update as base
from scripts.debug_dataset import execute_stage
from scripts.export_video import export_video
from scripts.hssd_targets import sha256


PROTOCOL = "hssd_language_navigation_dataset_v1"
CONDITIONS = single.CONDITIONS
BRANCH_FILES = ("episode.json", "metrics.json", "trajectory.json", "events.json",
                "replans.json", "final_task_state.json", "final_policy_state.json",
                "terminal_diagnostics.json")
EVIDENCE_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    """Freeze orchestration as well as the unchanged single-task audit code."""
    names = ("scripts/language_navigation_dataset.py", "scripts/debug_dataset.py",
             "scripts/export_video.py")
    return {**single.code_hashes(), **{name: sha256(base.ROOT / name) for name in names}}


def prompt_hash():
    return hashlib.sha256(system_prompt(base.INSTRUCTIONS).encode("utf-8")).hexdigest()


def controls_from(args):
    controls = {"model": args.model, "base_url": args.base_url,
                "temperature": args.temperature, "max_tokens": args.max_tokens,
                "timeout_seconds": args.timeout, "thinking_mode": args.thinking_mode}
    validate_controls(controls)
    return controls


def validate_controls(controls):
    """Validate before creating a manifest or launching any model/renderer."""
    if (set(controls) != {"model", "base_url", "temperature", "max_tokens", "timeout_seconds", "thinking_mode"}
            or not controls["model"].strip() or not math.isfinite(controls["temperature"])
            or controls["temperature"] < 0 or not math.isfinite(controls["timeout_seconds"])
            or controls["timeout_seconds"] <= 0 or type(controls["max_tokens"]) is not int
            or controls["max_tokens"] < 1 or controls["thinking_mode"] not in ("off", "on", "default")):
        raise ValueError("Invalid frozen model controls")
    single.client.endpoint_for(controls["base_url"])


def roster(source):
    manifest = base.read_json(source / "manifest.json")
    hssd_dataset.validate_inputs(source, manifest)
    identities = [task["task_id"] for task in manifest["tasks"]]
    if (not identities or len(set(identities)) != len(identities)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in identities)):
        raise ValueError("Source requires unique safe task IDs")
    return manifest


def disjoint_output(output, protected):
    """Never create reports inside a source or reused evidence directory."""
    output = output.resolve()
    for path in protected:
        path = path.resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError(f"Output overlaps protected source/evidence: {path}")


def result_dir(output, task):
    return (Path(task["reuse"]["run_dir"]) if task["reuse"] else
            hssd_dataset.within(output, task["output_subdir"]))


def bind_run(directory, prepared, controls):
    """Compare configurations/controls, not just a claimed task ID or success.

    Older preparations may have different directory names. Their actual content
    must equal the fresh preparation, including source/reference hashes. GPU IDs
    may differ across tasks, but must agree within each three-branch experiment.
    """
    expected_manifest, expected_config = single.load_preparation(prepared)
    run = base.read_json(directory / "run.json")
    previous = Path(run["prepared_dir"])
    manifest, config = single.load_preparation(previous)
    if (run.get("kind") != single.PROTOCOL or manifest != expected_manifest or config != expected_config
            or sha256(previous / "manifest.json") != run["prepared_manifest_sha256"]
            or run["controls"] != controls or run["code_sha256"] != single.code_hashes()
            or run["system_prompt_sha256"] != prompt_hash()):
        raise ValueError("Run task, timing, sources, model controls, prompt or code differs from batch")
    if type(run["gpu_id"]) is not int or run["gpu_id"] < 0:
        raise ValueError("Invalid saved GPU ID")
    for condition in CONDITIONS:
        episode = directory / condition / "episode.json"
        if episode.exists() and base.read_json(episode).get("gpu_device_id") != run["gpu_id"]:
            raise ValueError("Branch GPU differs from its run metadata")
    return run


def evidence_hashes(directory):
    """Freeze consumed raw evidence, excluding optional videos/cached summaries.

    Include every recorded sensor payload and terminal diagnostic attachment.
    Resolve paths below the branch root to prevent an attachment escaping into
    another experiment. Adding a video later does not invalidate this snapshot.
    """
    files = {"run.json", *single.JOURNAL_FILES, "model/policy_updates.json"}
    for condition in CONDITIONS:
        branch = hssd_dataset.within(directory, condition)
        files.update(f"{condition}/{name}" for name in BRANCH_FILES)
        diagnostics = base.read_json(branch / "terminal_diagnostics.json")
        attachments = [item["path"] for item in diagnostics["files"]]
        for row in base.read_json(branch / "trajectory.json"):
            attachments.extend(row.get("frames", {}).values())
        for name in attachments:
            path = hssd_dataset.within(branch, name)
            files.add(str(path.relative_to(directory.resolve())))
    return {name: sha256(hssd_dataset.within(directory, name)) for name in sorted(files)}


def prepare(args):
    """Preflight the entire collection and all reuse before creating output."""
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    source_manifest, controls = roster(source), controls_from(args)
    reused = {}
    for folder in args.reuse_run:
        directory = folder.resolve()
        run = base.read_json(directory / "run.json")
        manifest, _ = single.load_preparation(Path(run["prepared_dir"]))
        identity = manifest["task_id"]
        if identity in reused or identity not in [t["task_id"] for t in source_manifest["tasks"]]:
            raise ValueError("Duplicate or out-of-roster reused task: " + identity)
        if manifest["timing"] != args.timing:
            raise ValueError("Reused timing differs: " + identity)
        reused[identity] = directory
    protected = [source, *reused.values()]
    for task in source_manifest["tasks"]:
        config = base.read_json(hssd_dataset.within(source, task["base_config"]))
        protected.extend(single.project_path(config[key]) for key in ("scene", "scene_dataset_config", "navmesh"))
    disjoint_output(output, protected)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    tasks, audits = [], {}
    # Staging is temporary and CPU only. Invalid reuse leaves no batch directory
    # and cannot silently turn into a new GPU run with different evidence.
    with tempfile.TemporaryDirectory(prefix="language-batch-") as temporary:
        staging = Path(temporary)
        for item in source_manifest["tasks"]:
            identity = item["task_id"]
            prepared = staging / identity
            with redirect_stdout(io.StringIO()):
                single.prepare(SimpleNamespace(source_dir=source, task_id=identity,
                                               timing=args.timing, output_dir=prepared))
            record = {"task_id": identity, "preparation_dir": f"preparations/{identity}",
                      "preparation_manifest_sha256": sha256(prepared / "manifest.json"),
                      "config_sha256": sha256(prepared / "config.json"),
                      "output_subdir": f"tasks/{identity}", "reuse": None}
            if identity in reused:
                directory = reused[identity]
                run = bind_run(directory, prepared, controls)
                audit = single.check_experiment(directory)
                # A complete, honestly failed model result is reusable. Requiring
                # `passed` here would bias the batch towards successful cases.
                if not audit["integrity_passed"]:
                    raise ValueError("Reused evidence failed integrity audit: " + identity)
                audits[identity] = audit
                record["reuse"] = {"run_dir": str(directory), "gpu_id": run["gpu_id"],
                    "artifacts_sha256": evidence_hashes(directory),
                    "audit_snapshot": f"reuse_audits/{identity}.json"}
            tasks.append(record)
        base.fresh_output(output)
        (output / "preparations").mkdir()
        for task in tasks:
            shutil.copytree(staging / task["task_id"], output / task["preparation_dir"])
            if task["reuse"]:
                path = output / task["reuse"]["audit_snapshot"]
                path.parent.mkdir(exist_ok=True)
                base.write_json(path, audits[task["task_id"]])
                task["reuse"]["audit_snapshot_sha256"] = sha256(path)
    count, reused_count = len(tasks), len(reused)
    base.write_json(output / "manifest.json", {"schema_version": 1, "kind": PROTOCOL,
        "source_dir": str(source), "source_manifest_sha256": sha256(source / "manifest.json"),
        "timing": args.timing, "conditions": list(CONDITIONS), "controls": controls,
        "system_prompt_sha256": prompt_hash(), "code_sha256": source_hashes(),
        "task_count": count, "planned_episodes": count * 3, "planned_model_calls": count,
        "reused_tasks": reused_count, "new_tasks": count - reused_count,
        "new_episodes": (count - reused_count) * 3, "new_model_calls": count - reused_count,
        "tasks": tasks, "created_at": now(),
        "scope": "One replacement per task; text model plus privileged Oracle navigator; fixed source roster"})
    print(f"Prepared {count} tasks/{count * 3} episodes; reuse {reused_count}, "
          f"new {count - reused_count} tasks/{(count - reused_count) * 3} episodes. No GPU/model calls.")
    return 0


def validate_batch(output, verify_reuse=True):
    """Fail before launching if any frozen input, roster or reused file drifts."""
    manifest = base.read_json(output / "manifest.json")
    if (manifest.get("kind") != PROTOCOL or manifest.get("schema_version") != 1
            or manifest.get("conditions") != list(CONDITIONS)
            or manifest["code_sha256"] != source_hashes()
            or manifest["system_prompt_sha256"] != prompt_hash()):
        raise ValueError("Batch protocol, code or system prompt changed")
    validate_controls(manifest["controls"])
    source = Path(manifest["source_dir"])
    original = roster(source)
    if sha256(source / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    tasks = manifest["tasks"]
    if [t["task_id"] for t in tasks] != [t["task_id"] for t in original["tasks"]]:
        raise ValueError("Batch must contain the entire fixed source roster in order")
    count, reused = len(tasks), sum(t["reuse"] is not None for t in tasks)
    expected = {"task_count": count, "planned_episodes": 3 * count, "planned_model_calls": count,
                "reused_tasks": reused, "new_tasks": count - reused,
                "new_episodes": 3 * (count - reused), "new_model_calls": count - reused}
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("Frozen denominators changed")
    disjoint_output(output, [source, *[Path(t["reuse"]["run_dir"]) for t in tasks if t["reuse"]]])
    for task, source_task in zip(tasks, original["tasks"]):
        identity = task["task_id"]
        if task["preparation_dir"] != f"preparations/{identity}" or task["output_subdir"] != f"tasks/{identity}":
            raise ValueError("Task output/preparation ownership changed")
        prepared = hssd_dataset.within(output, task["preparation_dir"])
        info, _ = single.load_preparation(prepared)
        expected_source = hssd_dataset.within(source, source_task["output_subdir"]) / f"timing/configs/{manifest['timing']}.json"
        if (info["task_id"] != identity or info["timing"] != manifest["timing"]
                or Path(info["source_config"]).resolve() != expected_source.resolve()
                or sha256(prepared / "manifest.json") != task["preparation_manifest_sha256"]
                or sha256(prepared / "config.json") != task["config_sha256"]):
            raise ValueError("Task preparation changed: " + identity)
        if verify_reuse and task["reuse"]:
            reuse = task["reuse"]
            directory = Path(reuse["run_dir"])
            run = bind_run(directory, prepared, manifest["controls"])
            if run["gpu_id"] != reuse["gpu_id"] or evidence_hashes(directory) != reuse["artifacts_sha256"]:
                raise ValueError("Reused raw evidence changed: " + identity)
            snapshot = hssd_dataset.within(output, reuse["audit_snapshot"])
            if sha256(snapshot) != reuse["audit_snapshot_sha256"]:
                raise ValueError("Reuse audit snapshot changed: " + identity)
    return manifest


def child_command(output, task, manifest, args):
    """All scientific controls come from the manifest, never runtime overrides."""
    controls = manifest["controls"]
    command = [sys.executable, "-u", "-m", "scripts.language_navigation", "run",
               "--prepared-dir", str(hssd_dataset.within(output, task["preparation_dir"])),
               "--output-dir", str(result_dir(output, task)), "--gpu-id", str(args.gpu_id),
               "--api-key-env", args.api_key_env]
    for flag, key in (("model", "model"), ("base-url", "base_url"), ("temperature", "temperature"),
                      ("max-tokens", "max_tokens"), ("timeout", "timeout_seconds"), ("thinking-mode", "thinking_mode")):
        command.extend(["--" + flag, str(controls[key])])
    return command


def run(args):
    output = args.output_dir.resolve()
    manifest = validate_batch(output)
    if type(args.gpu_id) is not int or args.gpu_id < 0:
        raise ValueError("GPU ID must be non-negative")
    # A second run is an error, even after interruption: never overwrite or retry
    # selectively based on outcomes. Start a fresh explicitly prepared batch.
    for name in ("batch_run.json", "run_status.json", "run_logs", "tasks"):
        if (output / name).exists():
            raise FileExistsError("Batch already started or output occupied: " + name)
    from scripts.run_navigation import scene_provenance, validate_config
    for task in manifest["tasks"]:
        if task["reuse"]:
            audit = single.check_experiment(result_dir(output, task))
            if not audit["integrity_passed"]:
                raise ValueError("Reused evidence no longer passes audit")
        elif result_dir(output, task).exists():
            raise FileExistsError("Pending task output already exists")
        else:
            _, config = single.load_preparation(output / task["preparation_dir"])
            validate_config(config)
            scene_provenance(config, single.project_path(config["scene"]))
    statuses = [{"task_id": t["task_id"], "origin": "reused" if t["reuse"] else "new",
                 "status": "reused" if t["reuse"] else "pending", "returncode": None,
                 "gpu_id": t["reuse"]["gpu_id"] if t["reuse"] else args.gpu_id, "error": None}
                for t in manifest["tasks"]]
    record = {"kind": PROTOCOL, "manifest_sha256": sha256(output / "manifest.json"),
              "code_sha256": source_hashes(), "gpu_id": args.gpu_id, "started_at": now()}
    base.write_json(output / "batch_run.json", record)
    base.write_json(output / "run_status.json", statuses)
    for task, status in zip(manifest["tasks"], statuses):
        if task["reuse"]:
            continue
        status.update(status="running", started_at=now())
        base.write_json(output / "run_status.json", statuses)
        interrupted = False
        try:
            print(f"Running {task['task_id']} (Oracle/model/Ignore)...", flush=True)
            code = execute_stage(child_command(output, task, manifest, args),
                                 output / "run_logs" / f"{task['task_id']}.log")
            status.update(status="finished", returncode=code)
        except KeyboardInterrupt:
            status.update(status="interrupted", error="Interrupted; future slots remain pending")
            interrupted = True
        except OSError as error:
            status.update(status="launch_failed", error=f"{type(error).__name__}: {error}")
        status["finished_at"] = now()
        base.write_json(output / "run_status.json", statuses)
        if interrupted:
            break
        # A non-zero exit can be a valid model failure. Keep its evidence and
        # continue the fixed roster rather than selecting successful cases.
    record.update(finished_at=now(), run_status_sha256=sha256(output / "run_status.json"))
    base.write_json(output / "batch_run.json", record)
    return 0 if summarize(output)["passed"] else 1


def latency_summary(values):
    return {"count": len(values), "mean_seconds": statistics.mean(values) if values else None,
            "median_seconds": statistics.median(values) if values else None,
            "max_seconds": max(values) if values else None}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    """Re-audit raw runs, retaining three rows even for missing/invalid tasks.

    Rates always divide by planned tasks, including unrun/invalid slots. A
    provisional flag prevents a pre-run snapshot being mistaken for final model
    accuracy. Cached old summaries and preparation-time audit snapshots never
    provide scores; each available task is independently re-audited.
    """
    output = output.resolve()
    # Reuse corruption becomes an explicit failed task in reports. Running uses
    # strict validation, so corrupt imports cannot trigger fresh GPU fallback.
    manifest = validate_batch(output, verify_reuse=False)
    statuses = None
    if (output / "batch_run.json").exists():
        record = base.read_json(output / "batch_run.json")
        statuses = base.read_json(output / "run_status.json")
        if (record["manifest_sha256"] != sha256(output / "manifest.json")
                or record["code_sha256"] != source_hashes()
                or [s["task_id"] for s in statuses] != [t["task_id"] for t in manifest["tasks"]]
                or ("run_status_sha256" in record and record["run_status_sha256"] != sha256(output / "run_status.json"))):
            raise ValueError("Batch execution metadata changed")
    tasks, cases, calls = [], [], []
    for index, task in enumerate(manifest["tasks"]):
        directory, report, error, gpu_id = result_dir(output, task), None, None, None
        _, config = single.load_preparation(output / task["preparation_dir"])
        event = config["events"][0]
        available = (directory / "run.json").is_file()
        state = statuses[index]["status"] if statuses else ("reused" if task["reuse"] else "pending")
        try:
            if not available:
                raise FileNotFoundError("Not run" if state == "pending" else "No run.json; see run_status/log")
            metadata = bind_run(directory, output / task["preparation_dir"], manifest["controls"])
            gpu_id = metadata["gpu_id"]
            if statuses and gpu_id != statuses[index]["gpu_id"]:
                raise ValueError("Task GPU differs from batch execution metadata")
            if task["reuse"]:
                reuse = task["reuse"]
                if (gpu_id != reuse["gpu_id"] or evidence_hashes(directory) != reuse["artifacts_sha256"]
                        or sha256(hssd_dataset.within(output, reuse["audit_snapshot"])) != reuse["audit_snapshot_sha256"]):
                    raise ValueError("Reused raw evidence or audit snapshot changed")
            report = single.check_experiment(directory)
        except EVIDENCE_ERRORS as failure:
            error = f"{type(failure).__name__}: {failure}"
        origin = "reused" if task["reuse"] else "new"
        item = {"task_id": task["task_id"], "origin": origin, "run_dir": str(directory),
                "scene_id": config["scene_id"], "seed": config["seed"], "timing": manifest["timing"],
                "trigger_step": event["trigger"]["value"], "initial_goal": config["initial_goal"],
                "replacement_goal": event["task_update"]["new_goal"],
                "execution_status": state, "returncode": statuses[index]["returncode"] if statuses else None,
                "gpu_id": gpu_id, "available": available, "audit_returned": report is not None,
                "integrity_passed": bool(report and report["integrity_passed"]),
                "prefix_matched": bool(report and report["prefix_matched"]),
                "prefix_error": report["prefix_error"] if report else None,
                "model_update_correct": bool(report and report["model_update_correct"]),
                "model_pipeline_success": bool(report and report["model_pipeline_success"]),
                "model_failure_stage": report["model_failure_stage"] if report else ("not_run" if state == "pending" and not available else "evidence"),
                "model_outcome": report["model_outcome"] if report else None,
                "model_callback_seconds": report["model_callback_seconds"] if report else None,
                "passed": bool(report and report["passed"]), "error": error}
        # Cost comparisons are paired within a task and only available when both
        # branches pass their evidence audit. Unrun/invalid costs remain null.
        branches = {row["condition"]: row for row in report["conditions"]} if report else {}
        paired = all(branches.get(c, {}).get("integrity_passed") for c in ("oracle", "model"))
        for metric in ("steps", "path_length_m"):
            item["model_minus_oracle_" + metric] = (
                branches["model"][metric] - branches["oracle"][metric] if paired else None)
        tasks.append(item)
        calls.append({key: item[key] for key in ("task_id", "origin", "gpu_id", "model_update_correct",
            "model_pipeline_success", "model_failure_stage", "model_outcome", "model_callback_seconds", "error")})
        for condition in CONDITIONS:
            observed = next((row for row in report["conditions"] if row["condition"] == condition), {}) if report else {}
            valid = bool(observed.get("integrity_passed"))
            cases.append({"task_id": task["task_id"], "condition": condition, "origin": origin,
                "run_dir": str(directory), "gpu_id": gpu_id, "task_integrity_passed": item["integrity_passed"],
                "integrity_passed": valid, "dynamic_success": valid and bool(observed.get("dynamic_success")),
                "joint_success": valid and bool(observed.get("joint_success")),
                "ignore_expected_failure": valid and bool(observed.get("ignore_expected_failure")),
                "steps": observed.get("steps"), "path_length_m": observed.get("path_length_m"),
                "navigation_error_m": observed.get("navigation_error_m"),
                "error": error or observed.get("error") or (report["prefix_error"] if report and not report["prefix_matched"] else None)})
    count = manifest["task_count"]
    grouped = {}
    for condition in CONDITIONS:
        rows = [row for row in cases if row["condition"] == condition]
        grouped[condition] = {"planned": count, "audited": sum(r["integrity_passed"] for r in rows)}
        for metric in ("dynamic_success", "joint_success", "ignore_expected_failure"):
            total = sum(r[metric] for r in rows)
            grouped[condition].update({metric + "_count": total, metric + "_rate": total / count})
    audited = sum(t["integrity_passed"] for t in tasks)
    complete = audited == count
    samples = [t for t in tasks if t["model_callback_seconds"] is not None]
    latencies = {"all": latency_summary([t["model_callback_seconds"] for t in samples])}
    for origin in ("reused", "new"):
        latencies[origin] = latency_summary([t["model_callback_seconds"] for t in samples if t["origin"] == origin])
    stages = {stage: sum(t["model_failure_stage"] == stage for t in tasks)
              for stage in ("not_run", "evidence", "model_call", "update", "navigation", None)}
    summary = {"kind": PROTOCOL, "task_count": count, "planned_episodes": 3 * count,
        "planned_model_calls": count, "reused_tasks": manifest["reused_tasks"], "new_tasks": manifest["new_tasks"],
        "available_tasks": sum(t["available"] for t in tasks), "audited_tasks": audited,
        "audited_episodes": sum(r["integrity_passed"] for r in cases),
        "not_run_tasks": stages["not_run"], "results_provisional": not complete,
        "integrity_passed": complete, "available_results_integrity_passed": bool(any(t["available"] for t in tasks)) and all(t["integrity_passed"] for t in tasks if t["available"]),
        "by_condition": grouped, "model_update_correct_count": sum(t["model_update_correct"] for t in tasks),
        "model_update_correct_rate": sum(t["model_update_correct"] for t in tasks) / count,
        "model_pipeline_success_count": sum(t["model_pipeline_success"] for t in tasks),
        "model_pipeline_success_rate": sum(t["model_pipeline_success"] for t in tasks) / count,
        "model_failure_stages": {key if key is not None else "none": value for key, value in stages.items()},
        "callback_latency": latencies, "passed": complete and all(t["passed"] for t in tasks),
        "controls": manifest["controls"], "gpu_ids": sorted({t["gpu_id"] for t in tasks if t["gpu_id"] is not None}),
        "audit_code_sha256": source_hashes(), "tasks": tasks,
        "limitations": ["Fixed reviewed tasks in one scene with one replacement and a closed candidate catalog",
            "Text model predicts update; navigation uses privileged coordinates/navmesh/instance centers",
            "Rates use all planned slots; provisional snapshots are not final accuracy",
            "GPU IDs can differ across tasks; callback wall time is not isolated model inference time"]}
    base.write_json(output / "summary.json", summary)
    base.write_jsonl(output / "cases.jsonl", cases)
    base.write_jsonl(output / "tasks.jsonl", tasks)
    write_csv(output / "tasks.csv", tasks)
    write_csv(output / "cases.csv", cases)
    write_csv(output / "model_calls.csv", calls)
    print(f"Audit {audited}/{count} tasks, {summary['audited_episodes']}/{3 * count} episodes; "
          f"model updates {summary['model_update_correct_count']}/{count}; provisional={not complete}")
    return summary


def videos(args):
    output = args.output_dir.resolve()
    manifest = validate_batch(output)
    if args.task_id and args.task_id not in [t["task_id"] for t in manifest["tasks"]]:
        raise ValueError("Video task is not in batch roster")
    failures = []
    for task in manifest["tasks"]:
        if args.task_id and task["task_id"] != args.task_id:
            continue
        for condition in CONDITIONS:
            destination = hssd_dataset.within(output, f"videos/{task['task_id']}/{condition}.mp4")
            try:
                print(export_video(result_dir(output, task) / condition, args.fps, output=destination))
            except (OSError, ValueError, RuntimeError) as error:
                failures.append({"task_id": task["task_id"], "condition": condition, "error": str(error)})
    base.write_json(output / "video_status.json", {"failures": failures})
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="CPU: freeze all source tasks, controls and verified reuse")
    preparing.add_argument("--source-dir", type=Path, required=True)
    preparing.add_argument("--timing", choices=("early", "middle", "late"), default="middle")
    preparing.add_argument("--model", required=True)
    preparing.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    preparing.add_argument("--temperature", type=float, default=0.0)
    preparing.add_argument("--max-tokens", type=int, default=512)
    preparing.add_argument("--timeout", type=float, default=120.0)
    preparing.add_argument("--thinking-mode", choices=("off", "on", "default"), default="off")
    preparing.add_argument("--reuse-run", type=Path, action="append", default=[])
    running = commands.add_parser("run", help="GPU/model: sequential new tasks, preserve all failures")
    running.add_argument("--gpu-id", type=int, default=0)
    running.add_argument("--api-key-env", default="TASK_UPDATE_API_KEY")
    commands.add_parser("summarize", help="CPU Habitat: independently audit every planned slot")
    video = commands.add_parser("videos", help="FFmpeg: write new batch videos, including reused evidence")
    video.add_argument("--task-id")
    video.add_argument("--fps", type=float, default=10.0)
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        return 0 if summarize(args.output_dir)["passed"] else 1
    return {"prepare": prepare, "run": run, "videos": videos}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
