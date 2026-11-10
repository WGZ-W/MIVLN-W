"""Run five predeclared paraphrase/reversed-order navigation representatives.

Selection is the smallest task ID for each reviewed destination, independent of
offline prediction success. The existing runner, public callback and per-episode
physics auditor are unchanged. This versioned loader preserves candidate ORDER
and paraphrased language, which the original preparation intentionally sorted.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

from mivln import TaskState
from mivln.navigation_updates import navigation_request
from mivln.task_update_prompts import fingerprint, system_prompt
from scripts import hssd_dataset, language_navigation as single
from scripts import language_navigation_candidates as five, language_navigation_dataset as batch
from scripts import language_navigation_robustness as offline, task_update as base
from scripts.hssd_targets import sha256


PROTOCOL = "hssd_paraphrase_reversed_navigation_v1"
WORDING, ORDER = "paraphrase_2", "reversed"
CONDITIONS = single.CONDITIONS


def code_hashes():
    return {**offline.code_hashes(), "scripts/language_navigation_representatives.py": sha256(Path(__file__))}


def load_offline(prepared, run_dir):
    """Independently bind raw offline calls without rescoring into old folders."""
    manifest, records = offline.load_prepared(prepared)
    public, index, requests, instructions = offline.load_public(prepared / "public")
    seal = base.read_json(run_dir / "sealed_run.json")
    if (seal.get("kind") != offline.PROTOCOL or seal["code_sha256"] != offline.code_hashes()
            or seal["public_manifest_sha256"] != sha256(prepared / "public/manifest.json")
            or seal["files_sha256"] != {name: sha256(run_dir / name) for name in offline.RUN_FILES}):
        raise ValueError("Offline raw run changed or is incomplete")
    audited = offline.audit.audit_run(run_dir, "baseline", public, index, requests, instructions)
    if audited["legacy_reconstruction"] or any(audited["run"][k] != v for k, v in manifest["controls"].items()):
        raise ValueError("Offline model controls differ from the frozen plan")
    responses = {r["request_id"]: r for r in single.lines(run_dir / "responses.jsonl")}
    rows = {r["request"]["request_id"]: offline.evaluate(r, r["request_sha256"],
        audited["raw"][r["request"]["request_id"]], responses[r["request"]["request_id"]]["error"]) for r in records}
    return manifest, records, rows


def select_records(records):
    """One task per candidate destination, chosen solely by stable task ID.

    No correctness, response, cost or success field is consulted. This rule is
    fixed before launching any new navigation episode and retains failed cases.
    """
    chosen = {}
    for record in records:
        if record["wording"] != WORDING or record["order"] != ORDER:
            continue
        target = record["expected_prediction"]["task_update"]["new_goal"]
        if target not in chosen or record["task_id"] < chosen[target]["task_id"]:
            chosen[target] = record
    candidates = {c["goal_id"] for c in records[0]["request"]["candidate_goals"]}
    if len(candidates) != 5 or set(chosen) != candidates:
        raise ValueError("Source must cover all five reviewed destinations")
    return [chosen[target] for target in sorted(chosen)]


def semantic_request(request):
    """Offline and navigation opaque IDs differ; all semantic fields must match."""
    return {key: value for key, value in request.items() if key != "request_id"}


def derive_config(original, record):
    """Change episode identity, utterance and candidate order, preserving physics."""
    config = deepcopy(original)
    config["episode_id"] += "_paraphrase2_reversed"
    config["events"][0]["utterance"] = record["request"]["message"]
    config["language_candidates"] = deepcopy(record["request"]["candidate_goals"])
    if config["language_candidates"] != list(reversed(single.reviewed_candidates(config))):
        raise ValueError("Use exactly the original five reviewed candidates in reversed order")
    event = config["events"][0]
    request = navigation_request(event["trigger"]["value"], event["utterance"],
                                 config["language_candidates"], TaskState(config["initial_goal"]))
    if (semantic_request(request) != semantic_request(record["request"])
            or event["task_update"] != record["expected_prediction"]["task_update"]):
        raise ValueError("Offline language/state does not describe the same navigation task")
    return config


def task_plan(prepared_manifest, records):
    source = Path(prepared_manifest["source_dir"])
    _, inner = five.validate(source)
    source_tasks = {t["task_id"]: t for t in inner["tasks"]}
    tasks, configs = [], {}
    for record in select_records(records):
        identity = record["task_id"]
        source_task = source_tasks[identity]
        previous = source / "batch" / source_task["preparation_dir"]
        info, original = single.load_preparation(previous)
        configs[identity] = derive_config(original, record)
        tasks.append({"task_id": identity, "target_id": record["expected_prediction"]["task_update"]["new_goal"],
            "config_path": f"configs/{identity}.json", "output_subdir": f"tasks/{identity}",
            "baseline_prepared_dir": str(previous), "baseline_run_dir": str(batch.result_dir(source / "batch", source_task)),
            "reference_dir": info["reference_dir"], "offline_request_id": record["request"]["request_id"],
            "semantic_request_sha256": fingerprint(semantic_request(record["request"]))})
    return tasks, configs


def geometry(config, reference_dir):
    """CPU-only stop/instance checks, using the unchanged pre-action reference pose."""
    from scripts.run_navigation import scene_provenance, validate_config
    validate_config(config)
    scene_provenance(config, single.project_path(config["scene"]))
    single.subgoal.reviewed_centers(config, config["goal_positions"])
    pathfinder = single.subgoal.pathfinder_for(config)
    trigger = config["events"][0]["trigger"]["value"]
    reference = base.read_json(reference_dir / "trajectory.json")
    distances = {}
    for identity, point in config["goal_positions"].items():
        if not pathfinder.is_navigable(point):
            raise ValueError("Reviewed stop is not navigable: " + identity)
        distances[identity] = {key: single.subgoal.distance_between(pathfinder, pose, point)
            for key, pose in (("start_m", config["start_position"]), ("event_m", reference[trigger]["position"]))}
        if any(value is None for value in distances[identity].values()):
            raise ValueError("Candidate unreachable; do not drop or replace the planned task")
    return {"trigger_step": trigger, "candidate_count": len(distances), "distances": distances}


def prepare(args):
    prepared, run_dir, output = args.prepared_dir.resolve(), args.offline_run.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new representative-navigation directory")
    source, records, _ = load_offline(prepared, run_dir)
    batch.disjoint_output(output, [prepared, run_dir, Path(source["source_dir"])])
    tasks, configs = task_plan(source, records)
    checks = []
    for task in tasks:
        if not single.check_experiment(Path(task["baseline_run_dir"]))["integrity_passed"]:
            raise ValueError("Baseline navigation evidence is incomplete")
        checks.append({"task_id": task["task_id"], **geometry(configs[task["task_id"]], Path(task["reference_dir"]))})
    base.fresh_output(output)
    (output / "configs").mkdir()
    previews = []
    for task in tasks:
        config = configs[task["task_id"]]
        base.write_json(output / task["config_path"], config)
        task["config_sha256"] = sha256(output / task["config_path"])
        event = config["events"][0]
        previews.append(navigation_request(event["trigger"]["value"], event["utterance"],
            config["language_candidates"], TaskState(config["initial_goal"])))
    base.write_json(output / "manifest.json", {"schema_version": 1, "kind": PROTOCOL,
        "prepared_dir": str(prepared), "offline_run": str(run_dir), "wording": WORDING, "order": ORDER,
        "selection": "Smallest task_id per reviewed destination; independent of model outcomes",
        "robustness_manifest_sha256": sha256(prepared / "manifest.json"),
        "offline_files_sha256": {name: sha256(run_dir / name) for name in (*offline.RUN_FILES, "sealed_run.json")},
        "controls": source["controls"], "code_sha256": code_hashes(), "conditions": list(CONDITIONS),
        "task_count": 5, "planned_episodes": 15, "planned_model_calls": 5, "tasks": tasks,
        "created_at": batch.now(), "scope": "Five new model calls; text update plus privileged navigator; offline answers never used as callbacks"})
    base.write_jsonl(output / "request_previews.jsonl", previews)
    base.write_json(output / "cpu_check.json", {"passed": True, "no_renderer_or_model_calls": True,
        "task_count": 5, "planned_episodes": 15, "tasks": checks})
    print("Prepared five destination representatives, paraphrase_2/reversed, 15 NEW episodes and 5 NEW model calls. No GPU/API calls.")
    return 0


def load_experiment(output):
    manifest = base.read_json(output / "manifest.json")
    if (manifest.get("kind") != PROTOCOL or manifest.get("schema_version") != 1
            or manifest["code_sha256"] != code_hashes() or manifest["conditions"] != list(CONDITIONS)
            or manifest["wording"] != WORDING or manifest["order"] != ORDER
            or (manifest["task_count"], manifest["planned_episodes"], manifest["planned_model_calls"]) != (5, 15, 5)):
        raise ValueError("Representative protocol, code, factors or denominators changed")
    prepared, run_dir = Path(manifest["prepared_dir"]), Path(manifest["offline_run"])
    if (sha256(prepared / "manifest.json") != manifest["robustness_manifest_sha256"]
            or manifest["offline_files_sha256"] != {name: sha256(run_dir / name) for name in (*offline.RUN_FILES, "sealed_run.json")}):
        raise ValueError("Offline source evidence changed")
    source, records, rows = load_offline(prepared, run_dir)
    batch.disjoint_output(output, [prepared, run_dir, Path(source["source_dir"])])
    expected, configs = task_plan(source, records)
    for task in expected:
        path = hssd_dataset.within(output, task["config_path"])
        if base.read_json(path) != configs[task["task_id"]]:
            raise ValueError("Representative changed more than identity, wording and candidate order")
        task["config_sha256"] = sha256(path)
    if manifest["tasks"] != expected or manifest["controls"] != source["controls"]:
        raise ValueError("Representative selection, source binding or controls changed")
    return manifest, configs, rows


def check(output):
    manifest, configs, _ = load_experiment(output)
    checks = []
    for task in manifest["tasks"]:
        if not single.check_experiment(Path(task["baseline_run_dir"]))["integrity_passed"]:
            raise ValueError("Baseline evidence failed integrity audit")
        checks.append({"task_id": task["task_id"], **geometry(configs[task["task_id"]], Path(task["reference_dir"]))})
    base.write_json(output / "cpu_check.json", {"passed": True, "no_renderer_or_model_calls": True,
        "task_count": 5, "planned_episodes": 15, "tasks": checks})
    print("CPU preflight passed: five selected tasks, reversed catalogs, unchanged poses/triggers/scoring.")
    return manifest, configs


def run_task(args):
    """Child process: real HTTP callback once at delivery, never offline replay."""
    output = args.output_dir.resolve()
    manifest, configs, _ = load_experiment(output)
    task = next((t for t in manifest["tasks"] if t["task_id"] == args.task_id), None)
    if task is None or args.gpu_id < 0:
        raise ValueError("Unknown representative or invalid GPU")
    config = configs[task["task_id"]]
    geometry(config, Path(task["reference_dir"]))
    from scripts.run_navigation import run as navigate
    directory = base.fresh_output(hssd_dataset.within(output, task["output_subdir"]))
    system = system_prompt(base.INSTRUCTIONS)
    (directory / "model_system_prompt.txt").write_text(system, encoding="utf-8")
    for name in single.JOURNAL_FILES[:2]:
        (directory / name).touch(exist_ok=False)
    updater = single.ModelClient(manifest["controls"], system, directory, args.api_key_env)
    statuses = [{"condition": name, "completed": False, "error": "Not run"} for name in CONDITIONS]
    record = {"kind": PROTOCOL, "task_id": task["task_id"], "experiment_dir": str(output),
        "manifest_sha256": sha256(output / "manifest.json"), "controls": manifest["controls"],
        "gpu_id": args.gpu_id, "code_sha256": code_hashes(),
        "system_prompt_sha256": sha256(directory / "model_system_prompt.txt"), "started_at": batch.now()}
    base.write_json(directory / "run.json", record)
    base.write_json(directory / "run_status.json", statuses)
    for status in statuses:
        branch = deepcopy(config)
        branch["episode_id"] += "_" + status["condition"]
        interrupted = False
        try:
            print("Running " + status["condition"], flush=True)
            navigate(branch, single.project_path(config["scene"]), args.gpu_id, directory / status["condition"],
                intervention_policy={"oracle": "follow", "model": "model", "ignore": "ignore"}[status["condition"]],
                task_updater=updater if status["condition"] == "model" else None)
            status.update(completed=True, error=None)
        except KeyboardInterrupt:
            status["error"], interrupted = "Interrupted", True
        except Exception as failure:
            status["error"] = f"{type(failure).__name__}: {failure}"
        base.write_json(directory / "run_status.json", statuses)
        if interrupted:
            break
    record.update(finished_at=batch.now(), files_sha256={name: sha256(directory / name) for name in single.JOURNAL_FILES})
    base.write_json(directory / "run.json", record)
    report = audit_task(output, manifest, task, config)
    base.write_json(directory / "summary.json", report)
    return 0 if report["passed"] else 1


def audit_task(output, manifest, task, config):
    """Versioned journal loader plus the unchanged raw-response/physics auditors.

    The old single-task loader sorts candidates by design. Only that envelope is
    replaced here; native displacement, STOP, event timing, truth/policy replay,
    measured heading and instance visibility still use its original functions.
    """
    directory = hssd_dataset.within(output, task["output_subdir"])
    run = base.read_json(directory / "run.json")
    if (run.get("kind") != PROTOCOL or run["task_id"] != task["task_id"]
            or Path(run["experiment_dir"]).resolve() != output.resolve()
            or run["manifest_sha256"] != sha256(output / "manifest.json")
            or run["controls"] != manifest["controls"] or run["code_sha256"] != code_hashes()
            or type(run["gpu_id"]) is not int or run["gpu_id"] < 0):
        raise ValueError("Task run binding, model controls, code or GPU changed")
    for name in single.JOURNAL_FILES:
        if sha256(directory / name) != run["files_sha256"][name]:
            raise ValueError("Actual run journal changed: " + name)
    if ((directory / "model_system_prompt.txt").read_text(encoding="utf-8") != system_prompt(base.INSTRUCTIONS)
            or sha256(directory / "model_system_prompt.txt") != run["system_prompt_sha256"]):
        raise ValueError("Actual system prompt changed")
    statuses = base.read_json(directory / "run_status.json")
    if [row["condition"] for row in statuses] != list(CONDITIONS):
        raise ValueError("Condition roster changed")
    pathfinder = single.subgoal.pathfinder_for(config)
    centers = single.subgoal.reviewed_centers(config, config["goal_positions"])
    model_state, record, correct, model_error = None, None, False, None
    try:
        model_state, record, correct = single.replay_model(config, directory, run)
    except batch.EVIDENCE_ERRORS as failure:
        model_error = str(failure)
    conditions = []
    for status in statuses:
        condition = status["condition"]
        try:
            if not status["completed"]:
                raise ValueError(status["error"] or "Condition did not complete")
            if condition == "model" and model_error:
                raise ValueError(model_error)
            branch = directory / condition
            if base.read_json(branch / "episode.json")["gpu_device_id"] != run["gpu_id"]:
                raise ValueError("Branch GPU differs from task metadata")
            conditions.append(single.audit_episode(config, branch, condition, pathfinder, centers, model_state, record))
        except batch.EVIDENCE_ERRORS as failure:
            conditions.append({"condition": condition, "integrity_passed": False, "dynamic_success": False,
                "joint_success": False, "ignore_expected_failure": False, "error": str(failure)})
    prefix, prefix_error = False, None
    try:
        trigger = config["events"][0]["trigger"]["value"]
        first = base.read_json(directory / "oracle/trajectory.json")[:trigger + 1]
        keys = ("step", "action", "position", "rotation_xyzw", "collided", "distance_m")
        project = lambda rows: [{key: row[key] for key in keys} for row in rows]
        for condition in ("model", "ignore"):
            other = base.read_json(directory / condition / "trajectory.json")[:trigger + 1]
            if project(first) != project(other) or not hssd_dataset.matching_frames(directory / "oracle", first, directory / condition, other):
                raise ValueError("Pre-event physical trajectory or sensor prefix differs")
        prefix = True
    except batch.EVIDENCE_ERRORS as failure:
        prefix_error = str(failure)
    integrity = prefix and all(c["integrity_passed"] for c in conditions)
    pipeline = bool(integrity and correct and conditions[1]["joint_success"])
    stage = ("evidence" if not integrity else "model_call" if record["outcome"] == "call_failed"
             else "update" if not correct else "navigation" if not conditions[1]["joint_success"] else None)
    return {"task_id": task["task_id"], "gpu_id": run["gpu_id"], "planned_episodes": 3,
        "conditions": conditions, "integrity_passed": integrity, "prefix_matched": prefix,
        "prefix_error": prefix_error, "model_update_correct": correct,
        "model_pipeline_success": pipeline, "model_failure_stage": stage,
        "model_callback_seconds": record["callback_seconds"] if record else None,
        "prediction": record["prediction"] if record else None,
        "model_state": model_state.to_dict() if model_state else None,
        "passed": bool(pipeline and conditions[0]["joint_success"] and conditions[2]["ignore_expected_failure"])}


def run(args):
    output = args.output_dir.resolve()
    if type(args.gpu_id) is not int or args.gpu_id < 0:
        raise ValueError("GPU ID must be non-negative")
    for name in ("batch_run.json", "run_status.json", "run_logs", "tasks"):
        if (output / name).exists():
            raise FileExistsError("Batch already started or outputs occupied: " + name)
    manifest, _ = check(output)
    statuses = [{"task_id": t["task_id"], "status": "pending", "returncode": None, "error": None}
                for t in manifest["tasks"]]
    record = {"kind": PROTOCOL, "manifest_sha256": sha256(output / "manifest.json"),
        "code_sha256": code_hashes(), "gpu_id": args.gpu_id, "started_at": batch.now()}
    base.write_json(output / "batch_run.json", record)
    base.write_json(output / "run_status.json", statuses)
    for task, status in zip(manifest["tasks"], statuses):
        status.update(status="running", started_at=batch.now())
        base.write_json(output / "run_status.json", statuses)
        interrupted = False
        command = [sys.executable, "-u", "-m", "scripts.language_navigation_representatives", "run-task",
            "--output-dir", str(output), "--task-id", task["task_id"], "--gpu-id", str(args.gpu_id),
            "--api-key-env", args.api_key_env]
        try:
            print("Running representative " + task["task_id"], flush=True)
            code = batch.execute_stage(command, output / "run_logs" / (task["task_id"] + ".log"))
            status.update(status="finished", returncode=code)
        except KeyboardInterrupt:
            status.update(status="interrupted", error="Interrupted; remaining slots stay pending")
            interrupted = True
        except OSError as failure:
            status.update(status="launch_failed", error=f"{type(failure).__name__}: {failure}")
        status["finished_at"] = batch.now()
        base.write_json(output / "run_status.json", statuses)
        if interrupted:
            break
        # Non-zero child codes can describe honest model failures. Keep their
        # artifacts and continue all other predeclared destinations.
    record.update(finished_at=batch.now(), run_status_sha256=sha256(output / "run_status.json"))
    base.write_json(output / "batch_run.json", record)
    return 0 if summarize(output)["passed"] else 1


def summarize(output):
    """Keep all 15 slots and compare fresh online updates against offline ones."""
    manifest, configs, offline_rows = load_experiment(output)
    statuses, batch_run = None, None
    if (output / "batch_run.json").exists():
        batch_run = base.read_json(output / "batch_run.json")
        statuses = base.read_json(output / "run_status.json")
        if (batch_run["manifest_sha256"] != sha256(output / "manifest.json")
                or batch_run["code_sha256"] != code_hashes()
                or [s["task_id"] for s in statuses] != [t["task_id"] for t in manifest["tasks"]]
                or ("run_status_sha256" in batch_run and batch_run["run_status_sha256"] != sha256(output / "run_status.json"))):
            raise ValueError("Batch execution metadata changed")
    tasks, cases = [], []
    for index, task in enumerate(manifest["tasks"]):
        directory = hssd_dataset.within(output, task["output_subdir"])
        state = statuses[index]["status"] if statuses else "pending"
        report, error = None, None
        try:
            report = audit_task(output, manifest, task, configs[task["task_id"]])
            if batch_run and report["gpu_id"] != batch_run["gpu_id"]:
                raise ValueError("Task GPU differs from batch metadata")
        except batch.EVIDENCE_ERRORS as failure:
            report, error = None, str(failure)
        comparisons = {c + suffix: None for c in ("oracle", "ignore")
                       for suffix in ("_behavior_matched", "_observations_matched")}
        if report:
            baseline = single.check_experiment(Path(task["baseline_run_dir"]))
            if not baseline["integrity_passed"]:
                raise ValueError("Original five-candidate baseline failed audit")
            for condition in ("oracle", "ignore"):
                branch = next(c for c in report["conditions"] if c["condition"] == condition)
                if branch["integrity_passed"]:
                    try:
                        physical, observations = five.physical_match(Path(task["baseline_run_dir"]), directory, condition)
                        comparisons[condition + "_behavior_matched"] = physical
                        comparisons[condition + "_observations_matched"] = observations
                    except batch.EVIDENCE_ERRORS as failure:
                        comparisons[condition + "_behavior_matched"] = False
                        error = str(failure)
        previous = offline_rows[task["offline_request_id"]]
        model_valid = bool(report and report["conditions"][1]["integrity_passed"])
        same = (report["prediction"] == previous["prediction"] and report["model_state"] == previous["after_state"]
                if model_valid and previous["schema_valid"] and report["prediction"] is not None else None)
        row = {"task_id": task["task_id"], "target_id": task["target_id"], "execution_status": state,
            "returncode": statuses[index]["returncode"] if statuses else None,
            "integrity_passed": bool(report and report["integrity_passed"]),
            "model_update_correct": bool(report and report["model_update_correct"]),
            "model_pipeline_success": bool(report and report["model_pipeline_success"]),
            "model_failure_stage": report["model_failure_stage"] if report else ("not_run" if state == "pending" and not directory.exists() else "evidence"),
            "model_callback_seconds": report["model_callback_seconds"] if report else None,
            "offline_correct": previous["correct"], "offline_online_same_update_and_state": same,
            "passed": bool(report and report["passed"]), **comparisons, "error": error or (report["prefix_error"] if report else None)}
        tasks.append(row)
        for condition in CONDITIONS:
            observed = next((c for c in report["conditions"] if c["condition"] == condition), {}) if report else {}
            valid = bool(observed.get("integrity_passed"))
            cases.append({"task_id": task["task_id"], "condition": condition, "integrity_passed": valid,
                "dynamic_success": valid and bool(observed.get("dynamic_success")),
                "joint_success": valid and bool(observed.get("joint_success")),
                "ignore_expected_failure": valid and bool(observed.get("ignore_expected_failure")),
                "steps": observed.get("steps"), "path_length_m": observed.get("path_length_m"),
                "navigation_error_m": observed.get("navigation_error_m"), "error": error or observed.get("error")})
    count = manifest["task_count"]
    by_condition = {}
    for condition in CONDITIONS:
        rows = [c for c in cases if c["condition"] == condition]
        by_condition[condition] = {"planned": count, "audited": sum(r["integrity_passed"] for r in rows)}
        for metric in ("joint_success", "dynamic_success", "ignore_expected_failure"):
            total = sum(r[metric] for r in rows)
            by_condition[condition].update({metric + "_count": total, metric + "_rate": total / count})
    audited = sum(r["integrity_passed"] for r in tasks)
    matched = sum(r["oracle_behavior_matched"] is True and r["ignore_behavior_matched"] is True for r in tasks)
    report = {"kind": PROTOCOL, "task_count": count, "planned_episodes": 15, "planned_model_calls": 5,
        "wording": WORDING, "order": ORDER, "audited_tasks": audited,
        "audited_episodes": sum(r["integrity_passed"] for r in cases), "by_condition": by_condition,
        "model_update_correct_count": sum(r["model_update_correct"] for r in tasks),
        "model_update_correct_rate": sum(r["model_update_correct"] for r in tasks) / count,
        "model_pipeline_success_count": sum(r["model_pipeline_success"] for r in tasks),
        "model_pipeline_success_rate": sum(r["model_pipeline_success"] for r in tasks) / count,
        "offline_online_same_count": sum(r["offline_online_same_update_and_state"] is True for r in tasks),
        "oracle_ignore_behavior_matched_tasks": matched, "results_provisional": audited != count,
        "integrity_passed": audited == count, "comparison_complete": audited == count and matched == count,
        "passed": audited == count and matched == count and all(r["passed"] for r in tasks),
        "model_failure_stages": {stage: sum(r["model_failure_stage"] == (None if stage == "none" else stage) for r in tasks)
            for stage in ("not_run", "evidence", "model_call", "update", "navigation", "none")},
        "callback_latency": batch.latency_summary([r["model_callback_seconds"] for r in tasks if r["model_callback_seconds"] is not None]),
        "tasks": tasks, "controls": manifest["controls"], "code_sha256": code_hashes(),
        "limitations": ["Five predeclared destination representatives in one scene; one paraphrase and one reversed order",
            "Online opaque IDs differ from offline IDs; all semantic request fields are checked equal",
            "New real calls, not playback of offline predictions; privileged navigation remains",
            "Cross-batch pixel equality is reported separately from physical controls"]}
    base.write_json(output / "summary.json", report)
    base.write_jsonl(output / "cases.jsonl", cases)
    base.write_jsonl(output / "tasks.jsonl", tasks)
    batch.write_csv(output / "cases.csv", cases)
    batch.write_csv(output / "comparison.csv", tasks)
    print(f"Representatives audited {audited}/5 tasks, {report['audited_episodes']}/15 episodes; "
          f"updates {report['model_update_correct_count']}/5; pipeline {report['model_pipeline_success_count']}/5.")
    return report


def videos(args):
    output = args.output_dir.resolve()
    manifest, _, _ = load_experiment(output)
    if args.task_id and args.task_id not in [t["task_id"] for t in manifest["tasks"]]:
        raise ValueError("Unknown representative video task")
    failures = []
    for task in manifest["tasks"]:
        if args.task_id and task["task_id"] != args.task_id:
            continue
        for condition in CONDITIONS:
            destination = hssd_dataset.within(output, f"videos/{task['task_id']}/{condition}.mp4")
            try:
                print(batch.export_video(output / task["output_subdir"] / condition, args.fps, output=destination))
            except (OSError, ValueError, RuntimeError) as failure:
                failures.append({"task_id": task["task_id"], "condition": condition, "error": str(failure)})
    base.write_json(output / "video_status.json", {"failures": failures})
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="CPU Habitat: freeze destination-based representatives")
    preparing.add_argument("--prepared-dir", type=Path, required=True)
    preparing.add_argument("--offline-run", type=Path, required=True)
    commands.add_parser("check", help="CPU: recheck source, candidates and start/event reachability")
    for name in ("run", "run-task"):
        running = commands.add_parser(name, help="GPU: batch" if name == "run" else "Internal single-task child")
        running.add_argument("--gpu-id", type=int, default=0)
        running.add_argument("--api-key-env", default="TASK_UPDATE_API_KEY")
        if name == "run-task":
            running.add_argument("--task-id", required=True)
    commands.add_parser("summarize", help="CPU: audit new navigation and compare offline updates/baselines")
    video = commands.add_parser("videos", help="FFmpeg: export fifteen new trajectories")
    video.add_argument("--fps", type=float, default=10.0)
    video.add_argument("--task-id")
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        check(args.output_dir.resolve())
        return 0
    if args.command == "summarize":
        return 0 if summarize(args.output_dir.resolve())["passed"] else 1
    return {"prepare": prepare, "run": run, "run-task": run_task, "videos": videos}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
