"""Expand a frozen two-candidate batch to five reviewed navigation candidates.

This is a preparation/comparison layer over the unchanged language_navigation
and language_navigation_dataset engines. Old evidence stays independently
auditable under its original code hashes. Only executor goal maps and the
derived public candidate list expand; messages, poses, timing and scoring stay
fixed. All new episodes and model calls must actually run again.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from mivln import TaskState
from mivln.navigation_updates import navigation_request
from scripts import hssd_dataset, language_navigation as single
from scripts import language_navigation_dataset as batch, task_update as base
from scripts.hssd_targets import sha256, validate_catalog


PROTOCOL = "hssd_language_five_candidates_v1"
CANDIDATE_COUNT = 5
REFERENCE_FILES = ("episode.json", "metrics.json", "trajectory.json")


def code_hashes():
    return {**batch.source_hashes(), "scripts/language_navigation_candidates.py": sha256(Path(__file__))}


def expand_config(config, targets):
    """Add executable distractors without changing any existing task field.

    Public IDs must also exist in the executor, otherwise a valid wrong model
    prediction could not be tested honestly. Existing points/metadata must match
    their reviewed instance; new points are metres in the same scene frame.
    """
    expanded = deepcopy(config)
    by_id = {target["target_id"]: target for target in targets}
    if not set(config["goal_positions"]) <= set(by_id):
        raise ValueError("Original goal is outside the reviewed candidate catalog")
    single.reviewed_candidates(config)
    for identity in sorted(by_id):
        target = by_id[identity]
        if identity not in expanded["goal_positions"]:
            expanded["goal_positions"][identity] = deepcopy(target["stop_position"])
            expanded["goal_metadata"][identity] = {
                "template_id": target["template_id"], "source_instance_index": target["source_instance_index"],
                "category": target["category"], "description": target["review"]["description"]}
    # This check validates the complete executor/catalog correspondence, including
    # descriptions. Geometry never enters the public model request.
    single.reviewed_candidates(expanded)
    return expanded


def baseline_info(directory):
    """Validate all baseline slots without writing into the old batch or runs."""
    manifest = batch.validate_batch(directory)
    reports = {}
    for task in manifest["tasks"]:
        prepared = directory / task["preparation_dir"]
        _, config = single.load_preparation(prepared)
        if len(config["language_candidates"]) != 2:
            raise ValueError("Baseline must contain exactly two candidates per task")
        folder = batch.result_dir(directory, task)
        batch.bind_run(folder, prepared, manifest["controls"])
        report = single.check_experiment(folder)
        # Accept honestly failed model tasks too: baseline inclusion must not
        # depend on semantic correctness or success of the proposed method.
        if not report["integrity_passed"]:
            raise ValueError("Baseline evidence is incomplete: " + task["task_id"])
        reports[task["task_id"]] = report
    return manifest, reports


def collection_plan(baseline_dir, manifest):
    """Plan an expanded source collection and unchanged reference evidence.

    Reference files are byte-for-byte copies of ORIGINAL two-candidate runs,
    used only to select the same intervention step. They are not presented as
    newly executed five-candidate reference episodes, and no old observations
    or predictions are reused as new-condition outcomes.
    """
    source = Path(manifest["source_dir"])
    source_manifest = batch.roster(source)
    catalog_source = source_manifest["catalog_source"]
    targets = sorted(validate_catalog(base.read_json(Path(catalog_source["path"]))), key=lambda t: t["target_id"])
    if len(targets) != CANDIDATE_COUNT:
        raise ValueError("Use a catalog with exactly five explicitly reviewed targets")
    configs, copies, records = {}, {}, []
    for task, baseline_task in zip(source_manifest["tasks"], manifest["tasks"]):
        identity = task["task_id"]
        old_prepared = baseline_dir / baseline_task["preparation_dir"]
        info, _ = single.load_preparation(old_prepared)
        # The catalog is shared by all source tasks. Never introduce targets
        # reviewed against another scene, mesh or candidate evidence bundle.
        for name, original in (("base", hssd_dataset.within(source, task["base_config"])),
                               ("reference", hssd_dataset.within(source, task["reference_config"])),
                               ("timing", Path(info["source_config"]))):
            config = base.read_json(original)
            if config["catalog_source"] != catalog_source:
                raise ValueError("Task catalog differs from the source collection")
            relative = (f"tasks/{identity}/timing/configs/{manifest['timing']}.json" if name == "timing"
                        else f"configs/{identity}/{name}.json")
            configs[relative] = expand_config(config, targets)
        for name in REFERENCE_FILES:
            copies[f"tasks/{identity}/reference/{name}"] = Path(info["reference_dir"]) / name
        records.append({"task_id": identity, "output_subdir": f"tasks/{identity}",
                        "base_config": f"configs/{identity}/base.json",
                        "reference_config": f"configs/{identity}/reference.json"})
    header = {"schema_version": 1, "kind": "hssd_semantic_dataset", "task_count": len(records),
        "catalog_source": catalog_source, "tasks": records,
        "derivation": {"kind": PROTOCOL, "parent_source": str(source),
            "parent_manifest_sha256": sha256(source / "manifest.json"), "candidate_count": CANDIDATE_COUNT,
            "candidate_ids": [t["target_id"] for t in targets], "timing": manifest["timing"],
            "reference_usage": "Unmodified original two-candidate evidence; trigger selection only"}}
    return configs, copies, header


def write_collection(folder, configs, copies, header):
    folder.mkdir()
    for relative, config in configs.items():
        path = hssd_dataset.within(folder, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        base.write_json(path, config)
    for relative, original in copies.items():
        path = hssd_dataset.within(folder, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, path)
    header = deepcopy(header)
    for task in header["tasks"]:
        for key in ("base_config", "reference_config"):
            task[key + "_sha256"] = sha256(folder / task[key])
    base.write_json(folder / "manifest.json", header)


def frozen_inputs(baseline_dir, manifest):
    """Freeze original sources and all consumed raw baseline evidence."""
    files = {baseline_dir / "manifest.json"}
    for name in ("batch_run.json", "run_status.json"):
        if (baseline_dir / name).exists():
            files.add(baseline_dir / name)
    hashes = {}
    for task in manifest["tasks"]:
        prepared = baseline_dir / task["preparation_dir"]
        info, _ = single.load_preparation(prepared)
        files.update((prepared / "manifest.json", prepared / "config.json"))
        files.update(Path(item["path"]) for item in info["sources"])
        folder = batch.result_dir(baseline_dir, task)
        hashes.update({str((folder / name).resolve()): digest for name, digest in batch.evidence_hashes(folder).items()})
    hashes.update({str(path.resolve()): sha256(path) for path in files})
    return dict(sorted(hashes.items()))


def prepare(args):
    baseline_dir, output = args.baseline_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new candidate-expansion directory")
    manifest, _ = baseline_info(baseline_dir)
    protected = [baseline_dir, Path(manifest["source_dir"]),
                 *[batch.result_dir(baseline_dir, task) for task in manifest["tasks"]]]
    batch.disjoint_output(output, protected)
    configs, copies, header = collection_plan(baseline_dir, manifest)
    inputs = frozen_inputs(baseline_dir, manifest)
    # Preflight the full derived source using the unchanged single-task preparer.
    # No final directory is created if a source, timing or review is invalid.
    with tempfile.TemporaryDirectory(prefix="five-candidates-") as temporary:
        staging = Path(temporary)
        write_collection(staging / "collection", configs, copies, header)
        for task in header["tasks"]:
            with redirect_stdout(io.StringIO()):
                single.prepare(SimpleNamespace(source_dir=staging / "collection", task_id=task["task_id"],
                    timing=manifest["timing"], output_dir=staging / task["task_id"]))
        output.mkdir(parents=True)
        shutil.copytree(staging / "collection", output / "collection")
    controls = manifest["controls"]
    batch.prepare(SimpleNamespace(source_dir=output / "collection", output_dir=output / "batch",
        timing=manifest["timing"], reuse_run=[], model=controls["model"], base_url=controls["base_url"],
        temperature=controls["temperature"], max_tokens=controls["max_tokens"],
        timeout=controls["timeout_seconds"], thinking_mode=controls["thinking_mode"]))
    inner = base.read_json(output / "batch/manifest.json")
    base.write_json(output / "manifest.json", {"schema_version": 1, "kind": PROTOCOL,
        "baseline_dir": str(baseline_dir), "baseline_manifest_sha256": sha256(baseline_dir / "manifest.json"),
        "candidate_count": CANDIDATE_COUNT, "candidate_ids": header["derivation"]["candidate_ids"],
        "controls": controls, "timing": manifest["timing"], "task_count": manifest["task_count"],
        "planned_episodes": manifest["planned_episodes"], "planned_model_calls": manifest["planned_model_calls"],
        "collection_manifest_sha256": sha256(output / "collection/manifest.json"),
        "batch_manifest_sha256": sha256(output / "batch/manifest.json"), "code_sha256": code_hashes(),
        "baseline_inputs_sha256": inputs, "created_at": batch.now(), "new_tasks": inner["new_tasks"],
        "scope": "Candidate-count intervention only; all expanded outcomes require new execution"})
    requests = []
    for task in inner["tasks"]:
        _, config = single.load_preparation(output / "batch" / task["preparation_dir"])
        event = config["events"][0]
        requests.append(navigation_request(event["trigger"]["value"], event["utterance"],
                                          config["language_candidates"], TaskState(config["initial_goal"])))
    base.write_jsonl(output / "request_previews.jsonl", requests)
    print(f"Five-candidate experiment ready: {manifest['task_count']} NEW tasks/"
          f"{manifest['planned_episodes']} episodes; {manifest['planned_model_calls']} NEW model calls. No GPU/API calls.")
    return 0


def validate(output):
    """Re-derive the expansion; self-consistent edited hashes cannot hide drift."""
    manifest = base.read_json(output / "manifest.json")
    if (manifest.get("kind") != PROTOCOL or manifest.get("schema_version") != 1
            or manifest["candidate_count"] != CANDIDATE_COUNT or manifest["code_sha256"] != code_hashes()):
        raise ValueError("Candidate expansion protocol or code changed")
    baseline_dir = Path(manifest["baseline_dir"])
    original = batch.validate_batch(baseline_dir)
    # Reconstruct the complete evidence roster as well as its digests. Deleting
    # entries from the outer manifest cannot hide a modified baseline file.
    if frozen_inputs(baseline_dir, original) != manifest["baseline_inputs_sha256"]:
        raise ValueError("Frozen baseline/source evidence changed")
    batch.disjoint_output(output, [baseline_dir, Path(original["source_dir"]),
        *[batch.result_dir(baseline_dir, task) for task in original["tasks"]]])
    if sha256(baseline_dir / "manifest.json") != manifest["baseline_manifest_sha256"]:
        raise ValueError("Baseline manifest changed")
    configs, copies, expected_header = collection_plan(baseline_dir, original)
    collection = output / "collection"
    header = base.read_json(collection / "manifest.json")
    for relative, expected in configs.items():
        if base.read_json(hssd_dataset.within(collection, relative)) != expected:
            raise ValueError("Expanded task changed something beyond candidate maps: " + relative)
    for relative, path in copies.items():
        if sha256(hssd_dataset.within(collection, relative)) != sha256(path):
            raise ValueError("Original trigger reference changed")
    for task in expected_header["tasks"]:
        for key in ("base_config", "reference_config"):
            task[key + "_sha256"] = sha256(collection / task[key])
    if header != expected_header or sha256(collection / "manifest.json") != manifest["collection_manifest_sha256"]:
        raise ValueError("Expanded collection roster/provenance changed")
    inner = batch.validate_batch(output / "batch")
    if (sha256(output / "batch/manifest.json") != manifest["batch_manifest_sha256"]
            or inner["source_dir"] != str(collection.resolve()) or inner["reused_tasks"] != 0
            or inner["new_tasks"] != original["task_count"]):
        raise ValueError("Expanded batch binding changed; old outcomes cannot be reused")
    for key in ("controls", "timing", "task_count", "planned_episodes", "planned_model_calls"):
        if manifest[key] != original[key] or inner[key] != original[key]:
            raise ValueError("Frozen experiment control changed: " + key)
    if manifest["candidate_ids"] != header["derivation"]["candidate_ids"] or manifest["new_tasks"] != inner["new_tasks"]:
        raise ValueError("Frozen candidate IDs or denominator changed")
    return manifest, inner


def check(output):
    """CPU-only reachability at start/event for every executable candidate.

    Unreachable distractors are errors, not silently dropped candidates. The
    event pose comes from original reference row k, before action k. This is
    sufficient for fixed Oracle prefixes; actual execution is audited separately.
    """
    manifest, inner = validate(output)
    from scripts.run_navigation import scene_provenance, validate_config
    rows = []
    for task in inner["tasks"]:
        info, config = single.load_preparation(output / "batch" / task["preparation_dir"])
        validate_config(config)
        scene_provenance(config, single.project_path(config["scene"]))
        single.subgoal.reviewed_centers(config, config["goal_positions"])
        pathfinder = single.subgoal.pathfinder_for(config)
        trigger = config["events"][0]["trigger"]["value"]
        reference = base.read_json(Path(info["reference_dir"]) / "trajectory.json")
        distances = {}
        for identity, point in config["goal_positions"].items():
            if not pathfinder.is_navigable(point):
                raise ValueError("Reviewed candidate stop is not navigable: " + identity)
            distances[identity] = {name: single.subgoal.distance_between(pathfinder, position, point)
                for name, position in (("start_m", config["start_position"]), ("event_m", reference[trigger]["position"]))}
            if any(value is None for value in distances[identity].values()):
                raise ValueError("Candidate is unreachable; retain the planned task: " + identity)
        rows.append({"task_id": task["task_id"], "candidate_count": len(distances), "trigger_step": trigger,
                     "distances": distances})
    report = {"kind": PROTOCOL, "passed": True, "no_renderer_or_model_calls": True,
        "task_count": manifest["task_count"], "candidate_count": CANDIDATE_COUNT, "tasks": rows}
    base.write_json(output / "cpu_check.json", report)
    print(f"CPU check passed: {len(rows)} tasks, five reachable candidates per task. No GPU/API calls.")
    return report


def physical_match(old, new, condition):
    left = base.read_json(old / condition / "trajectory.json")
    right = base.read_json(new / condition / "trajectory.json")
    keys = ("step", "action", "position", "rotation_xyzw", "collided", "distance_m")
    project = lambda rows: [{key: row[key] for key in keys} for row in rows]
    # The text model sees no rendered images. Across batches GPU IDs may differ,
    # so record pixel equality separately from physical execution equality. The
    # single-task auditor still enforces identical pre-event sensor evidence
    # WITHIN each new Oracle/model/Ignore experiment on its assigned GPU.
    return (project(left) == project(right),
            hssd_dataset.matching_frames(old / condition, left, new / condition, right))


def summarize(output):
    """Re-score both batches and check Oracle/Ignore physical controls.

    The model may legitimately select another executable distractor; its actual
    target is reported and executed, with no reference repair. All planned tasks
    remain in both the expanded summary and paired comparison, including missing
    runs, invalid evidence, and semantically wrong but valid predictions.
    """
    manifest, inner = validate(output)
    baseline_dir = Path(manifest["baseline_dir"])
    original, baseline = baseline_info(baseline_dir)
    expanded = batch.summarize(output / "batch")
    cases = {(row["task_id"], row["condition"]): row for row in single.lines(output / "batch/cases.jsonl")}
    paired = []
    for old_task, new_task, result in zip(original["tasks"], inner["tasks"], expanded["tasks"]):
        identity = new_task["task_id"]
        old, new = batch.result_dir(baseline_dir, old_task), batch.result_dir(output / "batch", new_task)
        before = baseline[identity]
        comparisons = {}
        error = None
        for condition in ("oracle", "ignore"):
            comparisons[condition + "_behavior_matched"] = None
            comparisons[condition + "_observations_matched"] = None
            if cases[identity, condition]["integrity_passed"]:
                try:
                    physical, observations = physical_match(old, new, condition)
                    comparisons[condition + "_behavior_matched"] = physical
                    comparisons[condition + "_observations_matched"] = observations
                except batch.EVIDENCE_ERRORS as failure:
                    comparisons[condition + "_behavior_matched"] = False
                    error = str(failure)
        predicted = None
        if cases[identity, "model"]["integrity_passed"]:
            _, config = single.load_preparation(output / "batch" / new_task["preparation_dir"])
            _, record, _ = single.replay_model(config, new, base.read_json(new / "run.json"))
            if record["prediction"] and record["prediction"]["status"] == "ok":
                predicted = record["prediction"]["task_update"].get("new_goal")
        row = {"task_id": identity, "baseline_candidate_count": 2, "expanded_candidate_count": CANDIDATE_COUNT,
            "baseline_update_correct": before["model_update_correct"], "expanded_update_correct": result["model_update_correct"],
            "baseline_pipeline_success": before["model_pipeline_success"], "expanded_pipeline_success": result["model_pipeline_success"],
            "expanded_integrity_passed": result["integrity_passed"], "expected_goal": result["replacement_goal"],
            "predicted_goal": predicted, "selected_distractor": predicted is not None and predicted not in (result["initial_goal"], result["replacement_goal"]),
            "model_failure_stage": result["model_failure_stage"], **comparisons, "error": error or result["error"]}
        paired.append(row)
    count = manifest["task_count"]
    matched = sum(all(row[key] is True for key in ("oracle_behavior_matched", "ignore_behavior_matched")) for row in paired)
    report = {"kind": PROTOCOL, "task_count": count, "planned_episodes": manifest["planned_episodes"],
        "planned_model_calls": manifest["planned_model_calls"], "baseline_candidate_count": 2,
        "expanded_candidate_count": CANDIDATE_COUNT, "task_configs_matched": count,
        "baseline_update_correct_count": sum(r["baseline_update_correct"] for r in paired),
        "expanded_update_correct_count": expanded["model_update_correct_count"],
        "expanded_update_correct_rate": expanded["model_update_correct_rate"],
        "baseline_pipeline_success_count": sum(r["baseline_pipeline_success"] for r in paired),
        "expanded_pipeline_success_count": expanded["model_pipeline_success_count"],
        "expanded_pipeline_success_rate": expanded["model_pipeline_success_rate"],
        "selected_distractor_count": sum(r["selected_distractor"] for r in paired),
        "oracle_ignore_behavior_matched_tasks": matched, "results_provisional": expanded["results_provisional"],
        "comparison_complete": expanded["integrity_passed"] and matched == count,
        "passed": expanded["passed"] and matched == count, "expanded_summary": "batch/summary.json",
        "pairs": paired, "limitations": ["Single scene, five known distinct reviewed instances, same explicit template messages",
            "Candidate order remains sorted; no paraphrase/order experiment or same-category ambiguity claim",
            "Cross-batch GPU IDs may differ; pixel equality is reported separately from physical controls",
            "Text model plus privileged navigator; this is not visual grounding or a history/state comparison"]}
    base.write_json(output / "summary.json", report)
    base.write_jsonl(output / "comparison.jsonl", paired)
    batch.write_csv(output / "comparison.csv", paired)
    print(f"Five candidates: updates {report['expanded_update_correct_count']}/{count}; "
          f"pipeline {report['expanded_pipeline_success_count']}/{count}; Oracle/Ignore controls matched {matched}/{count}.")
    return report


def run(args):
    output = args.output_dir.resolve()
    check(output)
    batch.run(SimpleNamespace(output_dir=output / "batch", gpu_id=args.gpu_id, api_key_env=args.api_key_env))
    return 0 if summarize(output)["passed"] else 1


def videos(args):
    output = args.output_dir.resolve()
    validate(output)
    return batch.videos(SimpleNamespace(output_dir=output / "batch", fps=args.fps, task_id=args.task_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="CPU: freeze a five-candidate derivation of a complete baseline")
    preparing.add_argument("--baseline-dir", type=Path, required=True)
    commands.add_parser("check", help="CPU Habitat: check all five stopping points and reachability")
    running = commands.add_parser("run", help="GPU/model: run all NEW three-condition episodes")
    running.add_argument("--gpu-id", type=int, default=0)
    running.add_argument("--api-key-env", default="TASK_UPDATE_API_KEY")
    commands.add_parser("summarize", help="CPU: audit new runs and compare against original raw evidence")
    video = commands.add_parser("videos", help="FFmpeg: export the new batch videos")
    video.add_argument("--fps", type=float, default=10.0)
    video.add_argument("--task-id")
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        return 0 if check(args.output_dir.resolve())["passed"] else 1
    if args.command == "summarize":
        return 0 if summarize(args.output_dir.resolve())["passed"] else 1
    return {"prepare": prepare, "run": run, "videos": videos}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
