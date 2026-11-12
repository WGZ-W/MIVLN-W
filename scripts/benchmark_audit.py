"""Index and re-audit existing Oracle assets without modifying their sources.

Run with --index-only in the core environment, or without it in the Habitat
environment for CPU native mesh/instance replay. Neither mode runs a policy,
creates a Simulator, calls a language model, or changes historical summaries.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from scripts.benchmark_manifest import (
    CONDITIONS, ERRORS, PROFILES, PROJECT_ROOT, build_inventory, coverage,
    file_binding, issue, read_json, sha256, within,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(roots):
    """Hash every source file, including images, before and after the audit.

    This verifies bytes and the file roster, not just summary timestamps. Source
    directory symlinks are rejected so traversal cannot silently omit an external
    evidence tree. External dependencies (mesh/catalog) remain hash-checked by
    the legacy validators and are explicitly outside this tree comparison.
    """
    result = {}
    for name, root in roots.items():
        root = Path(root).resolve()
        if not root.is_dir():
            result[name] = None
            continue
        files = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symlink in source evidence tree: {path}")
            if path.is_file():
                files[str(path.relative_to(root))] = sha256(path)
        result[name] = files
    return result


def validate_output(output, roots):
    """Fail before creating output if it overlaps any protected source tree."""
    output = Path(output).resolve()
    for root in roots.values():
        root = Path(root).resolve()
        if output == root or output in root.parents or root in output.parents:
            raise ValueError(f"Audit output overlaps source: {root}")
    if output.exists():
        raise FileExistsError(f"Use a fresh --output-dir: {output}")
    return output


def finish_episode(row, passed, evidence=None, error=None):
    """Attach raw reports separately from accepted task outcomes."""
    row["audit_status"] = "accepted" if passed else "rejected"
    row["condition_expectation_passed"] = bool(passed)
    row["audit_error"] = str(error) if error else (None if passed else "Condition expectation or evidence validation failed")
    row["audit_evidence"] = evidence or {}


def family_errors(inventory, family):
    return [entry["detail"] for entry in inventory["issues"]
            if entry["scope"] == family and entry["severity"] == "error"]


def read_metrics(inventory):
    for row in inventory["episodes"]:
        path = Path(row["source_dir"]) / "metrics.json"
        row["metrics_available"] = False
        row["reported_dynamic_success"] = None
        try:
            metrics = read_json(path)
            if not isinstance(metrics, dict) or type(metrics.get("dynamic_success")) is not bool:
                raise ValueError("Metrics must contain a boolean dynamic_success")
            row.update({"metrics_available": True, "reported_dynamic_success": metrics["dynamic_success"],
                        "metrics_binding": file_binding(path)})
        except ERRORS as error:
            row["metrics_error"] = str(error)


def audit_replacement(inventory, root):
    """Reuse pure checkers, never hssd_dataset.summarize (which writes files).

    This is the existing replacement acceptance profile: pose/event timelines,
    paired observation hashes, STOP metrics, and frozen terminal diagnostics.
    It is not a new native replay of all replacement geodesic distances.
    """
    from scripts import hssd_dataset as legacy
    from scripts import terminal_view
    from scripts.intervention_timing import check_reference, select_triggers
    from scripts.run_intervention import check_pair
    from scripts.run_ignore_intervention import check_negative

    errors = family_errors(inventory, "replacement")
    try:
        legacy.validate_inputs(root, read_json(root / "manifest.json"))
    except ERRORS as error:
        errors.append(str(error))
    report = {"input_errors": errors, "audit_profile": "legacy_replacement_terminal_evidence_v1"}
    for base in inventory["base_tasks"]:
        task_id = base["source_task_id"]
        rows = [row for row in inventory["episodes"]
                if row["task_type"] == "replacement" and row["base_task_id"] == base["base_task_id"]]
        reference_row = next(row for row in rows if row["role"] == "reference")
        refdir = Path(reference_row["source_dir"])
        config, reference, triggers, referror = None, None, {}, None
        try:
            config = read_json(base["source_config"]["path"])
            reference = check_reference(config, refdir)
            legacy.same_task(config, read_json(refdir / "episode.json"))
            metrics = read_json(refdir / "metrics.json")
            legacy.check_episode_records(config, config, "control", refdir, metrics, reference)
            terminal = legacy.terminal_record(config, metrics, reference, refdir, "reference")
            if terminal_view.enabled(config, "terminal_visibility") and not terminal["passed"]:
                raise ValueError(terminal["error"])
            triggers = {item["name"]: item for item in select_triggers(reference)}
            status = reference_row["original_run_status"]
            if not isinstance(status, list) or not any(s.get("stage") == "reference" and s.get("returncode") == 0 for s in status):
                raise ValueError("Missing successful original reference run status")
            finish_episode(reference_row, not errors, {"terminal": terminal}, "; ".join(errors))
        except ERRORS as error:
            referror = str(error)
            finish_episode(reference_row, False, error=error)
        expected_stages = {"reference", "timing_prepare", *[f"{name}_{part}" for name in ("early", "middle", "late")
                                                         for part in ("pair", "ignore")]}
        for row in rows:
            if row["role"] == "reference":
                continue
            evidence = {}
            try:
                if errors or referror:
                    raise ValueError("; ".join(errors + ([referror] if referror else [])))
                status = row["original_run_status"]
                if (not isinstance(status, list) or len(status) != len(expected_stages)
                        or {s["stage"] for s in status} != expected_stages
                        or any(s["returncode"] != 0 for s in status)):
                    raise ValueError("Original replacement stages missing or nonzero")
                directory = Path(row["source_dir"])
                variant = read_json(row["source_config"]["path"])
                legacy.same_task(config, variant)
                event = variant["events"][0]
                if (len(variant["events"]) != 1 or event["trigger"]["value"] != triggers[row["timing"]]["trigger_step"]
                        or event["task_update"] != config["events"][0]["task_update"]
                        or event["utterance"] != config["events"][0]["utterance"]):
                    raise ValueError("Timing changed the frozen update or trigger")
                condition = row["source_condition"]
                episode = read_json(directory / "episode.json")
                legacy.same_task(config, episode)
                if episode["events"] != ([] if condition == "control" else variant["events"]):
                    raise ValueError("Episode event configuration differs from its slot")
                metrics, trajectory = read_json(directory / "metrics.json"), read_json(directory / "trajectory.json")
                legacy.check_episode_records(config, variant, condition, directory, metrics, trajectory)
                terminal = legacy.terminal_record(config, metrics, trajectory, directory, condition)
                evidence["terminal"] = terminal
                if terminal_view.enabled(config, "terminal_visibility") and not terminal["passed"]:
                    raise ValueError(terminal["error"])
                control_dir = directory.parent / "control"
                control = read_json(control_dir / "trajectory.json")
                if condition == "control":
                    passed = bool(trajectory == reference and metrics["dynamic_success"]
                        and trajectory[-1]["action"] == "STOP"
                        and legacy.matching_frames(refdir, reference, directory, trajectory))
                elif condition == "intervention":
                    checked = check_pair(directory.parent, event)
                    evidence["checks"] = checked["checks"]
                    count = event["trigger"]["value"] + 1
                    passed = bool(checked["passed"] and legacy.matching_frames(
                        control_dir, control[:count], directory, trajectory[:count]))
                else:
                    checked = check_negative(directory, variant, control_dir)
                    evidence["checks"] = checked["checks"]
                    passed = bool(checked["negative_check_passed"] and legacy.matching_frames(
                        control_dir, control, directory, trajectory))
                finish_episode(row, passed, evidence)
            except ERRORS as error:
                finish_episode(row, False, evidence, error)
    report["input_validation_passed"] = not errors
    return report


def run_completed(status, audit, allow_native_recovery=False):
    """Preserve historical nonzero exits; require fresh native proof to recover."""
    recovered = bool(allow_native_recovery and isinstance(status, dict)
        and status.get("attempted") is True and status.get("returncode") == 1
        and audit and audit.get("passed") is True
        and any(item.get("native_projection_checks") for item in audit["conditions"]))
    completed = bool(isinstance(status, dict) and status.get("attempted") is True
                     and (status.get("returncode") == 0 or recovered))
    return completed, recovered


def audit_derived(inventory, family, root):
    """Read-only raw-evidence replay with this manifest's exact child binding."""
    if family == "avoid":
        from scripts import avoid_dataset as dataset, avoid_region as checker
    else:
        from scripts import subgoal_dataset as dataset, subgoal as checker
    errors = family_errors(inventory, family)
    try:
        dataset.validate_batch(root, read_json(root / "manifest.json"))
    except ERRORS as error:
        errors.append(str(error))
    report = {"input_errors": errors, "cases": [], "audit_profile":
              "native_projection_endpoint_v2" if family == "subgoal" else "add_avoid_region_v1_raw_replay"}
    for case in inventory["cases"]:
        if case["task_type"] != family:
            continue
        audit, error = None, case["preparation_error"]
        try:
            if not case["prepared"]:
                raise ValueError(error or "Preparation missing")
            folder = Path(case["source_dir"])
            audit = checker.check_experiment(folder)
            experiment = read_json(folder / "experiment.json")
            binding = case["source_config"]
            if (experiment["source_config"] != binding["path"] or experiment["config_sha256"] != binding["sha256"]):
                raise ValueError("Child experiment belongs to a different manifest slot")
        except ERRORS as caught:
            audit, error = None, str(caught)
        completed, recovered = run_completed(case["original_run_status"], audit, family == "subgoal")
        if not completed:
            error = error or "Original child run missing, incomplete or nonzero without fresh native proof"
        if errors:
            error = "; ".join(errors)
        report["cases"].append({"case_id": case["case_id"], "source_case_id": case["source_case_id"],
                                "original_run_status": case["original_run_status"],
                                "audit_recovered": recovered, "error": error, "raw_audit": audit})
        for row in inventory["episodes"]:
            if row["case_id"] != case["case_id"]:
                continue
            checked = next((item for item in (audit or {}).get("conditions", [])
                            if item["condition"] == row["source_condition"]), {})
            passed = bool(audit and audit["passed"] and checked.get("passed") and completed and not errors)
            row["audit_recovered"] = recovered
            finish_episode(row, passed, {"condition": checked, "pairing_errors": (audit or {}).get("pairing_errors")},
                           error or checked.get("error"))
    report["input_validation_passed"] = not errors
    return report


def historical_reaudit(root, issues):
    """Bind supplementary historical evidence; never use its success as a score."""
    result = {"path": str(root), "used_for_scoring": False, "status": "missing"}
    try:
        record = read_json(root / "acceptance.json")
        if not record.get("original_files_unchanged") or not record.get("checker_sha256"):
            raise ValueError("Historical acceptance lacks original-file/checker hash bindings")
        result.update({"binding": file_binding(root / "acceptance.json"),
                       "reported_passed": record.get("passed"), "audit_protocol": record.get("audit_protocol"),
                       "manual_video_review_note": record.get("manual_video_review"), "status": "bound"})
        mismatches = []
        for name, digest in record.get("original_files_unchanged", {}).items():
            if not Path(name).is_file() or sha256(name) != digest:
                mismatches.append(name)
        for name, digest in record.get("checker_sha256", {}).items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256(path) != digest:
                mismatches.append(name)
        result["mismatches"] = mismatches
        result["evidence_files"] = [file_binding(path) for path in sorted(root.glob("*.json"))]
        if mismatches:
            result["status"] = "stale"
            issue(issues, "warning", "historical_evidence_stale", "historical_reaudit", mismatches)
    except ERRORS as error:
        issue(issues, "warning", "historical_evidence_missing", "historical_reaudit", error)
    return result


def export_rows(output, name, rows):
    """JSONL is lossless; CSV encodes nested evidence as JSON for inspection."""
    with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def run(args):
    """Write only a fresh report tree; a failed audit still exports all slots."""
    sources = {"replacement": args.replacement_dir.resolve(), "avoid": args.avoid_dir.resolve(),
               "subgoal": args.subgoal_dir.resolve()}
    protected = {**sources, "historical_reaudit": args.reaudit_dir.resolve()}
    output = validate_output(args.output_dir, protected)
    print("Hashing source evidence before audit (including image/video bytes)...", flush=True)
    before = snapshot(protected)
    inventory = build_inventory(sources)
    read_metrics(inventory)
    historical = historical_reaudit(protected["historical_reaudit"], inventory["issues"])
    output.mkdir(parents=True)
    audits = {}
    if not args.index_only:
        for family, root in sources.items():
            print(f"Read-only CPU audit: {family}", flush=True)
            try:
                audits[family] = (audit_replacement(inventory, root) if family == "replacement"
                                  else audit_derived(inventory, family, root))
            except ERRORS as error:
                audits[family] = {"input_errors": [str(error)]}
                for row in inventory["episodes"]:
                    if row["task_type"] == family:
                        finish_episode(row, False, error=error)
            for error in audits[family]["input_errors"]:
                issue(inventory["issues"], "error", "family_audit_error", family, error)
            write_json(output / "audits" / f"{family}.json", audits[family])
    for case in inventory["cases"]:
        rows = [row for row in inventory["episodes"] if row["case_id"] == case["case_id"]]
        case["audit_status"] = ("not_audited" if args.index_only else
                                "accepted" if all(row["audit_status"] == "accepted" for row in rows) else "rejected")
        case["audit_evidence"] = f"audits/{case['task_type']}.json" if not args.index_only else None
    for row in inventory["episodes"]:
        if row["audit_status"] == "rejected":
            issue(inventory["issues"], "error", "episode_rejected", row["episode_id"], row["audit_error"])
    for code, detail in {
        "input_visibility_unfrozen": "Oracle inputs are privileged; public agent inputs and avoid-region observability need B1 specification",
        "timing_profiles_differ": "Replacement uses remaining geodesic reduction; avoid/subgoal use cumulative reference travel",
        "split_unfrozen": "One development scene; no scene-disjoint split is defined",
        "manual_review_unrecorded": "Historical representative frames do not establish full independent language/video review",
        "efficiency_metric_unfrozen": "Constrained/subgoal optimal routes and common efficiency scores are not defined",
    }.items():
        issue(inventory["issues"], "warning", code, "release", detail)
    print("Hashing source evidence after audit...", flush=True)
    after = snapshot(protected)
    unchanged = before == after
    if not unchanged:
        issue(inventory["issues"], "error", "source_tree_changed", "readonly", "Source bytes or file roster changed during audit")
    write_json(output / "source_snapshot.json", {"scope": "all files in the four source trees; external dependencies checked by validators",
                "before": before, "after": after, "unchanged": unchanged})
    counts = coverage(inventory)
    passed = bool(not args.index_only and unchanged and all(row["audit_status"] == "accepted" for row in inventory["episodes"])
                  and not any(item["severity"] == "error" for item in inventory["issues"]))
    report = {"passed": passed, "mode": "index_only" if args.index_only else "readonly_audit",
              "source_files_unchanged": unchanged, "coverage": counts,
              "input_integrity": {name: audit.get("input_validation_passed", False) for name, audit in audits.items()},
              "error_count": sum(item["severity"] == "error" for item in inventory["issues"]),
              "warning_count": sum(item["severity"] == "warning" for item in inventory["issues"]),
              "release_ready": False, "historical_reaudit": historical}
    write_json(output / "summary.json", report)
    export_rows(output, "base_tasks", inventory["base_tasks"])
    export_rows(output, "task_index", inventory["cases"])
    export_rows(output, "episode_index", inventory["episodes"])
    export_rows(output, "issues", inventory["issues"])
    write_json(output / "manifest.json", {"schema_version": 1, "kind": inventory["kind"],
        "created_at": datetime.now(timezone.utc).isoformat(), "sources": inventory["sources"],
        "profiles": PROFILES, "historical_summaries": inventory["historical_summaries"],
        "coverage": counts, "audit_code": [file_binding(PROJECT_ROOT / "scripts" / name) for name in (
            "benchmark_manifest.py", "benchmark_audit.py", "hssd_dataset.py", "avoid_dataset.py", "avoid_region.py",
            "subgoal_dataset.py", "subgoal.py", "subgoal_cpu.py", "run_intervention.py", "run_ignore_intervention.py")],
        "inventory_visibility": "private_evaluator_only", "release_ready": False})
    print(f"Bases={counts['base_tasks']}; cases={counts['intervention_configurations']}; "
          f"episodes={counts['planned_episodes']}; passed={passed}; sources unchanged={unchanged}", flush=True)
    return 0 if passed or (args.index_only and unchanged and report["error_count"] == 0) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in (("replacement", "hssd_dataset_facing_001"), ("avoid", "hssd_avoid_batch_001"),
                          ("subgoal", "hssd_subgoal_timing_001"), ("reaudit", "hssd_subgoal_acceptance_2026-11-03")):
        parser.add_argument(f"--{flag}-dir", type=Path, default=PROJECT_ROOT / "outputs" / default)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-only", action="store_true", help="Inventory only; no raw-evidence acceptance claimed")
    args = parser.parse_args()
    try:
        return run(args)
    except ERRORS as error:
        print(f"Benchmark audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
