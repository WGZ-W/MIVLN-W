"""Private evaluator inventory for the three existing HSSD Oracle datasets.

This adapter only reads files. It deliberately plans slots from the canonical
replacement roster, rather than counting successful runs or cached summaries.
Inventory construction needs only the standard library; native replay is kept
in benchmark_audit so indexing also works in the core Python environment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMINGS = {"early": .25, "middle": .5, "late": .75}
CONDITIONS = ("control", "intervention", "ignore")
PROFILES = {
    "replacement": ("replace_goal_v0.1", "reference_remaining_geodesic_reduction_before_action"),
    "avoid": ("add_avoid_region_v1", "25_percent_reference_travel_before_action"),
    "subgoal": ("insert_subgoal_v1", "cumulative_reference_travel_before_action"),
}
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, ImportError)


def sha256(path):
    """Stream large observations instead of loading a dataset into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def within(root, relative):
    """Reject absolute/relative escapes, including symlinks, before any read."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Source path escapes dataset: {relative}")
    return path


def issue(issues, severity, code, scope, detail):
    issues.append({"severity": severity, "code": code, "scope": scope, "detail": str(detail)})


def optional_json(path, default, issues, scope, required=True):
    try:
        return read_json(path)
    except ERRORS as error:
        if required:
            issue(issues, "error", "unreadable_input", scope, f"{path}: {error}")
        return default


def file_binding(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256(path) if path.is_file() else None}


def safe_slot(root, record, key, fallback, issues, scope):
    try:
        return within(root, record.get(key, fallback))
    except ERRORS as error:
        issue(issues, "error", "unsafe_path", scope, error)
        # The canonical slot remains in the denominator, but this family's
        # input error prevents accepting files found at the fallback location.
        return within(root, fallback)


def unique_records(records, key):
    result = {}
    for record in records:
        identity = record[key]
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"Ambiguous or duplicate {key}: {identity}")
        result[identity] = record
    return result


def build_inventory(sources):
    """Return all planned cases/episodes, even if derived manifests are absent.

    A readable, unambiguous canonical roster is required: inventing base task
    identities after losing that roster would create an unjustified denominator.
    Derived family omissions instead retain their full 1/3-timing slot budgets.
    """
    sources = {name: Path(path).resolve() for name, path in sources.items()}
    issues, manifests, statuses, history = [], {}, {}, {}
    for family, root in sources.items():
        manifests[family] = optional_json(root / "manifest.json", {}, issues, family)
        statuses[family] = optional_json(root / "run_status.json", {}, issues, family)
        cached = optional_json(root / "summary.json", {}, issues, family, required=False)
        history[family] = {"binding": file_binding(root / "summary.json"),
                           "reported_passed": cached.get("passed"), "used_for_scoring": False}
        if not isinstance(manifests[family], dict) or not isinstance(statuses[family], dict):
            raise ValueError(f"{family}: manifest and run_status must be objects")
    canonical = manifests["replacement"]
    originals = unique_records(canonical["tasks"], "task_id")
    if not originals:
        raise ValueError("Canonical base task roster is empty")
    count = len(originals)
    if canonical.get("task_count") != count or canonical.get("expected_episodes") != count * 10:
        issue(issues, "error", "denominator_mismatch", "replacement", "Expected ten slots per canonical base task")
    for family in ("avoid", "subgoal"):
        manifest = manifests[family]
        try:
            if (Path(manifest["source_dir"]).resolve() != sources["replacement"]
                    or manifest["source_manifest_sha256"] != sha256(sources["replacement"] / "manifest.json")):
                raise ValueError("Derived dataset does not bind the selected replacement source")
            roster = unique_records(manifest["tasks"], "task_id")
            if set(roster) != set(originals):
                raise ValueError("Derived base task roster differs from the canonical roster")
            if manifest["task_count"] != count or manifest["expected_episodes"] != count * (3 if family == "avoid" else 9):
                raise ValueError("Derived planned denominator differs from the B0 profile")
            if manifest["trigger_protocol"] != PROFILES[family][1]:
                raise ValueError("Unexpected trigger protocol")
            if family == "subgoal" and (manifest["timings"] != list(TIMINGS)
                    or manifest["timing_progress"] != TIMINGS or manifest["case_count"] != count * 3):
                raise ValueError("B0 inventory requires all three frozen subgoal timings")
        except ERRORS as error:
            issue(issues, "error", "source_lineage", family, error)

    bases, cases, episodes = [], [], []
    for task_id, task in originals.items():
        root = sources["replacement"]
        # Task IDs also form fallback paths. Refuse traversal instead of trying
        # to sanitize two distinct identities into the same filesystem slot.
        if Path(task_id).name != task_id or task_id in (".", ".."):
            raise ValueError(f"Unsafe canonical task ID: {task_id}")
        base_path = safe_slot(root, task, "base_config", f"configs/{task_id}/base.json", issues, "replacement")
        config = optional_json(base_path, {}, issues, "replacement")
        scene_id, scene_hash = config.get("scene_id"), config.get("scene_sha256")
        scene_key = f"{scene_id or 'unknown'}@{scene_hash or 'unknown'}"
        base_id = f"{scene_key}/{task_id}"
        destination = safe_slot(root, task, "output_subdir", f"tasks/{task_id}", issues, "replacement")
        reference_dir = destination / "reference"
        reference_id = f"{base_id}/reference"
        shared = {"scene_id": scene_id, "scene_sha256": scene_hash, "scene_group": scene_key,
                  "base_task_id": base_id, "source_task_id": task_id,
                  "leakage_group": base_id, "split": None, "split_status": "not_frozen",
                  "manual_review_status": "not_recorded", "reference_id": reference_id}
        bases.append({**shared, "source_config": file_binding(base_path),
                      "reference_episode": file_binding(reference_dir / "episode.json"),
                      "catalog_source": config.get("catalog_source"),
                      "initial_goal": config.get("initial_goal"), "seed": config.get("seed")})
        reference = {**shared, "episode_id": reference_id, "case_id": None, "task_type": "replacement",
                     "role": "reference", "condition": "reference", "source_condition": "reference",
                     "protocol_version": PROFILES["replacement"][0], "trigger_protocol": None,
                     "evaluation_truth": "original_task", "expected_dynamic_success": True,
                     "timing": None, "source_dir": str(reference_dir), "source_config": file_binding(
                         safe_slot(root, task, "reference_config", f"configs/{task_id}/reference.json", issues, "replacement")),
                     "original_run_status": statuses["replacement"].get(task_id), "audit_status": "not_audited"}
        episodes.append(reference)
        for family in PROFILES:
            manifest, root = manifests[family], sources[family]
            key = "case_id" if family == "subgoal" else "task_id"
            try:
                records = unique_records(manifest.get("cases" if family == "subgoal" else "tasks", []), key)
            except ERRORS as error:
                issue(issues, "error", "ambiguous_roster", family, error)
                records = {}
            for timing in (["early"] if family == "avoid" else TIMINGS):
                source_id = f"{task_id}_{timing}" if family == "subgoal" else task_id
                record = records.get(source_id, {})
                if not record:
                    issue(issues, "error", "missing_case", family, source_id)
                default = f"tasks/{task_id}" + (f"/{timing}" if family == "subgoal" else "")
                folder = safe_slot(root, record, "output_subdir", default, issues, family)
                if family == "replacement":
                    folder = folder / "timing" / timing
                    config_path = folder.parent / "configs" / f"{timing}.json"
                else:
                    prep = f"preparations/{task_id}" + (f"/{timing}" if family == "subgoal" else "")
                    config_path = safe_slot(root, record, "preparation_dir", prep, issues, family) / "config.json"
                variant = optional_json(config_path, {}, issues, family)
                events = variant.get("events", [])
                case_id = f"{base_id}/{family}/{timing}"
                item = {**shared, "case_id": case_id, "task_type": family,
                        "source_case_id": source_id, "protocol_version": PROFILES[family][0],
                        "protocol_status": "legacy_profile_not_unified_release", "timing": timing,
                        "trigger_protocol": PROFILES[family][1], "progress_fraction": TIMINGS[timing],
                        "trigger_step": events[0].get("trigger", {}).get("value") if events else None,
                        "source_dir": str(folder), "source_config": file_binding(config_path),
                        "source_manifest": file_binding(root / "manifest.json"),
                        "declared_config_sha256": record.get("config_sha256"),
                        "preparation_sha256": record.get("preparation_sha256"),
                        "reference_hashes": record.get("reference_hashes"),
                        "prepared": bool(config_path.is_file()) if family == "replacement" else record.get("prepared", False),
                        "preparation_error": record.get("preparation_error"),
                        "original_run_status": statuses[family].get(source_id),
                        "input_visibility_status": "not_frozen",
                        "baseline": "privileged_oracle_and_ignore_controls",
                        "privileged_inputs": ["exact_geometry", "structured_task_updates", "navmesh"],
                        "audit_status": "not_audited"}
                cases.append(item)
                for condition in CONDITIONS:
                    episodes.append({**item, "episode_id": f"{case_id}/{condition}", "role": "branch",
                                     "source_condition": condition,
                                     "condition": "follow" if condition == "intervention" else condition,
                                     "evaluation_truth": "original_task" if condition == "control" else "updated_task",
                                     "source_dir": str(folder / condition),
                                     "expected_dynamic_success": condition != "ignore"})
    return {"schema_version": 1, "kind": "private_benchmark_evaluator_inventory",
            "sources": {name: {"path": str(root), "manifest": file_binding(root / "manifest.json")}
                        for name, root in sources.items()},
            "base_tasks": bases, "cases": cases, "episodes": episodes,
            "historical_summaries": history, "issues": issues}


def coverage(inventory):
    """Separate sample counts, expectation acceptance, and task success.

    A valid Ignore episode is accepted precisely when it fails the updated task
    for the expected reason. Unverified success flags never enter numerators.
    These are development asset statistics, not a public benchmark leaderboard.
    """
    episodes = inventory["episodes"]
    result = {"base_tasks": len(inventory["base_tasks"]),
              "scenes": len({row["scene_group"] for row in inventory["base_tasks"]}),
              "task_types": len(PROFILES), "intervention_configurations": len(inventory["cases"]),
              "planned_episodes": len(episodes), "reference_episodes": sum(row["role"] == "reference" for row in episodes),
              "manual_reviewed_configurations": sum(row["manual_review_status"] == "approved" for row in inventory["cases"]),
              "split_status": "not_frozen", "by_task_type": {}}
    for family in PROFILES:
        rows = [row for row in episodes if row["task_type"] == family]
        groups = {}
        for condition in ("reference", "control", "follow", "ignore"):
            selected = [row for row in rows if row["condition"] == condition]
            if not selected:
                continue
            accepted = [row for row in selected if row["audit_status"] == "accepted"]
            groups[condition] = {"planned": len(selected),
                "metrics_available": sum(row.get("metrics_available", False) for row in selected),
                "accepted": len(accepted),
                "verified_task_successes": sum(row.get("reported_dynamic_success") is True for row in accepted),
                "verified_expected_failures": sum(row.get("reported_dynamic_success") is False for row in accepted)
                    if condition == "ignore" else 0,
                "task_success_rate_fixed_denominator": sum(row.get("reported_dynamic_success") is True for row in accepted) / len(selected)}
        result["by_task_type"][family] = {"planned_episodes": len(rows), "conditions": groups,
            "oracle_feasible_configurations": sum(all(
                row["audit_status"] == "accepted" and row.get("reported_dynamic_success") is True
                for row in rows if row["case_id"] == case["case_id"] and row["condition"] == "follow")
                for case in inventory["cases"] if case["task_type"] == family)}
    return result
