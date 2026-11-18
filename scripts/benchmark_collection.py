"""Assemble and verify the first HSSD intervention collection package.

The package is an evaluator-side index of reviewed cases and audited runs.
Media and trajectories stay in their source folders. This command never starts
a renderer or a policy. Build uses Habitat CPU replay for the strong pilot.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

from mivln.benchmark_navigation import validate_case
from scripts import strong_relatedness_navigation as strong
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import PROJECT_ROOT, read_json, sha256, unique_records, within
from scripts.export_visual_videos import verified_batch
from scripts.hssd_visual_dataset import disjoint_output
from scripts.intervention_relatedness import digest
from scripts.public_inputs import write_json, write_lines
from scripts.relatedness_geometry import load_review


KIND = "hssd_intervention_collection_v0"
QUOTAS = {family: {level: 15 for level in ("strong", "weak", "independent")}
          for family in ("replacement", "subgoal")}
QUOTAS["cancel_task"] = {"not_applicable": 10}
DEFAULTS = {
    "middle_run": "hssd_v_middle_batch_run_001", "middle_audit": "hssd_v_middle_batch_audit_001",
    "strong_run": "hssd_strong_pilot_run_001", "strong_audit": "hssd_strong_pilot_audit_001",
    "replacement_candidates": "benchmark_relatedness_candidates_001",
    "replacement_review": "benchmark_relatedness_review_001",
    "subgoal_candidates": "benchmark_subgoal_relatedness_candidates_001",
    "subgoal_review": "benchmark_subgoal_relatedness_review_001",
}


def subgoal_labels(candidate_path, review_path):
    package, review = read_json(candidate_path), read_json(review_path)
    if (package["kind"] != "subgoal_relatedness_review_candidates_v1"
            or review["kind"] != "subgoal_relatedness_review_v1"
            or package["scene_id"] != review["scene_id"]
            or (review_path.parent / review["candidate_manifest"]).resolve() != candidate_path.resolve()
            or sha256(candidate_path) != review["candidate_manifest_sha256"]):
        raise ValueError("Subgoal review does not bind this candidate package")
    candidates = unique_records(package["candidates"], "case_id")
    decisions = unique_records(review["decisions"], "case_id")
    if not candidates or set(candidates) != set(decisions):
        raise ValueError("Subgoal review roster differs from the collection")
    for key, row in candidates.items():
        decision = decisions[key]
        if (decision["candidate_sha256"] != digest(row) or decision["decision"] != "approve"
                or decision["relatedness_level"] not in QUOTAS["subgoal"]):
            raise ValueError(f"Subgoal case lacks an approved label bound to its evidence: {key}")
    return candidates, decisions


def inventory(cases, episodes):
    """Keep scene, base task, intervention and execution denominators separate."""
    indexed = unique_records(cases, "case_id")
    unique_records(episodes, "episode_id")
    conditions = defaultdict(set)
    for row in episodes:
        case_id, condition = row["case_id"], row["condition"]
        if case_id not in indexed or condition not in ("oracle", "ignore") or condition in conditions[case_id]:
            raise ValueError("Unexpected or duplicate episode condition")
        if row["accepted"] is not True or row["task_success"] is not (condition == "oracle"):
            raise ValueError(f"Episode does not meet its planned outcome: {case_id}/{condition}")
        conditions[case_id].add(condition)
    bases, counts = {}, defaultdict(Counter)
    for row in cases:
        case_id, base_id = row["case_id"], row["base_task_id"]
        if conditions[case_id] != {"oracle", "ignore"}:
            raise ValueError(f"Missing execution evidence: {case_id}")
        family, level = row["task_type"], row["relatedness_level"]
        if family not in QUOTAS or level not in QUOTAS[family]:
            raise ValueError(f"Invalid task and relatedness combination: {case_id}")
        base = {key: row[key] for key in ("base_task_id", "scene_id", "scene_sha256", "scene_group")}
        base.update({key: row["config"][key] for key in ("start_position", "start_rotation_xyzw", "initial_goal")})
        base.update(split="unassigned")
        if bases.setdefault(base_id, base) != base:
            raise ValueError(f"Base identity groups different initial geometry: {base_id}")
        counts[row["scene_group"]][family, level] += 1
    if not bases:
        raise ValueError("Empty collection")
    by_scene = {scene: {family: {level: {"count": group[family, level], "target": target,
                                         "shortfall": max(0, target - group[family, level])}
                                for level, target in levels.items()}
                        for family, levels in QUOTAS.items()}
                for scene, group in counts.items()}
    return list(bases.values()), {"scenes": len(counts), "base_tasks": len(bases), "cases": len(cases),
               "audited_episodes": len(episodes),
               "oracle_successes": sum(row["condition"] == "oracle" for row in episodes),
               "ignore_expected_failures": sum(row["condition"] == "ignore" for row in episodes),
               "quota_by_scene": by_scene, "split_status": "unassigned", "release_ready": False}


def members(folder, root_snapshot, root):
    """Bind all PNGs, raw policy I/O, measurements and diagnostics in a branch."""
    relative = folder.relative_to(root).as_posix() + "/"
    files = {name[len(relative):]: value for name, value in root_snapshot.items() if name.startswith(relative)}
    required = {"private/config.json", "private/events.json", "private/trajectory.json",
                "private/policy_records.jsonl", "public/requests.jsonl", "public/actions.jsonl"}
    if not required <= files.keys() or not any(name.startswith("public/assets/") for name in files):
        raise ValueError(f"Incomplete audited branch: {folder}")
    return files


def build(args):
    prepared, middle, middle_roots, middle_before = verified_batch(args.middle_run, args.middle_audit)
    origin, repl_rows, repl_decisions = load_review(args.replacement_candidates, args.replacement_review)
    if origin.resolve() != Path(prepared["source_dir"]).resolve():
        raise ValueError("Replacement labels refer to a different navigation preparation")
    replacement = unique_records(repl_rows, "case_id")
    sub_path, sub_review = args.subgoal_candidates / "manifest.json", args.subgoal_review / "review.json"
    subgoal, sub_decisions = subgoal_labels(sub_path, sub_review)
    strong_run = args.strong_run.resolve()
    strong_manifest = read_json(strong_run / "manifest.json")
    strong_prepared = Path(strong_manifest["prepared_dir"])
    _, strong_protected = strong.load_prepared(strong_prepared)
    protected = [*middle_roots.values(), *strong_protected, strong_run,
                 args.middle_audit, args.strong_audit, args.replacement_candidates, args.replacement_review,
                 args.subgoal_candidates, args.subgoal_review]
    output = disjoint_output(args.output_dir, protected)
    strong_roots = {"strong_run": strong_run, "strong_prepared": strong_prepared}
    strong_before = snapshot(strong_roots)
    pilot = strong.check_run(strong_run)
    if (pilot["passed"] is not True or pilot != read_json(strong_run / "summary.json")
            or {**pilot, "audited_files_unchanged": True} != read_json(args.strong_audit / "audit.json")):
        raise ValueError("Strong pilot changed or failed a fresh native replay")
    middle_snapshots = {Path(path).resolve(): middle_before[name] for name, path in middle_roots.items()}
    cases, episodes = [], []

    def collect(audit_row, folder, files, *, is_strong):
        config = validate_case(read_json(folder / "private/config.json"))
        raw_id, family, condition = audit_row["case_id"], audit_row["task_type"], audit_row["condition"]
        scene = f"{config['scene_id']}@{config['scene_sha256']}"
        case_id = f"{scene}/{raw_id}"
        if config["episode_id"] != raw_id or config["task_type"] != family or config["timing"] != "middle":
            raise ValueError(f"Wrong executed configuration for {raw_id}")
        if condition == "oracle":
            if is_strong:
                level = "strong"
                approval = "strong_relation_and_grounding_review_bound_by_preparation"
            elif family == "cancel_task":
                level, approval = "not_applicable", "cancel_has_no_goal_pair"
            else:
                if family == "replacement":
                    decision, candidate = repl_decisions[raw_id], replacement[raw_id]
                    remote = candidate["updated_goal"]
                    update = config["events"][0]["task_update"]["new_goal"]
                else:
                    decision, candidate = sub_decisions[raw_id], subgoal[raw_id]
                    remote = candidate["inserted_goal"]
                    update = config["events"][0]["task_update"]["subgoal"]
                    for name in ("config.json", "events.json", "trajectory.json"):
                        if files[f"private/{name}"] != candidate["evidence_files"][name]["sha256"]:
                            raise ValueError(f"Subgoal label does not bind the executed {name}")
                if (decision["decision"] != "approve" or candidate["initial_goal"] != config["initial_goal"]
                        or candidate["base_task_id"] != config["base_task_id"] or remote != update):
                    raise ValueError(f"Missing compatible relatedness approval for {raw_id}")
                level, approval = decision["relatedness_level"], "human_relatedness_review"
            cases.append({"case_id": case_id, "source_case_id": raw_id, "base_task_id": config["base_task_id"],
                          "scene_group": scene, "scene_id": config["scene_id"], "scene_sha256": config["scene_sha256"],
                          "task_type": family, "relatedness_level": level, "label_provenance": approval,
                          "timing": "middle", "split": "unassigned", "config": config})
        metrics = audit_row if is_strong else audit_row["metrics"]
        episodes.append({"episode_id": f"{case_id}/{condition}", "case_id": case_id, "condition": condition,
                         "accepted": audit_row["passed" if is_strong else "accepted"],
                         "task_success": audit_row["task_success"], "episode_dir": str(folder),
                         "files_sha256": files, "metrics": metrics})

    for row in middle["episodes"]:
        root = Path(row["source_run"]).resolve()
        folder = within(root, row["metrics"]["directory"])
        # The batch snapshot uses the batch root for newly executed tasks and
        # the original external run root for reused episodes.
        snapshot_root = next((path for path in middle_snapshots if path in folder.parents), None)
        if snapshot_root is None:
            raise ValueError(f"Episode is outside the audited source roots: {folder}")
        collect(row, folder, members(folder, middle_snapshots[snapshot_root], snapshot_root), is_strong=False)
    for row in pilot["episodes"]:
        folder = within(strong_run, row["directory"])
        collect(row, folder, members(folder, strong_before["strong_run"], strong_run), is_strong=True)
    bases, coverage = inventory(cases, episodes)
    sources = [args.middle_audit / "audit.json", args.middle_audit / "source_bindings.json",
               args.strong_audit / "audit.json", strong_prepared / "manifest.json",
               args.replacement_candidates / "manifest.json", args.replacement_review / "review.json", sub_path, sub_review]
    if snapshot(strong_roots) != strong_before or snapshot(middle_roots) != middle_before:
        raise ValueError("Audited source trees changed during collection")
    output.mkdir(parents=True)
    write_lines(output / "base_tasks.jsonl", bases)
    write_lines(output / "private/cases.jsonl", cases)
    write_lines(output / "private/episodes.jsonl", episodes)
    write_json(output / "coverage.json", coverage)
    relative_files = ("base_tasks.jsonl", "private/cases.jsonl", "private/episodes.jsonl", "coverage.json")
    write_json(output / "manifest.json", {"kind": KIND, "visibility": "evaluator_only",
               "source_files": {str(path.resolve()): sha256(path) for path in sources},
               "output_files": {name: sha256(output / name) for name in relative_files},
               "code_sha256": sha256(Path(__file__))})
    print(coverage)
    return 0


def verify(directory):
    """Check an exported index and its referenced episode bytes without Habitat."""
    manifest = read_json(directory / "manifest.json")
    if manifest["kind"] != KIND or manifest["visibility"] != "evaluator_only":
        raise ValueError("Unknown collection package")
    for name, expected in manifest["output_files"].items():
        if sha256(within(directory, name)) != expected:
            raise ValueError(f"Changed collection index: {name}")
    for name, expected in manifest["source_files"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Changed collection source: {name}")
    from scripts.benchmark_splits import read_jsonl
    episodes = read_jsonl(directory / "private/episodes.jsonl")
    bases, coverage = inventory(read_jsonl(directory / "private/cases.jsonl"), episodes)
    if coverage != read_json(directory / "coverage.json") or bases != read_jsonl(directory / "base_tasks.jsonl"):
        raise ValueError("Collection counts or base task identity changed")
    for row in episodes:
        root = Path(row["episode_dir"])
        actual = snapshot({"episode": root})["episode"]
        if actual != row["files_sha256"]:
            raise ValueError(f"Changed or missing episode evidence: {row['episode_id']}")
    print(f"Verified {len(episodes)} episodes and {coverage['cases']} cases")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--output-dir", required=True, type=Path)
    for name, default in DEFAULTS.items():
        builder.add_argument("--" + name.replace("_", "-"), type=Path, default=PROJECT_ROOT / "outputs" / default)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        return build(args) if args.command == "build" else verify(args.input_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"Collection operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
