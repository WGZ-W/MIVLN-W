"""Propose strong-relatedness replacements and check distance matching.

This is private collection planning, not executable navigation preparation.
Category rules suggest relations; people must verify functional/spatial context,
target identity and language before any suggestion becomes an approved task.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
from urllib.parse import quote

from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import PROJECT_ROOT, read_json, sha256
from scripts.hssd_targets import candidate_identity, check_preview_record, validate_catalog
from scripts.hssd_visual_backend import NativeReplay
from scripts.hssd_visual_dataset import disjoint_output
from scripts.intervention_relatedness import digest
from scripts.public_inputs import write_json
from scripts.relatedness_geometry import measure_prefix


# Deliberately small collection heuristics. Spatial proximity alone cannot
# establish functional relatedness, and a shared category need not imply it.
FUNCTIONAL_PAIRS = {frozenset(pair) for pair in
                    (("table", "bench"), ("table", "chair"), ("table", "couch"), ("counter", "table"))}


def image_link(path, directory):
    """Markdown resolves relative paths from the index, not the shell's cwd."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise ValueError(f"Preview image is missing: {path}")
    return quote(os.path.relpath(path.resolve(), directory.resolve()), safe="/.")


def write_review_index(output, suggestions):
    """Rebuild only the advisory index; candidate and provenance files stay intact."""
    index = ["# 强相关候选审核索引", "",
             "全部为待审核建议。距离匹配不等于强相关或完整难度匹配；原有分类保持不变。", "",
             "工作表仅用于记录，审核导入和新任务执行配置尚未实现。", "",
             "使用 Markdown 预览查看下方图片；图片目录需与项目一同保留。", ""]
    for row in sorted(suggestions, key=lambda item: (not item["distance_match"], item["proposal_id"])):
        # Preview bundles keep each instance in a sibling target_ID directory.
        # This also repairs old proposal versions without rerunning Habitat.
        target_path = Path(row["target_rgb"])
        original_path = target_path.parent.parent / row["initial_goal"] / "rgb.png"
        original = image_link(original_path, output)
        target = image_link(target_path, output)
        index.extend([f"## {row['proposal_id']}", "",
                      f"初始指令：{row['initial_instruction']}", "",
                      f"候选类别：{row['target_category']}；建议依据：{row['relation_hint']}。", "",
                      "原目标近景：", "", f"![原目标近景]({original})", "",
                      "新目标近景：", "", f"![新目标近景]({target})", "",
                      f"新目标审核状态：{row['target_review_status']}；介入文本草稿：{row['intervention_draft'] or '待根据图片编写唯一描述'}", "",
                      f"介入后距离：{row['distance_to_proposed_stop_m']} m；原案例：{row['baseline_distance_m']} m；距离匹配：{row['distance_match']}。", ""])
    (output / "review_index.md").write_text("\n".join(index), encoding="utf-8")


def repair_index(directory):
    package = read_json(directory / "proposals.json")
    if package.get("kind") != "strong_relatedness_proposals_v1":
        raise ValueError("Unknown strong candidate package")
    write_review_index(directory, package["proposals"])
    print(f"Rebuilt {directory / 'review_index.md'}; no geometry calculation executed")
    return 0


def relation_hint(original, target):
    if original["target_id"] == target["target_id"]:
        return None
    a, b = original["category"], target["category"]
    if a == b:
        return "same_category_instance_correction"
    if frozenset((a, b)) in FUNCTIONAL_PAIRS:
        return "possible_functional_relation_requires_context"
    return None


def difficulty(distance, baseline, absolute_tolerance, relative_tolerance):
    """Require both absolute and relative closeness; never divide by zero.

    These are configurable pilot sampling tolerances, not a frozen Benchmark
    difficulty definition. They only compare distance at a shared prefix pose.
    """
    if distance is None or not math.isfinite(distance) or distance < 0:
        return {"distance_match": False, "distance_delta_m": None,
                "relative_distance_delta": None, "status": "unreachable"}
    delta = abs(distance-baseline)
    relative = delta / baseline if baseline > 0 else (0.0 if delta == 0 else None)
    matched = delta <= absolute_tolerance and relative is not None and relative <= relative_tolerance
    return {"distance_match": matched, "distance_delta_m": delta,
            "relative_distance_delta": relative, "status": "matched" if matched else "distance_mismatch"}


def run(args):
    for value in (args.absolute_tolerance_m, args.relative_tolerance):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Matching tolerances must be finite and nonnegative")
    geometry = read_json(args.geometry_dir / "geometry.json")
    summary = read_json(args.geometry_dir / "summary.json")
    seal = read_json(args.geometry_dir / "source_bindings.json")
    if (geometry.get("kind") != "relatedness_native_geometry_v1" or summary.get("passed") is not True
            or summary["verified_candidates"] != summary["planned_candidates"]
            or len(geometry["rows"]) != summary["planned_candidates"]):
        raise ValueError("Require complete successful geometry evidence")
    for name, expected in seal["code_bindings"].items():
        if sha256(PROJECT_ROOT / name) != expected:
            raise ValueError(f"Geometry producer code changed: {name}")
    roots = {name: Path(path) for name, path in seal["roots"].items()}
    if seal.get("source_files_unchanged") is not True or snapshot(roots) != seal["before"]:
        raise ValueError("Geometry source evidence changed")
    catalog, preview = read_json(args.catalog), read_json(args.previews)
    validate_catalog(catalog)
    if preview["candidate_identity"] != candidate_identity(catalog):
        raise ValueError("Preview belongs to another catalog")
    targets = {target["target_id"]: target for target in catalog["targets"]}
    if len(targets) != len(catalog["targets"]):
        raise ValueError("Duplicate catalog target identities")
    roots["geometry_package"] = args.geometry_dir.resolve()
    # Bind the entire evidence directory: adding or modifying a near-view after
    # proposal generation invalidates the collection version.
    roots["target_previews"] = args.previews.resolve().parent
    before = snapshot(roots)
    files = {str(path.resolve()): sha256(path) for path in (args.catalog, args.previews)}
    output = disjoint_output(args.output_dir, [*roots.values(), args.catalog, args.previews])
    suggestions, failures = [], []
    for base in geometry["rows"]:
        replay = None
        try:
            folder = Path(base["source_episode"])
            for name, expected in base["evidence_files"].items():
                if sha256(folder / name) != expected:
                    raise ValueError("Recorded native episode changed")
            config = read_json(folder / "config.json")
            scene = catalog["scene_config"]
            if any(config[key] != scene[key] for key in ("scene_id", "scene_sha256", "navmesh_sha256")):
                raise ValueError("Target catalog and trajectory use different scene/navmesh")
            replay = NativeReplay(config)
            measured = measure_prefix(config, read_json(folder / "trajectory.json"),
                                      read_json(folder / "events.json"), replay)
            if any(measured[key] != base[key] for key in measured):
                raise ValueError("Geometry evidence differs from reconstructed native replay")
            original = targets[config["initial_goal"]]
            for target in targets.values():
                hint = relation_hint(original, target)
                # Never duplicate an existing case under a new 'strong' label.
                if not hint or target["target_id"] == config["events"][0]["task_update"]["new_goal"]:
                    continue
                record = preview["targets"][target["target_id"]]
                check_preview_record(record, target["target_id"])
                rgb = next(item["path"] for item in record["files"] if Path(item["path"]).name == "rgb.png")
                description = target["review"].get("description", "")
                distance = replay.distance(target["stop_position"])
                if distance is not None and (not math.isfinite(distance) or distance < 0):
                    distance = None
                identity = f"{base['case_id']}__to_{target['target_id']}"
                suggestions.append({
                    "proposal_id": identity, "source_case_id": base["case_id"],
                    "base_task_id": base["base_task_id"], "scene_id": config["scene_id"],
                    "initial_goal": original["target_id"], "updated_goal": target["target_id"],
                    "relation_hint": hint, "proposed_relatedness": "strong",
                    "relatedness_status": "pending_human_review", "target_review_status": target["review"]["status"],
                    "target_category": target["category"], "target_rgb": rgb,
                    "initial_instruction": config["initial_instruction"],
                    "intervention_draft": f"Cancel the previous destination. Go to {description} instead." if description else None,
                    "language_status": "draft_requires_review" if description else "requires_visual_description",
                    "actual_intervention_pose": measured["actual_intervention_pose"],
                    "delivered_step": measured["delivered_step"], "max_steps": config["max_steps"],
                    "prefix_protocol": "reuse_exact_source_prefix_before_delivery",
                    "distance_to_proposed_stop_m": distance,
                    "baseline_relatedness": base["classification"]["relatedness_level"],
                    "baseline_distance_m": measured["updated_goal_geodesic_distance_m"],
                    **difficulty(distance, measured["updated_goal_geodesic_distance_m"],
                                 args.absolute_tolerance_m, args.relative_tolerance),
                    "pair_status": "collection_proposal_not_executable", "release_ready": False,
                })
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            failures.append({"case_id": base["case_id"], "error": str(error)})
        finally:
            if replay is not None:
                replay.close()
    unchanged = before == snapshot(roots) and all(sha256(Path(path)) == value for path, value in files.items())
    output.mkdir(parents=True)
    write_json(output / "proposals.json", {"kind": "strong_relatedness_proposals_v1", "proposals": suggestions})
    fields = ("proposal_id", "proposal_sha256", "decision", "reviewer", "relation_rationale",
              "spatial_evidence", "unique_target_description", "intervention_text")
    with (output / "proposal_review_sheet.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in suggestions:
            writer.writerow({"proposal_id": row["proposal_id"], "proposal_sha256": digest(row),
                             "decision": "pending"})
    write_review_index(output, suggestions)
    result = {"passed": unchanged and not failures and bool(suggestions),
              "planned_source_cases": len(geometry["rows"]), "proposed_candidates": len(suggestions),
              "distance_matched_proposals": sum(row["distance_match"] for row in suggestions),
              "absolute_tolerance_m": args.absolute_tolerance_m, "relative_tolerance": args.relative_tolerance,
              "source_files_unchanged": unchanged, "failures": failures,
              "approved_strong_candidates": 0, "release_ready": False,
              "difficulty_scope": "distance_only_same_prefix; language_and_visual_difficulty_unmatched",
              "review_import_status": "worksheet_only_import_not_implemented"}
    write_json(output / "summary.json", result)
    write_json(output / "source_bindings.json", {"files": files, "roots": {key: str(path) for key, path in roots.items()},
               "before": before, "source_files_unchanged": unchanged,
               "code_sha256": sha256(Path(__file__))})
    print(result)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-index", type=Path, help="Rebuild an existing package's image index without native replay")
    for name in ("geometry-dir", "catalog", "previews", "output-dir"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--absolute-tolerance-m", type=float, default=2.0)
    parser.add_argument("--relative-tolerance", type=float, default=.25)
    try:
        args = parser.parse_args()
        if args.repair_index is not None:
            return repair_index(args.repair_index)
        if any(getattr(args, name) is None for name in ("geometry_dir", "catalog", "previews", "output_dir")):
            parser.error("Provide --geometry-dir, --catalog, --previews and --output-dir, or --repair-index")
        return run(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"Strong candidate operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
