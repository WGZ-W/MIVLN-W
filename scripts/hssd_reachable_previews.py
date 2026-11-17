"""Prepare and render close-ups for the complete reachable instance roster.

Prepare uses CPU navmesh sampling to reconstruct the inventory; preview uses
the existing Habitat renderer. Neither operation approves targets or runs a
navigation episode. Output directories must be fresh and separate from inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from urllib.parse import quote
import os

from scripts import hssd_reachability as reach
from scripts import hssd_targets as targets
from scripts.benchmark_manifest import PROJECT_ROOT, read_json, sha256
from scripts.hssd_visual_dataset import disjoint_output
from scripts.public_inputs import write_json


def make_targets(rows, source_instances, runtime_objects):
    """Bridge unique identities and measured stop points into pending targets."""
    sources = {obj["source_instance_index"]: obj for obj in source_instances}
    objects = {obj["object_id"]: obj for obj in runtime_objects}
    if len(sources) != len(source_instances) or len(objects) != len(runtime_objects):
        raise ValueError("Duplicate source/runtime identities")
    result = []
    for row in rows:
        if row["status"] != "reachable":
            continue
        obj = objects[row["runtime_object_id"]]
        original = sources[row["source_instance_index"]]
        if row["template_id"] != obj["template_id"] or obj["template_id"] != original["template_id"]:
            raise ValueError("Reachable source/runtime template mismatch")
        geometry = row["geometry"]
        result.append({"target_id": f"target_{row['source_instance_index']:03d}",
                       "source_instance_index": row["source_instance_index"],
                       "template_id": row["template_id"], "category": row["category"],
                       "semantic_id": obj["semantic_id"], "display_name": obj.get("display_name"),
                       "world_aabb": obj["world_aabb"], "stop_position": geometry["stop_position"],
                       "geometry": {key: value for key, value in geometry.items() if key != "stop_position"},
                       "review": {"status": "pending"}})
    if not result or len({row["target_id"] for row in result}) != len(result):
        raise ValueError("Require a nonempty unique reachable target roster")
    return sorted(result, key=lambda row: row["target_id"])


def prepare(args):
    seal = read_json(args.input_dir / "source_bindings.json")
    summary = read_json(args.input_dir / "summary.json")
    saved = read_json(args.input_dir / "instances.json")
    if summary.get("passed") is not True or saved.get("kind") != "hssd_instance_reachability_v1":
        raise ValueError("Require a successful full-scene reachability inventory")
    for name, expected in seal["code_bindings"].items():
        if sha256(PROJECT_ROOT / name) != expected:
            raise ValueError(f"Inventory producer changed: {name}")
    for path, expected in seal["files"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Inventory source changed: {path}")
    source = args.source_dir.resolve()
    config = read_json(args.base_config)
    # Explicit input flags cannot substitute an unrelated scene or smoke table.
    for path in (args.base_config.resolve(), source / "objects.json", source / "source_objects.json"):
        if seal["files"].get(str(path)) != sha256(path):
            raise ValueError(f"Input is not bound by inventory: {path}")
    mesh, provenance = targets.load_geometry(config)
    source_instances = read_json(source / "source_objects.json")
    objects = read_json(source / "objects.json")
    reconstructed = reach.inventory(source_instances, objects,
                                   lambda obj: reach.sample_stop(mesh, obj, config["start_position"]))
    if reconstructed != saved["instances"] or len(reconstructed) != summary["planned_instances"]:
        raise ValueError("Saved reachability rows differ from native reconstruction")
    roster = make_targets(reconstructed, source_instances, objects)
    if len(roster) != summary["by_status"]["reachable"]:
        raise ValueError("Reachable target denominator changed")
    files = dict(seal["files"])
    for name in ("source_bindings.json", "summary.json", "instances.json"):
        path = (args.input_dir / name).resolve()
        files[str(path)] = sha256(path)
    output = disjoint_output(args.output_dir, [args.input_dir, source, *map(Path, files)])
    config["semantic_sensor"] = True
    catalog = {"schema_version": 1, "kind": "hssd_target_catalog", "scene_config": config,
               "sources": [{"path": path, "sha256": value} for path, value in files.items()],
               "targets": roster, "rejected": [],
               "note": "Full reachable roster; unknown/conflicting class labels require visual review. No target is auto-approved."}
    targets.approved_targets(catalog, minimum=0)
    output.mkdir(parents=True)
    write_json(output / "catalog.json", catalog)
    write_json(output / "summary.json", {"planned_source_instances": len(reconstructed),
               "reachable_targets": len(roster), "by_category": dict(Counter(row["category"] for row in roster)),
               "rendered_targets": 0, "human_approved": 0, "scene_provenance": provenance})
    print(f"Prepared {len(roster)} pending close-up targets; rendering not started")
    return 0


def preview(args):
    catalog = read_json(args.catalog)
    targets.approved_targets(catalog, minimum=0)
    targets.check_sources(catalog["sources"])
    output = disjoint_output(args.output_dir, [args.catalog.resolve(),
                              *[targets.project_path(item["path"]) for item in catalog["sources"]]])
    failure = None
    try:
        targets.preview(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        # Preserve the full planned roster if the existing renderer stops early.
        # A partly written folder is not counted as a completed evidence bundle.
        failure = f"{type(error).__name__}: {error}"
    if not output.exists():
        raise ValueError(failure or "Renderer did not create output")
    evidence = read_json(output / "preview.json") if (output / "preview.json").exists() else None
    completed = set(evidence["targets"]) if evidence else set()
    lines = ["# 全场景可达目标近景", "",
             "近景在可达停止点离线拍摄，不是导航实验。所有实例待人工审核。", "",
             "语义图是类别标签；同类像素或标签 0 的背景不能证明目标实例可见。", ""]
    for row in catalog["targets"]:
        identity = row["target_id"]
        lines.extend([f"## {identity} · {row['category']}", "",
                      f"源实例 {row['source_instance_index']}；距起点 {row['geometry']['anchor_distance_m']:.2f} m。", ""])
        if identity in completed:
            targets.check_preview_record(evidence["targets"][identity], identity)
            relative = quote(os.path.relpath(output / identity / "rgb.png", output), safe="/.")
            lines.extend([f"![{identity}]({relative})", ""])
        else:
            lines.extend(["渲染未完成，不计为已生成近景。", ""])
    unchanged = all(sha256(targets.project_path(item["path"])) == item["sha256"] for item in catalog["sources"])
    (output / "review_index.md").write_text("\n".join(lines), encoding="utf-8")
    passed = failure is None and len(completed) == len(catalog["targets"]) and unchanged
    write_json(output / "render_summary.json", {"passed": passed,
               "planned_targets": len(catalog["targets"]), "rendered_targets": len(completed),
               "pending_target_ids": [row["target_id"] for row in catalog["targets"] if row["target_id"] not in completed],
               "source_files_unchanged": unchanged, "error": failure, "human_approved": 0})
    print(f"Rendered {len(completed)}/{len(catalog['targets'])}; index: {output / 'review_index.md'}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--input-dir", type=Path, required=True)
    prepare_parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "outputs/hssd_smoke_001")
    prepare_parser.add_argument("--base-config", type=Path, default=PROJECT_ROOT / "configs/simulator/hssd_navigation_001.json")
    preview_parser = commands.add_parser("preview")
    preview_parser.add_argument("--catalog", type=Path, required=True)
    preview_parser.add_argument("--gpu-id", type=int, default=0)
    for command in (prepare_parser, preview_parser):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        return prepare(args) if args.command == "prepare" else preview(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"Reachable preview operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
