"""Audit assets for every locally present HSSD scene configuration, without Habitat.

This checks declared asset availability using the existing single-scene checker.
It does not render, generate navmeshes, certify GLB payloads, or claim that local
scene configurations cover all scenes in the official dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import sys

from scripts.benchmark_manifest import PROJECT_ROOT, read_json, sha256
from scripts.check_hssd import inspect_assets, LFS_POINTER
from scripts.hssd_visual_dataset import disjoint_output
from scripts.public_inputs import write_json


def required_annotation_files(root, scene_id):
    """Benchmark collection requires semantics/metadata that smoke treats as optional.

    Missing annotations are reported separately from unavailable geometry;
    category conflicts themselves do not make an asset unavailable.
    """
    return [root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json",
            root / "metadata" / "objects.json",
            root / "metadata" / "hssd_obj_semantics_condensed.csv"]


def annotation_issues(root, scene_id):
    issues = []
    for path in required_annotation_files(root, scene_id):
        try:
            with path.open("rb") as stream:
                header = stream.read(80)
            if not header or header.startswith(LFS_POINTER):
                raise ValueError("empty annotation or Git LFS pointer")
            if path.suffix == ".json" and not isinstance(read_json(path), dict):
                raise ValueError("annotation JSON must be an object")
            if path.suffix == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    fields = csv.DictReader(stream).fieldnames or []
                if "Object Hash" not in fields or not any("CONDENSED" in key for key in fields):
                    raise ValueError("category CSV lacks expected header fields")
        except (OSError, ValueError, UnicodeError) as error:
            issues.append({"path": str(path), "error": str(error)})
    return issues


def summarize(rows, unchanged):
    counts = Counter(row["status"] for row in rows)
    return {"planned_scene_configurations": len(rows),
            "by_status": {key: counts[key] for key in ("complete", "incomplete", "error", "not_checked")},
            "all_checked": bool(rows) and counts["not_checked"] == 0,
            "all_assets_complete": bool(rows) and counts["complete"] == len(rows) and unchanged is True,
            "scene_roster_unchanged": unchanged,
            "scope": "locally present scene configurations; rigid-scene declared assets and Benchmark annotation availability",
            "gpu_used": False, "rendering_verified": False, "navmesh_verified": False,
            "binary_validation": "existing checker tests file headers, not complete GLB decoding",
            "official_dataset_coverage": "not_certified"}


def run(args, inspector=inspect_assets):
    dataset_config = args.dataset_config.resolve()
    root = dataset_config.parent
    # Freeze the denominator before checking content; broken/LFS scenes remain.
    paths = sorted((root / "scenes").glob("*.scene_instance.json"))
    if not paths:
        raise ValueError("No local scene_instance.json configurations found")
    roster = {str(path): sha256(path) for path in paths}
    shared_before = {str(dataset_config): sha256(dataset_config)}
    rows = [{"scene_id": path.name.removesuffix(".scene_instance.json"),
             "scene_config": str(path), "scene_config_sha256": roster[str(path)],
             "status": "not_checked", "declared_instances": None, "resolved_instances": None,
             "checked_files": None, "asset_issues": [], "annotation_issues": [], "error": None}
            for path in paths]
    if len({row["scene_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate local scene identities")
    output = disjoint_output(args.output_dir, [root])
    output.mkdir(parents=True)
    write_json(output / "manifest.json", {"kind": "hssd_local_scene_asset_inventory_v1",
               "dataset_config": str(dataset_config), "dataset_config_sha256": shared_before[str(dataset_config)],
               "planned_scenes": [{key: row[key] for key in ("scene_id", "scene_config", "scene_config_sha256")} for row in rows],
               "code_bindings": {name: sha256(PROJECT_ROOT / name) for name in
                                 ("scripts/hssd_scene_inventory.py", "scripts/check_hssd.py")}})
    write_json(output / "scenes.json", rows)
    write_json(output / "summary.json", summarize(rows, None))
    for number, row in enumerate(rows, 1):
        try:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", row["scene_id"]):
                raise ValueError("Unsafe scene identifier")
            report, _, _, _ = inspector(dataset_config, row["scene_id"])
            notes = annotation_issues(root, row["scene_id"])
            row.update(status="complete" if report["passed"] and not notes else "incomplete",
                       declared_instances=report["declared_object_instances"],
                       resolved_instances=report["resolved_object_instances"], checked_files=report["checked_files"],
                       asset_issues=report["issues"], annotation_issues=notes)
            write_json(output / "scenes" / (row["scene_id"] + ".json"),
                       {"asset_report": report, "required_annotation_issues": notes})
        except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError) as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
        # A stopped run retains not_checked slots, never a reduced denominator.
        write_json(output / "scenes.json", rows)
        write_json(output / "summary.json", summarize(rows, None))
        print(f"[{number}/{len(rows)}] {row['scene_id']}: {row['status']}", flush=True)
    current = sorted((root / "scenes").glob("*.scene_instance.json"))
    unchanged = (current == paths and all(sha256(Path(path)) == expected for path, expected in roster.items())
                 and all(sha256(Path(path)) == expected for path, expected in shared_before.items()))
    summary = summarize(rows, unchanged)
    write_json(output / "summary.json", summary)
    fields = ("scene_id", "status", "declared_instances", "resolved_instances", "checked_files", "error")
    with (output / "scenes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    affected = defaultdict(set)
    for row in rows:
        for issue in row["asset_issues"] + row["annotation_issues"]:
            affected[(issue["path"], issue["error"])].add(row["scene_id"])
    write_json(output / "issues.json", [{"path": path, "error": error, "affected_scenes": sorted(scenes)}
                                       for (path, error), scenes in sorted(affected.items())])
    print(summary)
    return 0 if summary["all_assets_complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path,
                        default=PROJECT_ROOT / "data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    try:
        return run(parser.parse_args())
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Scene inventory failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
