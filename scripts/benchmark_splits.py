"""Create and validate scene-disjoint Benchmark split assignments.

This entry only prepares split metadata. It never runs navigation, changes
source inventories, or assigns scenes automatically: a maintainer must review
and fill the generated CSV before validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


SPLITS = ("train", "dev", "test")
CSV_COLUMNS = ("scene_group", "scene_id", "scene_sha256", "split")


def read_jsonl(path: Path) -> list[dict]:
    """Load JSONL while reporting the exact line with malformed inventory."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: each inventory row must be an object")
        rows.append(row)
    return rows


def scene_roster(base_rows: list[dict]) -> list[dict]:
    """Collapse repeated base-task scene fields into a verified scene roster."""
    scenes = {}
    for row in base_rows:
        scene_group = row.get("scene_group")
        scene_id = row.get("scene_id")
        scene_sha256 = row.get("scene_sha256")
        # Missing identity cannot safely be treated as a shared scene: doing so
        # could silently mix unknown scenes in one split.
        if not all(isinstance(value, str) and value.strip()
                   for value in (scene_group, scene_id, scene_sha256)):
            raise ValueError(f"Base task {row.get('base_task_id')} has incomplete scene identity")
        identity = (scene_group, scene_id, scene_sha256)
        previous = scenes.get(scene_group)
        if previous is not None and previous != identity:
            raise ValueError(f"Scene group {scene_group!r} maps to conflicting scene identities")
        scenes[scene_group] = identity
    if not scenes:
        raise ValueError("Inventory has no base tasks; cannot prepare scene splits")
    return [dict(zip(CSV_COLUMNS[:3], identity), split="unassigned")
            for identity in sorted(scenes.values())]


def write_template(inventory_dir: Path, output_dir: Path) -> int:
    """Write one editable assignment row per scene to a fresh output folder."""
    inventory_dir = inventory_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh --output-dir: {output_dir}")
    if output_dir == inventory_dir or output_dir in inventory_dir.parents or inventory_dir in output_dir.parents:
        raise ValueError("Output directory must be disjoint from the source inventory")
    roster = scene_roster(read_jsonl(inventory_dir / "base_tasks.jsonl"))
    output_dir.mkdir(parents=True)
    with (output_dir / "scene_assignments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(roster)
    report = {
        "kind": "scene_split_template_v1",
        "source_inventory": str(inventory_dir),
        "scene_count": len(roster),
        "split_status": "unassigned",
        "note": "Review scene identities, then assign whole scenes to train/dev/test; do not split base tasks from one scene across sets.",
    }
    (output_dir / "template_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def validate_assignments(base_rows: list[dict], assignment_rows: list[dict]) -> dict:
    """Check complete scene-level assignment and reject a fake one-scene split."""
    expected = {row["scene_group"]: row for row in scene_roster(base_rows)}
    actual = {}
    errors = []
    for number, row in enumerate(assignment_rows, 2):
        group = row.get("scene_group")
        if group in actual:
            errors.append(f"CSV row {number}: duplicate scene_group {group!r}")
            continue
        if group not in expected:
            errors.append(f"CSV row {number}: unknown scene_group {group!r}")
            continue
        identity = {key: row.get(key, "") for key in CSV_COLUMNS[:3]}
        if identity != {key: expected[group][key] for key in CSV_COLUMNS[:3]}:
            errors.append(f"CSV row {number}: scene identity changed for {group!r}")
        split = row.get("split", "")
        if split not in SPLITS:
            errors.append(f"CSV row {number}: split must be one of {', '.join(SPLITS)}")
        actual[group] = split
    for group in sorted(set(expected) - set(actual)):
        errors.append(f"Missing scene assignment: {group}")
    for group in sorted(set(actual) - set(expected)):
        errors.append(f"Unexpected scene assignment: {group}")

    used = set(actual.values()) - {"", "unassigned"}
    if len(expected) < 2:
        errors.append("At least two distinct scenes are required for a scene-disjoint split")
    if not {"train", "test"}.issubset(used):
        errors.append("A valid split must assign at least one scene to both train and test")
    return {
        "kind": "scene_split_validation_v1",
        "scene_count": len(expected),
        "assigned_scene_count": sum(value in SPLITS for value in actual.values()),
        "by_split": {split: sum(value == split for value in actual.values()) for split in SPLITS},
        "scene_disjoint": not errors,
        "passed": not errors,
        "errors": errors,
        "split_status": "frozen" if not errors else "not_frozen",
    }


def validate(inventory_dir: Path, assignments: Path, output_dir: Path) -> int:
    """Validate a human-edited CSV and preserve the result in a new directory."""
    inventory_dir = inventory_dir.resolve()
    assignments = assignments.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh --output-dir: {output_dir}")
    for source in (inventory_dir, assignments):
        if output_dir == source or output_dir in source.parents or source in output_dir.parents:
            raise ValueError(f"Output directory must be disjoint from input: {source}")
    with assignments.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"CSV header must be exactly: {','.join(CSV_COLUMNS)}")
        rows = list(reader)
    report = validate_assignments(read_jsonl(inventory_dir / "base_tasks.jsonl"), rows)
    output_dir.mkdir(parents=True)
    (output_dir / "split_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="Create a scene-level assignment CSV")
    template.add_argument("--inventory-dir", type=Path, required=True,
                          help="Completed benchmark audit directory containing base_tasks.jsonl")
    template.add_argument("--output-dir", type=Path, required=True)
    check = commands.add_parser("validate", help="Validate a reviewed scene assignment CSV")
    check.add_argument("--inventory-dir", type=Path, required=True)
    check.add_argument("--assignments", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            return write_template(args.inventory_dir, args.output_dir)
        return validate(args.inventory_dir, args.assignments, args.output_dir)
    except (OSError, ValueError, csv.Error) as error:
        print(f"benchmark_splits: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
