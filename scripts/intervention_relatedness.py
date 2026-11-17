"""Prepare and review private relatedness candidates without running navigation.

The first revision adapts the complete reviewed replacement roster at middle
timing. It does not invent room relations, geometry measurements or human
approvals. Pair entries describe a future experiment, not executable runs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from mivln.language_annotations import person
from scripts import benchmark_navigation as source
from scripts.benchmark_manifest import read_json, sha256, within
from scripts.hssd_visual_dataset import disjoint_output
from scripts.public_inputs import write_json


KIND = "replacement_relatedness_candidates_v1"
LEVELS = ("strong", "weak", "independent")
LOCKED = ("case_id", "candidate_sha256")
FIELDS = (*LOCKED, "decision", "relatedness_level", "reviewer",
          "relatedness_rationale", "semantic_relation", "spatial_relation")
DECISIONS = ("pending", "approve", "reject", "needs_revision")


def digest(value):
    """Bind a review to every candidate field, not just its display identity."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def build_candidates(directory):
    """Validate reviewed origins and retain all middle replacement cases.

    Geometry below is explicitly a private configuration target, not an actual
    intervention pose or a measured distance. Those require later replay.
    """
    directory = directory.resolve()
    manifest = source.load_preparation(directory)
    selected = [row for row in source.select_cases(manifest, "all", "middle")
                if row["task_type"] == "replacement"]
    if not selected or len(selected) != manifest["base_tasks"]:
        raise ValueError("Expected one middle replacement candidate per base task")
    candidates = []
    for row in selected:
        if not row["prepared"]:
            raise ValueError(f"Unprepared source case: {row['case_id']}")
        config_path = within(directory, row["config"])
        config = read_json(config_path)
        if len(config["events"]) != 1:
            raise ValueError("Relatedness v1 requires exactly one intervention")
        event = config["events"][0]
        update = event["task_update"]
        if update["operation"] != "replace_goal":
            raise ValueError("Only replacement semantics are supported in this revision")
        old, new = config["initial_goal"], update["new_goal"]
        if old == new or update["old_goal"] != old:
            raise ValueError("Replacement must bind two distinct verified goals")
        base = f"{config['scene_id']}@{config['scene_sha256']}/{row['source_task_id']}"
        candidates.append({
            "case_id": row["case_id"], "base_task_id": base,
            "scene_id": config["scene_id"], "scene_sha256": config["scene_sha256"],
            "task_type": "replacement", "timing": "middle",
            "source_config": str(config_path), "source_config_sha256": sha256(config_path),
            "initial_instruction": config["initial_instruction"],
            "intervention_instruction": event["utterance"],
            "initial_goal": old, "updated_goal": new,
            "goal_metadata": config["goal_metadata"],
            "goal_positions_private": config["goal_positions"],
            "trigger": event["trigger"], "seed": config["seed"],
            "max_steps": config["max_steps"],
            "actual_intervention_pose": None,
            "updated_goal_geodesic_distance_m": None,
            "reference_route_overlap": None,
            "updated_goal_previously_seen": None,
            "geometry_evidence_status": "pending_replay",
            "pair_id": f"{base}/replacement/middle/continue_vs_restart",
            "pair_conditions": ["continue", "restart_memory"],
            "pair_status": "identity_only_not_executable",
        })
    if len({row["case_id"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate source candidate identities")
    return candidates


def worksheet(candidates):
    return [{"case_id": row["case_id"], "candidate_sha256": digest(row),
             "decision": "pending", "relatedness_level": "unassigned",
             "reviewer": "", "relatedness_rationale": "",
             "semantic_relation": "", "spatial_relation": ""} for row in candidates]


def review_rows(candidates, rows):
    """Import a full roster, allowing explicit rejections and pending cases.

    An approved level is a human classification only. It does not certify the
    missing replay measurements, language changes, or future paired execution.
    """
    expected = {row["case_id"]: row for row in worksheet(candidates)}
    if any(set(row) != set(FIELDS) or any(type(v) is not str for v in row.values()) for row in rows):
        raise ValueError("Keep every CSV column and cell, including empty strings")
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Keep every candidate exactly once; do not drop failures")
    by_id = {row["case_id"]: row for row in rows}
    result = []
    for identity, original in expected.items():
        row = by_id[identity]
        if any(row[key] != original[key] for key in LOCKED):
            raise ValueError(f"{identity}: candidate binding changed")
        if row["decision"] not in DECISIONS:
            raise ValueError(f"{identity}: unknown decision")
        if row["decision"] == "pending":
            if row != original:
                raise ValueError(f"{identity}: filled rows need an explicit decision")
        else:
            person(row["reviewer"])
            if not row["relatedness_rationale"].strip():
                raise ValueError(f"{identity}: explain the classification or rejection")
            if row["relatedness_level"] not in (*LEVELS, "unassigned"):
                raise ValueError(f"{identity}: unknown relatedness level")
            if row["decision"] == "approve":
                if row["relatedness_level"] not in LEVELS:
                    raise ValueError(f"{identity}: approval requires a relatedness level")
                if not all(row[key].strip() for key in ("semantic_relation", "spatial_relation")):
                    raise ValueError(f"{identity}: document semantic and spatial relations separately")
        result.append(dict(row))
    if all(row["decision"] == "pending" for row in result):
        raise ValueError("No explicit classification decisions were supplied")
    return result


def report(candidates, reviews):
    """Keep the full denominator; class balance alone is not release readiness."""
    return {
        "planned_candidates": len(candidates),
        "base_tasks": len({row["base_task_id"] for row in candidates}),
        "scenes": len({row["scene_id"] for row in candidates}),
        "by_decision": {key: sum(row["decision"] == key for row in reviews) for key in DECISIONS},
        "approved_by_level": {key: sum(row["decision"] == "approve" and row["relatedness_level"] == key
                                       for row in reviews) for key in LEVELS},
        "geometry_evidence_status": "pending_replay", "paired_runs_executed": 0,
        "release_ready": False,
    }


def prepare(args):
    candidates = build_candidates(args.input_dir)
    output = disjoint_output(args.output_dir, [args.input_dir])
    output.mkdir(parents=True)
    reviews = worksheet(candidates)
    write_json(output / "manifest.json", {
        "kind": KIND, "schema_version": 1,
        "source_preparation": str(args.input_dir.resolve()),
        "source_manifest_sha256": sha256(args.input_dir / "manifest.json"),
        "candidates": candidates,
    })
    with (output / "relatedness_sheet.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(reviews)
    write_json(output / "coverage.json", report(candidates, reviews))
    print(f"Prepared {len(candidates)} unassigned candidates; no navigation or classification executed")


def review(args):
    manifest_path = args.input_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("kind") != KIND or manifest.get("schema_version") != 1:
        raise ValueError("Unknown candidate package")
    origin = Path(manifest["source_preparation"])
    if sha256(origin / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source preparation changed")
    candidates = build_candidates(origin)
    if candidates != manifest["candidates"]:
        raise ValueError("Candidates differ from validated reviewed origins")
    with args.sheet.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Keep the generated CSV header")
        decisions = review_rows(candidates, list(reader))
    output = disjoint_output(args.output_dir, [args.input_dir, origin, args.sheet])
    output.mkdir(parents=True)
    from datetime import datetime, timezone
    write_json(output / "review.json", {
        "kind": "replacement_relatedness_review_v1",
        "candidate_manifest": str(manifest_path.resolve()),
        "candidate_manifest_sha256": sha256(manifest_path),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reviewers_are_claimed_identities": True, "decisions": decisions,
    })
    write_json(output / "coverage.json", report(candidates, decisions))
    print("Saved classification declarations; geometry and paired execution remain pending")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "review"):
        command = commands.add_parser(name)
        command.add_argument("--input-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "review":
            command.add_argument("--sheet", type=Path, required=True)
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else review)(args)
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        print(f"Relatedness operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
