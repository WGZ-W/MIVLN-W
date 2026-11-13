"""CPU-only language drafting, immutable revisions, explicit review and export.

The whole annotation bundle is evaluator/annotator-private. Only a separately
constructed approved language corpus can be exported; it is not an online V
request or an executable navigation dataset. No model calls or GPU work occur.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

from mivln import language_annotations as model
from mivln.task_update import strict_json
from mivln.task_update_prompts import fingerprint
from scripts.benchmark_manifest import PROJECT_ROOT, sha256, within
from scripts.public_inputs import fresh_output, write_json, write_lines


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def collect_sources(source):
    """Rebuild deterministic origin data from frozen tasks and approved previews.

    This verifies identity/geometry evidence, not the new language. New approval
    remains pending even when a source object's old review was approved.
    """
    from scripts import hssd_dataset
    from scripts.hssd_targets import validate_catalog

    source, files = source.resolve(), {}
    def bind(path, expected=None, load=False):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Frozen source changed: {path}")
        if str(path) in files and files[str(path)] != digest:
            raise ValueError("Source changed while collecting evidence")
        files[str(path)] = digest
        return read_json(path) if load else {"path": str(path), "sha256": digest}

    manifest = bind(source / "manifest.json", load=True)
    hssd_dataset.validate_inputs(source, manifest)
    catalog = bind(manifest["catalog_source"]["path"], manifest["catalog_source"]["sha256"], True)
    approved = {target["target_id"]: target for target in validate_catalog(catalog)}
    for item in catalog["sources"]:
        bind(PROJECT_ROOT / item["path"], item["sha256"])
    targets = {}
    for identity, target in approved.items():
        preview = target["review"]["preview"]
        report = bind(PROJECT_ROOT / preview["path"], preview["sha256"], True)
        record = report["targets"][identity]
        rgb = None
        for item in record["files"]:
            binding = bind(PROJECT_ROOT / item["path"], item["sha256"])
            if Path(item["path"]).name == "rgb.png":
                rgb = binding
        if rgb is None:
            raise ValueError("Reviewed instance lacks actual RGB evidence")
        targets[identity] = {"description": target["review"]["description"],
            "template_id": target["template_id"], "source_instance_index": target["source_instance_index"],
            "stop_position": target["stop_position"], "rgb": rgb,
            "evidence_id": "ev_" + fingerprint([identity, rgb["sha256"]])[:20],
            "source_object_review": target["review"]}
    bases = []
    for task in manifest["tasks"]:
        config_path = within(source, task["base_config"])
        config = bind(config_path, task["base_config_sha256"], True)
        bind(within(source, task["reference_config"]), task["reference_config_sha256"])
        event = config["events"][0]
        if len(config["events"]) != 1 or event["task_update"]["operation"] != "replace_goal":
            raise ValueError("Source must be the canonical reviewed replacement roster")
        root, other = config["initial_goal"], event["task_update"]["new_goal"]
        if root == other or root not in targets or other not in targets:
            raise ValueError("Source goal pair does not bind two approved instances")
        for identity in (root, other):
            target, metadata = targets[identity], config["goal_metadata"][identity]
            if (config["goal_positions"][identity] != target["stop_position"]
                    or metadata["template_id"] != target["template_id"]
                    or metadata["source_instance_index"] != target["source_instance_index"]
                    or metadata["description"] != target["description"]):
                raise ValueError("Task differs from reviewed instance binding")
        if event["task_update"]["old_goal"] != root:
            raise ValueError("Initial/replacement source goals disagree")
        bases.append({"base_task_id": f"{config['scene_id']}@{config['scene_sha256']}/{task['task_id']}",
            "source_task_id": task["task_id"], "scene_id": config["scene_id"], "scene_sha256": config["scene_sha256"],
            "source_config": {"path": str(config_path), "sha256": files[str(config_path)]},
            "start_position": config["start_position"], "start_rotation_xyzw": config["start_rotation_xyzw"],
            "seed": config["seed"], "initial_goal": root, "paired_goal": other,
            "initial_instruction": config["initial_instruction"], "replacement_instruction": event["utterance"],
            "targets": {identity: targets[identity] for identity in (root, other)}})
    check_sources(files)
    return bases, files


def check_sources(files):
    for path, digest in files.items():
        if not Path(path).is_file() or sha256(path) != digest:
            raise ValueError(f"Source/evidence missing or changed: {path}")


def load_bundle(directory):
    """Validate every ancestor and replay the current authority against live sources.

    A revision must append events to its parent's history, never replace earlier
    declarations. The chain is checked without trusting editable CSV/review sheets;
    source assets are hashed once after the full lineage has been verified.
    """
    directory = directory.resolve()
    seal = read_json(directory / "manifest.json")
    if seal.get("kind") != model.KIND or seal.get("annotations_sha256") != sha256(directory / "annotations.json"):
        raise ValueError("Annotation bundle seal changed")
    dataset = read_json(directory / "annotations.json")
    model.replay(dataset)
    if any(seal.get(key) != dataset[key] for key in ("dataset_id", "revision_id", "revision_number")):
        raise ValueError("Bundle identity/revision mismatch")
    child, child_seal = dataset, seal
    seen = {(directory / "manifest.json").resolve()}
    while child_seal.get("parent") is not None:
        parent = child_seal["parent"]
        path = Path(parent["path"])
        if path.resolve() in seen or sha256(path) != parent["sha256"]:
            raise ValueError("Parent revision missing or changed")
        seen.add(path.resolve())
        ancestor = read_json(path)
        if sha256(path.parent / "annotations.json") != ancestor["annotations_sha256"]:
            raise ValueError("Parent annotation contents changed")
        parent_data = read_json(path.parent / "annotations.json")
        model.replay(parent_data)
        if (ancestor.get("kind") != model.KIND
                or any(ancestor.get(key) != parent_data[key] for key in ("dataset_id", "revision_id", "revision_number"))
                or parent_data["dataset_id"] != child["dataset_id"]
                or parent_data["revision_number"] + 1 != child["revision_number"]
                or parent_data["revision_id"] == child["revision_id"]):
            raise ValueError("Invalid parent revision lineage")
        if (any(child[key] != parent_data[key] for key in ("source_dir", "source_files", "base_tasks"))
                or len(child["events"]) <= len(parent_data["events"])
                or child["events"][:len(parent_data["events"])] != parent_data["events"]):
            raise ValueError("A revision must append events to its unchanged parent's history")
        child, child_seal = parent_data, ancestor
    if child["revision_number"] != 0 or child["events"]:
        raise ValueError("A revised bundle requires its parent binding")
    bases, files = collect_sources(Path(dataset["source_dir"]))
    if bases != dataset["base_tasks"] or files != dataset["source_files"]:
        raise ValueError("Origin bindings differ from the frozen source roster/evidence")
    return dataset


def write_bundle(dataset, output, parent=None):
    """Write a fresh private revision, seal its authority and create editable sheets."""
    output = fresh_output(output, parent or Path(dataset["source_dir"]))
    model.replay(dataset)
    check_sources(dataset["source_files"])
    output.mkdir(parents=True)
    write_json(output / "annotations.json", dataset)
    write_json(output / "manifest.json", {"kind": model.KIND, "dataset_id": dataset["dataset_id"],
        "revision_id": dataset["revision_id"], "revision_number": dataset["revision_number"],
        "annotations_sha256": sha256(output / "annotations.json"),
        "parent": {"path": str((parent / "manifest.json").resolve()), "sha256": sha256(parent / "manifest.json")} if parent else None,
        "code_bindings": {name: sha256(PROJECT_ROOT / name) for name in ("mivln/language_annotations.py", "scripts/language_annotations.py")}})
    with (output / "annotation_sheet.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=model.SHEET_COLUMNS)
        writer.writeheader()
        for record in dataset["records"]:
            writer.writerow({**{key: record[key] for key in model.SHEET_COLUMNS if key not in ("author", "note")}, "author": "", "note": ""})
    write_json(output / "review_decisions.json", model.review_template(dataset))
    write_json(output / "status.json", model.summary(dataset))
    evidence = {target["evidence_id"]: target for base in dataset["base_tasks"] for target in base["targets"].values()}
    lines = ["# 私有语言标注审核包", "", "这里只含待审或已声明审核的语言，不代表导航验收。不要作为 Agent 输入。", "",
        "先查看实例 RGB，再核对初始指令及其介入。approve 需填真实审核者、具体 note、全部 checklist 和 checked_evidence_ids。", "",
        "修改文本：编辑 annotation_sheet.csv 的 text/language，填写 author/note，再使用 revise 生成新版本。", "",
        "## 实例证据", ""]
    for identity, target in evidence.items():
        relative = os.path.relpath(target["rgb"]["path"], output)
        lines.extend([f"- `{identity}`：{target['description']} — [打开 RGB](<{relative}>)"])
    for record in dataset["records"]:
        lines.extend(["", f"## {record['annotation_id']}", "", f"类型：{record['task_type']}；状态：{record['review']['status']}；语言：{record['language']}",
            "", record["text"], "", f"私有语义绑定：`{json.dumps(record['semantic_label'], ensure_ascii=False)}`", "",
            "需核对证据：" + ", ".join(f"`{identity}`" for identity in record["required_evidence_ids"])])
    (output / "review_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def prepare(args):
    """Create pending template drafts from existing reviewed instance evidence."""
    source = args.source_dir.resolve()
    fresh_output(args.output_dir, source)
    bases, files = collect_sources(source)
    dataset = {"schema_version": 1, "kind": model.KIND, "dataset_id": uuid.uuid4().hex,
        "revision_id": uuid.uuid4().hex, "revision_number": 0, "created_at": now(),
        "source_dir": str(source), "source_files": files, "base_tasks": bases,
        "events": [], "records": model.seed_records(bases)}
    write_bundle(dataset, args.output_dir)
    print(f"Drafts: {len(bases)} base tasks, {len(dataset['records'])} annotations; all pending, no model/GPU")
    return 0


def revise(args):
    """Import a full-roster CSV, preserving labels and recording human edits."""
    dataset = load_bundle(args.input_dir)
    records, events = model.replay(dataset), []
    with args.sheet.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != model.SHEET_COLUMNS:
            raise ValueError("Use the generated annotation sheet header, without added fields")
        rows = list(reader)
    ids = [row.get("annotation_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(records):
        raise ValueError("The sheet must retain every planned annotation exactly once")
    for row in rows:
        model.exact(row, model.SHEET_COLUMNS, "CSV row")
        record = records[row["annotation_id"]]
        if any(row[key] != record[key] for key in ("base_task_id", "role", "task_type")):
            raise ValueError("Task bindings are immutable; edit only language/text/author/note")
        if row["text"] == record["text"] and row["language"] == record["language"] and not row["author"] and not row["note"]:
            continue
        events.append({"sequence": 0, "event_id": uuid.uuid4().hex, "at": now(), "action": "revise",
            "annotation_id": row["annotation_id"], "payload": {key: row[key] for key in ("language", "text", "author", "note")}})
    if not events:
        raise ValueError("No explicit text changes or human adoption declarations")
    changed = model.with_events(dataset, events, uuid.uuid4().hex, now())
    write_bundle(changed, args.output_dir, args.input_dir)
    print(f"Saved {len(events)} language edits in revision {changed['revision_number']}; affected reviews invalidated")
    return 0


def review(args):
    """Import explicit decisions bound to the current revision and text/context."""
    dataset = load_bundle(args.input_dir)
    decisions = read_json(args.decisions)
    model.exact(decisions, ("schema_version", "dataset_id", "revision_id", "decisions"), "Review file")
    if (type(decisions["schema_version"]) is not int or decisions["schema_version"] != 1
            or decisions["dataset_id"] != dataset["dataset_id"] or decisions["revision_id"] != dataset["revision_id"]):
        raise ValueError("Review file belongs to another dataset or revision")
    if type(decisions["decisions"]) is not list:
        raise ValueError("Review decisions must be a list")
    events, seen = [], set()
    for item in decisions["decisions"]:
        model.exact(item, ("annotation_id", "content_sha256", "context_sha256", "decision", "reviewer", "note",
                           "checks", "checked_evidence_ids"), "Review row")
        identity = item["annotation_id"]
        if identity in seen or identity not in {record["annotation_id"] for record in dataset["records"]}:
            raise ValueError("Duplicate or unknown review identity")
        seen.add(identity)
        if item["decision"] is None:
            if item["reviewer"] or item["note"] or item["checked_evidence_ids"] or item["checks"] != {key: False for key in model.CHECKS}:
                raise ValueError("Set an explicit decision for a filled review row")
            continue
        events.append({"sequence": 0, "event_id": uuid.uuid4().hex, "at": now(), "action": "review",
                       "annotation_id": identity, "payload": {key: value for key, value in item.items() if key != "annotation_id"}})
    if not events:
        raise ValueError("No explicit human decisions; template rows are not approvals")
    changed = model.with_events(dataset, events, uuid.uuid4().hex, now())
    write_bundle(changed, args.output_dir, args.input_dir)
    print(f"Saved {len(events)} explicit review decisions in a new revision")
    return 0


def export(args):
    """Gate the whole offline corpus; preserve the planned denominator on failure."""
    dataset = load_bundle(args.input_dir)
    report = model.summary(dataset)
    output = fresh_output(args.output_dir, args.input_dir)
    output.mkdir(parents=True)
    write_json(output / "export_report.json", report)
    write_lines(output / "private/planned_cases.jsonl", report["cases"])
    if not report["all_language_approved"]:
        print(f"Export blocked: {report['language_ready_cases']}/{report['planned_task_cases']} language-ready; report saved, no public corpus")
        return 1
    records = model.replay(dataset)
    bases = {base["base_task_id"]: base for base in dataset["base_tasks"]}
    public, private = [], []
    for case in report["cases"]:
        record = records[case["annotation_id"]]
        initial = records[record["initial_annotation_id"]]
        base = bases[record["base_task_id"]]
        identity = uuid.uuid4().hex
        public.append({"sample_id": identity, "initial_instruction": initial["text"],
            "initial_language": initial["language"], "intervention_message": record["text"],
            "intervention_language": record["language"]})
        private.append({"sample_id": identity, "base_task_id": record["base_task_id"], "scene_id": base["scene_id"],
            "scene_group": f"{base['scene_id']}@{base['scene_sha256']}", "leakage_group": record["base_task_id"],
            "task_type": record["task_type"], "task_update": record["semantic_label"],
            "source_config": base["source_config"], "initial_annotation_id": initial["annotation_id"],
            "intervention_annotation_id": record["annotation_id"], "dataset_id": dataset["dataset_id"],
            "revision_id": dataset["revision_id"], "split": None, "trigger_status": "not_frozen_for_new_release",
            "new_language_navigation_executed": False})
    write_lines(output / "public/language.jsonl", public)
    write_lines(output / "private/tasks.jsonl", private)
    write_json(output / "public/manifest.json", {"kind": "reviewed_language_corpus_v1", "samples": len(public),
        "content_sha256": sha256(output / "public/language.jsonl"), "online_agent_input": False,
        "release_ready": False, "navigation_executed": False})
    write_json(output / "private/annotation_source.json", {"manifest": str((args.input_dir / 'manifest.json').resolve()),
        "sha256": sha256(args.input_dir / "manifest.json"), "reviewers_are_claimed_identities": True})
    print(f"Exported {len(public)} approved language cases; timing/splits/navigation release still pending")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "revise", "review", "status", "inspect", "export", "audit"):
        child = sub.add_parser(command)
        if command == "prepare":
            child.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "outputs/hssd_dataset_facing_001")
        else:
            child.add_argument("--input-dir", type=Path, required=True)
        if command not in ("status", "inspect"):
            child.add_argument("--output-dir", type=Path, required=True)
        if command == "revise":
            child.add_argument("--sheet", type=Path, required=True)
        if command == "review":
            child.add_argument("--decisions", type=Path, required=True)
        if command == "inspect":
            child.add_argument("--annotation-id", required=True)
    args = parser.parse_args()
    try:
        if args.command in ("prepare", "revise", "review", "export"):
            return globals()[args.command](args)
        before = {name: sha256(args.input_dir / name) for name in ("manifest.json", "annotations.json")}
        dataset = load_bundle(args.input_dir)
        report = model.summary(dataset)
        if args.command == "inspect":
            records = model.replay(dataset)
            if args.annotation_id not in records:
                raise ValueError("Unknown annotation ID")
            record = records[args.annotation_id]
            evidence = {target["evidence_id"]: target["rgb"] for base in dataset["base_tasks"] for target in base["targets"].values()}
            print(json.dumps({"record": record, "evidence": {key: evidence[key] for key in record["required_evidence_ids"]}}, ensure_ascii=False, indent=2))
        elif args.command == "status":
            print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
        else:
            output = fresh_output(args.output_dir, args.input_dir)
            after = {name: sha256(args.input_dir / name) for name in before}
            if before != after:
                raise ValueError("Annotation authority changed during audit")
            output.mkdir(parents=True)
            write_json(output / "audit.json", {"audit_passed": True, "authority_files_unchanged": True,
                "before": before, "after": after, "language_status": report})
            print(f"Annotation audit passed; language-ready {report['language_ready_cases']}/{report['planned_task_cases']}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, csv.Error) as error:
        print(f"Language annotation operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
