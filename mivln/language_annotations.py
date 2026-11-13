"""Versioned language annotation edits and explicit, replayable human reviews.

No simulator, model, reference-state parser or filesystem dependency. Structural
checks cannot establish that language is correct or that a reviewer really saw
an image; approvals record an explicit human declaration, not authentication.
"""

from __future__ import annotations

from copy import deepcopy
import re
import unicodedata

from .task_update_prompts import fingerprint


KIND = "benchmark_language_annotations_v1"
TASK_TYPES = ("replacement", "subgoal", "cancel_task")
CHECKS = ("instance_binding", "unambiguous", "semantic_consistency", "public_text_only", "natural_language")
LANGUAGES = ("en", "zh-CN")
SHEET_COLUMNS = ("annotation_id", "base_task_id", "role", "task_type", "language", "text", "author", "note")


def exact(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label}: unexpected or missing fields")


def person(name):
    if type(name) is not str or not name.strip():
        raise ValueError("A named human author/reviewer is required")
    normalized = unicodedata.normalize("NFKC", name).strip().casefold()
    if normalized in ("your_name", "your_reviewer_name", "reviewer", "todo", "template", "template_v1"):
        raise ValueError("Replace the placeholder with the actual person's name")
    return normalized


def validate_text(text, language):
    if language not in LANGUAGES or type(text) is not str or not text.strip():
        raise ValueError("Use nonempty text and a supported language")
    # This catches known internal identifiers and obvious source paths only.
    # Arbitrary free-text answer leakage still requires the human checklist.
    forbidden = (r"\b(?:target|task)_[0-9]+\b", r"(?:file://|/mnt/|/home/|/tmp/|[A-Za-z]:\\)",
                 r"\b(?:goal_positions|task_update|source_instance_index|bounds_xz|new_goal|resume_goal)\b")
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in forbidden):
        raise ValueError("Language contains an internal identifier, private path or answer field")


def pending_review():
    return {"status": "pending", "reviewer": None, "note": None, "reviewed_at": None,
            "content_sha256": None, "context_sha256": None, "checks": {}, "checked_evidence_ids": []}


def seed_records(bases):
    """One shared initial instruction and three interventions per source base.

    Templates are drafts. Existing instance approval is evidence about physical
    identity, never inherited as approval of these new language annotations.
    """
    records = []
    for base in bases:
        task, root, other = base["source_task_id"], base["initial_goal"], base["paired_goal"]
        initial_id = task + "_initial"
        a, b = base["targets"][root], base["targets"][other]
        definitions = ((initial_id, "initial", "initial", base["initial_instruction"],
                        {"initial_goal": root}, [a["evidence_id"]]),
            (task + "_replacement", "intervention", "replacement", base["replacement_instruction"],
             {"operation": "replace_goal", "old_goal": root, "new_goal": other}, [a["evidence_id"], b["evidence_id"]]),
            (task + "_subgoal", "intervention", "subgoal",
             f"First go to {b['description']}. Then resume going to {a['description']}.",
             {"operation": "insert_subgoal", "subgoal": other, "resume_goal": root}, [a["evidence_id"], b["evidence_id"]]),
            (task + "_cancel_task", "intervention", "cancel_task", "Cancel the task. Stop now.",
             {"operation": "cancel_task"}, [a["evidence_id"]]))
        for identity, role, family, text, label, evidence in definitions:
            validate_text(text, "en")
            records.append({"annotation_id": identity, "base_task_id": base["base_task_id"], "role": role,
                "task_type": family, "initial_annotation_id": None if role == "initial" else initial_id,
                "language": "en", "text": text, "semantic_label": label,
                "required_evidence_ids": sorted(set(evidence)), "origin": "template_derived",
                "author": {"kind": "template", "name": "template_v1", "at": None},
                "human_contributors": [], "text_revision": 0, "review": pending_review()})
    return records


def content_digest(record):
    return fingerprint({key: record[key] for key in ("annotation_id", "text", "language", "semantic_label", "required_evidence_ids")})


def context_digest(record, records):
    identity = record["initial_annotation_id"]
    return content_digest(records[identity]) if identity is not None else None


def invalidate(records, record):
    record["review"] = pending_review()
    if record["role"] == "initial":
        for dependent in records.values():
            if dependent["initial_annotation_id"] == record["annotation_id"]:
                dependent["review"] = pending_review()


def apply_event(records, event):
    """Replay immutable label-preserving edits/reviews in their recorded order."""
    exact(event, ("sequence", "event_id", "at", "action", "annotation_id", "payload"), "Audit event")
    if type(event["at"]) is not str or not event["at"] or type(event["event_id"]) is not str or not event["event_id"]:
        raise ValueError("Audit event requires an identity and timestamp")
    if event["annotation_id"] not in records:
        raise ValueError("Unknown annotation identity")
    record, payload = records[event["annotation_id"]], event["payload"]
    if event["action"] == "revise":
        exact(payload, ("language", "text", "author", "note"), "Language edit")
        person(payload["author"])
        validate_text(payload["text"], payload["language"])
        if type(payload["note"]) is not str or not payload["note"].strip():
            raise ValueError("Explain the edit or explicit adoption of a template")
        record.update(language=payload["language"], text=payload["text"],
            author={"kind": "human", "name": payload["author"].strip(), "at": event["at"]},
            text_revision=record["text_revision"] + 1)
        if person(payload["author"]) not in [person(name) for name in record["human_contributors"]]:
            record["human_contributors"].append(payload["author"].strip())
        invalidate(records, record)
    elif event["action"] == "review":
        exact(payload, ("decision", "reviewer", "note", "checks", "checked_evidence_ids",
                        "content_sha256", "context_sha256"), "Review decision")
        reviewer = person(payload["reviewer"])
        if payload["decision"] not in ("approve", "reject", "needs_revision"):
            raise ValueError("Unknown review decision")
        if type(payload["note"]) is not str or not payload["note"].strip():
            raise ValueError("A substantive review note is required")
        exact(payload["checks"], CHECKS, "Review checklist")
        if any(type(value) is not bool for value in payload["checks"].values()):
            raise ValueError("Checklist values must be booleans")
        checked = payload["checked_evidence_ids"]
        if (type(checked) is not list or any(type(value) is not str for value in checked)
                or len(set(checked)) != len(checked) or not set(checked) <= set(record["required_evidence_ids"])):
            raise ValueError("Unknown/duplicate evidence declaration")
        if payload["content_sha256"] != content_digest(record) or payload["context_sha256"] != context_digest(record, records):
            raise ValueError("Review refers to stale text or initial context")
        if payload["decision"] == "approve":
            if reviewer in [person(name) for name in record["human_contributors"]]:
                raise ValueError("Approval requires a reviewer independent of the text's human contributors")
            if not all(payload["checks"].values()) or set(checked) != set(record["required_evidence_ids"]):
                raise ValueError("Approval requires every checklist item and required evidence declaration")
            if record["initial_annotation_id"] and records[record["initial_annotation_id"]]["review"]["status"] != "approved":
                raise ValueError("Approve the shared initial instruction before its interventions")
        else:
            # Rejecting a shared initial context also blocks its dependents.
            invalidate(records, record)
        record["review"] = {"status": {"approve": "approved", "reject": "rejected", "needs_revision": "needs_revision"}[payload["decision"]],
            "reviewer": payload["reviewer"].strip(), "note": payload["note"], "reviewed_at": event["at"],
            **{key: deepcopy(payload[key]) for key in ("content_sha256", "context_sha256", "checks", "checked_evidence_ids")}}
    else:
        raise ValueError("Unknown annotation audit action")


def replay(dataset):
    exact(dataset, ("schema_version", "kind", "dataset_id", "revision_id", "revision_number", "created_at",
                    "source_dir", "source_files", "base_tasks", "events", "records"), "Annotation dataset")
    if type(dataset["schema_version"]) is not int or dataset["schema_version"] != 1 or dataset["kind"] != KIND:
        raise ValueError("Unsupported annotation dataset")
    if type(dataset["revision_number"]) is not int or dataset["revision_number"] < 0:
        raise ValueError("Invalid bundle revision")
    bases = dataset["base_tasks"]
    if not bases or len({base["base_task_id"] for base in bases}) != len(bases):
        raise ValueError("Base roster is empty or ambiguous")
    original = seed_records(bases)
    records = {record["annotation_id"]: record for record in original}
    if len(records) != len(original):
        raise ValueError("Annotation IDs are ambiguous")
    seen = set()
    for index, event in enumerate(dataset["events"], 1):
        if type(event.get("sequence")) is not int or event["sequence"] != index or event.get("event_id") in seen:
            raise ValueError("Audit history is not unique and sequential")
        apply_event(records, event)
        seen.add(event["event_id"])
    expected = list(records.values())
    if dataset["records"] != expected:
        raise ValueError("Records differ from the immutable-source edit/review history")
    return records


def with_events(dataset, events, revision_id, at):
    result = deepcopy(dataset)
    records = replay(result)
    for event in events:
        event = deepcopy(event)
        event["sequence"] = len(result["events"]) + 1
        apply_event(records, event)
        result["events"].append(event)
    result.update(records=list(records.values()), revision_id=revision_id,
                  revision_number=result["revision_number"]+1, created_at=at)
    replay(result)
    return result


def review_template(dataset):
    records = replay(dataset)
    return {"schema_version": 1, "dataset_id": dataset["dataset_id"], "revision_id": dataset["revision_id"],
        "decisions": [{"annotation_id": record["annotation_id"], "content_sha256": content_digest(record),
            "context_sha256": context_digest(record, records), "decision": None, "reviewer": "", "note": "",
            "checks": {key: False for key in CHECKS}, "checked_evidence_ids": []} for record in records.values()]}


def summary(dataset):
    records = replay(dataset)
    statuses = {status: sum(record["review"]["status"] == status for record in records.values())
                for status in ("pending", "approved", "rejected", "needs_revision")}
    cases = []
    for record in records.values():
        if record["role"] != "intervention":
            continue
        initial = records[record["initial_annotation_id"]]
        blocked = [item["annotation_id"] for item in (initial, record) if item["review"]["status"] != "approved"]
        cases.append({"base_task_id": record["base_task_id"], "task_type": record["task_type"],
            "annotation_id": record["annotation_id"], "language_ready": not blocked, "blocked_annotations": blocked})
    ready = sum(case["language_ready"] for case in cases)
    return {"base_tasks": len(dataset["base_tasks"]), "planned_annotations": len(records), "review_status": statuses,
        "planned_task_cases": len(cases), "language_ready_cases": ready,
        "all_language_approved": bool(cases) and ready == len(cases), "cases": cases,
        "release_ready": False, "split_status": "not_frozen", "new_language_navigation_executed": False}
