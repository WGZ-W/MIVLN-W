"""Simulator-free conditional task-update evaluation with independent references.

Every example is one current update given a fixed language history. A shared
rule-based context builder derives prior policy state from that history. Current
model predictions never influence another example's supplied context. This is
not a test of an LLM maintaining an entire dialogue or perceiving a scene.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from .task_state import TaskState
from .regions import validate_region
from .rule_task_updater import initial_goal, predict_rule
from .task_update import (MODES, STATE_KEYS, apply_prediction, candidates, exact_keys,
                          make_request, normalize_prediction, public_state, strict_json)


def digest(value):
    """Canonical JSON binds each response to its exact public request."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def reference_states(dataset, case):
    """Only the evaluator calls this path; no reference field reaches a predictor."""
    reference = case["reference"]
    state = TaskState(reference["initial_goal"])
    goals, regions = dataset["candidate_goals"], dataset["candidate_regions"]
    for index, update in enumerate(reference["context_updates"]):
        view = make_request(f"reference_{index}", case["history"][index], goals, regions,
                            "state", case["initial_instruction"], [], state)
        state = apply_prediction(state, {"status": "ok", "task_update": update}, view, dataset["region_registry"])
    before = deepcopy(state)
    view = make_request("reference_current", case["message"], goals, regions, "state", "", [], state)
    prediction = normalize_prediction(reference["prediction"], view)
    after = apply_prediction(state, prediction, view, dataset["region_registry"])
    return before, after


def validate_dataset(dataset):
    exact_keys(dataset, ("schema_version", "kind", "purpose", "candidate_goals", "candidate_regions",
                         "region_registry", "cases"), "Development dataset")
    if type(dataset["schema_version"]) is not int or dataset["schema_version"] != 1 or dataset["kind"] != "task_update_development_set":
        raise ValueError("Unsupported task-update dataset")
    if not isinstance(dataset["purpose"], str) or not dataset["purpose"].strip():
        raise ValueError("Dataset must declare its purpose")
    goal_ids = candidates(dataset["candidate_goals"], "goal_id")
    region_ids = candidates(dataset["candidate_regions"], "region_id")
    registry = dataset["region_registry"]
    if not isinstance(registry, dict) or set(registry) != region_ids:
        raise ValueError("Executor region registry must match public region IDs")
    for identity, region in registry.items():
        if validate_region(region)["region_id"] != identity:
            raise ValueError("Executor region identity mismatch")
    if not isinstance(dataset["cases"], list) or not dataset["cases"]:
        raise ValueError("Dataset must contain a fixed nonempty case list")
    ids = set()
    for case in dataset["cases"]:
        exact_keys(case, ("case_id", "group_id", "language", "initial_instruction", "history", "message", "reference"), "Case")
        for key in ("case_id", "group_id"):
            if not isinstance(case[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case[key]):
                raise ValueError("Case/group IDs must be simple identifiers")
        if case["case_id"] in ids:
            raise ValueError("Duplicate case ID")
        ids.add(case["case_id"])
        if case["language"] not in ("zh", "en"):
            raise ValueError("Case language must be zh or en")
        if (any(not isinstance(case[key], str) or not case[key].strip() for key in ("initial_instruction", "message"))
                or not isinstance(case["history"], list)
                or any(not isinstance(text, str) or not text.strip() for text in case["history"])):
            raise ValueError("Case instructions/history must be nonempty strings")
        reference = case["reference"]
        exact_keys(reference, ("initial_goal", "context_updates", "prediction"), "Evaluator reference")
        if (not isinstance(reference["initial_goal"], str) or reference["initial_goal"] not in goal_ids
                or not isinstance(reference["context_updates"], list)
                or len(reference["context_updates"]) != len(case["history"])):
            raise ValueError("Reference initial goal/history update count is invalid")
        reference_states(dataset, case)
    return deepcopy(dataset)


def build_context(initial_instruction, history, goals, regions, registry):
    """Derive policy context from prior language only, using the declared builder.

    The registry is used by the fixed executor after predicting region IDs; the
    predictor receives only the public candidate catalog and projected state.
    No corrected state is substituted from reference_states on failure.
    """
    state = TaskState(initial_goal(initial_instruction, goals))
    for index, message in enumerate(history):
        request = make_request(f"context_{index}", message, goals, regions, "state", initial_instruction, history[:index], state)
        prediction = predict_rule(request)
        if prediction["status"] != "ok":
            raise ValueError(f"Context message unresolved: {prediction['reason']}")
        state = apply_prediction(state, prediction, request, registry)
    return state


def prepare_records(dataset, modes):
    """Keep every planned sample, including failed context construction."""
    if not modes or any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("Select unique modes from latest, history, state")
    records = []
    for case in dataset["cases"]:
        expected_before, expected_after = reference_states(dataset, case)
        before, context_error = None, None
        try:
            before = build_context(case["initial_instruction"], case["history"], dataset["candidate_goals"],
                                   dataset["candidate_regions"], dataset["region_registry"])
        except (ValueError, KeyError, TypeError) as error:
            context_error = f"{type(error).__name__}: {error}"
        for mode in modes:
            request_id = f"{case['case_id']}__{mode}"
            request = (make_request(request_id, case["message"], dataset["candidate_goals"], dataset["candidate_regions"],
                        mode, case["initial_instruction"], case["history"], before) if before is not None else None)
            records.append({"case": case, "mode": mode, "request_id": request_id, "request": request,
                "request_sha256": digest(request) if request is not None else None,
                "before": before, "expected_before": expected_before, "expected_after": expected_after,
                "context_error": context_error})
    return records


def evaluate_record(dataset, record, raw_prediction, input_error=None):
    """Score status, operation, arguments, full state and untouched requirements."""
    case, reference = record["case"], record["case"]["reference"]["prediction"]
    expected_before, expected_after = public_state(record["expected_before"]), public_state(record["expected_after"])
    before = public_state(record["before"]) if record["before"] is not None else None
    after, normalized, schema_valid, application_valid = before, None, False, False
    error = record["context_error"] or input_error
    if not error:
        try:
            value = strict_json(raw_prediction) if isinstance(raw_prediction, str) else raw_prediction
            normalized = normalize_prediction(value, record["request"])
            schema_valid = True
            state = apply_prediction(record["before"], normalized, record["request"], dataset["region_registry"])
            after, application_valid = public_state(state), True
        except (ValueError, KeyError, TypeError) as failure:
            error = f"{type(failure).__name__}: {failure}"
    actionable = reference["status"] == "ok"
    predicted_update = normalized.get("task_update", {}) if normalized else {}
    expected_update = reference.get("task_update", {})
    status_correct = schema_valid and normalized["status"] == reference["status"]
    operation_correct = bool(actionable and status_correct and predicted_update.get("operation") == expected_update["operation"])
    arguments_correct = bool(operation_correct and all(predicted_update.get(key) == value
        for key, value in expected_update.items() if key != "operation"))
    preserved = sorted(key for key in STATE_KEYS if expected_before[key] == expected_after[key])
    fields = {key: bool(application_valid and after[key] == expected_after[key]) for key in sorted(STATE_KEYS)}
    prediction_exact = bool(application_valid and normalized == reference)
    context_correct = before == expected_before
    return {"request_id": record["request_id"], "request_sha256": record["request_sha256"],
        "case_id": case["case_id"], "group_id": case["group_id"], "language": case["language"], "mode": record["mode"],
        "expected_kind": expected_update.get("operation", "abstain"), "expected_actionable": actionable,
        "context_correct": context_correct, "schema_valid": schema_valid, "application_valid": application_valid,
        "status_correct": bool(status_correct), "operation_correct": operation_correct, "arguments_correct": arguments_correct,
        "prediction_exact": prediction_exact, "state_exact": all(fields.values()), "field_correct": fields,
        "preserved_fields": preserved, "requirements_preserved": bool(application_valid and all(after[key] == expected_before[key] for key in preserved)),
        "correct": bool(prediction_exact and all(fields.values()) and context_correct), "error": error,
        "prediction": normalized, "reference_prediction": reference,
        "before_state": before, "after_state": after, "expected_after_state": expected_after}


def statistics(rows):
    count = len(rows)
    report = {"planned": count}
    for key in ("context_correct", "schema_valid", "application_valid", "status_correct", "prediction_exact", "state_exact", "correct"):
        total = sum(row[key] is True for row in rows)
        report[key+"_count"] = total
        report[key+"_rate"] = total/count if count else None
    actionable = [row for row in rows if row["expected_actionable"]]
    report["actionable_count"] = len(actionable)
    for key in ("operation_correct", "arguments_correct", "requirements_preserved"):
        total = sum(row[key] is True for row in actionable)
        report[key+"_count"] = total
        report[key+"_rate"] = total/len(actionable) if actionable else None
    report["expected_abstention_count"] = count-len(actionable)
    report["expected_abstention_correct_count"] = sum(row["correct"] for row in rows if not row["expected_actionable"])
    report["invalid_or_missing_count"] = sum(row["error"] is not None for row in rows)
    return report


def summarize_records(rows):
    """Report fixed denominators; rule correctness is not learned-model quality."""
    return {"kind": "conditional_task_update_evaluation", "expected_records": len(rows),
        "statistics": statistics(rows),
        "by_mode": {mode: statistics([row for row in rows if row["mode"] == mode])
                    for mode in MODES if any(row["mode"] == mode for row in rows)},
        "by_kind": {kind: statistics([row for row in rows if row["expected_kind"] == kind])
                    for kind in sorted({row["expected_kind"] for row in rows})},
        "field_accuracy": {key: sum(row["field_correct"][key] for row in rows)/len(rows) if rows else None
                           for key in sorted(STATE_KEYS)},
        "limitations": ["Synthetic development cases, no held-out or visual navigation claim",
            "Prior state comes from a fixed language-only rule context builder",
            "Untouched-field retention is guaranteed by the shared symbolic executor",
            "Rule results do not establish LLM or structured-memory superiority"]}
