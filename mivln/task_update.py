"""Public task-update contracts and a transactional adapter to the task state.

Language updaters receive symbolic candidate references, messages and permitted
context. Region geometry and reference answers are kept outside that boundary.
The same checked executor applies predictions from any updater, so preserving
untouched fields is an executor guarantee, not evidence of model intelligence.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Mapping

from .regions import validate_region
from .task_state import TaskState


MODES = ("latest", "history", "state")
REASONS = ("missing_context", "unknown_goal", "ambiguous_goal", "unknown_region",
           "ambiguous_region", "unsupported_message", "invalid_context")
STATE_KEYS = {"active_goal", "suspended_goals", "active_constraints", "completed_subgoals", "cancelled_goals", "status"}


def strict_json(text: str) -> Any:
    """Reject duplicate keys and non-JSON numbers instead of repairing answers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def finite_float(value):
        # Even standard JSON exponents can overflow Python's finite float range.
        # Reject them here so imported raw model text follows the same contract.
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite JSON number: {value}")
        return number

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)


def exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} requires exactly these fields: {sorted(keys)}")


def candidates(values, id_key):
    """Validate the entire public catalog; coordinates cannot hide in extras."""
    if not isinstance(values, list):
        raise ValueError("Candidate catalog must be an array")
    ids = set()
    for value in values:
        exact_keys(value, (id_key, "description", "aliases"), "Public candidate")
        identity = value[id_key]
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError("Candidate IDs must be nonempty and unique")
        if not isinstance(value["description"], str) or not value["description"].strip():
            raise ValueError("Candidate description must be nonempty")
        if (not isinstance(value["aliases"], list) or not value["aliases"]
                or any(not isinstance(alias, str) or not alias.strip() for alias in value["aliases"])):
            raise ValueError("Candidate aliases must be nonempty strings")
        ids.add(identity)
    return ids


def public_state(state: TaskState) -> dict:
    """Project policy-owned state to references; never expose region coordinates."""
    result = state.to_dict()
    result["active_constraints"] = [region["region_id"] for region in state.active_constraints]
    return result


def validate_state_view(value, goal_ids, region_ids):
    exact_keys(value, STATE_KEYS, "Previous policy state")
    if (not isinstance(value["active_goal"], str) or value["active_goal"] not in goal_ids
            or value["status"] not in ("active", "paused", "completed", "failed")):
        raise ValueError("Invalid active goal or state status")
    for key in ("suspended_goals", "completed_subgoals", "cancelled_goals", "active_constraints"):
        items = value[key]
        allowed = region_ids if key == "active_constraints" else goal_ids
        if (not isinstance(items, list) or any(not isinstance(item, str) or item not in allowed for item in items)
                or len(set(items)) != len(items)):
            raise ValueError(f"Invalid references in previous state: {key}")


def validate_request(value):
    """One input schema shared by rule and future model-backed updaters."""
    exact_keys(value, ("schema_version", "request_id", "mode", "message", "candidate_goals",
                       "candidate_regions", "context"), "Task-update request")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["mode"] not in MODES:
        raise ValueError("Unsupported task-update schema or mode")
    if not all(isinstance(value[key], str) and value[key].strip() for key in ("request_id", "message")):
        raise ValueError("Request ID and message must be nonempty strings")
    goal_ids = candidates(value["candidate_goals"], "goal_id")
    region_ids = candidates(value["candidate_regions"], "region_id")
    if not goal_ids:
        raise ValueError("At least one candidate goal is required")
    context, mode = value["context"], value["mode"]
    if mode == "latest":
        exact_keys(context, (), "Latest-message context")
    elif mode == "history":
        exact_keys(context, ("initial_instruction", "message_history"), "History context")
        if (not isinstance(context["initial_instruction"], str) or not context["initial_instruction"].strip()
                or not isinstance(context["message_history"], list)
                or any(not isinstance(item, str) or not item.strip() for item in context["message_history"])):
            raise ValueError("History must contain an initial instruction and prior message strings")
    else:
        exact_keys(context, ("previous_state",), "Structured-state context")
        validate_state_view(context["previous_state"], goal_ids, region_ids)
    return deepcopy(value)


def make_request(request_id, message, goal_candidates, region_candidates, mode,
                 initial_instruction, message_history, previous_state):
    """Build a whitelisted view, never pass an episode/reference dictionary through.

    ``previous_state`` belongs to the policy/context builder. It must not be
    replaced with evaluator truth after a prediction fails or abstains.
    """
    if mode == "latest":
        context = {}
    elif mode == "history":
        context = {"initial_instruction": initial_instruction, "message_history": list(message_history)}
    elif mode == "state":
        context = {"previous_state": public_state(previous_state)}
    else:
        raise ValueError("Unknown context mode")
    return validate_request({"schema_version": 1, "request_id": request_id, "mode": mode,
        "message": message, "candidate_goals": deepcopy(goal_candidates),
        "candidate_regions": deepcopy(region_candidates), "context": context})


def normalize_prediction(value, request):
    """Check exact output fields and candidate references; no hidden inference."""
    validate_request(request)
    if not isinstance(value, dict):
        raise ValueError("Prediction must be an object")
    if value.get("status") == "abstain":
        exact_keys(value, ("status", "reason"), "Abstention")
        if value["reason"] not in REASONS:
            raise ValueError("Unknown abstention reason")
        return deepcopy(value)
    exact_keys(value, ("status", "task_update"), "Actionable prediction")
    if value["status"] != "ok":
        raise ValueError("Prediction status must be ok or abstain")
    update = value["task_update"]
    if not isinstance(update, dict):
        raise ValueError("task_update must be an object")
    operation = update.get("operation")
    fields = {
        "replace_goal": ("operation", "old_goal", "new_goal"),
        "add_avoid_region": ("operation", "region_id"),
        "insert_subgoal": ("operation", "subgoal", "resume_goal"),
    }
    if not isinstance(operation, str) or operation not in fields:
        raise ValueError("Unsupported predicted operation")
    exact_keys(update, fields[operation], "Predicted update")
    goals = {item["goal_id"] for item in request["candidate_goals"]}
    regions = {item["region_id"] for item in request["candidate_regions"]}
    for field in fields[operation][1:]:
        allowed = regions if field == "region_id" else goals
        if not isinstance(update[field], str) or update[field] not in allowed:
            raise ValueError(f"Unknown predicted reference: {field}")
    if operation == "replace_goal" and update["old_goal"] == update["new_goal"]:
        raise ValueError("Goal replacement must change the goal")
    if operation == "insert_subgoal" and update["subgoal"] == update["resume_goal"]:
        raise ValueError("Temporary and resume goals must differ")
    return deepcopy(value)


def apply_prediction(state: TaskState, prediction, request, region_registry: Mapping) -> TaskState:
    """Apply on an owned copy; any invalid prediction leaves the input untouched.

    A region ID is resolved at this executor boundary. Its geometry is never
    included in an updater request or accepted from the updater's output.
    """
    prediction = normalize_prediction(prediction, request)
    validate_state_view(public_state(state),
        {item["goal_id"] for item in request["candidate_goals"]},
        {item["region_id"] for item in request["candidate_regions"]})
    updated = deepcopy(state)
    if prediction["status"] == "abstain":
        return updated
    update = deepcopy(prediction["task_update"])
    if update["operation"] == "add_avoid_region":
        identity = update.pop("region_id")
        if identity not in region_registry:
            raise ValueError("Executor has no geometry for the predicted region")
        region = validate_region(region_registry[identity])
        if region["region_id"] != identity:
            raise ValueError("Region registry identity mismatch")
        update["region"] = region
    updated.apply_update(update)
    return updated
