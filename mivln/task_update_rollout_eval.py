"""Private independent labels and diagnostics for sequential update experiments.

Generation never imports this module or opens the source labels. References
advance in a separate trajectory; an incorrect model state is never corrected.
"""

from __future__ import annotations

from copy import deepcopy
import re

from .regions import validate_region
from .rule_task_updater import initial_goal
from .task_state import TaskState
from .task_update import apply_prediction, candidates, exact_keys, make_request, public_state
from .task_update_prompts import fingerprint
from .task_update_rollout import MODES, validate_dialogue


def public_dialogue(dataset, case):
    goals, regions = deepcopy(dataset["candidate_goals"]), deepcopy(dataset["candidate_regions"])
    if case["order"] == 1:
        goals.reverse()
        regions.reverse()
    events = [{key: value for key, value in event.items() if key != "prediction"} for event in case["events"]]
    view = {"initial_instruction": case["initial_instruction"], "candidate_goals": goals,
            "candidate_regions": regions, "events": events}
    # Hash ONLY public content, not family/language labels or reference answers.
    return validate_dialogue({"dialogue_id": "d" + fingerprint(view)[:24], **view})


def reference_steps(dataset, case):
    """Advance declared labels, never derive answers from predict_rule."""
    public = public_dialogue(dataset, case)
    state = TaskState(case["reference_initial_goal"])
    result = []
    for event in case["events"]:
        before = public_state(state)
        prediction = event.get("prediction")
        if event["type"] == "complete_subgoal":
            state.complete_subgoal(event["goal_id"])
        else:
            if prediction.get("status") != "ok":
                raise ValueError("This protocol contains only clear actionable updates")
            request = make_request("reference", event["message"], public["candidate_goals"],
                public["candidate_regions"], "state", "", [], state)
            state = apply_prediction(state, prediction, request, dataset["region_registry"])
        result.append({"before_state": before, "after_state": public_state(state), "prediction": deepcopy(prediction)})
    return result


def validate_dataset(value):
    exact_keys(value, ("schema_version", "kind", "purpose", "candidate_goals", "candidate_regions",
                       "region_registry", "dialogues"), "Rollout dataset")
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != "task_update_rollout_dataset" or not isinstance(value["purpose"], str)
            or not value["purpose"].strip()):
        raise ValueError("Unsupported sequential dataset")
    goals = candidates(value["candidate_goals"], "goal_id")
    regions = candidates(value["candidate_regions"], "region_id")
    if not isinstance(value["region_registry"], dict) or set(value["region_registry"]) != regions:
        raise ValueError("Executor region registry does not match candidates")
    for identity, region in value["region_registry"].items():
        if validate_region(region)["region_id"] != identity:
            raise ValueError("Executor region identity mismatch")
    if not isinstance(value["dialogues"], list) or not value["dialogues"]:
        raise ValueError("Dataset requires dialogues")
    identities, public_ids = set(), set()
    for case in value["dialogues"]:
        exact_keys(case, ("case_id", "group_id", "language", "order", "initial_instruction", "reference_initial_goal", "events"), "Labelled dialogue")
        if (any(not isinstance(case[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", case[key]) for key in ("case_id", "group_id"))
                or case["case_id"] in identities or case["language"] not in ("zh", "en")
                or type(case["order"]) is not int or case["order"] not in (0, 1)
                or not isinstance(case["reference_initial_goal"], str) or case["reference_initial_goal"] not in goals):
            raise ValueError("Invalid dialogue metadata")
        if not isinstance(case["events"], list):
            raise ValueError("Events must be an array")
        for event in case["events"]:
            if not isinstance(event, dict):
                raise ValueError("Event must be an object")
            fields = ("type", "message", "prediction") if event.get("type") == "message" else ("type", "goal_id")
            exact_keys(event, fields, "Labelled event")
            if event.get("type") == "message" and not isinstance(event["prediction"], dict):
                raise ValueError("Independent prediction label must be an object")
        view = public_dialogue(value, case)
        if view["dialogue_id"] in public_ids:
            raise ValueError("Duplicate public dialogue content")
        # A common initial state is legitimate. Subsequent reference snapshots
        # are independent labels and can never initialize later policy turns.
        if initial_goal(view["initial_instruction"], view["candidate_goals"]) != case["reference_initial_goal"]:
            raise ValueError("Initial reference and public instruction disagree")
        reference_steps(value, case)
        identities.add(case["case_id"])
        public_ids.add(view["dialogue_id"])
    return deepcopy(value)


def score_records(dataset, trace, modes):
    """Score exact deltas and persistent states using fixed ALL-event rosters."""
    expected = []
    for case in dataset["dialogues"]:
        dialogue = public_dialogue(dataset, case)
        references = reference_steps(dataset, case)
        for mode in modes:
            for index, reference in enumerate(references, 1):
                expected.append((f"{dialogue['dialogue_id']}__{mode}__e{index:02d}", case, reference))
    if [row["record_id"] for row in trace] != [identity for identity, _, _ in expected]:
        raise ValueError("Scoring requires the entire ordered event roster")
    rows = []
    for actual, (_, case, reference) in zip(trace, expected):
        message = actual["event"]["type"] == "message"
        expected_update = reference["prediction"].get("task_update", {}) if message else {}
        prediction = actual["prediction"] or {}
        predicted_update = prediction.get("task_update", {})
        before_exact = actual["before_state"] == reference["before_state"]
        after_exact = actual["after_state"] == reference["after_state"]
        operation_correct = bool(message and prediction.get("status") == "ok"
                                 and predicted_update.get("operation") == expected_update["operation"])
        reference_fields = [key for key in ("old_goal", "resume_goal") if key in expected_update]
        target_fields = [key for key in ("new_goal", "subgoal", "region_id") if key in expected_update]
        prediction_exact = prediction == reference["prediction"] if message else None
        fields = {key: actual["after_state"][key] == reference["after_state"][key] for key in reference["after_state"]}
        origin = ("maintained" if before_exact else "recovered") if after_exact else ("new_error" if before_exact else "continued_error")
        rows.append({"record_id": actual["record_id"], "dialogue_id": actual["dialogue_id"], "case_id": case["case_id"],
            "group_id": case["group_id"], "language": case["language"], "order": case["order"], "mode": actual["mode"],
            "event_index": actual["event_index"], "turn_index": actual["turn_index"], "is_update": message,
            "before_state_exact": before_exact, "state_exact": after_exact, "field_correct": fields,
            "prediction_exact": prediction_exact, "operation_correct": operation_correct,
            "target_argument_correct": bool(operation_correct and all(predicted_update.get(key) == expected_update[key] for key in target_fields)),
            "reference_argument_correct": (bool(operation_correct and all(predicted_update.get(key) == expected_update[key] for key in reference_fields)) if reference_fields else None),
            "step_correct": bool(actual["application_valid"] and after_exact and (prediction_exact if message else True)),
            "completion_restoration_correct": bool(not message and actual["application_valid"]
                and all(fields[key] for key in ("active_goal", "suspended_goals", "completed_subgoals"))),
            "error_origin": origin, "schema_valid": actual["schema_valid"], "application_valid": actual["application_valid"],
            "outcome": actual["outcome"], "error": actual["error"], "prediction": actual["prediction"],
            "expected_prediction": reference["prediction"], "before_state": actual["before_state"],
            "after_state": actual["after_state"], "expected_before_state": reference["before_state"],
            "expected_after_state": reference["after_state"]})
    return rows


def statistics(rows):
    updates = [row for row in rows if row["is_update"]]
    completions = [row for row in rows if not row["is_update"]]
    count = len(updates)
    result = {"planned_updates": count, "planned_completion_events": len(completions)}
    for key in ("prediction_exact", "state_exact", "step_correct", "operation_correct", "target_argument_correct"):
        total = sum(row[key] is True for row in updates)
        result[key + "_count"] = total
        result[key + "_rate"] = total / count if count else None
    result.update(schema_invalid_count=sum(not row["schema_valid"] for row in updates),
        update_errors=sum(row["error"] is not None for row in updates),
        carried_error_count=sum(not row["before_state_exact"] for row in updates),
        new_error_count=sum(row["error_origin"] == "new_error" for row in updates),
        continued_error_count=sum(row["error_origin"] == "continued_error" for row in updates),
        recovered_count=sum(row["error_origin"] == "recovered" for row in updates),
        active_goal_correct_count=sum(row["field_correct"]["active_goal"] for row in updates),
        completion_correct_count=sum(row["step_correct"] for row in completions),
        completion_restoration_correct_count=sum(row["completion_restoration_correct"] for row in completions))
    result["field_correct_count"] = {key: sum(row["field_correct"][key] for row in updates)
                                     for key in ("active_goal", "suspended_goals", "active_constraints", "completed_subgoals", "cancelled_goals", "status")}
    return result


def summarize(rows):
    trajectories = []
    for identity, mode in dict.fromkeys((row["dialogue_id"], row["mode"]) for row in rows):
        group = [row for row in rows if row["dialogue_id"] == identity and row["mode"] == mode]
        first = next((row for row in group if not row["step_correct"]), None)
        trajectories.append({"dialogue_id": identity, "mode": mode,
            **{key: group[0][key] for key in ("case_id", "group_id", "language", "order")},
            "all_steps_correct": all(row["step_correct"] for row in group),
            "final_state_exact": group[-1]["state_exact"], "final_active_goal_correct": group[-1]["field_correct"]["active_goal"],
            "first_error_event": first["event_index"] if first else None,
            "first_error_turn": first["turn_index"] if first else None})
    by_mode = {}
    for mode in dict.fromkeys(row["mode"] for row in rows):
        selected = [row for row in rows if row["mode"] == mode]
        chains = [row for row in trajectories if row["mode"] == mode]
        by_mode[mode] = {**statistics(selected), "planned_dialogues": len(chains),
            "all_steps_correct_count": sum(row["all_steps_correct"] for row in chains),
            "final_state_exact_count": sum(row["final_state_exact"] for row in chains),
            "final_active_goal_correct_count": sum(row["final_active_goal_correct"] for row in chains),
            "by_turn": {str(turn): statistics([row for row in selected if row["is_update"] and row["turn_index"] == turn])
                        for turn in sorted({row["turn_index"] for row in selected if row["is_update"]})}}
    paired = {}
    if set(by_mode) == set(MODES):
        mapping = {(row["dialogue_id"], row["mode"]): row for row in trajectories}
        for key in ("all_steps_correct", "final_state_exact", "final_active_goal_correct"):
            pairs = [(mapping[(identity, "history")][key], mapping[(identity, "state")][key])
                     for identity in dict.fromkeys(row["dialogue_id"] for row in trajectories)]
            paired[key] = {"planned": len(pairs), "state_only": sum(second and not first for first, second in pairs),
                "history_only": sum(first and not second for first, second in pairs),
                "both_correct": sum(first and second for first, second in pairs),
                "both_wrong": sum(not first and not second for first, second in pairs)}
    return {"kind": "task_update_rollout_evaluation", "statistics": statistics(rows), "by_mode": by_mode,
        "paired_dialogues": paired,
        "trajectories": trajectories, "limitations": [
            "Synthetic clear-update dev dialogues, not independent natural-language generalization tasks",
            "Initial goal uses one shared public-language parser; no later gold state repair",
            "Model predicts deltas; shared executor preserves fields and handles completion",
            "Completion notices are synthetic external fixtures, not measured navigation",
            "History includes its own execution feedback; state compresses its own execution history",
            "Full-state errors can persist in cancelled/completed logs even after active-goal recovery"]}
