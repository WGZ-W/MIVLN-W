"""Public-only sequential task updates, with no evaluator state or simulator.

Each branch owns a TaskState. Accepted model deltas change that state, failures
leave it untouched, and the next request uses the result without gold repair.
History carries actual execution feedback; state is a projection of the same
kind of information. Synthetic completion notices exercise restoration only,
and must never be described as measured navigation completion.
"""

from __future__ import annotations

from copy import deepcopy
import re

from .rule_task_updater import initial_goal, predict_rule
from .task_state import TaskState
from .task_update import (apply_prediction, candidates, exact_keys, make_request,
                          normalize_prediction, public_state, strict_json)
from .task_update_prompts import encode, fingerprint


PROTOCOL = "task_update_rollout_v1"
MODES = ("history", "state")
FEEDBACK_PREFIX = "执行反馈 / Execution feedback: "


def validate_dialogue(value):
    """Strict public schema: labels and reference snapshots cannot be exported."""
    exact_keys(value, ("dialogue_id", "initial_instruction", "candidate_goals", "candidate_regions", "events"), "Public dialogue")
    if not isinstance(value["dialogue_id"], str) or not re.fullmatch(r"d[0-9a-f]{24}", value["dialogue_id"]):
        raise ValueError("Use an opaque public dialogue ID")
    goals = candidates(value["candidate_goals"], "goal_id")
    candidates(value["candidate_regions"], "region_id")
    instruction = value["initial_instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Initial instruction must be nonempty")
    # This is the ONE public-language initialization, shared by both modes.
    # It does not read reference_initial_goal from the source dataset.
    initial_goal(instruction, value["candidate_goals"])
    if not isinstance(value["events"], list) or not value["events"]:
        raise ValueError("Dialogue requires an event sequence")
    messages = 0
    for event in value["events"]:
        if not isinstance(event, dict):
            raise ValueError("Event must be an object")
        if event.get("type") == "message":
            exact_keys(event, ("type", "message"), "Public message event")
            if not isinstance(event["message"], str) or not event["message"].strip():
                raise ValueError("Message must be nonempty")
            messages += 1
        elif event.get("type") == "complete_subgoal":
            exact_keys(event, ("type", "goal_id"), "Synthetic completion event")
            if not isinstance(event["goal_id"], str) or event["goal_id"] not in goals:
                raise ValueError("Unknown completion goal")
        else:
            raise ValueError("Unsupported dialogue event")
    if not messages:
        raise ValueError("Dialogue requires at least one model update")
    return deepcopy(value)


def render_user(request, history_format):
    """Timeline labels acknowledge failed requests instead of declaring success."""
    if history_format not in ("timeline", "json"):
        raise ValueError("Unknown rollout history format")
    if request["mode"] != "history" or history_format == "json":
        return encode(request)
    context = request["context"]
    public = {key: value for key, value in request.items() if key not in ("context", "message")}
    lines = ["公共候选与请求标识：", encode(public), "初始指令：", encode(context["initial_instruction"]),
             "历史记录（按时间；用户消息是请求，是否生效以随后执行反馈为准）："]
    lines.extend(f"{number}. {encode(item)}" for number, item in enumerate(context["message_history"], 1))
    lines.extend(["当前消息（只预测这一条更新）：", encode(request["message"])])
    return "\n".join(lines)


def step(state, event, request, raw, registry, call_error=None):
    """Transactional transition, including erroneous yet applicable predictions.

    No semantic reference is available here. For example, replacing with the
    wrong but valid candidate really changes the next input. An external
    completion can restore a goal only if this branch actually has the matching
    active subgoal and a suspended goal; it never repairs a divergent branch.
    """
    normalized, schema_valid, application_valid = None, False, False
    updated, error = deepcopy(state), None
    if event["type"] == "complete_subgoal":
        try:
            updated.complete_subgoal(event["goal_id"])
            application_valid, outcome = True, "completed"
        except ValueError as failure:
            updated, outcome, error = deepcopy(state), "completion_failed", str(failure)
        feedback = {"type": "completion_feedback", "goal_id": event["goal_id"], "outcome": outcome}
    else:
        outcome = "call_failed"
        if call_error is not None or raw is None:
            error = call_error or "Missing prediction"
        else:
            outcome = "prediction_invalid"
            try:
                if not isinstance(raw, str):
                    raise ValueError("Preserve raw model text as a string")
                normalized = normalize_prediction(strict_json(raw), request)
                schema_valid = True
                outcome = "application_failed"
                updated = apply_prediction(state, normalized, request, registry)
                application_valid = True
                outcome = "applied" if normalized["status"] == "ok" else "abstained"
            except (ValueError, TypeError, KeyError) as failure:
                error = str(failure)
        # Feedback includes only the accepted/attempted symbolic delta and a
        # result code. Private exception text, geometry and the full state are
        # deliberately excluded from the history prompt.
        feedback = {"type": "update_feedback", "prediction": normalized, "outcome": outcome}
    return updated, {"prediction": normalized, "schema_valid": schema_valid,
        "application_valid": application_valid, "outcome": outcome, "error": error,
        "feedback": feedback}


def iter_rollout(dialogues, modes, registry, predictor):
    """Yield ALL events; predictor(request) returns (raw_text, metadata, error).

    Branches restart only at dialogue boundaries. Completion events make no
    model call. The predictor handles transport interruption by returning null
    failure slots; subsequent state is still replayed without correction.
    """
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("Select unique history/state modes")
    for source in dialogues:
        dialogue = validate_dialogue(source)
        for mode in modes:
            state = TaskState(initial_goal(dialogue["initial_instruction"], dialogue["candidate_goals"]))
            history, turn = [], 0
            for index, event in enumerate(dialogue["events"], 1):
                identity = f"{dialogue['dialogue_id']}__{mode}__e{index:02d}"
                request, raw, metadata, call_error = None, None, {}, None
                before = public_state(state)
                if event["type"] == "message":
                    turn += 1
                    request = make_request(identity, event["message"], dialogue["candidate_goals"],
                        dialogue["candidate_regions"], mode, dialogue["initial_instruction"], history, state)
                    raw, metadata, call_error = predictor(request)
                    history.append("用户消息 / User message: " + event["message"])
                state, result = step(state, event, request, raw, registry, call_error)
                history.append(FEEDBACK_PREFIX + encode(result["feedback"]))
                yield {"record_id": identity, "dialogue_id": dialogue["dialogue_id"], "mode": mode,
                    "event_index": index, "turn_index": turn, "event": deepcopy(event),
                    "request": request, "request_sha256": fingerprint(request) if request is not None else None,
                    "raw_prediction": raw, "response_metadata": metadata, "call_error": call_error,
                    "before_state": before, "after_state": public_state(state), **result}


def predict_rules(request, registry):
    """CPU sanity baseline, reconstructing history from PUBLIC feedback only.

    Rule scores verify the plumbing and grammar, not model performance. Failed
    requests are ignored because their public feedback says they were not
    applied; no fixed-history rule builder or evaluator state is substituted.
    """
    if request["mode"] == "state":
        return predict_rule(request)
    state = TaskState(initial_goal(request["context"]["initial_instruction"], request["candidate_goals"]))
    for item in request["context"]["message_history"]:
        if not item.startswith(FEEDBACK_PREFIX):
            continue
        feedback = strict_json(item[len(FEEDBACK_PREFIX):])
        if feedback["type"] == "completion_feedback" and feedback["outcome"] == "completed":
            state.complete_subgoal(feedback["goal_id"])
        elif feedback["type"] == "update_feedback" and feedback["outcome"] == "applied":
            state = apply_prediction(state, feedback["prediction"], request, registry)
    view = make_request(request["request_id"], request["message"], request["candidate_goals"],
        request["candidate_regions"], "state", "", [], state)
    return predict_rule(view)
