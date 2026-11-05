"""A deliberately limited bilingual rule baseline for task-update development.

This is not an LLM or a learned language model. It recognizes documented phrase
patterns and exact candidate aliases. Ambiguous/unknown references abstain; no
reference labels, geometry or simulator inputs are consulted.
"""

from __future__ import annotations

import re

from .task_update import normalize_prediction, validate_request


REPLACE = re.compile(r"改去|改为前往|换去|\b(?:go\s+to|visit)\b", re.I)
INSERT = re.compile(r"先(?:去|前往)|\bfirst\s+(?:go\s+to|visit)\b", re.I)
RESUME = re.compile(r"再继续|然后继续|之后继续|再去|\bthen\s+(?:resume|continue|go\s+to)\b", re.I)
AVOID = re.compile(r"避开|不要经过|不要进入|\bavoid\b|\bdo\s+not\s+enter\b", re.I)


def mentions(text, catalog, id_key):
    """Keep the longest matching alias at a span, exposing genuine ambiguity.

    '深色沙发' should override a contained generic '沙发', while an unqualified
    '沙发' shared by two public candidates remains ambiguous. ASCII word
    boundaries prevent an alias such as 'bed' matching inside 'bedroom'.
    """
    found = []
    for candidate in catalog:
        for alias in set(candidate["aliases"] + [candidate["description"]]):
            pattern = re.escape(alias)
            if re.search(r"[a-zA-Z]", alias):
                pattern = r"(?<!\w)"+pattern+r"(?!\w)"
            for match in re.finditer(pattern, text, re.I):
                found.append((match.start(), match.end(), candidate[id_key]))
    return {identity for start, end, identity in found if not any(
        other_start <= start and end <= other_end and other_end-other_start > end-start
        for other_start, other_end, _ in found)}


def abstain(reason):
    return {"status": "abstain", "reason": reason}


def reference(text, catalog, id_key):
    matches = mentions(text, catalog, id_key)
    name = "region" if id_key == "region_id" else "goal"
    if len(matches) != 1:
        return None, abstain(("ambiguous_" if len(matches) > 1 else "unknown_")+name)
    return next(iter(matches)), None


def initial_goal(instruction, goal_candidates):
    """Resolve the initial goal from public language, never from initial_goal truth."""
    goal, error = reference(instruction, goal_candidates, "goal_id")
    if error:
        raise ValueError(f"Initial instruction cannot be resolved: {error['reason']}")
    return goal


def interpret(message, current_goal, goals, regions):
    """Predict one supported update from a small, explicit phrase grammar."""
    inserting, avoiding = INSERT.search(message), AVOID.search(message)
    replacing = re.search(r"改去|改为前往|换去|\binstead\b", message, re.I)
    if sum(bool(value) for value in (inserting, avoiding, replacing)) != 1:
        return abstain("unsupported_message")
    if avoiding:
        region, error = reference(message[avoiding.end():], regions, "region_id")
        if error:
            if error["reason"] == "unknown_region" and len(regions) > 1 and re.search(
                    r"标记区域|那片区域|\bmarked\s+(?:floor\s+)?region\b", message, re.I):
                return abstain("ambiguous_region")
            return error
        return {"status": "ok", "task_update": {"operation": "add_avoid_region", "region_id": region}}
    if inserting:
        resume = RESUME.search(message, inserting.end())
        if not resume:
            return abstain("unsupported_message")
        temporary, error = reference(message[inserting.end():resume.start()], goals, "goal_id")
        if error:
            return error
        remainder = message[resume.end():]
        found = mentions(remainder, goals, "goal_id")
        if len(found) > 1:
            return abstain("ambiguous_goal")
        if found:
            restored = next(iter(found))
        elif re.search(r"原(?:来)?(?:的)?任务|原目标|\boriginal\s+(?:task|goal)\b", remainder, re.I):
            restored = current_goal
        else:
            return abstain("unknown_goal")
        if restored is None:
            return abstain("missing_context")
        if current_goal is not None and restored != current_goal or temporary == restored:
            return abstain("invalid_context")
        return {"status": "ok", "task_update": {"operation": "insert_subgoal",
                "subgoal": temporary, "resume_goal": restored}}
    # 'instead of A, go to B' and 'go to B instead' both isolate the destination
    # after go-to, rather than counting old and new referents as an ambiguity.
    marker = REPLACE.search(message)
    if not marker:
        return abstain("unsupported_message")
    destination, error = reference(message[marker.end():], goals, "goal_id")
    if error:
        return error
    prefix_goals = mentions(message[:marker.start()], goals, "goal_id")
    if len(prefix_goals) > 1:
        return abstain("ambiguous_goal")
    previous = next(iter(prefix_goals)) if prefix_goals else current_goal
    if previous is None:
        return abstain("missing_context")
    if current_goal is not None and previous != current_goal or previous == destination:
        return abstain("invalid_context")
    return {"status": "ok", "task_update": {"operation": "replace_goal",
            "old_goal": previous, "new_goal": destination}}


def predict_rule(request):
    """The three modes differ only in accessible context, not parser strength."""
    request = validate_request(request)
    goals, regions = request["candidate_goals"], request["candidate_regions"]
    current = None
    if request["mode"] == "state":
        current = request["context"]["previous_state"]["active_goal"]
    elif request["mode"] == "history":
        try:
            current = initial_goal(request["context"]["initial_instruction"], goals)
        except ValueError:
            return abstain("missing_context")
        for message in request["context"]["message_history"]:
            prior = interpret(message, current, goals, regions)
            if prior["status"] != "ok":
                return abstain("missing_context")
            update = prior["task_update"]
            if update["operation"] == "replace_goal":
                current = update["new_goal"]
            elif update["operation"] == "insert_subgoal":
                current = update["subgoal"]
    return normalize_prediction(interpret(request["message"], current, goals, regions), request)
