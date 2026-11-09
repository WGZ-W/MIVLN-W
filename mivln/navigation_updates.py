"""Public model boundary for one language-driven navigation goal replacement.

The callback receives no InterventionEvent, scene, pose, target coordinates or
evaluator state. Only the checked symbolic prediction can change policy state;
neither a failed call nor an invalid update falls back to a reference answer.
"""

from __future__ import annotations

from copy import deepcopy

from .task_update import (apply_prediction, candidates, make_request,
                          normalize_prediction, public_state, strict_json)
from .task_update_prompts import fingerprint


def validate_navigation_candidates(catalog, goal_ids):
    """Candidate identities must map exactly to the executor's configured goals."""
    if candidates(catalog, "goal_id") != set(goal_ids):
        raise ValueError("Language candidates must match configured navigation goals")


def navigation_request(step, message, catalog, state):
    """Use an opaque identity rather than a reference-bearing event identifier."""
    identity = "nav_" + fingerprint([step, message, catalog, public_state(state)])[:24]
    return make_request(identity, message, catalog, [], "state", "", [], state)


def apply_navigation_prediction(state, request, raw, metadata=None, call_error=None):
    """Transactional target replacement with explicit scope and failure records.

    The existing generic schema still recognizes three update operations. This
    first navigation entry executes ONLY replace_goal: other operations require
    their own planner/completion contracts, and are logged rather than coerced.
    Semantic correctness is evaluated later, outside this policy adapter.
    """
    updated, prediction = deepcopy(state), None
    schema_valid, application_valid = False, False
    outcome, error = "call_failed", call_error
    if call_error is None and raw is not None:
        outcome = "prediction_invalid"
        try:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Preserve nonempty raw model text")
            prediction = normalize_prediction(strict_json(raw), request)
            schema_valid = True
            if prediction["status"] == "abstain":
                outcome, application_valid = "abstained", True
            elif prediction["task_update"]["operation"] != "replace_goal":
                outcome, error = "unsupported_navigation_operation", "Only replace_goal is enabled in this entry"
            else:
                outcome = "application_failed"
                updated = apply_prediction(state, prediction, request, {})
                application_valid, outcome = True, "applied"
        except (ValueError, TypeError, KeyError) as failure:
            error = str(failure)
    elif error is None:
        error = "Missing model response"
    return updated, {"request": deepcopy(request), "request_sha256": fingerprint(request),
        "raw_prediction": raw, "response_metadata": deepcopy(metadata or {}), "call_error": call_error,
        "prediction": prediction, "schema_valid": schema_valid, "application_valid": application_valid,
        "outcome": outcome, "error": error, "before_state": public_state(state), "after_state": public_state(updated)}
