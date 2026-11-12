"""V RGB navigation request/action contracts, separate from frozen U/G schemas."""

from __future__ import annotations

from copy import deepcopy

from .public_messages import ACTIONS, _validate, array_schema, object_schema
from .task_update import strict_json
from .task_update_prompts import fingerprint


CONTRACT = "mivln_visual_navigation_v1"


def request_schema():
    text = {"type": "string", "minLength": 1}
    integer = {"type": "integer", "minimum": 0}
    positive = {"type": "number", "exclusiveMinimum": 0}
    token = {"type": "string", "pattern": r"^[0-9a-f]{32}$"}
    resource = object_schema({"asset_id": token, "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "width": {"type": "integer", "minimum": 1}, "height": {"type": "integer", "minimum": 1},
        "media_type": {"const": "image/png"}})
    message = object_schema({"delivered_step": integer, "message": text})
    properties = {"schema_version": {"type": "integer", "const": 1}, "contract": {"const": CONTRACT},
        "track": {"const": "V"}, "session_id": token,
        "request_id": {"type": "string", "pattern": r"^vis_[0-9a-f]{24}$"}, "step": integer,
        "initial_instruction": text,
        "observation": object_schema({"rgb": resource, "collided": {"type": "boolean"}}),
        "history": object_schema({
            "observations": array_schema(object_schema({"step": integer, "rgb": resource})),
            "actions": array_schema(object_schema({"step": integer, "action": {"enum": list(ACTIONS)},
                                                   "collided": {"type": "boolean"}}))}),
        "message_history": array_schema(message, 0, 1),
        "current_intervention": {"oneOf": [{"const": None}, message]},
        "history_policy": object_schema({"strategy": {"const": "recent_stride_v1"},
            "max_frames": integer, "stride": {"type": "integer", "minimum": 1}}),
        "observation_profile": {"enum": ["V_RGB_v1", "V_RGB_MARKED_v1"]},
        "camera": object_schema({"width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1}, "fx": positive, "fy": positive,
            "cx": {"type": "number"}, "cy": {"type": "number"}}),
        "actions": {"const": list(ACTIONS)},
        "limits": object_schema({"max_steps": {"type": "integer", "minimum": 1},
            "remaining_actions": {"type": "integer", "minimum": 1},
            "forward_step_m": positive, "turn_angle_degrees": positive, "success_radius_m": positive})}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "MI-VLN V RGB request v1",
            **object_schema(properties)}


def action_schema():
    """Host-bound action envelope; raw model output is only {action: ...}."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "MI-VLN V action envelope v1",
        **object_schema({"schema_version": {"type": "integer", "const": 1}, "contract": {"const": CONTRACT},
            "session_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
            "request_id": {"type": "string", "pattern": r"^vis_[0-9a-f]{24}$"},
            "step": {"type": "integer", "minimum": 0}, "action": {"enum": list(ACTIONS)}})}


def seal_request(request):
    value = deepcopy(request)
    value["request_id"] = "vis_" + fingerprint({key: item for key, item in value.items() if key != "request_id"})[:24]
    return validate_request(value)


def validate_request(request):
    """Check body shape and causal alignment, without reading private assets."""
    _validate(request, request_schema())
    expected = "vis_" + fingerprint({key: value for key, value in request.items() if key != "request_id"})[:24]
    if request["request_id"] != expected:
        raise ValueError("V request checksum mismatch")
    step, limits, camera = request["step"], request["limits"], request["camera"]
    if step >= limits["max_steps"] or limits["remaining_actions"] != limits["max_steps"] - step:
        raise ValueError("V budget does not match the pre-action step")
    if not 0 <= camera["cx"] < camera["width"] or not 0 <= camera["cy"] < camera["height"]:
        raise ValueError("Camera principal point is outside the image")
    if camera["width"] > 4096 or camera["height"] > 4096:
        raise ValueError("Camera dimensions exceed the RGB collector limit")
    observations = request["history"]["observations"]
    steps = [item["step"] for item in observations]
    policy = request["history_policy"]
    eligible = range(0, step, policy["stride"])
    if len(steps) != min(len(eligible), policy["max_frames"]):
        raise ValueError("Unexpected number of historical images")
    selected = list(eligible[-policy["max_frames"]:]) if policy["max_frames"] else []
    if steps != selected:
        raise ValueError("Historical images do not follow the frozen sampling policy")
    actions = request["history"]["actions"]
    if len(actions) != step or any(item["step"] != k or item["action"] == "STOP" for k, item in enumerate(actions)):
        raise ValueError("Action history must be actual, contiguous and before terminal STOP")
    if request["observation"]["collided"] != (actions[-1]["collided"] if actions else False):
        raise ValueError("Current collision feedback disagrees with the last executed action")
    history, current = request["message_history"], request["current_intervention"]
    if any(item["delivered_step"] >= step for item in history) or (current and current["delivered_step"] != step):
        raise ValueError("A new message belongs only to this pre-action slot; past messages belong in history")
    if len(history) + int(current is not None) > 1:
        raise ValueError("First V protocol permits at most one intervention")
    resources = [request["observation"]["rgb"], *[item["rgb"] for item in observations]]
    ids = [item["asset_id"] for item in resources]
    if len(ids) != len(set(ids)):
        raise ValueError("Each acquisition must use a distinct public asset ID")
    if any((item["width"], item["height"]) != (camera["width"], camera["height"]) for item in resources):
        raise ValueError("Images disagree with the frozen camera dimensions")
    return deepcopy(request)


def parse_action(raw_text, request):
    """No JSON repair, retries, reference answer, or fallback action."""
    validate_request(request)
    if type(raw_text) is not str:
        raise ValueError("Preserve a raw JSON model response")
    response = strict_json(raw_text)
    _validate(response, object_schema({"action": {"enum": list(ACTIONS)}}))
    return {"schema_version": 1, "contract": CONTRACT, "session_id": request["session_id"],
            "request_id": request["request_id"], "step": request["step"], "action": response["action"]}


def validate_action(action, request):
    validate_request(request)
    _validate(action, action_schema())
    if any(action[key] != request[key] for key in ("session_id", "request_id", "step")):
        raise ValueError("Action is bound to a different session, request or step")
    return deepcopy(action)
