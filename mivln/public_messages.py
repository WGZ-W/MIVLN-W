"""Versioned U/G public message schemas and dependency-free validation.

This is a data boundary, not a sandbox. The caller remains responsible for
reviewing free text and collecting truthful observations. We deliberately do
not import evaluators, simulators, scene readers, or reference-answer builders.
"""

from __future__ import annotations

from copy import deepcopy
import math
import re

from .regions import validate_region
from .task_update import validate_request
from .task_update_prompts import fingerprint


CONTRACT = "mivln_public_messages_v1"
CONTEXT_SOURCES = ("public_language_constructed", "provided_initial_goal_then_policy_rollout", "model_rollout")
ACTIONS = ("move_forward", "turn_left", "turn_right", "STOP", "COMPLETE_SUBGOAL")


def object_schema(properties):
    """Every property is required; extensions require a versioned contract."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array_schema(items, minimum=0, maximum=None):
    result = {"type": "array", "items": items, "minItems": minimum}
    if maximum is not None:
        result["maxItems"] = maximum
    return result


def message_schema():
    """Return a portable JSON Schema; runtime adds cross-field invariants.

    Only the small documented keyword subset below is needed, so the core
    stays standard-library-only. A fresh object prevents callers mutating the
    process-wide contract. UUIDs are host-assigned, never private sample hashes.
    """
    text = {"type": "string", "minLength": 1}
    symbol = {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"}
    token = {"type": "string", "pattern": r"^[0-9a-f]{32}$"}
    number = {"type": "number"}
    vector = array_schema(number, 3, 3)
    pose = object_schema({"position": vector, "rotation_xyzw": array_schema(number, 4, 4)})
    goal = object_schema({"goal_id": symbol, "description": text, "aliases": array_schema(text, 1)})
    region_candidate = object_schema({"region_id": symbol, "description": text, "aliases": array_schema(text, 1)})
    region = object_schema({"region_id": symbol, "type": {"const": "rectangle_xz"},
        "bounds_xz": array_schema(array_schema(number, 2, 2), 2, 2),
        "floor_y": number, "floor_tolerance_m": {"type": "number", "exclusiveMinimum": 0}})
    state = object_schema({"active_goal": symbol, "suspended_goals": array_schema(symbol),
        "active_constraints": array_schema(symbol), "completed_subgoals": array_schema(symbol),
        "cancelled_goals": array_schema(symbol), "status": {"enum": ["active", "paused", "completed", "failed"]}})
    requests = []
    contexts = {"latest": object_schema({}), "history": object_schema({
        "initial_instruction": text, "message_history": array_schema(text)}),
        "state": object_schema({"previous_state": state})}
    for mode, context in contexts.items():
        requests.append(object_schema({"schema_version": {"const": 1, "type": "integer"},
            "request_id": {"type": "string", "pattern": r"^(nav_|pub_)[0-9a-f]{24}$"},
            "mode": {"const": mode}, "message": text,
            "candidate_goals": array_schema(goal, 1), "candidate_regions": array_schema(region_candidate),
            "context": context}))
    payloads = {
        ("U", "initialize"): object_schema({"mode": {"enum": ["latest", "history", "state"]},
            "context_source": {"enum": ["none", "received_message_history", *CONTEXT_SOURCES]},
            "candidate_goals": array_schema(goal, 1), "candidate_regions": array_schema(region_candidate)}),
        ("U", "intervention"): object_schema({"request": {"oneOf": requests}}),
        ("G", "initialize"): object_schema({"initial_instruction": text,
            "candidate_goals": array_schema(goal, 1),
            "goal_geometry": array_schema(object_schema({"goal_id": symbol, "stop_position": vector}), 1),
            "map_ref": object_schema({"asset_id": token, "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"}}),
            "pose": pose, "limits": object_schema({"max_steps": {"type": "integer", "minimum": 1},
                "forward_step_m": {"type": "number", "exclusiveMinimum": 0},
                "turn_angle_degrees": {"type": "number", "exclusiveMinimum": 0},
                "success_radius_m": {"type": "number", "exclusiveMinimum": 0},
                "agent_radius_m": {"type": "number", "exclusiveMinimum": 0}}),
            "actions": {"const": list(ACTIONS)}}),
        ("G", "observation"): object_schema({"pose": pose, "collided": {"type": "boolean"},
            "last_action": {"enum": list(ACTIONS)}}),
        ("G", "intervention"): object_schema({"message": text,
            "regions": array_schema(object_schema({"region": region, "description": text,
                                                    "aliases": array_schema(text, 1)}))}),
    }
    variants = []
    for (track, phase), payload in payloads.items():
        variants.append(object_schema({"schema_version": {"const": 1, "type": "integer"},
            "contract": {"const": CONTRACT}, "track": {"const": track}, "phase": {"const": phase},
            "session_id": token, "message_id": {"type": "string", "pattern": r"^pub_[0-9a-f]{24}$"},
            "step": {"type": "integer", "minimum": 0}, "payload": payload}))
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "MI-VLN public U/G messages v1", "oneOf": variants}


def _validate(value, schema, path="$"):
    """Check our finite JSON subset; no external references or coercion."""
    if "oneOf" in schema:
        matches = 0
        for choice in schema["oneOf"]:
            try:
                _validate(value, choice, path)
                matches += 1
            except ValueError:
                pass
        if matches != 1:
            raise ValueError(f"{path}: expected exactly one public schema variant, got {matches}")
        return
    kind = schema.get("type")
    valid = {"object": type(value) is dict, "array": type(value) is list,
             "string": type(value) is str, "integer": type(value) is int,
             "number": type(value) in (int, float), "boolean": type(value) is bool}
    if kind is not None and not valid[kind]:
        raise ValueError(f"{path}: expected {kind}")
    if type(value) in (int, float):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{path}: nonfinite or unrepresentable number")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: constant mismatch")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: unknown enum")
    if kind == "object":
        keys, allowed = set(value), set(schema["properties"])
        if not set(schema["required"]) <= keys or keys - allowed:
            raise ValueError(f"{path}: missing or unauthorized fields")
        for key, item in value.items():
            _validate(item, schema["properties"][key], f"{path}.{key}")
    if kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: wrong array length")
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")
    if kind == "string":
        if len(value.strip()) < schema.get("minLength", 0):
            raise ValueError(f"{path}: empty text")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValueError(f"{path}: malformed identifier")
    if "minimum" in schema and value < schema["minimum"]:
        raise ValueError(f"{path}: below minimum")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        raise ValueError(f"{path}: nonpositive value")


def validate_message(message):
    """Validate structure plus references/geometry; return an owned JSON copy."""
    _validate(message, message_schema())
    # A checksum binds the public body; it is not a signature or a guarantee
    # that free text or the host's provenance claims are trustworthy.
    body = {key: value for key, value in message.items() if key != "message_id"}
    if message["message_id"] != "pub_" + fingerprint(body)[:24]:
        raise ValueError("Public message checksum mismatch")
    payload = message["payload"]
    if message["track"] == "U":
        if message["phase"] == "intervention":
            validate_request(payload["request"])
        else:
            from .task_update import candidates
            candidates(payload["candidate_goals"], "goal_id")
            candidates(payload["candidate_regions"], "region_id")
            expected = {"latest": ("none",), "history": ("received_message_history",), "state": CONTEXT_SOURCES}
            if payload["context_source"] not in expected[payload["mode"]]:
                raise ValueError("U mode and context source disagree")
    else:
        if message["phase"] == "intervention":
            ids = []
            for entry in payload["regions"]:
                ids.append(validate_region(entry["region"])["region_id"])
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate region announcement")
        if message["phase"] == "initialize":
            from .task_update import candidates
            goals = candidates(payload["candidate_goals"], "goal_id")
            geometry = [item["goal_id"] for item in payload["goal_geometry"]]
            if set(geometry) != goals or len(geometry) != len(goals):
                raise ValueError("G geometry must match public candidate identities exactly")
        if "pose" in payload:
            rotation = payload["pose"]["rotation_xyzw"]
            if not math.isclose(sum(v*v for v in rotation), 1, abs_tol=1e-5):
                raise ValueError("Pose requires an XYZW unit quaternion")
    return deepcopy(message)
