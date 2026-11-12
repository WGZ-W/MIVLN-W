"""Whitelisted public views with per-session, pre-action delivery guards.

The host supplies approved catalogs, actual policy state and observations. This
module never obtains policy context from evaluator truth, never reads a file,
and never applies a reference update to the policy. These are in-process data
contracts; a caller lying about provenance still needs runtime isolation/audit.
"""

from __future__ import annotations

from copy import deepcopy
import uuid

from .navigation_updates import navigation_request
from .public_messages import ACTIONS, CONTEXT_SOURCES, CONTRACT, validate_message
from .task_state import TaskState
from .task_update import candidates, make_request
from .task_update_prompts import fingerprint


def update_request(step, message, goals, regions, mode, policy_state=None,
                   initial_instruction="", message_history=()):
    """Reuse U contracts, preserving the existing navigation request byte shape.

    State is required explicitly; no ``config.initial_goal`` or reference-state
    fallback is available here. Non-state modes do not serialize policy state.
    The legacy navigation identity hashes public values only, including step.
    """
    if type(step) is not int or step < 0:
        raise ValueError("Step must be a nonnegative integer")
    if mode == "state" and type(policy_state) is not TaskState:
        raise ValueError("State mode requires the caller's actual policy TaskState")
    if mode == "state" and not regions:
        return navigation_request(step, message, goals, policy_state)
    # Build with a temporary constant ID before hashing the allowed body. Never
    # derive request identity from a private event ID, task ID or config hash.
    request = make_request("public", message, goals, regions, mode, initial_instruction,
                           message_history, policy_state)
    request["request_id"] = "pub_" + fingerprint([step, request])[:24]
    return request


class PublicInputSession:
    """Construct U or minimal G messages in an actual host's action sequence.

    U only emits initialization and intervention messages. Minimal G additionally
    emits precise pose/collision observations, without image/depth/map-serving
    APIs. V and sensor extensions are rejected rather than silently upgraded.
    Exactly one human event is supported, matching the first task protocol.
    """

    def __init__(self, track, max_steps, *, mode="state", context_source=None, session_id=None):
        if track not in ("U", "G"):
            raise ValueError("Only U and minimal G contracts are implemented")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.track, self.max_steps, self.mode = track, max_steps, mode
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self.context_source = context_source
        if track == "U":
            if mode not in ("latest", "history", "state"):
                raise ValueError("Unknown U context mode")
            expected = {"latest": "none", "history": "received_message_history"}
            if mode in expected:
                if context_source is not None and context_source != expected[mode]:
                    raise ValueError("Context source disagrees with U mode")
                self.context_source = expected[mode]
            elif context_source not in CONTEXT_SOURCES:
                raise ValueError("Declare the provenance of the supplied policy state")
        elif context_source is not None or mode != "state":
            raise ValueError("G does not use U context settings")
        self.step, self.closed, self.initialized = 0, False, False
        self._delivered = set()
        self._history = []

    def _message(self, phase, payload, step=None):
        body = {"schema_version": 1, "contract": CONTRACT, "track": self.track, "phase": phase,
                "session_id": self.session_id, "step": self.step if step is None else step,
                "payload": deepcopy(payload)}
        body["message_id"] = "pub_" + fingerprint(body)[:24]
        return validate_message(body)

    def initialize(self, private_config, goal_candidates, region_candidates=(), *, map_ref=None):
        """Project known fields, not a blacklist of private ones.

        Catalogs are independent approved inputs: reject their extra fields
        instead of stripping potentially mislabeled geometry silently. G selects
        stop points only for those candidate IDs; unrelated metadata is ignored.
        A map reference is a host-issued asset ID/digest, never a filesystem path.
        """
        if self.initialized:
            raise ValueError("Session already initialized")
        goals, regions = deepcopy(goal_candidates), deepcopy(list(region_candidates))
        candidates(goals, "goal_id")
        candidates(regions, "region_id")
        if type(private_config["max_steps"]) is not int or private_config["max_steps"] != self.max_steps:
            raise ValueError("Session budget differs from the frozen config")
        instruction = private_config.get("initial_instruction", "")
        if self.track == "U":
            if map_ref is not None:
                raise ValueError("Map resources are not permitted in U")
            payload = {"mode": self.mode, "context_source": self.context_source,
                       "candidate_goals": goals, "candidate_regions": regions}
        else:
            if regions:
                raise ValueError("Announce G regions at delivery, not during initialization")
            payload = {"initial_instruction": instruction, "candidate_goals": goals,
                "goal_geometry": [{"goal_id": item["goal_id"],
                    "stop_position": deepcopy(private_config["goal_positions"][item["goal_id"]])} for item in goals],
                "map_ref": deepcopy(map_ref),
                "pose": {"position": deepcopy(private_config["start_position"]),
                         "rotation_xyzw": deepcopy(private_config["start_rotation_xyzw"])},
                "limits": {"max_steps": self.max_steps,
                    **{key: private_config[key] for key in ("forward_step_m", "turn_angle_degrees", "success_radius_m")},
                    "agent_radius_m": private_config["agent_radius"]}, "actions": list(ACTIONS)}
        result = self._message("initialize", payload)
        # Commit local state only after all checks pass. Returned messages and
        # host config/candidate mutations cannot alter a session's stored views.
        self._goals, self._regions = goals, regions
        self._initial_instruction = instruction
        self._pose = deepcopy(payload.get("pose"))
        self.initialized = True
        return result

    def _active(self):
        if not self.initialized or self.closed or self.step >= self.max_steps:
            raise ValueError("No active pre-action slot in this session")

    def advance(self, action, observation=None):
        """Record one actually executed action; never execute or invent it.

        A G packet is host-stamped with this session and the post-action step
        (action k produces observation step k+1). This catches accidental branch
        mixing/future packets, but cannot verify a dishonest collector's pixels
        or timestamps. U counts actions without receiving any observation data.
        """
        self._active()
        if action not in ACTIONS:
            raise ValueError("Unknown action")
        if self.track == "U":
            if observation is not None:
                raise ValueError("U cannot receive observation packets")
            result = None
        else:
            if (type(observation) is not dict or set(observation) != {"session_id", "step", "pose", "collided"}
                    or observation["session_id"] != self.session_id
                    or type(observation["step"]) is not int or observation["step"] != self.step + 1):
                raise ValueError("Observation has wrong fields, session or post-action step")
            payload = {"pose": deepcopy(observation["pose"]), "collided": observation["collided"], "last_action": action}
            result = self._message("observation", payload, self.step + 1)
            if action in ("STOP", "COMPLETE_SUBGOAL") and payload["pose"] != self._pose:
                raise ValueError("Logical STOP/completion cannot change pose")
        self.step += 1
        self.closed = action == "STOP" or self.step == self.max_steps
        if result is not None:
            self._pose = deepcopy(result["payload"]["pose"])
        return result

    def intervention(self, event, *, policy_state=None, region_packet=None):
        """Publish an event only at its actual pre-action slot.

        Of a private event, only utterance is serialized. Trigger/event identity
        is used for delivery checks only; reference task_update is never read.
        G region announcements come from a separate explicit host packet, not
        from the private operation/region selection. U rejects geometry packets.
        """
        self._active()
        trigger = event.get("trigger", {})
        if (trigger.get("type") != "step" or type(trigger.get("value")) is not int
                or trigger["value"] != self.step):
            raise ValueError("Event is not due at the current pre-action step")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or self._delivered:
            raise ValueError("Only one once-delivered event is permitted")
        if self.track == "U":
            if region_packet is not None:
                raise ValueError("Geometry announcements are forbidden in U")
            request = update_request(self.step, event["utterance"], self._goals, self._regions,
                self.mode, policy_state, self._initial_instruction, self._history)
            payload = {"request": request}
        else:
            if policy_state is not None:
                raise ValueError("G does not receive symbolic evaluator/policy snapshots")
            if (type(region_packet) is not dict or set(region_packet) != {"session_id", "step", "regions"}
                    or region_packet["session_id"] != self.session_id
                    or type(region_packet["step"]) is not int or region_packet["step"] != self.step):
                raise ValueError("Region announcement has wrong fields, session or delivery step")
            payload = {"message": event["utterance"], "regions": deepcopy(region_packet["regions"])}
        result = self._message("intervention", payload)
        self._delivered.add(event_id)
        self._history.append(event["utterance"])
        return result
