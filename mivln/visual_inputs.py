"""Causal V RGB inputs, action binding, and request-scoped image access.

No policy state, goal positions, reference route or evaluator result is accepted
as model context. The host executes actions and collects frames; this class only
checks their ordering and builds independent public snapshots. Process isolation
and correct scene rendering remain host responsibilities.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import uuid

from .public_messages import ACTIONS
from .visual_frames import RGBFrame, token, verify_marker
from .visual_messages import CONTRACT, parse_action, seal_request, validate_action


MARKER_MIN_PIXELS = 16
MARKER_RGB = (255, 0, 0)


class VisualInputSession:
    """One initial instruction, one human intervention, a continuous RGB stream.

    Observation step t is before action t. Historical frames are steps <t that
    are divisible by stride, retaining the newest max_history_frames. Actual
    action history and delivered messages are never reconstructed from truth.
    Sampling controls retransmission, not an agent's own previously built memory.
    """

    def __init__(self, initial_instruction, camera, *, max_steps=500,
                 forward_step_m=.25, turn_angle_degrees=15, success_radius_m=.3,
                 max_history_frames=8, history_stride=1,
                 observation_profile="V_RGB_v1", session_id=None):
        if type(initial_instruction) is not str or not initial_instruction.strip():
            raise ValueError("An initial natural-language instruction is required")
        if type(camera) is not dict:
            raise ValueError("Camera calibration must be a public field dictionary")
        if (type(max_steps) is not int or max_steps < 1 or type(max_history_frames) is not int
                or max_history_frames < 0 or type(history_stride) is not int or history_stride < 1):
            raise ValueError("Invalid budget/history policy")
        if observation_profile not in ("V_RGB_v1", "V_RGB_MARKED_v1"):
            raise ValueError("Only declared RGB and visibly marked RGB profiles are supported")
        self.session_id = token(session_id if session_id is not None else uuid.uuid4().hex)
        self.initial_instruction, self.camera = initial_instruction, deepcopy(camera)
        self.max_steps = max_steps
        self.profile = observation_profile
        self.history_policy = {"strategy": "recent_stride_v1", "max_frames": max_history_frames, "stride": history_stride}
        self.limits = {"max_steps": max_steps, "forward_step_m": forward_step_m,
                       "turn_angle_degrees": turn_angle_degrees, "success_radius_m": success_radius_m}
        self.step, self.closed, self.initialized = 0, False, False
        self.actions, self.message_history = [], []
        self._frames, self._assets = {}, {}
        self._new_message, self._pending = None, None
        self._delivered = False
        self.marker_audits = []
        self.failure_reason = None

    def _check_frame(self, frame, step):
        if type(frame) is not RGBFrame:
            raise ValueError("Only raw RGBFrame observations are accepted")
        frame.validate()
        if (frame.session_id != self.session_id or frame.step != step
                or (frame.width, frame.height) != (self.camera.get("width"), self.camera.get("height"))):
            raise ValueError("RGB observation has wrong session, step or camera dimensions")
        if frame.asset_id in self._assets:
            raise ValueError("Do not reuse an asset ID across acquisitions")

    def initialize(self, frame):
        if self.initialized:
            raise ValueError("Already initialized")
        self._check_frame(frame, 0)
        self._frames[0], self._assets[frame.asset_id] = frame, frame
        try:
            # Validate camera/limits through the same public contract before
            # committing initialization. No model request is issued yet.
            self._build_request()
        except (ValueError, TypeError, KeyError):
            self._frames.clear()
            self._assets.clear()
            raise
        self.initialized = True

    def _active(self):
        if not self.initialized or self.closed or self.step >= self.max_steps:
            raise ValueError("No active V decision slot")

    def _build_request(self):
        eligible = range(0, self.step, self.history_policy["stride"])
        count = self.history_policy["max_frames"]
        selected = list(eligible[-count:]) if count else []
        value = {"schema_version": 1, "contract": CONTRACT, "track": "V", "session_id": self.session_id,
            "step": self.step, "initial_instruction": self.initial_instruction,
            "observation": {"rgb": self._frames[self.step].resource(),
                            "collided": self.actions[-1]["collided"] if self.actions else False},
            "history": {"observations": [{"step": k, "rgb": self._frames[k].resource()} for k in selected],
                        "actions": deepcopy(self.actions)},
            "message_history": deepcopy(self.message_history), "current_intervention": deepcopy(self._new_message),
            "history_policy": deepcopy(self.history_policy), "observation_profile": self.profile,
            "camera": deepcopy(self.camera), "actions": list(ACTIONS),
            "limits": {**self.limits, "remaining_actions": self.max_steps - self.step}}
        return seal_request(value)

    def request(self):
        """Idempotent snapshot; delivery must happen BEFORE issuing this request."""
        self._active()
        if self._pending is None:
            self._pending = self._build_request()
        return deepcopy(self._pending)

    def deliver(self, event, *, marked_frame=None, marker_evidence=None):
        """Apply visibility eligibility, then serialize only the human message.

        The private operation selects an eligibility check, not an answer sent to
        the agent. For avoid-region trials this first V version requires actual
        marker pixels at delivery; a textual 'marked region' without proof fails.
        No event ID, target ID, region coordinates or completion truth is exposed.
        """
        self._active()
        if self._pending is not None or self._delivered:
            raise ValueError("Deliver once before policy choice, without changing an issued request")
        if type(event) is not dict or type(event.get("trigger")) is not dict:
            raise ValueError("Malformed host event")
        trigger = event["trigger"]
        if trigger.get("type") != "step" or type(trigger.get("value")) is not int or trigger["value"] != self.step:
            raise ValueError("Event is not due at this pre-action step")
        message = event.get("utterance")
        update = event.get("task_update")
        if type(message) is not str or not message.strip() or type(update) is not dict:
            raise ValueError("Event needs text and private eligibility annotation")
        operation = update.get("operation")
        if operation not in ("replace_goal", "add_avoid_region", "insert_subgoal"):
            raise ValueError("Unsupported first-version task")
        audit = None
        if operation == "add_avoid_region":
            if self.profile != "V_RGB_MARKED_v1" or marked_frame is None or marker_evidence is None:
                raise ValueError("Avoid-region V trials require a visible marker in the actual RGB input")
            region = update.get("region", {})
            if type(region) is not dict or type(region.get("region_id")) is not str:
                raise ValueError("Private marker evidence must bind the annotated region")
            self._check_frame(marked_frame, self.step)
            visible = verify_marker(self._frames[self.step], marked_frame, marker_evidence, region["region_id"], MARKER_MIN_PIXELS)
            if marker_evidence.color_rgb != MARKER_RGB:
                raise ValueError("V_RGB_MARKED_v1 declares red markers; another color needs its own input profile")
            audit = {"step": self.step, "changed_pixels": visible, "evidence": asdict(marker_evidence),
                     "world_projection_verified": False}
        elif marked_frame is not None or marker_evidence is not None:
            raise ValueError("Unexpected overlay for this event; use the collector's declared RGB stream")
        # Everything above may fail without consuming a slot or mutating public
        # history. Correct world projection is NOT inferred from this byte proof.
        if marked_frame is not None:
            self._frames[self.step] = marked_frame
            self._assets[marked_frame.asset_id] = marked_frame
            self.marker_audits.append(audit)
        self._new_message = {"delivered_step": self.step, "message": message}
        self._delivered = True

    def assets_for(self, request_id):
        """Return only the current request's actual PNG bytes, with no paths.

        Previously seen pixels may be kept in a method's own memory. This API
        prevents browsing unpublished/future/unselected assets via the collector.
        It does not sandbox Python callbacks or stop reading external files.
        """
        self._active()
        if self._pending is None or request_id != self._pending["request_id"]:
            raise ValueError("No matching outstanding V request")
        resources = [self._pending["observation"]["rgb"],
                     *[entry["rgb"] for entry in self._pending["history"]["observations"]]]
        return {item["asset_id"]: self._assets[item["asset_id"]].png() for item in resources}

    def advance(self, action_envelope, frame_after=None, *, collided=False):
        """Record execution, not a model's claimed pose or task completion.

        STOP counts one action and ends the session using its current frame.
        COMPLETE_SUBGOAL counts without physical advance; the host provides a
        newly stamped frame with identical pixels. Its semantic validity still
        belongs to the evaluator; this interface never restores a truth goal.
        """
        self._active()
        if self._pending is None:
            raise ValueError("Issue the input before accepting an action")
        validate_action(action_envelope, self._pending)
        if type(collided) is not bool:
            raise ValueError("Collision feedback must be boolean")
        action = action_envelope["action"]
        if action == "STOP":
            if frame_after is not None or collided:
                raise ValueError("STOP has no new physical observation or collision")
        else:
            self._check_frame(frame_after, self.step + 1)
            if action == "COMPLETE_SUBGOAL" and (frame_after.pixels != self._frames[self.step].pixels or collided):
                raise ValueError("Logical completion cannot advance RGB physics or collide")
        self.actions.append({"step": self.step, "action": action, "collided": collided})
        if self._new_message is not None:
            self.message_history.append(deepcopy(self._new_message))
            self._new_message = None
        self.step += 1
        if frame_after is not None:
            self._frames[self.step], self._assets[frame_after.asset_id] = frame_after, frame_after
        self._pending = None
        self.closed = action == "STOP" or self.step == self.max_steps

    def fail(self, reason):
        self._active()
        if type(reason) is not str or not reason:
            raise ValueError("Record a failure reason")
        self.failure_reason, self.closed = reason, True


def call_policy(session, callback):
    """CPU-testable boundary for a future real model adapter.

    A callback gets only an owned public request and its allowed PNG bytes. Bad
    output or a call error closes the trial, preserving the raw result; no
    corrective Oracle action is returned and no automatic retry is performed.
    """
    request = session.request()
    record = {"request": deepcopy(request), "raw_response": None, "response_type": None, "action": None, "error": None}
    try:
        raw = callback(deepcopy(request), session.assets_for(request["request_id"]))
        record["response_type"] = type(raw).__name__
        if type(raw) is str:
            record["raw_response"] = raw
        action = parse_action(raw, request)
        record["action"] = action
    except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        session.fail("policy_call_or_action_contract_failed")
    return record
