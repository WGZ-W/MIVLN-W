"""Versioned cancel-and-stop semantics and independent pose-based evaluation.

The legacy goal-reaching state/evaluator files remain byte-identical for frozen
experiments. Opt into this extension explicitly; cancelling a task does not
select a dummy goal and does not count arrival at an object as success.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Mapping, Optional

from .task_state import TaskState
from .visual_inputs import VisualInputSession


PROTOCOL = "cancel_task_v1"
POSITION_TOLERANCE_M = 1e-5
ROTATION_TOLERANCE = 1e-6


def validate_cancel_update(update):
    if not isinstance(update, Mapping) or set(update) != {"operation"} or update["operation"] != "cancel_task":
        raise ValueError("Cancellation requires exactly {operation: cancel_task}")


@dataclass
class CancellationTaskState(TaskState):
    """Evaluator state with an explicit no-active-goal cancelled condition.

    Completed history and existing constraints are retained. All unfinished
    navigation goals (including any suspended ones) lose their obligations.
    This does not claim that the robot has physically stopped. Cancelled states
    cannot be revived by a later update/completion in this single-event version.
    """
    active_goal: Optional[str]

    def apply_update(self, update):
        if not isinstance(update, Mapping):
            raise ValueError("Task updates must be mappings")
        if self.status == "cancelled":
            raise ValueError("Cancelled navigation cannot be resumed by this protocol")
        if update.get("operation") != "cancel_task":
            return super().apply_update(update)
        validate_cancel_update(update)
        if self.status != "active" or not isinstance(self.active_goal, str) or not self.active_goal:
            raise ValueError("Only an active navigation task can be cancelled")
        for goal in [self.active_goal, *self.suspended_goals]:
            if goal not in self.cancelled_goals:
                self.cancelled_goals.append(goal)
        self.active_goal = None
        self.suspended_goals.clear()
        self.status = "cancelled"

    def complete_subgoal(self, goal):
        if self.status == "cancelled":
            raise ValueError("Cancellation cannot be reversed by a completion claim")
        return super().complete_subgoal(goal)


class CancellationVisualSession(VisualInputSession):
    """Opt-in V delivery extension; public RGB/action schemas stay unchanged.

    The model only gets the human message, not the cancel opcode, an evaluator
    state, or an automatically generated STOP. Its actual output is evaluated.
    """

    def deliver(self, event, *, marked_frame=None, marker_evidence=None):
        if type(event) is not dict or type(event.get("task_update")) is not dict:
            raise ValueError("Malformed host intervention")
        if event["task_update"].get("operation") != "cancel_task":
            return super().deliver(event, marked_frame=marked_frame, marker_evidence=marker_evidence)
        self._active()
        validate_cancel_update(event["task_update"])
        if type(event.get("event_id")) is not str or not event["event_id"].strip():
            raise ValueError("Host cancellation needs a unique event identity")
        if self._pending is not None or self._delivered:
            raise ValueError("Deliver cancellation once, before the policy chooses its action")
        trigger = event.get("trigger")
        if (type(trigger) is not dict or trigger.get("type") != "step"
                or type(trigger.get("value")) is not int or trigger["value"] != self.step):
            raise ValueError("Cancellation is not due at the current pre-action step")
        if type(event.get("utterance")) is not str or not event["utterance"].strip():
            raise ValueError("Cancellation needs a nonempty human message")
        if marked_frame is not None or marker_evidence is not None:
            raise ValueError("Cancellation has no floor-region marker payload")
        self._new_message = {"delivered_step": self.step, "message": event["utterance"]}
        self._delivered = True


def _vector(value, size, name):
    try:
        valid = type(value) is list and len(value) == size and all(type(v) in (int, float) and math.isfinite(v) for v in value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"Invalid {name}")
    if size == 4 and not math.isclose(sum(v*v for v in value), 1, abs_tol=1e-5):
        raise ValueError("Rotation requires an XYZW unit quaternion")
    return value


def pose(value):
    return {"position": _vector(value["position"], 3, "position"),
            "rotation_xyzw": _vector(value["rotation_xyzw"], 4, "rotation")}


def same_pose(left, right):
    a, b = pose(left), pose(right)
    # q and -q describe the same orientation; do not flag a sign convention as
    # a reset. A real rotation/translation during STOP still fails.
    rotation_error = min(math.dist(a["rotation_xyzw"], b["rotation_xyzw"]),
                         math.dist(a["rotation_xyzw"], [-v for v in b["rotation_xyzw"]]))
    return math.dist(a["position"], b["position"]) <= POSITION_TOLERANCE_M and rotation_error <= ROTATION_TOLERANCE


def validate_config(config):
    if config.get("cancel_protocol") != PROTOCOL:
        raise ValueError("Use the explicit cancel_task_v1 protocol")
    if type(config.get("max_steps")) is not int or config["max_steps"] < 1:
        raise ValueError("Cancellation needs a positive action budget")
    if type(config.get("initial_goal")) is not str or not config["initial_goal"]:
        raise ValueError("Record the initial goal obligation")
    if type(config.get("initial_instruction")) is not str or not config["initial_instruction"].strip():
        raise ValueError("Record the initial instruction")
    pose({"position": config["start_position"], "rotation_xyzw": config["start_rotation_xyzw"]})
    if config.get("subgoal_protocol") or config.get("initial_constraints"):
        raise ValueError("First cancellation experiments have one initial goal, without mixed obligations")
    if type(config.get("events")) is not list or len(config["events"]) != 1:
        raise ValueError("Cancellation episodes require one scheduled intervention")
    event = config["events"][0]
    if type(event) is not dict:
        raise ValueError("Malformed cancellation event")
    validate_cancel_update(event.get("task_update"))
    for key in ("event_id", "utterance"):
        if type(event.get(key)) is not str or not event[key].strip():
            raise ValueError(f"Missing cancellation {key}")
    trigger = event.get("trigger")
    if (type(trigger) is not dict or set(trigger) != {"type", "value"} or trigger["type"] != "step"
            or type(trigger["value"]) is not int or not 0 <= trigger["value"] < config["max_steps"]):
        raise ValueError("Cancellation trigger must precede an action within the budget")
    return deepcopy(config)


def inspect_trajectory(config, trajectory):
    """Read poses/actions, not success flags. This is not native mesh replay."""
    if type(trajectory) is not list or not trajectory or trajectory[0].get("action") != "INITIALIZE":
        raise ValueError("Missing initialized trajectory")
    if trajectory[0].get("step") is not None or trajectory[0].get("delivered_event_ids") != []:
        raise ValueError("Initialization cannot execute an action or deliver an event")
    start = {"position": config["start_position"], "rotation_xyzw": config["start_rotation_xyzw"]}
    if not same_pose(start, trajectory[0]):
        raise ValueError("Trajectory changed the frozen start pose")
    length, stopped_step, collisions = 0., None, 0
    for step, (before, after) in enumerate(zip(trajectory, trajectory[1:])):
        pose(before)
        pose(after)
        if type(after.get("step")) is not int or after["step"] != step or stopped_step is not None:
            raise ValueError("Actions must be contiguous and cannot continue after STOP")
        if after.get("action") not in ("move_forward", "turn_left", "turn_right", "STOP", "COMPLETE_SUBGOAL"):
            raise ValueError("Unknown recorded action")
        if type(after.get("collided")) is not bool or type(after.get("delivered_event_ids")) is not list:
            raise ValueError("Missing measured collision/event record")
        distance = math.dist(before["position"], after["position"])
        reported = after.get("distance_m")
        if (type(reported) not in (int, float) or not math.isfinite(reported) or reported < 0
                or not math.isclose(distance, reported, abs_tol=POSITION_TOLERANCE_M)):
            raise ValueError("Recorded distance differs from measured positions")
        if after["action"] in ("STOP", "COMPLETE_SUBGOAL"):
            if not same_pose(before, after) or after["collided"]:
                raise ValueError("Logical STOP/completion moved, rotated or collided")
        if after["action"] in ("turn_left", "turn_right") and distance > POSITION_TOLERANCE_M:
            raise ValueError("A turn translated the agent")
        length += distance
        collisions += int(after["collided"])
        if after["action"] == "STOP":
            stopped_step = step
    return {"steps": len(trajectory)-1, "stopped_step": stopped_step, "path_length_m": length,
            "collision_count": collisions}


def evaluate_cancellation(config, trajectory, events):
    """Immediate STOP at step k is required, with a fixed numerical tolerance.

    A delayed turn/STOP fails even if translation is zero. Net displacement
    cannot conceal walking out and back: cumulative post-cancel travel is also
    measured. No goal distance, heading-to-object or visibility is required.
    Evidence failures retain a failed planned slot and a separate validity flag.
    """
    config = validate_config(config)
    result = {"protocol": PROTOCOL, "evidence_valid": False, "cancellation_received": False,
        "cancellation_success": False, "immediate_stop": False, "stop_delay_actions": None,
        "post_cancel_action_count": None, "post_cancel_path_length_m": None,
        "post_cancel_net_displacement_m": None, "stop_pose_unchanged": False,
        "budget_satisfied": False, "task_state": None, "failure_reason": None, "errors": []}
    try:
        measured = inspect_trajectory(config, trajectory)
        result.update(measured)
        result["budget_satisfied"] = measured["steps"] <= config["max_steps"]
        if type(events) is not list or len(events) > 1:
            raise ValueError("Cancellation delivery must be logged exactly once if reached")
        planned = config["events"][0]
        k = planned["trigger"]["value"]
        n = measured["steps"]
        received = bool(events)
        if events:
            logged = events[0]
            if (type(logged) is not dict or logged.get("event_id") != planned["event_id"]
                    or type(logged.get("delivered_step")) is not int or logged["delivered_step"] != k
                    or logged.get("task_update") != planned["task_update"] or n < k
                    or measured["stopped_step"] is not None and measured["stopped_step"] < k):
                raise ValueError("Cancellation delivered at the wrong slot or after termination")
            if not same_pose(logged["pose_before"], trajectory[k]) or not same_pose(logged["pose_before"], logged["pose_after"]):
                raise ValueError("Pose changed during event delivery")
        elif n > k:
            raise ValueError("Scheduled cancellation was omitted before an executed action")
        for row in trajectory[1:]:
            expected = [planned["event_id"]] if received and row["step"] == k else []
            if row["delivered_event_ids"] != expected:
                raise ValueError("Per-action event IDs do not match the once-only delivery")
        truth = CancellationTaskState(config["initial_goal"])
        if received:
            truth.apply_update(planned["task_update"])
        result.update(evidence_valid=True, cancellation_received=received, task_state=truth.to_dict())
        stop = measured["stopped_step"]
        if received:
            positions = [row["position"] for row in trajectory[k:]]
            result["post_cancel_path_length_m"] = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
            result["post_cancel_net_displacement_m"] = math.dist(positions[0], positions[-1])
            result["post_cancel_action_count"] = n-k
            if stop is not None:
                result["stop_delay_actions"] = stop-k
        result["stop_pose_unchanged"] = bool(stop is not None and same_pose(trajectory[stop], trajectory[stop+1]))
        result["immediate_stop"] = bool(received and stop == k)
        invalid_completion = any(row["action"] == "COMPLETE_SUBGOAL" for row in trajectory[1:])
        success = bool(received and result["budget_satisfied"] and result["immediate_stop"]
            and result["stop_pose_unchanged"] and result["post_cancel_path_length_m"] <= POSITION_TOLERANCE_M
            and not invalid_completion)
        result["cancellation_success"] = success
        if not success:
            result["failure_reason"] = ("cancellation_not_received" if not received else
                "action_budget_exceeded" if not result["budget_satisfied"] else
                "invalid_subgoal_completion" if invalid_completion else
                "no_stop" if stop is None else "did_not_stop_immediately")
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        result["errors"].append(str(error))
        result["failure_reason"] = "invalid_execution_evidence"
    return result
