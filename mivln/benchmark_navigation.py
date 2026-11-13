"""Versioned three-task V loop and independent action/measurement scoring.

The policy boundary accepts only public V requests and their scoped PNG bytes.
The backend and evaluator are trusted host components with private geometry.
This module has no simulator, filesystem, NumPy or model-client dependency.
"""

from __future__ import annotations

from copy import deepcopy
import math

from .cancel_task import (PROTOCOL as CANCEL_PROTOCOL, CancellationTaskState,
    CancellationVisualSession, evaluate_cancellation, inspect_trajectory, pose, same_pose)
from .subgoals import completion_is_valid, sequence_satisfied
from .visual_frames import RGBFrame
from .visual_inputs import call_policy


PROTOCOL = "active_v_navigation_v1"
TASK_TYPES = ("replacement", "subgoal", "cancel_task")


def validate_case(config):
    """Validate the private single-event contract, never expose it to a policy."""
    if config.get("navigation_protocol") != PROTOCOL or config.get("task_type") not in TASK_TYPES:
        raise ValueError("Use the versioned active three-task navigation protocol")
    if type(config.get("max_steps")) is not int or config["max_steps"] < 1:
        raise ValueError("A positive, shared action budget is required")
    pose({"position": config["start_position"], "rotation_xyzw": config["start_rotation_xyzw"]})
    for key in ("forward_step_m", "turn_angle_degrees", "success_radius_m", "yaw_tolerance_degrees"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid navigation limit: {key}")
    if config["turn_angle_degrees"] > 180 or config["yaw_tolerance_degrees"] > 180:
        raise ValueError("Angles are in degrees and must not exceed 180")
    if type(config.get("initial_instruction")) is not str or not config["initial_instruction"].strip():
        raise ValueError("Missing initial language")
    root = config.get("initial_goal")
    if type(root) is not str or root not in config["goal_positions"]:
        raise ValueError("Initial goal must bind private geometry")
    for position in config["goal_positions"].values():
        pose({"position": position, "rotation_xyzw": [0., 0., 0., 1.]})
    if type(config.get("events")) is not list or len(config["events"]) != 1:
        raise ValueError("Each case requires exactly one planned intervention")
    event = config["events"][0]
    if any(type(event.get(key)) is not str or not event[key].strip() for key in ("event_id", "utterance")):
        raise ValueError("Missing intervention identity or language")
    trigger = event.get("trigger")
    if (type(trigger) is not dict or set(trigger) != {"type", "value"} or trigger["type"] != "step"
            or type(trigger["value"]) is not int or not 0 <= trigger["value"] < config["max_steps"]):
        raise ValueError("Freeze an external pre-action delivery step within the budget")
    update = event.get("task_update")
    keys = {"replacement": {"operation", "old_goal", "new_goal"},
            "subgoal": {"operation", "subgoal", "resume_goal"}, "cancel_task": {"operation"}}
    operations = {"replacement": "replace_goal", "subgoal": "insert_subgoal", "cancel_task": "cancel_task"}
    family = config["task_type"]
    if type(update) is not dict or set(update) != keys[family] or update["operation"] != operations[family]:
        raise ValueError("Task type and exact private update fields disagree")
    if family != "cancel_task":
        target = update["new_goal" if family == "replacement" else "subgoal"]
        if (type(target) is not str or target == root or target not in config["goal_positions"]
                or update["old_goal" if family == "replacement" else "resume_goal"] != root):
            raise ValueError("Intervention must bind a different instance and preserve the correct A")
    elif config.get("cancel_protocol") != CANCEL_PROTOCOL:
        raise ValueError("Cancellation must declare cancel_task_v1")
    return config


def goal_complete(config, measurements, goal):
    """Use measured distance, facing and instance diagnostics, never policy intent."""
    measurement = measurements[goal]
    return completion_is_valid(measurement["distance_m"], config["success_radius_m"],
        measurement["yaw_error_degrees"], config["yaw_tolerance_degrees"], measurement["diagnostic"])


def complete_action(config, state, action, measurements, step, completions, violations):
    """Only a valid measured B completion restores evaluator A.

    An invalid logical action is still an executed, counted policy action. Keep
    navigating and retain the violation; never silently repair it or restore A.
    """
    if action != "COMPLETE_SUBGOAL":
        return
    valid = bool(config["task_type"] == "subgoal" and state.suspended_goals
                 and goal_complete(config, measurements, state.active_goal))
    completions.append({"step": step, "goal_id": state.active_goal, "valid": valid})
    if valid:
        state.complete_subgoal(state.active_goal)
    else:
        violations.append({"step": step, "reason": "invalid_subgoal_completion"})


def evaluate_episode(config, trajectory, events, *, interface_error=None):
    """Reconstruct truth from private labels, actions and collector measurements.

    Delivered messages are required for intervention success. Early STOP, a
    missing response and budget exhaustion retain failed slots. Measurements
    come from the backend; a saved CPU run additionally replays its kinematics
    and synthetic diagnostics during the script-level read-only audit.
    """
    validate_case(config)
    result = {"task_type": config["task_type"], "task_success": False,
        "evidence_valid": False, "intervention_received": False, "errors": [],
        "completions": [], "semantic_violations": [], "interface_error": interface_error}
    try:
        measured = inspect_trajectory(config, trajectory)
        result.update(measured)
        k, event = config["events"][0]["trigger"]["value"], config["events"][0]
        if type(events) is not list or len(events) > 1:
            raise ValueError("An intervention can be delivered only once")
        received = bool(events)
        if received:
            expected = {"event_id": event["event_id"], "delivered_step": k,
                "task_update": event["task_update"], "pose_before": pose(trajectory[k]), "pose_after": pose(trajectory[k])}
            if events != [expected] or measured["stopped_step"] is not None and measured["stopped_step"] < k:
                raise ValueError("Event identity, label, time or unchanged delivery pose mismatch")
        elif measured["steps"] > k:
            raise ValueError("An executed action skipped its scheduled intervention")
        truth = CancellationTaskState(config["initial_goal"])
        for step, row in enumerate(trajectory[1:]):
            if step == k and received:
                truth.apply_update(event["task_update"])
            if row["delivered_event_ids"] != ([event["event_id"]] if step == k and received else []):
                raise ValueError("Per-action delivery IDs differ from actual delivery")
            complete_action(config, truth, row["action"], row["measurements"], step,
                            result["completions"], result["semantic_violations"])
        # Delivery can precede a failed policy call without an executed action.
        if received and measured["steps"] == k:
            truth.apply_update(event["task_update"])
        stopped = measured["stopped_step"] is not None
        budget = measured["steps"] <= config["max_steps"]
        result.update(evidence_valid=True, intervention_received=received, budget_satisfied=budget,
                      final_task_state=truth.to_dict())
        if config["task_type"] == "cancel_task":
            cancellation = evaluate_cancellation(config, trajectory, events)
            result["cancellation"] = cancellation
            result["evidence_valid"] = cancellation["evidence_valid"]
            success = cancellation["cancellation_success"]
        else:
            terminal = bool(stopped and goal_complete(config, trajectory[-1]["measurements"], truth.active_goal))
            sequence = True
            if config["task_type"] == "subgoal":
                sequence = sequence_satisfied(config["initial_goal"], event["task_update"]["subgoal"], truth, result["completions"])
            result.update(terminal_goal_complete=terminal, sequence_satisfied=sequence)
            success = bool(received and stopped and budget and terminal and sequence and not result["semantic_violations"])
        result["task_success"] = bool(success and interface_error is None)
        if not result["task_success"]:
            result["failure_reason"] = ("interface_failure" if interface_error else "intervention_not_received" if not received
                else "semantic_violation" if result["semantic_violations"] else "task_obligations_not_satisfied")
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        result.update(evidence_valid=False, task_success=False, failure_reason="invalid_evidence")
        result["errors"].append(str(error))
    return result


def run_episode(config, backend, callback, *, session_id=None, asset_ids=None):
    """Execute one continuous episode through a backend and a V callback.

    Backend contract: ``camera`` is public calibration; ``pose()``,
    ``measurements()`` and ``rgb()`` are trusted collector queries; ``act(name)``
    executes exactly one action and returns a boolean collision. Positions are
    metres, rotations XYZW, headings degrees. STOP/COMPLETE_SUBGOAL must not move.
    The backend starts at the configured pose. It is never reset at delivery or
    at completion. Optional IDs support independent replay of saved PNG bindings.
    """
    validate_case(config)
    session = CancellationVisualSession(config["initial_instruction"], backend.camera,
        max_steps=config["max_steps"], forward_step_m=config["forward_step_m"],
        turn_angle_degrees=config["turn_angle_degrees"], success_radius_m=config["success_radius_m"],
        max_history_frames=4, history_stride=1, session_id=session_id)
    truth = CancellationTaskState(config["initial_goal"])
    asset_ids = iter(asset_ids) if asset_ids is not None else None

    def capture(step):
        width, height = backend.camera["width"], backend.camera["height"]
        return RGBFrame.collect(session.session_id, step, width, height, backend.rgb(),
                                asset_id=next(asset_ids) if asset_ids is not None else None)

    def row(step, action, distance, collided, delivered, frame):
        before = deepcopy(backend.pose())
        measurements = deepcopy(backend.measurements())
        if not same_pose(before, backend.pose()):
            raise ValueError("Collector diagnostics changed the physical pose")
        return {"step": step, "action": action, **before, "distance_m": distance,
            "collided": collided, "delivered_event_ids": delivered,
            "measurements": measurements, "rgb_asset_id": frame.asset_id}

    current = capture(0)
    session.initialize(current)
    if not same_pose(backend.pose(), {"position": config["start_position"], "rotation_xyzw": config["start_rotation_xyzw"]}):
        raise ValueError("Backend did not start at the frozen pose")
    trajectory = [row(None, "INITIALIZE", 0., False, [], current)]
    events, records, assets, completions, violations = [], [], {}, [], []
    runtime_error = None
    event, k = config["events"][0], config["events"][0]["trigger"]["value"]
    while not session.closed:
        step, before = session.step, deepcopy(backend.pose())
        delivered = []
        if step == k:
            truth.apply_update(event["task_update"])
            session.deliver(event)
            events.append({"event_id": event["event_id"], "delivered_step": step,
                "task_update": deepcopy(event["task_update"]), "pose_before": before,
                "pose_after": deepcopy(backend.pose())})
            delivered = [event["event_id"]]
        request = session.request()
        assets.update(session.assets_for(request["request_id"]))
        record = call_policy(session, callback)
        records.append(record)
        if record["error"]:
            break
        try:
            if not same_pose(before, backend.pose()):
                raise ValueError("Policy callback changed backend pose outside action execution")
            action = record["action"]["action"]
            collided = backend.act(action)
            if type(collided) is not bool:
                raise ValueError("Backend collision feedback must be boolean")
            next_frame = None if action == "STOP" else capture(step+1)
            session.advance(record["action"], next_frame, collided=collided)
            current = next_frame or current
            after = row(step, action, math.dist(before["position"], backend.pose()["position"]), collided, delivered, current)
            trajectory.append(after)
            complete_action(config, truth, action, after["measurements"], step, completions, violations)
        except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError) as error:
            # Keep the last issued input and raw response even when the trusted
            # backend fails mid-action. An incomplete physical transition cannot
            # be reconstructed as valid evidence or repaired with an Oracle.
            runtime_error = f"{type(error).__name__}: {error}"
            record["execution_error"] = runtime_error
            break
    interface_error = runtime_error or session.failure_reason
    metrics = evaluate_episode(config, trajectory, events, interface_error=interface_error)
    if runtime_error:
        metrics["evidence_valid"] = False
        metrics["errors"].append(runtime_error)
    return {"trajectory": trajectory, "events": events, "policy_records": records, "assets": assets,
            "metrics": metrics, "final_task_state": truth.to_dict(), "interface_error": interface_error}
