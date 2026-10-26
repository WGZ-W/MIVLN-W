"""Run an Ignore Intervention negative control and validate its expected task failure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from scripts.export_video import export_video, find_ffmpeg
from scripts.intervention_timing import load_json, project_path, write_json
from scripts.navigation_scene import check_scene_match


def validate_reference(config: dict, reference: Path) -> None:
    episode = load_json(reference / "episode.json")
    if episode.get("events"):
        raise ValueError("The reference must have no intervention events")
    keys = ("start_position", "start_rotation_xyzw", "goal_positions", "seed", "initial_goal",
            "forward_step_m", "turn_angle_degrees", "success_radius_m", "sensor_height", "resolution", "max_steps")
    if any(episode[key] != config[key] for key in keys):
        raise ValueError("Reference and negative-control configurations must use the same task and navigation parameters")
    check_scene_match(config, episode)


def check_negative(output: Path, config: dict, reference: Path = None) -> dict:
    """Recheck saved records without importing Habitat or starting a renderer."""
    metrics = load_json(output / "metrics.json")
    trajectory = load_json(output / "trajectory.json")
    delivered = load_json(output / "events.json")
    truth = load_json(output / "final_task_state.json")
    policy = load_json(output / "final_policy_state.json")
    event = config["events"][0]
    trigger = event["trigger"]["value"]
    old_goal = config["initial_goal"]
    new_goal = event["task_update"]["new_goal"]
    once = (len(delivered) == 1 and delivered[0]["event_id"] == event["event_id"]
            and delivered[0]["task_update"] == event["task_update"])
    logged = delivered[0] if once else None
    reached_trigger = len(trajectory) > trigger + 1

    def within_radius(goal):
        distance = metrics["goal_distances_m"].get(goal)
        return distance is not None and math.isfinite(distance) and 0 <= distance <= config["success_radius_m"]

    def pose(row):
        return {key: row[key] for key in ("position", "rotation_xyzw")}

    distance_b = metrics["goal_distances_m"].get(new_goal)
    checks = {
        "ignore_policy_selected": metrics["intervention_policy"] == "ignore",
        "intervention_received": metrics["intervention_received"] is True,
        "event_delivered_once": once and metrics["delivered_event_count"] == 1,
        "event_delivered_at_expected_step": bool(once and logged["delivered_step"] == trigger),
        "truth_updated_before_action": bool(reached_trigger and once
            and logged["task_state_before"]["active_goal"] == old_goal
            and logged["task_state_after"]["active_goal"] == new_goal
            and trajectory[trigger + 1]["step"] == trigger
            and trajectory[trigger + 1]["active_goal"] == new_goal
            and trajectory[trigger + 1]["delivered_event_ids"] == [event["event_id"]]),
        "truth_keeps_new_goal_and_cancels_old": truth["active_goal"] == new_goal
        and old_goal in truth["cancelled_goals"] and metrics["final_goal"] == new_goal,
        "policy_keeps_old_goal": policy["active_goal"] == old_goal
        and metrics["policy_goal"] == old_goal
        and all(row["policy_goal"] == old_goal for row in trajectory),
        "policy_state_ignores_event": bool(once and logged["ignored_by_policy"]
            and logged["policy_task_state_before"] == logged["policy_task_state_after"]
            and metrics["ignored_event_count"] == 1),
        "no_pose_reset_at_event": bool(reached_trigger and once
            and logged["pose_before"] == pose(trajectory[trigger])
            and logged["pose_before"] == logged["pose_after"]),
        "no_policy_replan": load_json(output / "replans.json") == [],
        "stopped_at_old_goal": bool(trajectory) and trajectory[-1]["action"] == "STOP"
        and metrics["stopped"] and metrics["stopped_goal"] == old_goal and within_radius(old_goal),
        "new_goal_remains_reachable_but_unreached": distance_b is not None
        and math.isfinite(distance_b) and distance_b > config["success_radius_m"],
        "task_correctly_failed": metrics["dynamic_success"] is False
        and metrics["success"] is False and metrics["goal_update_compliance"] is False,
        "no_compliant_action_reported": metrics["first_compliant_action_step"] is None
        and metrics["response_latency_steps"] is None,
        "failure_is_cancelled_goal_not_timeout": metrics["termination"] == "stopped_at_cancelled_goal"
        and metrics["planner_error"] is None and metrics["steps"] == len(trajectory) - 1
        and metrics["steps"] <= config["max_steps"],
    }
    reference_error = None
    if reference is not None:
        try:
            validate_reference(config, reference)
            checks["reference_configuration_matches"] = True
            reference_metrics = load_json(reference / "metrics.json")
            reference_rows = load_json(reference / "trajectory.json")
            checks["reference_succeeded_at_old_goal"] = bool(reference_metrics["success"]
                and reference_metrics["stopped"] and reference_rows[-1]["action"] == "STOP"
                and reference_metrics["final_goal"] == old_goal)
            # Old outputs need not contain the new policy/truth fields. Compare
            # physical execution only; the task's truth must differ after delivery.
            keys = ("step", "action", "position", "rotation_xyzw", "collided", "distance_m")
            physical = lambda rows: [{key: row[key] for key in keys} for row in rows]
            checks["physical_trajectory_matches_reference"] = physical(trajectory) == physical(reference_rows)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            checks["reference_configuration_matches"] = False
            reference_error = str(error)
    return {
        "negative_check_passed": all(checks.values()), "checks": checks,
        "expected_dynamic_success": False, "observed_dynamic_success": metrics["dynamic_success"],
        "reference_checked": reference is not None, "reference_error": reference_error, "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, help="Existing no-event control with the same task parameters")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=10.0)
    args = parser.parse_args()
    if args.gpu_id < 0 or not math.isfinite(args.video_fps) or args.video_fps <= 0:
        parser.error("--gpu-id must be non-negative and --video-fps finite and positive")
    config = load_json(args.config)
    from scripts.run_navigation import run, validate_config

    validate_config(config)
    if (len(config.get("events", [])) != 1
            or config["events"][0]["task_update"].get("old_goal") != config["initial_goal"]):
        parser.error("The negative control requires one goal-replacement event with an explicit matching old_goal")
    if args.reference_dir is not None:
        validate_reference(config, args.reference_dir)
    if not args.no_video:
        find_ffmpeg()
    run(config, project_path(config["scene"]), args.gpu_id, args.output_dir, intervention_policy="ignore")
    report = check_negative(args.output_dir, config, args.reference_dir)
    write_json(args.output_dir / "negative_check.json", report)
    print(json.dumps(report, indent=2))
    if not args.no_video:
        print(f"Video written to: {export_video(args.output_dir, args.video_fps)}")
    # Task failure is expected. The command succeeds only when all negative-control
    # checks pass, excluding planner errors, timeouts, and missing events.
    return 0 if report["negative_check_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
