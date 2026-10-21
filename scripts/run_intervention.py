"""Run and validate matched control/target-replacement Oracle branches."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from scripts.export_video import export_video, find_ffmpeg


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_pair(output: Path, event: dict) -> dict:
    """Check recorded trajectories without importing Habitat or using a GPU."""

    def load(branch: str, name: str):
        return json.loads((output / branch / name).read_text(encoding="utf-8"))

    control = load("control", "trajectory.json")
    intervention = load("intervention", "trajectory.json")
    control_metrics = load("control", "metrics.json")
    intervention_metrics = load("intervention", "metrics.json")
    delivered = load("intervention", "events.json")
    trigger = event["trigger"]["value"]
    old_goal = event["task_update"]["old_goal"]
    new_goal = event["task_update"]["new_goal"]
    reached_trigger = len(control) > trigger + 1 and len(intervention) > trigger + 1
    once = len(delivered) == 1 and delivered[0]["event_id"] == event["event_id"]
    logged = delivered[0] if once else None

    def pose(row: dict) -> dict:
        return {key: row[key] for key in ("position", "rotation_xyzw")}

    def stopped_at(trajectory, metrics, goal):
        distance = metrics["goal_distances_m"].get(goal)
        return (bool(trajectory) and trajectory[-1]["action"] == "STOP"
                and metrics["stopped_goal"] == goal and metrics["final_goal"] == goal
                and distance is not None and math.isfinite(distance)
                and 0 <= distance <= metrics["success_radius_m"])

    checks = {
        "control_success": bool(control_metrics["dynamic_success"])
        and stopped_at(control, control_metrics, old_goal),
        "intervention_success": bool(intervention_metrics["dynamic_success"])
        and stopped_at(intervention, intervention_metrics, new_goal),
        "trigger_reached_in_both_branches": reached_trigger,
        # Index 0 is INITIALIZE; indices 1..k are actions 0..k-1.
        "shared_prefix_matches": reached_trigger
        and control[:trigger + 1] == intervention[:trigger + 1],
        "event_delivered_once": once,
        "event_delivered_at_expected_step": once and logged["delivered_step"] == trigger,
        "no_pose_reset_at_event": bool(reached_trigger and once
            and logged["pose_before"] == pose(intervention[trigger])
            and logged["pose_after"] == logged["pose_before"]),
        "task_state_updated": bool(once
            and logged["task_state_before"]["active_goal"] == old_goal
            and logged["task_state_after"]["active_goal"] == new_goal
            and old_goal in logged["task_state_after"]["cancelled_goals"]),
        "event_applied_before_action": bool(reached_trigger and once
            and intervention[trigger + 1]["step"] == trigger
            and intervention[trigger + 1]["active_goal"] == new_goal
            and intervention[trigger + 1]["delivered_event_ids"] == [event["event_id"]]),
        "control_has_no_events": load("control", "events.json") == [],
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "trigger_step": trigger, "shared_prefix_actions": trigger,
        "shared_prefix_frames": trigger + 1,
        "control": control_metrics, "intervention": intervention_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs/simulator/intervention_001.json")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=10.0)
    args = parser.parse_args()
    if args.gpu_id < 0:
        parser.error("--gpu-id must be non-negative")
    if not math.isfinite(args.video_fps) or args.video_fps <= 0:
        parser.error("--video-fps must be finite and positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if len(config.get("events", [])) != 1:
        parser.error("The paired experiment requires exactly one replace_goal event")
    # Import after argument parsing so --help works without Habitat installed.
    from scripts.run_navigation import run, validate_config
    from scripts.check_habitat import write_json

    validate_config(config)
    event = config["events"][0]
    if event["task_update"].get("old_goal") != config["initial_goal"]:
        parser.error("The paired experiment requires old_goal to match initial_goal")
    output = args.output_dir if args.output_dir is not None else PROJECT_ROOT / "outputs" / config["episode_id"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use a new --output-dir.")
    if not args.no_video:
        find_ffmpeg()
    scene = Path(config["scene"])
    if not scene.is_absolute():
        scene = PROJECT_ROOT / scene
    for branch in ("control", "intervention"):
        branch_config = deepcopy(config)
        branch_config["episode_id"] = f"{config['episode_id']}_{branch}"
        if branch == "control":
            branch_config["events"] = []
        print(f"Running {branch} branch...", flush=True)
        run(branch_config, scene.resolve(), args.gpu_id, output / branch)
    summary = check_pair(output, event)
    write_json(output / "pair_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written to: {output}")
    if not args.no_video:
        for branch in ("control", "intervention"):
            print(f"Video written to: {export_video(output / branch, args.video_fps)}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
