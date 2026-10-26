"""Prepare, run, and summarize early/middle/late target-replacement experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

from scripts.run_intervention import check_pair
from scripts.navigation_scene import check_scene_match, navmesh_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMINGS = (("early", 0.25), ("middle", 0.50), ("late", 0.75))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def project_path(path: str) -> Path:
    return (PROJECT_ROOT / path).resolve()


def select_triggers(trajectory: list) -> list:
    """Choose pre-action states using reduction in remaining geodesic distance."""
    if (len(trajectory) < 3 or trajectory[0]["action"] != "INITIALIZE"
            or trajectory[-1]["action"] != "STOP"):
        raise ValueError("The reference must be a completed navigation trajectory")
    initial_distance = trajectory[0]["distance_to_goal_m"]
    if initial_distance is None or not math.isfinite(initial_distance) or initial_distance <= 0:
        raise ValueError("The reference initial distance must be finite and positive")
    if any(row["step"] != step for step, row in enumerate(trajectory[1:])):
        raise ValueError("Reference action steps must be contiguous and start at zero")
    selected = []
    for name, fraction in TIMINGS:
        for index, row in enumerate(trajectory[:-1]):
            distance = row["distance_to_goal_m"]
            if (distance is not None and math.isfinite(distance) and distance >= 0
                    and 1.0 - distance / initial_distance >= fraction
                    and trajectory[index + 1]["action"] != "STOP"):
                selected.append({
                    "name": name, "target_progress": fraction,
                    "actual_progress": 1.0 - distance / initial_distance,
                    "trigger_step": index, "position": row["position"],
                    "distance_to_original_goal_m": distance,
                })
                break
        else:
            raise ValueError(f"No valid pre-STOP reference state for {name} intervention")
    if len({case["trigger_step"] for case in selected}) != len(TIMINGS):
        raise ValueError("The reference is too short to provide three distinct trigger steps")
    return selected


def check_reference(config: dict, reference: Path) -> list:
    episode = load_json(reference / "episode.json")
    metrics = load_json(reference / "metrics.json")
    trajectory = load_json(reference / "trajectory.json")
    if not metrics["success"] or not metrics["stopped"] or episode.get("events"):
        raise ValueError("Use a successful, intervention-free reference episode")
    for key in ("initial_goal", "start_position", "start_rotation_xyzw", "seed",
                "forward_step_m", "turn_angle_degrees", "success_radius_m", "sensor_height",
                "resolution", "max_steps"):
        if config[key] != episode[key]:
            raise ValueError(f"Base configuration and reference differ: {key}")
    check_scene_match(config, episode)
    goal = config["initial_goal"]
    if config["goal_positions"][goal] != episode["goal_positions"][goal]:
        raise ValueError("Base configuration and reference use different initial goal positions")
    if (math.dist(trajectory[0]["position"], config["start_position"]) > 1e-5
            or any(row["active_goal"] != goal for row in trajectory)):
        raise ValueError("Reference trajectory does not follow the configured initial goal")
    return trajectory


def inspect_target(pathfinder, config: dict, trajectory: list, cases: list, candidate,
                   min_separation: float, min_route_distance: float, max_distance: float):
    """Return geometry diagnostics, or None if the fixed target fails the filters."""
    import habitat_sim
    import numpy as np

    candidate = np.asarray(candidate, dtype=np.float32)
    start = np.asarray(config["start_position"], dtype=np.float32)
    if (candidate.shape != (3,) or not np.isfinite(candidate).all()
            or not pathfinder.is_navigable(candidate)
            or abs(float(candidate[1] - start[1])) > 0.5):
        return None
    route = np.asarray([row["position"] for row in trajectory], dtype=np.float32)
    route_distance = float(np.min(np.linalg.norm(route - candidate, axis=1)))
    if route_distance < min_route_distance:
        return None
    anchors = {"start": start, "point_a": np.asarray(
        config["goal_positions"][config["initial_goal"]], dtype=np.float32)}
    anchors.update({case["name"]: np.asarray(case["position"], dtype=np.float32) for case in cases})
    distances = {}
    for name, position in anchors.items():
        path = habitat_sim.ShortestPath()
        path.requested_start = position
        path.requested_end = candidate
        if (not pathfinder.find_path(path) or not math.isfinite(path.geodesic_distance)
                or not min_separation <= path.geodesic_distance <= max_distance):
            return None
        distances[name] = float(path.geodesic_distance)
    return {"position": candidate.tolist(), "distances_m": distances,
            "min_reference_frame_distance_m": route_distance}


def sample_target(pathfinder, config: dict, trajectory: list, cases: list, seed: int,
                  max_attempts: int, min_separation: float, min_route_distance: float,
                  max_distance: float) -> dict:
    # These navmesh queries never create a renderer or use a GPU.
    pathfinder.seed(seed)
    for attempt in range(1, max_attempts + 1):
        record = inspect_target(pathfinder, config, trajectory, cases,
                                pathfinder.get_random_navigable_point(), min_separation,
                                min_route_distance, max_distance)
        if record is not None:
            return {**record, "attempt": attempt}
    raise ValueError(f"No target satisfying the geometry filters after {max_attempts} samples")


def prepare(args) -> int:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new directory.")
    from scripts.run_navigation import validate_config
    import habitat_sim
    import numpy as np

    config = load_json(args.base_config)
    validate_config(config)
    if len(config.get("events", [])) != 1:
        raise ValueError("The base configuration must contain exactly one replace_goal event")
    if config.get("scene_dataset_config") and not getattr(args, "keep_target", False):
        raise ValueError("HSSD semantic targets must not be resampled; use --keep-target")
    if args.min_separation <= 2 * config["success_radius_m"]:
        raise ValueError("min-separation must exceed twice the success radius")
    reference = args.reference_dir.resolve()
    trajectory = check_reference(config, reference)
    cases = select_triggers(trajectory)
    navmesh = navmesh_path(config)
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh)):
        raise FileNotFoundError(f"Cannot load navigation mesh: {navmesh}")
    old_goal = config["initial_goal"]
    new_goal = config["events"][0]["task_update"]["new_goal"]
    if getattr(args, "keep_target", False):
        target = inspect_target(pathfinder, config, trajectory, cases, config["goal_positions"][new_goal],
                                args.min_separation, args.min_route_distance, args.max_distance)
        if target is None:
            raise ValueError("The fixed target fails the reference-route filters; retain this task as a failure")
        target["source"] = "base_config"
    else:
        target = sample_target(pathfinder, config, trajectory, cases, args.seed, args.max_attempts,
                               args.min_separation, args.min_route_distance, args.max_distance)
    for case in cases:
        variant = deepcopy(config)
        variant["episode_id"] = f"independent_target_{case['name']}"
        variant["goal_positions"] = {
            old_goal: config["goal_positions"][old_goal], new_goal: target["position"],
        }
        variant["events"] = [{
            "event_id": f"replace_goal_{case['name']}",
            "trigger": {"type": "step", "value": case["trigger_step"]},
            "utterance": (config["events"][0]["utterance"] if config.get("scene_dataset_config")
                          else "Cancel going to point A. Go to point B instead."),
            "task_update": {"operation": "replace_goal", "old_goal": old_goal, "new_goal": new_goal},
        }]
        validate_config(variant)
        case["config"] = f"configs/{case['name']}.json"
        case["output_subdir"] = case["name"]
        write_json(output / case["config"], variant)
    manifest = {
        "reference_dir": str(reference), "base_config": str(args.base_config.resolve()),
        "reference_trajectory_sha256": hashlib.sha256((reference / "trajectory.json").read_bytes()).hexdigest(),
        "navmesh": str(navmesh), "navmesh_sha256": hashlib.sha256(navmesh.read_bytes()).hexdigest(),
        "sampling_seed": args.seed, "target_b": target,
        "filters": {"min_separation_m": args.min_separation,
                    "min_reference_frame_distance_m": args.min_route_distance,
                    "max_geodesic_distance_m": args.max_distance, "max_height_difference_m": 0.5},
        "versions": {"python": platform.python_version(), "habitat_sim": habitat_sim.__version__,
                     "numpy": np.__version__},
        "cases": cases,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print(f"Prepared configurations: {output}")
    return 0


def summarize(output: Path) -> dict:
    manifest = load_json(output / "manifest.json")
    rows = []
    controls = []
    for case in manifest["cases"]:
        row = {key: case[key] for key in ("name", "target_progress", "actual_progress", "trigger_step")}
        row.update({"passed": False, "error": None})
        try:
            config = load_json(output / case["config"])
            case_dir = output / case["output_subdir"]
            summary = check_pair(case_dir, config["events"][0])
            row["passed"] = summary["passed"]
            row["failed_checks"] = [name for name, passed in summary["checks"].items() if not passed]
            row.update({
                "control_steps": summary["control"]["steps"],
                "intervention_steps": summary["intervention"]["steps"],
                "navigation_error_m": summary["intervention"]["navigation_error_m"],
                "path_length_m": summary["intervention"]["path_length_m"],
                "post_intervention_path_length_m": summary["intervention"]["post_intervention_path_length_m"],
                "post_intervention_spl": summary["intervention"]["post_intervention_spl"],
                "collision_count": summary["intervention"]["collision_count"],
            })
            controls.append(load_json(case_dir / "control/trajectory.json"))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            row["passed"] = False
            row["error"] = str(error)
        rows.append(row)
    controls_identical = bool(controls) and len(controls) == len(rows) and all(
        trajectory == controls[0] for trajectory in controls[1:])
    status = output / "run_status.json"
    failures = load_json(status)["failed_processes"] if status.is_file() else []
    report = {
        "passed": bool(rows) and controls_identical and not failures and all(row["passed"] for row in rows),
        "passed_cases": sum(row["passed"] for row in rows), "total_cases": len(rows),
        "controls_identical": controls_identical, "target_b": manifest["target_b"], "cases": rows,
        "failed_processes": failures,
    }
    write_json(output / "summary.json", report)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))
    return report


def run_cases(args) -> int:
    output = args.output_dir.resolve()
    manifest = load_json(output / "manifest.json")
    # Validate all destinations before launching the first GPU process.
    for case in manifest["cases"]:
        case_dir = output / case["output_subdir"]
        if case_dir.exists() and any(case_dir.iterdir()):
            raise FileExistsError(f"Case directory is not empty: {case_dir}. Prepare a new experiment directory.")
        if not (output / case["config"]).is_file():
            raise FileNotFoundError(output / case["config"])
    failures = []
    for case in manifest["cases"]:
        command = [sys.executable, "-m", "scripts.run_intervention", "--gpu-id", str(args.gpu_id),
                   "--config", str(output / case["config"]),
                   "--output-dir", str(output / case["output_subdir"])]
        if args.no_video:
            command.append("--no-video")
        print(f"Running {case['name']} intervention at step {case['trigger_step']}...", flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode:
            failures.append({"name": case["name"], "returncode": result.returncode})
    write_json(output / "run_status.json", {"failed_processes": failures})
    report = summarize(output)
    return 0 if report["passed"] and not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Prepare three configurations using only CPU navmesh queries")
    preparation.add_argument("--base-config", type=Path,
                             default=PROJECT_ROOT / "configs/simulator/intervention_001.json")
    preparation.add_argument("--reference-dir", type=Path,
                             default=PROJECT_ROOT / "outputs/intervention_001/control")
    preparation.add_argument("--seed", type=int, default=123)
    preparation.add_argument("--keep-target", action="store_true",
                             help="Validate and retain the base configuration's B target without resampling")
    preparation.add_argument("--max-attempts", type=int, default=1000)
    preparation.add_argument("--min-separation", type=float, default=2.0)
    preparation.add_argument("--min-route-distance", type=float, default=1.0)
    preparation.add_argument("--max-distance", type=float, default=20.0)
    running = commands.add_parser("run", help="Run all three matched pairs and write a summary")
    running.add_argument("--gpu-id", type=int, default=0)
    running.add_argument("--no-video", action="store_true")
    reporting = commands.add_parser("summarize", help="Recompute checks from existing trajectories without a GPU")
    for command in (preparation, running, reporting):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if (args.max_attempts < 1 or any(not math.isfinite(value) or value <= 0 for value in
                (args.min_separation, args.min_route_distance, args.max_distance))
                or args.max_distance < args.min_separation):
            parser.error("Sampling limits must be finite and positive, with max-distance >= min-separation")
        return prepare(args)
    if args.command == "run":
        if args.gpu_id < 0:
            parser.error("--gpu-id must be non-negative")
        return run_cases(args)
    return 0 if summarize(args.output_dir.resolve())["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
