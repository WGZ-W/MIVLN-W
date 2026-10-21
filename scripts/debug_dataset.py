"""Prepare and validate a small, fixed collection of Oracle intervention tasks."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import hashlib
import io
import math
from pathlib import Path
import platform
import random
import subprocess
import sys

from scripts.intervention_timing import (
    PROJECT_ROOT, TIMINGS, load_json, project_path, summarize as summarize_timing, write_json,
)


def distance_to_polyline(point, points) -> float:
    """Minimum Euclidean distance to the reference path's continuous segments."""
    distances = []
    for start, end in zip(points, points[1:]):
        direction = [b - a for a, b in zip(start, end)]
        length_squared = sum(value * value for value in direction)
        fraction = (sum((p - a) * d for p, a, d in zip(point, start, direction)) / length_squared
                    if length_squared else 0.0)
        fraction = min(1.0, max(0.0, fraction))
        closest = [a + fraction * d for a, d in zip(start, direction)]
        distances.append(math.dist(point, closest))
    return min(distances) if distances else float("inf")


def sample_task(pathfinder, template: dict, task_id: str, seed: int, previous: list,
                max_attempts: int) -> tuple:
    import numpy as np
    from scripts.run_navigation import shortest_path, validate_config

    pathfinder.seed(seed)
    for attempt in range(1, max_attempts + 1):
        start, goal_a, goal_b = [np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
                                  for _ in range(3)]
        points = (start, goal_a, goal_b)
        if any(not np.isfinite(point).all() or not pathfinder.is_navigable(point) for point in points):
            continue
        if any(abs(float(point[1] - start[1])) > 0.5 for point in (goal_a, goal_b)):
            continue
        if any(math.dist(point, old) < 1.0 for point in points for old in previous):
            continue
        initial = shortest_path(pathfinder, start, goal_a)
        to_b = shortest_path(pathfinder, start, goal_b)
        between = shortest_path(pathfinder, goal_a, goal_b)
        if (initial is None or to_b is None or between is None
                or not 6.0 <= initial.geodesic_distance <= 12.0
                or not 2.0 <= to_b.geodesic_distance <= 20.0
                or not 2.0 <= between.geodesic_distance <= 20.0):
            continue
        route_distance = distance_to_polyline(goal_b, initial.points)
        if route_distance < 2.0:
            continue
        yaw = random.Random(seed).uniform(-math.pi, math.pi)
        config = deepcopy(template)
        config.update({
            "episode_id": task_id, "seed": seed, "initial_goal": "point_a",
            "initial_instruction": "Go to point A.", "start_position": start.tolist(),
            "start_rotation_xyzw": [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)],
            "goal_positions": {"point_a": goal_a.tolist(), "point_b": goal_b.tolist()},
            # This placeholder is never run. Reference-derived triggers replace it.
            "events": [{"event_id": f"{task_id}_replace_goal", "trigger": {"type": "step", "value": 1},
                        "utterance": "Cancel going to point A. Go to point B instead.",
                        "task_update": {"operation": "replace_goal", "old_goal": "point_a", "new_goal": "point_b"}}],
        })
        config.pop("goal_position", None)
        validate_config(config)
        record = {
            "task_id": task_id, "seed": seed, "sampling_attempt": attempt,
            "start_position": start.tolist(), "goal_a": goal_a.tolist(), "goal_b": goal_b.tolist(),
            "start_to_a_m": float(initial.geodesic_distance),
            "start_to_b_m": float(to_b.geodesic_distance),
            "a_to_b_m": float(between.geodesic_distance),
            "b_distance_to_reference_polyline_m": route_distance,
            "base_config": f"configs/{task_id}/base.json",
            "reference_config": f"configs/{task_id}/reference.json",
            "output_subdir": f"tasks/{task_id}",
        }
        return config, record
    raise ValueError(f"Cannot sample {task_id} after {max_attempts} attempts; no GPU failures were filtered")


def prepare(args) -> int:
    from scripts.run_navigation import validate_config
    import habitat_sim
    import numpy as np

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new directory.")
    template = load_json(args.base_config)
    validate_config(template)
    navmesh = project_path(template["scene"]).with_suffix(".navmesh")
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh)):
        raise FileNotFoundError(navmesh)
    prepared = []
    previous = []
    for index in range(args.count):
        config, record = sample_task(pathfinder, template, f"task_{index + 1:03d}",
                                     args.seed + index, previous, args.max_attempts)
        previous.extend([record["start_position"], record["goal_a"], record["goal_b"]])
        prepared.append((config, record))
    # Finish sampling before creating files; a geometry failure leaves no partial dataset.
    for config, record in prepared:
        write_json(output / record["base_config"], config)
        reference = deepcopy(config)
        reference["episode_id"] += "_reference"
        reference["events"] = []
        write_json(output / record["reference_config"], reference)
    manifest = {
        "purpose": "Oracle debugging collection; not a semantic VLN benchmark",
        "task_count": args.count, "sampling_seed": args.seed,
        "expected_pairs": args.count * len(TIMINGS), "expected_episodes": args.count * 7,
        "navmesh": str(navmesh), "navmesh_sha256": hashlib.sha256(navmesh.read_bytes()).hexdigest(),
        "filters": {"start_to_a_m": [6.0, 12.0], "start_to_b_m": [2.0, 20.0],
                    "a_to_b_m": [2.0, 20.0], "max_height_difference_m": 0.5,
                    "b_distance_to_reference_polyline_m": 2.0, "cross_task_point_separation_m": 1.0},
        "versions": {"python": platform.python_version(), "habitat_sim": habitat_sim.__version__,
                     "numpy": np.__version__}, "tasks": [record for _, record in prepared],
    }
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {args.count} fixed tasks, {manifest['expected_pairs']} pairs: {output}")
    return 0


def execute_stage(command: list, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"Stage failed ({result.returncode}); log: {log}", flush=True)
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]))
    return result.returncode


def run(args) -> int:
    output = args.output_dir.resolve()
    manifest = load_json(output / "manifest.json")
    for task in manifest["tasks"]:
        destination = output / task["output_subdir"]
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Task output is not empty: {destination}. Prepare a new dataset directory.")
        for name in ("base_config", "reference_config"):
            if not (output / task[name]).is_file():
                raise FileNotFoundError(output / task[name])
    statuses = {}
    prefix = [sys.executable, "-u", "-m"]
    for index, task in enumerate(manifest["tasks"], start=1):
        destination = output / task["output_subdir"]
        commands = [
            ("reference", [*prefix, "scripts.run_navigation", "--config", str(output / task["reference_config"]),
                           "--gpu-id", str(args.gpu_id), "--output-dir", str(destination / "reference"), "--no-video"]),
            ("timing_prepare", [*prefix, "scripts.intervention_timing", "prepare", "--base-config",
                                str(output / task["base_config"]), "--reference-dir", str(destination / "reference"),
                                "--output-dir", str(destination / "timing"), "--keep-target"]),
            ("timing_run", [*prefix, "scripts.intervention_timing", "run", "--output-dir", str(destination / "timing"),
                            "--gpu-id", str(args.gpu_id), "--no-video"]),
        ]
        statuses[task["task_id"]] = []
        for stage, command in commands:
            print(f"[{index}/{len(manifest['tasks'])}] {task['task_id']}: {stage}", flush=True)
            log = destination / f"{stage}.log"
            code = execute_stage(command, log)
            statuses[task["task_id"]].append({"stage": stage, "returncode": code,
                                               "log": str(log.relative_to(output))})
            write_json(output / "run_status.json", statuses)
            if code:
                break
    return 0 if summarize(output)["passed"] else 1


def summarize(output: Path) -> dict:
    manifest = load_json(output / "manifest.json")
    statuses = load_json(output / "run_status.json") if (output / "run_status.json").is_file() else {}
    rows = []
    flat_cases = []
    for task in manifest["tasks"]:
        destination = output / task["output_subdir"]
        row = {"task_id": task["task_id"], "passed": False, "reference_success": False,
               "reference_matches_controls": False, "fixed_task_parameters": False,
               "failed_stages": [stage for stage in statuses.get(task["task_id"], [])
                   if stage["returncode"]], "errors": [], "timings": []}
        try:
            metrics = load_json(destination / "reference/metrics.json")
            reference = load_json(destination / "reference/trajectory.json")
            row["reference_success"] = bool(metrics["success"] and metrics["stopped"]
                and reference[-1]["action"] == "STOP" and metrics["navigation_error_m"] is not None
                and metrics["navigation_error_m"] <= metrics["success_radius_m"])
            with redirect_stdout(io.StringIO()):
                timing = summarize_timing(destination / "timing")
            row["timings"] = timing["cases"]
            controls = [load_json(destination / f"timing/{name}/control/trajectory.json") for name, _ in TIMINGS]
            row["reference_matches_controls"] = all(control == reference for control in controls)
            base = load_json(output / task["base_config"])
            variants = [load_json(destination / f"timing/{name}/{branch}/episode.json")
                        for name, _ in TIMINGS for branch in ("control", "intervention")]
            keys = ("start_position", "start_rotation_xyzw", "goal_positions", "seed", "initial_goal",
                    "forward_step_m", "turn_angle_degrees", "success_radius_m", "sensor_height", "resolution")
            row["fixed_task_parameters"] = all(
                project_path(variant["scene"]) == project_path(base["scene"])
                and all(variant[key] == base[key] for key in keys) for variant in variants)
            row["passed"] = bool(row["reference_success"] and row["reference_matches_controls"]
                                 and row["fixed_task_parameters"] and timing["passed"] and not row["failed_stages"])
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            row["errors"].append(str(error))
        for name, _ in TIMINGS:
            case = next((case for case in row["timings"] if case["name"] == name), None)
            flat_cases.append({"task_id": task["task_id"], **(case if case is not None else {
                "name": name, "passed": False, "error": "Timing result unavailable; inspect task logs"})})
        rows.append(row)
    report = {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "passed_tasks": sum(row["passed"] for row in rows), "total_tasks": len(rows),
        "passed_pairs": sum(case["passed"] for case in flat_cases), "total_pairs": len(flat_cases),
        "failed_tasks": [row["task_id"] for row in rows if not row["passed"]], "tasks": rows,
    }
    write_json(output / "summary.json", report)
    columns = list(dict.fromkeys(key for case in flat_cases for key in case))
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_cases)
    print(f"Tasks passed: {report['passed_tasks']}/{report['total_tasks']}; "
          f"pairs passed: {report['passed_pairs']}/{report['total_pairs']}; "
          f"failed tasks: {report['failed_tasks']}")
    print(f"Summary: {output / 'summary.json'}")
    return report


def videos(args) -> int:
    from scripts.export_video import export_video

    output = args.output_dir.resolve()
    manifest = load_json(output / "manifest.json")
    tasks = [task for task in manifest["tasks"] if args.task_id is None or task["task_id"] == args.task_id]
    if not tasks:
        raise ValueError(f"Unknown task ID: {args.task_id}")
    failures = []
    for task in tasks:
        destination = output / task["output_subdir"]
        episodes = [destination / "reference", *[destination / f"timing/{name}/intervention" for name, _ in TIMINGS]]
        for episode in episodes:
            if (episode / "trajectory.mp4").is_file():
                print(f"Already exported: {episode / 'trajectory.mp4'}")
                continue
            try:
                print(f"Video: {export_video(episode)}")
            except (OSError, RuntimeError, ValueError) as error:
                failures.append({"episode": str(episode), "error": str(error)})
    write_json(output / "video_status.json", {"failed_exports": failures})
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Sample and save fixed tasks using only CPU navmesh queries")
    preparation.add_argument("--base-config", type=Path,
                             default=PROJECT_ROOT / "configs/simulator/intervention_001.json")
    preparation.add_argument("--count", type=int, default=10)
    preparation.add_argument("--seed", type=int, default=20261021)
    preparation.add_argument("--max-attempts", type=int, default=5000)
    running = commands.add_parser("run", help="Run reference navigation and three paired timings per task")
    running.add_argument("--gpu-id", type=int, default=0)
    reporting = commands.add_parser("summarize", help="Recheck existing results without using a GPU")
    exporting = commands.add_parser("videos", help="Export one reference and three intervention videos per task")
    exporting.add_argument("--task-id", help="Export only this task, e.g. task_001")
    for command in (preparation, running, reporting, exporting):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.count < 1 or args.max_attempts < 1:
            parser.error("--count and --max-attempts must be positive")
        return prepare(args)
    if args.command == "run":
        if args.gpu_id < 0:
            parser.error("--gpu-id must be non-negative")
        return run(args)
    if args.command == "videos":
        return videos(args)
    return 0 if summarize(args.output_dir.resolve())["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
