"""Run one fixed, intervention-free Oracle navigation episode on Habitat-Sim."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_from_coeffs

from scripts.check_habitat import PROJECT_ROOT, pose, write_json
from scripts.export_video import export_video, find_ffmpeg


def make_configuration(config: dict, scene: Path, gpu_id: int) -> habitat_sim.Configuration:
    simulator = habitat_sim.SimulatorConfiguration()
    simulator.scene_id = str(scene)
    simulator.gpu_device_id = gpu_id
    simulator.enable_physics = True
    sensors = []
    for name, sensor_type in (("rgb", habitat_sim.SensorType.COLOR),
                              ("depth", habitat_sim.SensorType.DEPTH)):
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = name
        sensor.sensor_type = sensor_type
        sensor.resolution = config["resolution"]
        sensor.position = [0.0, config["sensor_height"], 0.0]
        sensors.append(sensor)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = sensors
    agent.action_space = {}
    for name, amount in (("move_forward", config["forward_step_m"]),
                         ("turn_left", config["turn_angle_degrees"]),
                         ("turn_right", config["turn_angle_degrees"])):
        agent.action_space[name] = habitat_sim.agent.ActionSpec(
            name, habitat_sim.agent.ActuationSpec(amount=amount)
        )
    return habitat_sim.Configuration(simulator, [agent])


def save_frame(output: Path, frame: int, observations: dict) -> dict:
    rgb = observations["rgb"]
    depth = observations["depth"]
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise RuntimeError(f"Unexpected RGB shape: {rgb.shape}")
    if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all():
        raise RuntimeError("Invalid depth observation")
    paths = {
        "rgb": f"rgb/{frame:06d}.png",
        "depth": f"depth/{frame:06d}.npy",
        "depth_preview": f"depth_preview/{frame:06d}.png",
    }
    Image.fromarray(rgb[:, :, :3]).save(output / paths["rgb"])
    np.save(output / paths["depth"], depth)
    Image.fromarray((np.clip(depth / 10.0, 0.0, 1.0) * 255).astype(np.uint8)).save(
        output / paths["depth_preview"]
    )
    return paths


def shortest_path(pathfinder, start: np.ndarray, goal: np.ndarray):
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = goal
    if not pathfinder.find_path(path) or not np.isfinite(path.geodesic_distance):
        return None
    return path


def validate_config(config: dict) -> None:
    if not isinstance(config["max_steps"], int) or config["max_steps"] < 1:
        raise ValueError("max_steps must be a positive integer")
    for name in ("success_radius_m", "forward_step_m", "turn_angle_degrees", "sensor_height"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("start_position", "goal_position"):
        value = np.asarray(config[name], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain three finite coordinates")
    rotation = np.asarray(config["start_rotation_xyzw"], dtype=float)
    if rotation.shape != (4,) or not np.isfinite(rotation).all() or not np.isclose(np.linalg.norm(rotation), 1.0):
        raise ValueError("start_rotation_xyzw must be a unit quaternion in x,y,z,w order")


def run(config: dict, scene: Path, gpu_id: int, output: Path) -> dict:
    validate_config(config)
    if not scene.is_file():
        raise FileNotFoundError(f"Scene not found: {scene}")
    # Keep each run's frames and metrics together; never mix them with an older run.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use a new --output-dir.")
    for name in ("rgb", "depth", "depth_preview"):
        (output / name).mkdir(parents=True, exist_ok=True)

    with habitat_sim.Simulator(make_configuration(config, scene, gpu_id)) as sim:
        sim.seed(config["seed"])
        if not sim.pathfinder.is_loaded:
            raise RuntimeError("The scene requires a loaded navigation mesh")
        sim.pathfinder.seed(config["seed"])
        start = np.asarray(config["start_position"], dtype=np.float32)
        goal = np.asarray(config["goal_position"], dtype=np.float32)
        if not sim.pathfinder.is_navigable(start) or not sim.pathfinder.is_navigable(goal):
            raise ValueError("The configured start and goal must be navigable in this scene")
        oracle = shortest_path(sim.pathfinder, start, goal)
        if oracle is None or oracle.geodesic_distance <= config["success_radius_m"]:
            raise ValueError("Start and goal must be connected and outside the success radius")
        optimal_length = float(oracle.geodesic_distance)
        write_json(output / "oracle_path.json", {
            "start": start.tolist(), "goal": goal.tolist(),
            "distance_m": optimal_length,
            "points": [np.asarray(point).tolist() for point in oracle.points],
        })

        state = habitat_sim.AgentState()
        state.position = start
        state.rotation = quat_from_coeffs(np.asarray(config["start_rotation_xyzw"]))
        agent = sim.initialize_agent(0, state)
        follower = habitat_sim.nav.GreedyGeodesicFollower(
            sim.pathfinder, agent, goal_radius=config["success_radius_m"], stop_key="STOP"
        )
        episode = {
            **config, "scene": str(scene), "gpu_device_id": gpu_id,
            "controller": "oracle_greedy_geodesic_follower", "events": [],
            "shortest_path_length_m": optimal_length,
            "versions": {"python": platform.python_version(),
                         "habitat_sim": habitat_sim.__version__, "numpy": np.__version__},
        }
        write_json(output / "episode.json", episode)
        observations = sim.get_sensor_observations()
        trajectory = [{
            "step": None, "action": "INITIALIZE", **pose(agent), "collided": False,
            "distance_m": 0.0, "distance_to_goal_m": optimal_length,
            "frames": save_frame(output, 0, observations),
        }]
        path_length = 0.0
        stopped = False
        collision_count = 0
        termination = "timeout"
        planner_error = None
        final_distance = optimal_length
        for step in range(config["max_steps"]):
            previous_position = agent.get_state().position.copy()
            try:
                action = follower.next_action_along(goal)
            except habitat_sim.errors.GreedyFollowerError as error:
                termination = "planner_error"
                planner_error = str(error) or type(error).__name__
                break
            # Habitat-Sim has movement actions only. STOP terminates this episode
            # without moving the agent, and is counted as one action in the budget.
            if action == "STOP":
                stopped = True
                collided = False
            else:
                observations = sim.step(action)
                collided = bool(observations["collided"])
            position = agent.get_state().position
            distance = float(np.linalg.norm(position - previous_position))
            path_length += distance
            collision_count += int(collided)
            remaining = shortest_path(sim.pathfinder, position, goal)
            final_distance = float(remaining.geodesic_distance) if remaining is not None else None
            trajectory.append({
                "step": step, "action": action, **pose(agent), "collided": collided,
                "distance_m": distance, "distance_to_goal_m": final_distance,
                "frames": save_frame(output, step + 1, observations),
            })
            if stopped:
                termination = "stopped"
                break

        success = stopped and final_distance is not None and final_distance <= config["success_radius_m"]
        if stopped and not success:
            termination = "stopped_outside_goal"
        metrics = {
            "episode_id": config["episode_id"], "success": bool(success),
            "success_rate": float(success),
            "spl": float(success) * optimal_length / max(optimal_length, path_length),
            "stopped": stopped, "termination": termination,
            "steps": len(trajectory) - 1, "max_steps": config["max_steps"],
            "path_length_m": path_length, "shortest_path_length_m": optimal_length,
            "navigation_error_m": final_distance,
            "success_radius_m": config["success_radius_m"],
            "collision_count": collision_count, "planner_error": planner_error,
        }
        write_json(output / "trajectory.json", trajectory)
        write_json(output / "metrics.json", metrics)
        return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/simulator/navigation_001.json")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--no-video", action="store_true", help="Save frames and metrics without encoding an MP4")
    args = parser.parse_args()
    if args.gpu_id < 0:
        parser.error("--gpu-id must be non-negative")
    if not math.isfinite(args.video_fps) or args.video_fps <= 0:
        parser.error("--video-fps must be finite and positive")
    if not args.no_video:
        find_ffmpeg()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scene = Path(config["scene"])
    if not scene.is_absolute():
        scene = PROJECT_ROOT / scene
    output = args.output_dir if args.output_dir is not None else PROJECT_ROOT / "outputs" / config["episode_id"]
    metrics = run(config, scene.resolve(), args.gpu_id, output)
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to: {output}")
    if not args.no_video:
        video = export_video(output, args.video_fps)
        print(f"Video written to: {video}")
    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
