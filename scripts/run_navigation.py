"""Run an Oracle navigation episode with optional target-replacement events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_from_coeffs

from mivln import EpisodeEvaluator, EventEngine, InterventionEvent, TaskState
from scripts.check_habitat import PROJECT_ROOT, pose, write_json
from scripts.export_video import export_video, find_ffmpeg
from scripts.navigation_scene import navmesh_path, project_path


def make_configuration(config: dict, scene: Path, gpu_id: int) -> habitat_sim.Configuration:
    simulator = habitat_sim.SimulatorConfiguration()
    if config.get("scene_dataset_config"):
        simulator.scene_dataset_config_file = str(project_path(config["scene_dataset_config"]))
        simulator.scene_id = config["scene_id"]
    else:
        simulator.scene_id = str(scene)
    simulator.gpu_device_id = gpu_id
    simulator.enable_physics = True
    sensors = []
    kinds = [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]
    if config.get("semantic_sensor", False):
        kinds.append(("semantic", habitat_sim.SensorType.SEMANTIC))
    for name, sensor_type in kinds:
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = name
        sensor.sensor_type = sensor_type
        sensor.resolution = config["resolution"]
        sensor.position = [0.0, config["sensor_height"], 0.0]
        sensors.append(sensor)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height = config.get("agent_height", 1.5)
    agent.radius = config.get("agent_radius", 0.1)
    agent.sensor_specifications = sensors
    agent.action_space = {}
    for name, amount in (("move_forward", config["forward_step_m"]),
                         ("turn_left", config["turn_angle_degrees"]),
                         ("turn_right", config["turn_angle_degrees"])):
        agent.action_space[name] = habitat_sim.agent.ActionSpec(
            name, habitat_sim.agent.ActuationSpec(amount=amount)
        )
    return habitat_sim.Configuration(simulator, [agent])


def save_frame(output: Path, frame: int, observations: dict, require_semantic: bool = False) -> dict:
    if require_semantic and "semantic" not in observations:
        raise RuntimeError("Semantic sensor was enabled but produced no observation")
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
    if "semantic" in observations:
        semantic = observations["semantic"]
        if semantic.shape != depth.shape or not np.issubdtype(semantic.dtype, np.integer):
            raise RuntimeError("Invalid semantic observation")
        paths["semantic"] = f"semantic/{frame:06d}.npy"
        np.save(output / paths["semantic"], semantic)
    return paths


def scene_provenance(config: dict, scene: Path) -> dict:
    """Validate and hash inputs before starting a renderer or creating outputs."""
    files = {"scene": scene.resolve()}
    if config.get("scene_dataset_config"):
        files["scene_dataset_config"] = project_path(config["scene_dataset_config"])
        expected = files["scene_dataset_config"].parent / "scenes" / f"{config['scene_id']}.scene_instance.json"
        if scene.resolve() != expected.resolve():
            raise ValueError("scene must be the scene_instance file for scene_id in this HSSD dataset")
    mesh = navmesh_path({**config, "scene": str(scene)})
    if config.get("navmesh") or mesh.is_file():
        files["navmesh"] = mesh
    record = {}
    for key, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{key} file not found: {path}")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if config.get(f"{key}_sha256") is not None and config[f"{key}_sha256"] != digest:
            raise ValueError(f"{key} SHA256 mismatch; do not reuse coordinates with changed scene inputs")
        record[key] = str(path)
        record[f"{key}_sha256"] = digest
    if config.get("scene_dataset_config"):
        instance = json.loads(scene.read_text(encoding="utf-8"))
        if instance.get("articulated_object_instances"):
            raise ValueError("This navigation entry supports rigid HSSD scenes, not articulated scenes")
        record["expected_rigid_objects"] = len(instance.get("object_instances", []))
    return record


def load_navigation_mesh(pathfinder, config: dict) -> dict:
    """Load an explicitly configured mesh; never silently fall back to another one."""
    if config.get("navmesh") and not pathfinder.load_nav_mesh(str(navmesh_path(config))):
        raise RuntimeError("Cannot load the configured navigation mesh")
    if not pathfinder.is_loaded:
        raise RuntimeError("The scene requires a loaded navigation mesh")
    if not config.get("scene_dataset_config"):
        return {}
    settings = pathfinder.nav_mesh_settings
    if (not math.isclose(settings.agent_height, config["agent_height"], abs_tol=1e-6)
            or not math.isclose(settings.agent_radius, config["agent_radius"], abs_tol=1e-6)
            or not settings.include_static_objects):
        raise ValueError("HSSD navmesh must include static objects and match agent_height/agent_radius")
    return {"include_static_objects": bool(settings.include_static_objects),
            "agent_height": float(settings.agent_height), "agent_radius": float(settings.agent_radius),
            "navigable_area_m2": float(pathfinder.navigable_area), "island_count": pathfinder.num_islands}


def shortest_path(pathfinder, start: np.ndarray, goal: np.ndarray):
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = goal
    if not pathfinder.find_path(path) or not np.isfinite(path.geodesic_distance):
        return None
    return path


def configured_goals(config: dict) -> dict:
    goals = config.get("goal_positions")
    if goals is None:
        goals = {config["initial_goal"]: config["goal_position"]}
    if not isinstance(goals, dict) or not goals:
        raise ValueError("goal_positions must be a non-empty goal-to-position mapping")
    if any(not isinstance(goal, str) or not goal.strip() for goal in goals):
        raise ValueError("Goal identifiers must be non-empty strings")
    if config["initial_goal"] not in goals:
        raise ValueError("initial_goal must have a configured position")
    return goals


def validate_config(config: dict) -> None:
    for key in ("scene_dataset_config", "navmesh"):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f"{key} must be a non-empty path string")
    dataset = config.get("scene_dataset_config")
    if str(config.get("scene", "")).endswith(".scene_instance.json") and not dataset:
        raise ValueError("A scene_instance file requires scene_dataset_config and scene_id")
    if dataset:
        if not isinstance(config.get("scene_id"), str) or not config["scene_id"] or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in config["scene_id"]):
            raise ValueError("HSSD scene_id must be an identifier, not a path")
        for key in ("navmesh", "navmesh_sha256", "agent_height", "agent_radius"):
            if key not in config:
                raise ValueError(f"HSSD configuration requires {key}")
    for key in ("agent_height", "agent_radius"):
        if key in config and (not np.isfinite(config[key]) or config[key] <= 0):
            raise ValueError(f"{key} must be finite and positive")
    if "semantic_sensor" in config and type(config["semantic_sensor"]) is not bool:
        raise ValueError("semantic_sensor must be a boolean")
    for key in ("navmesh_sha256", "scene_sha256", "scene_dataset_config_sha256"):
        if key in config and (not isinstance(config[key], str) or len(config[key]) != 64
                              or any(char not in "0123456789abcdef" for char in config[key])):
            raise ValueError(f"{key} must be a lowercase SHA256 digest")
    if type(config["max_steps"]) is not int or config["max_steps"] < 1:
        raise ValueError("max_steps must be a positive integer")
    for name in ("success_radius_m", "forward_step_m", "turn_angle_degrees", "sensor_height"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    goals = configured_goals(config)
    for name, position in [("start_position", config["start_position"]), *goals.items()]:
        value = np.asarray(position, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain three finite coordinates")
    rotation = np.asarray(config["start_rotation_xyzw"], dtype=float)
    if rotation.shape != (4,) or not np.isfinite(rotation).all() or not np.isclose(np.linalg.norm(rotation), 1.0):
        raise ValueError("start_rotation_xyzw must be a unit quaternion in x,y,z,w order")
    events = config.get("events", [])
    if not isinstance(events, list) or len(events) > 1:
        raise ValueError("Version 0.1 supports zero or one intervention")
    state = TaskState(active_goal=config["initial_goal"])
    for value in events:
        event = InterventionEvent.from_dict(value)
        if type(event.trigger_step) is not int or event.trigger_step >= config["max_steps"]:
            raise ValueError("Event trigger must be an integer within the action budget")
        old_goal = state.active_goal
        state.apply_update(event.task_update)
        if state.active_goal not in goals or state.active_goal == old_goal:
            raise ValueError("Intervention must select a different, configured goal")


def path_record(path, start, goal, goal_id: str, step=None) -> dict:
    return {
        "step": step, "goal_id": goal_id,
        "start": np.asarray(start).tolist(), "goal": np.asarray(goal).tolist(),
        "distance_m": float(path.geodesic_distance),
        "points": [np.asarray(point).tolist() for point in path.points],
    }


def run(config: dict, scene: Path, gpu_id: int, output: Path,
        intervention_policy: str = "follow") -> dict:
    if intervention_policy not in ("follow", "ignore"):
        raise ValueError("intervention_policy must be 'follow' or 'ignore'")
    validate_config(config)
    goals = {goal: np.asarray(position, dtype=np.float32)
             for goal, position in configured_goals(config).items()}
    events = [InterventionEvent.from_dict(value) for value in config.get("events", [])]
    task_state = TaskState(active_goal=config["initial_goal"])
    # Evaluator truth and the policy's belief are separate objects. Only the
    # evaluator is unconditionally updated by delivered intervention events.
    policy_state = TaskState(active_goal=config["initial_goal"])
    event_engine = EventEngine(events)
    provenance = scene_provenance(config, scene)
    # Keep each run's frames and metrics together; never mix them with an older run.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use a new --output-dir.")
    frame_directories = ["rgb", "depth", "depth_preview"]
    if config.get("semantic_sensor", False):
        frame_directories.append("semantic")
    for name in frame_directories:
        (output / name).mkdir(parents=True, exist_ok=True)

    with habitat_sim.Simulator(make_configuration(config, scene, gpu_id)) as sim:
        sim.seed(config["seed"])
        mesh_settings = load_navigation_mesh(sim.pathfinder, config)
        loaded_objects = None
        if config.get("scene_dataset_config"):
            loaded_objects = len(sim.get_rigid_object_manager().get_object_handles())
            if loaded_objects != provenance["expected_rigid_objects"]:
                raise RuntimeError("Loaded rigid object count does not match the configured HSSD scene instance")
        sim.pathfinder.seed(config["seed"])
        start = np.asarray(config["start_position"], dtype=np.float32)
        goal = goals[policy_state.active_goal]
        if not sim.pathfinder.is_navigable(start):
            raise ValueError("The configured start and goal must be navigable in this scene")
        for goal_id, position in goals.items():
            if not sim.pathfinder.is_navigable(position) or shortest_path(sim.pathfinder, start, position) is None:
                raise ValueError(f"Goal is not navigable and reachable from the start: {goal_id}")
        for event in events:
            replacement = goals[event.task_update["new_goal"]]
            separation = shortest_path(sim.pathfinder, goal, replacement)
            if separation is None or separation.geodesic_distance <= 2 * config["success_radius_m"]:
                raise ValueError("Old and new goals must have separate, reachable success regions")
        oracle = shortest_path(sim.pathfinder, start, goal)
        if oracle is None or oracle.geodesic_distance <= config["success_radius_m"]:
            raise ValueError("Start and goal must be connected and outside the success radius")
        optimal_length = float(oracle.geodesic_distance)
        write_json(output / "oracle_path.json", path_record(oracle, start, goal, task_state.active_goal))

        state = habitat_sim.AgentState()
        state.position = start
        state.rotation = quat_from_coeffs(np.asarray(config["start_rotation_xyzw"]))
        agent = sim.initialize_agent(0, state)
        follower = habitat_sim.nav.GreedyGeodesicFollower(
            sim.pathfinder, agent, goal_radius=config["success_radius_m"], stop_key="STOP"
        )
        episode = {
            **config, **provenance, "gpu_device_id": gpu_id,
            "navmesh_settings": mesh_settings, "loaded_rigid_objects": loaded_objects,
            "controller": ("ignore_intervention_oracle" if intervention_policy == "ignore"
                           else "oracle_dynamic_planner") if events else "oracle_greedy_geodesic_follower",
            "intervention_policy": intervention_policy,
            "events": [event.to_dict() for event in events],
            "shortest_path_length_m": optimal_length,
            "versions": {"python": platform.python_version(),
                         "habitat_sim": habitat_sim.__version__, "numpy": np.__version__},
        }
        write_json(output / "episode.json", episode)
        observations = sim.get_sensor_observations()
        trajectory = [{
            "step": None, "action": "INITIALIZE", **pose(agent), "collided": False,
            "distance_m": 0.0, "distance_to_goal_m": optimal_length,
            "active_goal": task_state.active_goal, "task_state": task_state.to_dict(),
            "policy_goal": policy_state.active_goal, "policy_task_state": policy_state.to_dict(),
            "distance_to_policy_goal_m": optimal_length,
            "delivered_event_ids": [],
            "frames": save_frame(output, 0, observations, config.get("semantic_sensor", False)),
        }]
        path_length = 0.0
        stopped = False
        collision_count = 0
        termination = "timeout"
        planner_error = None
        final_distance = optimal_length
        event_log = []
        replans = []
        intervention_step = None
        first_compliant_action_step = None
        post_intervention_optimal_length = None
        pre_intervention_path_length = None
        for step in range(config["max_steps"]):
            # Events update task state between actions. The simulator and agent
            # stay alive; recreating a follower clears only its planner state.
            before_pose = pose(agent)
            before_task = task_state.to_dict()
            before_policy = policy_state.to_dict()
            delivered = event_engine.step(step, task_state)
            if delivered:
                intervention_step = step
                if intervention_policy == "follow":
                    for event in delivered:
                        policy_state.apply_update(event.task_update)
                for event in delivered:
                    event_log.append({
                        **event.to_dict(), "delivered_step": step,
                        "pose_before": before_pose, "pose_after": pose(agent),
                        "task_state_before": before_task, "task_state_after": task_state.to_dict(),
                        "policy_task_state_before": before_policy,
                        "policy_task_state_after": policy_state.to_dict(),
                        "ignored_by_policy": intervention_policy == "ignore",
                    })
                updated_path = shortest_path(sim.pathfinder, agent.get_state().position,
                                             goals[task_state.active_goal])
                if updated_path is None:
                    termination = "updated_goal_unreachable"
                    break
                post_intervention_optimal_length = float(updated_path.geodesic_distance)
                pre_intervention_path_length = path_length
                if intervention_policy == "follow":
                    goal = goals[policy_state.active_goal]
                    replans.append(path_record(updated_path, agent.get_state().position, goal,
                                               policy_state.active_goal, step))
                    follower = habitat_sim.nav.GreedyGeodesicFollower(
                        sim.pathfinder, agent, goal_radius=config["success_radius_m"], stop_key="STOP"
                    )
            previous_position = agent.get_state().position.copy()
            try:
                action = follower.next_action_along(goal)
            except habitat_sim.errors.GreedyFollowerError as error:
                termination = "planner_error"
                planner_error = str(error) or type(error).__name__
                break
            # This is the Oracle's first action planned for the updated target;
            # it measures planner compliance, not learned language understanding.
            if (intervention_step is not None and first_compliant_action_step is None
                    and policy_state.active_goal == task_state.active_goal):
                first_compliant_action_step = step
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
            remaining = shortest_path(sim.pathfinder, position, goals[task_state.active_goal])
            final_distance = float(remaining.geodesic_distance) if remaining is not None else None
            policy_remaining = (remaining if policy_state.active_goal == task_state.active_goal
                                else shortest_path(sim.pathfinder, position, goal))
            trajectory.append({
                "step": step, "action": action, **pose(agent), "collided": collided,
                "distance_m": distance, "distance_to_goal_m": final_distance,
                "active_goal": task_state.active_goal, "task_state": task_state.to_dict(),
                "policy_goal": policy_state.active_goal, "policy_task_state": policy_state.to_dict(),
                "distance_to_policy_goal_m": (float(policy_remaining.geodesic_distance)
                                             if policy_remaining is not None else None),
                "delivered_event_ids": [event.event_id for event in delivered],
                "frames": save_frame(output, step + 1, observations, config.get("semantic_sensor", False)),
            })
            if stopped:
                termination = "stopped"
                break

        distances = {}
        for goal_id, position in goals.items():
            remaining = shortest_path(sim.pathfinder, agent.get_state().position, position)
            distances[goal_id] = float(remaining.geodesic_distance) if remaining is not None else None
        final_distance = distances[task_state.active_goal]
        evaluated = EpisodeEvaluator().evaluate_navigation(
            task_state=task_state, stopped=stopped, goal_distances_m=distances,
            success_radius_m=config["success_radius_m"], intervention_step=intervention_step,
            first_compliant_action_step=first_compliant_action_step,
            path_length=path_length, steps=len(trajectory) - 1, max_steps=config["max_steps"],
        )
        success = evaluated.dynamic_success
        if stopped and not success:
            termination = ("stopped_at_cancelled_goal" if evaluated.stopped_goal in task_state.cancelled_goals
                           else "stopped_outside_goal")
        post_spl = None
        post_length = None
        if post_intervention_optimal_length is not None:
            post_length = path_length - pre_intervention_path_length
            denominator = max(post_intervention_optimal_length, post_length)
            post_spl = (float(success) * post_intervention_optimal_length / denominator
                        if denominator > 0 else float(success))
        metrics = {
            **evaluated.to_dict(),
            "episode_id": config["episode_id"], "success": bool(success),
            "success_rate": float(success),
            # Ordinary SPL is meaningful only for the fixed-goal control.
            "spl": None if events else float(success) * optimal_length / max(optimal_length, path_length),
            "post_intervention_spl": post_spl,
            "post_intervention_path_length_m": post_length,
            "post_intervention_shortest_path_length_m": post_intervention_optimal_length,
            "stopped": stopped, "termination": termination,
            "steps": len(trajectory) - 1, "max_steps": config["max_steps"],
            "path_length_m": path_length, "shortest_path_length_m": optimal_length,
            "navigation_error_m": final_distance,
            "success_radius_m": config["success_radius_m"],
            "collision_count": collision_count, "planner_error": planner_error,
            "goal_distances_m": distances,
            "intervention_step": intervention_step,
            "first_compliant_action_step": first_compliant_action_step,
            "scheduled_event_count": len(events), "delivered_event_count": len(event_log),
            "intervention_policy": intervention_policy, "policy_goal": policy_state.active_goal,
            "ignored_event_count": len(event_log) if intervention_policy == "ignore" else 0,
            "scene_id": config.get("scene_id"), "scene_provenance": provenance,
            "navmesh_settings": mesh_settings, "loaded_rigid_objects": loaded_objects,
            "goal_evaluation": "STOP within geodesic success radius of a configured point; not object-instance visibility",
        }
        write_json(output / "trajectory.json", trajectory)
        write_json(output / "events.json", event_log)
        write_json(output / "replans.json", replans)
        write_json(output / "final_task_state.json", task_state.to_dict())
        write_json(output / "final_policy_state.json", policy_state.to_dict())
        write_json(output / "metrics.json", metrics)
        return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/simulator/navigation_001.json")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--no-video", action="store_true", help="Save frames and metrics without encoding an MP4")
    parser.add_argument("--intervention-policy", choices=("follow", "ignore"), default="follow")
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
    metrics = run(config, scene.resolve(), args.gpu_id, output, args.intervention_policy)
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to: {output}")
    if not args.no_video:
        video = export_video(output, args.video_fps)
        print(f"Video written to: {video}")
    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
