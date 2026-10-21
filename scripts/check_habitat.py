"""Check RGB-D rendering, actions, and pathfinding without importing MIVLN."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pose(agent: habitat_sim.Agent) -> dict:
    state = agent.get_state()
    rotation = state.rotation
    return {
        "position": state.position.tolist(),
        "rotation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
    }


def save_observations(output: Path, observations: dict, suffix: str = "") -> None:
    rgb = observations["rgb"]
    depth = observations["depth"]
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise RuntimeError(f"Unexpected RGB shape: {rgb.shape}")
    if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all():
        raise RuntimeError("Invalid depth observation")
    Image.fromarray(rgb[:, :, :3]).save(output / f"rgb{suffix}.png")
    # Preserve metric depth separately; the PNG is a fixed 0-10 m preview.
    np.save(output / f"depth{suffix}.npy", depth)
    preview = (np.clip(depth / 10.0, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(preview).save(output / f"depth{suffix}.png")


def run(config: dict, scene: Path, gpu_id: int, output: Path) -> dict:
    if not scene.is_file():
        raise FileNotFoundError(f"Scene not found: {scene}")
    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = str(scene)
    sim_config.gpu_device_id = gpu_id
    sim_config.enable_physics = True

    sensors = []
    for name, sensor_type in (("rgb", habitat_sim.SensorType.COLOR),
                              ("depth", habitat_sim.SensorType.DEPTH)):
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = name
        sensor.sensor_type = sensor_type
        sensor.resolution = config["resolution"]
        sensor.position = [0.0, config["sensor_height"], 0.0]
        sensors.append(sensor)

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = sensors
    agent_config.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=config["forward_step_m"])
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=config["turn_angle_degrees"])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=config["turn_angle_degrees"])
        ),
    }
    if not config["actions"] or any(action not in agent_config.action_space for action in config["actions"]):
        raise ValueError("Configure a non-empty sequence of supported actions")

    output.mkdir(parents=True, exist_ok=True)
    with habitat_sim.Simulator(habitat_sim.Configuration(sim_config, [agent_config])) as sim:
        sim.seed(config["seed"])
        if not sim.pathfinder.is_loaded:
            raise RuntimeError("The scene requires a loaded navigation mesh")
        sim.pathfinder.seed(config["seed"])
        initial_state = habitat_sim.AgentState()
        initial_state.position = sim.pathfinder.get_random_navigable_point()
        agent = sim.initialize_agent(0, initial_state)
        initial_pose = pose(agent)
        observations = sim.get_sensor_observations()
        save_observations(output, observations)

        path = habitat_sim.ShortestPath()
        path.requested_start = initial_state.position
        for _ in range(100):
            path.requested_end = sim.pathfinder.get_random_navigable_point()
            if sim.pathfinder.find_path(path) and np.isfinite(path.geodesic_distance) and path.geodesic_distance > 1.0:
                break
        else:
            raise RuntimeError("Could not find a reachable point more than 1 m away")
        shortest_path = {
            "start": np.asarray(path.requested_start).tolist(),
            "end": np.asarray(path.requested_end).tolist(),
            "geodesic_distance_m": float(path.geodesic_distance),
            "points": [np.asarray(point).tolist() for point in path.points],
        }

        trajectory = [{"step": 0, "action": "INITIALIZE", **initial_pose, "collided": False}]
        path_length = 0.0
        collisions = 0
        rotation_changed = False
        for step, action in enumerate(config["actions"], start=1):
            previous = pose(agent)
            observations = sim.step(action)
            current = pose(agent)
            distance = float(np.linalg.norm(np.asarray(current["position"]) - previous["position"]))
            path_length += distance
            collided = bool(observations["collided"])
            collisions += int(collided)
            if action.startswith("turn_"):
                rotation_changed |= not np.allclose(previous["rotation_xyzw"], current["rotation_xyzw"])
            trajectory.append({"step": step, "action": action, **current,
                               "collided": collided, "distance_m": distance})

        save_observations(output, observations, "_final")
        write_json(output / "trajectory.json", trajectory)
        write_json(output / "shortest_path.json", shortest_path)
        result = {
            "success": path_length > 0.01 and rotation_changed,
            "scene": str(scene),
            "gpu_device_id": gpu_id,
            "seed": config["seed"],
            "physics_enabled": True,
            "steps": len(config["actions"]),
            "path_length_m": path_length,
            "rotation_changed": rotation_changed,
            "collision_count": collisions,
            "collision_note": "Reports motion-filter collision flags; zero collisions is valid.",
            "shortest_path_distance_m": shortest_path["geodesic_distance_m"],
            "rgb_shape": list(observations["rgb"].shape),
            "depth_shape": list(observations["depth"].shape),
            "versions": {"python": platform.python_version(),
                         "habitat_sim": habitat_sim.__version__, "numpy": np.__version__},
        }
        write_json(output / "config.json", {**config, "scene": str(scene), "gpu_device_id": gpu_id})
        write_json(output / "metrics.json", result)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/simulator/minimal.json")
    parser.add_argument("--scene", type=Path, help="Override the configured scene")
    parser.add_argument("--gpu-id", type=int, default=0, help="Habitat CUDA device index; the eighth GPU is 7")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/habitat_smoke_test")
    args = parser.parse_args()
    if args.gpu_id < 0:
        parser.error("--gpu-id must be non-negative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scene = args.scene if args.scene is not None else Path(config["scene"])
    if not scene.is_absolute():
        scene = PROJECT_ROOT / scene
    result = run(config, scene.resolve(), args.gpu_id, args.output_dir)
    print(json.dumps(result, indent=2))
    print(f"Artifacts written to: {args.output_dir}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
