"""Habitat backend for the frozen three-task V loop, with private diagnostics.

Native imports are lazy so preparation and source validation remain usable in
the dependency-free environment. Only this adapter creates a rendered Simulator;
NativeReplay uses a no-sensor SceneGraph/Agent and the frozen navigation mesh.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from mivln.benchmark_navigation import validate_case
from mivln.cancel_task import CancellationTaskState, same_pose
from scripts.benchmark_manifest import sha256
from scripts.public_inputs import write_json


SCOPE = "hssd_visual_navigation_middle_v1"


def native_modules():
    import habitat_sim
    import numpy as np
    from habitat_sim.utils.common import quat_from_coeffs
    from scripts import run_navigation as legacy
    from scripts import terminal_view
    return habitat_sim, np, quat_from_coeffs, legacy, terminal_view


def camera_calibration(configuration):
    """Read actual RGB HFOV; square pixels give fx=fy for Habitat perspective."""
    sensor = next(spec for spec in configuration.agents[0].sensor_specifications if spec.uuid == "rgb")
    height, width = sensor.resolution
    hfov = float(sensor.hfov)
    if not 0 < hfov < 180:
        raise ValueError("Unsupported camera horizontal field of view")
    focal = width / (2*math.tan(math.radians(hfov)/2))
    return {"width": int(width), "height": int(height), "fx": focal, "fy": focal,
            "cx": (width-1)/2, "cy": (height-1)/2}


class HabitatBackend:
    """Continuous real RGB/physics collector; diagnoses only near stopping points.

    Distances are geodesic metres; facing uses the resolved object's world centre.
    Instance masks are evaluator-private, saved only at poses within the stopping
    radius. Relabeling is restored before subsequent RGB or policy calls. Querying
    diagnostics never changes pose, calls step_physics or advances the action clock.
    """

    def __init__(self, config, gpu_id, output):
        validate_case(config)
        if config.get("execution_scope") != SCOPE or config.get("timing") != "middle":
            raise ValueError("This entry accepts only the explicit real-HSSD middle configuration")
        habitat, np, quat, legacy, terminal = native_modules()
        self.config, self.output = deepcopy(config), output.resolve()
        self.modules = habitat, np, quat, legacy, terminal
        self.sim, self.step_count, self._cache = None, 0, None
        scene = legacy.project_path(config["scene"])
        provenance = legacy.scene_provenance(config, scene)
        terminal.validate(config, config["goal_positions"])
        if not terminal.enabled(config, "terminal_alignment") or not terminal.enabled(config, "terminal_visibility"):
            raise ValueError("Real object navigation requires facing and instance visibility")
        configuration = legacy.make_configuration(config, scene, gpu_id)
        self.camera = camera_calibration(configuration)
        try:
            self.sim = habitat.Simulator(configuration)
            self.sim.seed(config["seed"])
            mesh = legacy.load_navigation_mesh(self.sim.pathfinder, config)
            self.sim.pathfinder.seed(config["seed"])
            loaded = len(self.sim.get_rigid_object_manager().get_object_handles())
            if loaded != provenance["expected_rigid_objects"]:
                raise ValueError("Loaded rigid object count differs from the frozen scene")
            self.targets = terminal.resolve_instances(self.sim, scene, config["goal_metadata"])
            from scripts.subgoal import reviewed_centers
            centres = reviewed_centers(config, config["goal_positions"])
            for goal, target in self.targets.items():
                if math.dist(target["world_center"], centres[goal]) > 1e-4:
                    raise ValueError("Runtime target centre differs from reviewed instance geometry")
            start = np.asarray(config["start_position"], dtype=np.float32)
            if not self.sim.pathfinder.is_navigable(start):
                raise ValueError("The configured start is not navigable")
            for position in config["goal_positions"].values():
                point = np.asarray(position, dtype=np.float32)
                if not self.sim.pathfinder.is_navigable(point) or legacy.shortest_path(self.sim.pathfinder, start, point) is None:
                    raise ValueError("A reviewed stopping point is not reachable")
            state = habitat.AgentState()
            state.position, state.rotation = start, quat(np.asarray(config["start_rotation_xyzw"]))
            self.agent = self.sim.initialize_agent(0, state)
            self.provenance = {**provenance, "execution_scope": SCOPE, "gpu_id": gpu_id,
                "seed": config["seed"], "habitat_sim_version": habitat.__version__,
                "numpy_version": np.__version__, "navmesh_settings": mesh, "camera": self.camera,
                "loaded_rigid_objects": loaded, "agent_initializations": 1,
                "targets": {goal: {key: target[key] for key in ("template_id", "source_instance_index", "world_center", "object_id")}
                            for goal, target in self.targets.items()}}
            write_json(self.output / "private/backend.json", self.provenance)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def pose(self):
        return self.modules[4].pose_snapshot(self.agent)

    def rgb(self):
        np = self.modules[1]
        rgb = np.asarray(self.sim.get_sensor_observations()["rgb"])
        expected = (self.camera["height"], self.camera["width"])
        if rgb.dtype != np.uint8 or rgb.shape[:2] != expected or rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError("Actual RGB shape/type differs from camera calibration")
        return np.ascontiguousarray(rgb[..., :3]).tobytes()

    def act(self, action):
        before = self.pose()
        if action in ("STOP", "COMPLETE_SUBGOAL"):
            collided = False  # Logical actions count, with no physical simulator step.
        elif action in ("move_forward", "turn_left", "turn_right"):
            observations = self.sim.step(action)
            if "collided" not in observations:
                raise ValueError("Habitat action did not report collision feedback")
            collided = bool(observations["collided"])
        else:
            raise ValueError("Unsupported Habitat action")
        self.step_count += 1
        self._cache = None
        if action in ("STOP", "COMPLETE_SUBGOAL") and not same_pose(before, self.pose()):
            raise ValueError("Logical completion/STOP changed the physical pose")
        return collided

    def measurements(self):
        current = self.pose()
        if self._cache is not None and self._cache[0] == current:
            return deepcopy(self._cache[1])
        _, np, _, legacy, terminal = self.modules
        state, values = self.agent.get_state(), {}
        for goal, point in self.config["goal_positions"].items():
            path = legacy.shortest_path(self.sim.pathfinder, np.asarray(current["position"], dtype=np.float32),
                                        np.asarray(point, dtype=np.float32))
            distance = float(path.geodesic_distance) if path is not None else None
            yaw = terminal.yaw_action(state, self.targets[goal]["world_center"], self.config["yaw_tolerance_degrees"])[1]
            diagnostic = {"evaluated": False, "reason": "outside_stopping_radius"}
            evidence = None
            if distance is not None and distance <= self.config["success_radius_m"]:
                relative = f"private/diagnostics/{self.step_count:04d}/{goal}"
                folder = self.output / relative
                diagnostic = legacy.save_instance_diagnostic(folder, self.sim, self.agent, self.targets[goal], goal,
                    self.config["terminal_visibility"]["min_pixels"])
                evidence = {"directory": relative, "sha256": sha256(folder / "terminal_diagnostics.json")}
                if diagnostic.get("evaluated") is not True:
                    raise RuntimeError("Instance diagnostic failed: " + str(diagnostic.get("error")))
            values[goal] = {"distance_m": distance, "yaw_error_degrees": yaw,
                            "diagnostic": diagnostic, "evidence": evidence}
        if not same_pose(current, self.pose()):
            raise RuntimeError("Private measurements changed agent pose")
        self._cache = current, deepcopy(values)
        return values


class OraclePolicy:
    """Privileged geometry/update controller, independent of evaluator state.

    The arrival of public language triggers the supplied correct private update;
    this is not language understanding. Ignore receives identical messages but
    retains A. Follower resets change only controller memory, never agent pose.
    """

    def __init__(self, config, backend, condition):
        if condition not in ("oracle", "ignore"):
            raise ValueError("Choose oracle or ignore")
        self.config, self.backend, self.condition = config, backend, condition
        self.state = CancellationTaskState(config["initial_goal"])
        self.updated, self.turns, self.trace = False, 0, []
        self._new_follower()

    def _new_follower(self):
        habitat = self.backend.modules[0]
        self.follower = habitat.nav.GreedyGeodesicFollower(self.backend.sim.pathfinder, self.backend.agent,
            goal_radius=self.config["success_radius_m"], stop_key="STOP")
        self.turns = 0

    def __call__(self, request, assets):
        if request["current_intervention"] is not None and not self.updated:
            self.updated = True
            if self.condition == "oracle":
                self.state.apply_update(self.config["events"][0]["task_update"])
                if self.state.active_goal is not None:
                    self._new_follower()
        goal, phase = self.state.active_goal, "navigate"
        if goal is None:
            action, phase = "STOP", "cancel_stop"
        else:
            np, terminal = self.backend.modules[1], self.backend.modules[4]
            action = self.follower.next_action_along(np.asarray(self.config["goal_positions"][goal], dtype=np.float32))
            if action == "STOP":
                action, _ = terminal.yaw_action(self.backend.agent.get_state(), self.backend.targets[goal]["world_center"],
                                                self.config["yaw_tolerance_degrees"])
                phase = "terminal_align"
                if action != "STOP":
                    self.turns += 1
                    if self.turns > self.config["terminal_alignment"]["max_turn_actions"]:
                        raise RuntimeError("Oracle terminal alignment exceeded its counted turn limit")
                elif self.state.suspended_goals:
                    action, phase = "COMPLETE_SUBGOAL", "complete_subgoal_intent"
                    self.state.complete_subgoal(goal)
                    self._new_follower()
        self.trace.append({"step": request["step"], "policy_goal": goal, "phase": phase, "action": action})
        return json.dumps({"action": action})


class NativeReplay:
    """No-render action replay on the frozen mesh; never reset at intervention."""

    def __init__(self, config):
        habitat, np, quat, legacy, terminal = native_modules()
        self.modules, self.config = (habitat, np, quat, legacy, terminal), config
        self.pathfinder = habitat.PathFinder()
        legacy.load_navigation_mesh(self.pathfinder, config)
        self.graph = habitat.SceneGraph()
        specification = habitat.agent.AgentConfiguration()
        specification.sensor_specifications = []
        specification.height, specification.radius = config["agent_height"], config["agent_radius"]
        specification.action_space = {name: habitat.agent.ActionSpec(name, habitat.agent.ActuationSpec(amount=amount))
            for name, amount in (("move_forward", config["forward_step_m"]), ("turn_left", config["turn_angle_degrees"]),
                                 ("turn_right", config["turn_angle_degrees"]))}
        self.agent = habitat.Agent(self.graph.get_root_node().create_child(), specification,
            controls=habitat.agent.ObjectControls(move_filter_fn=self.pathfinder.try_step))
        state = habitat.AgentState()
        state.position = np.asarray(config["start_position"], dtype=np.float32)
        state.rotation = quat(np.asarray(config["start_rotation_xyzw"]))
        self.agent.set_state(state)

    def pose(self):
        return self.modules[4].pose_snapshot(self.agent)

    def act(self, action):
        return bool(self.agent.act(action)) if action not in ("STOP", "COMPLETE_SUBGOAL") else False

    def distance(self, position):
        _, np, _, legacy, _ = self.modules
        path = legacy.shortest_path(self.pathfinder, self.agent.get_state().position, np.asarray(position, dtype=np.float32))
        return float(path.geodesic_distance) if path is not None else None

    def yaw(self, centre):
        return self.modules[4].yaw_action(self.agent.get_state(), centre, self.config["yaw_tolerance_degrees"])[1]

    def close(self):
        self.agent.close()
