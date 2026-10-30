"""Constraint semantics and matched navigation using a CPU fake simulator."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln import EpisodeEvaluator, EventEngine, InterventionEvent, TaskState
from mivln.regions import segment_intersects_region, trajectory_violations, validate_region
from scripts.avoid_planner import AvoidPlanningError, validate_settings


REGION = {"region_id": "r1", "type": "rectangle_xz", "bounds_xz": [[-.3, -2.3], [.3, -1.7]],
          "floor_y": 0, "floor_tolerance_m": .5}


class AvoidSemanticsTests(unittest.TestCase):
    def test_add_region_preserves_goal_and_owns_constraint(self):
        region = deepcopy(REGION)
        state = TaskState("a")
        state.apply_update({"operation": "add_avoid_region", "region": region})
        region["bounds_xz"][0][0] = -20
        self.assertEqual(state.active_goal, "a")
        self.assertEqual(state.cancelled_goals, [])
        self.assertEqual(state.active_constraints, [REGION])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            state.apply_update({"operation": "add_avoid_region", "region": REGION})

    def test_invalid_geometry_rejected_without_mutating_state(self):
        for changes in ({"bounds_xz": [[0, 0], [0, 1]]}, {"floor_y": float("nan")},
                        {"floor_tolerance_m": 0}, {"region_id": ""}, {"bounds_xz": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_region({**REGION, **changes})
        for settings in ([], {"clearance_m": .1}, {"waypoint_radius_m": float("inf")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_settings({"avoid_planner": settings})

    def test_crossing_boundary_body_radius_stop_and_other_floor(self):
        self.assertTrue(segment_intersects_region([-1, 0, -2], [1, 0, -2], REGION))
        self.assertTrue(segment_intersects_region([.3, 0, -2], [.3, 0, -2], REGION))
        self.assertTrue(segment_intersects_region([.39, 0, -2], [.39, 0, -2], REGION, .1))
        self.assertFalse(segment_intersects_region([.41, 0, -2], [.41, 0, -2], REGION, .1))
        self.assertFalse(segment_intersects_region([-1, 2, -2], [1, 2, -2], REGION, .1))

    def test_event_delivered_once_and_only_post_event_segments_count(self):
        state = TaskState("a")
        event = InterventionEvent("e", 1, "Avoid marked area", {"operation": "add_avoid_region", "region": REGION})
        engine = EventEngine([event])
        initial = {"step": None, "position": [-1, 0, -2], "task_state": state.to_dict()}
        self.assertEqual(engine.step(0, state), [])
        before = {"step": 0, "position": [1, 0, -2], "task_state": state.to_dict()}
        self.assertEqual(len(engine.step(1, state)), 1)
        after = {"step": 1, "position": [-1, 0, -2], "task_state": state.to_dict()}
        self.assertEqual(engine.step(2, state), [])
        self.assertEqual(trajectory_violations([initial, before, after], .1), [{"step": 1, "region_ids": ["r1"]}])

    def test_reaching_same_goal_with_violation_fails_and_unknown_check_is_rejected(self):
        state = TaskState("a", active_constraints=[deepcopy(REGION)])
        inputs = dict(task_state=state, stopped=True, goal_distances_m={"a": .1}, success_radius_m=.3,
                      intervention_step=2, first_compliant_action_step=None,
                      path_length=4, steps=20, max_steps=30)
        evaluator = EpisodeEvaluator()
        with self.assertRaisesRegex(ValueError, "measured violation"):
            evaluator.evaluate_navigation(**inputs)
        self.assertFalse(evaluator.evaluate_navigation(**inputs, constraint_violation_count=1).dynamic_success)
        self.assertTrue(evaluator.evaluate_navigation(**inputs, constraint_violation_count=0).dynamic_success)
        self.assertFalse(evaluator.evaluate_navigation(**{**inputs, "stopped": False}, constraint_violation_count=0).dynamic_success)
        self.assertFalse(evaluator.evaluate_navigation(**{**inputs, "steps": 31}, constraint_violation_count=0).dynamic_success)


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires native imports, no rendering")
class AvoidNavigationTests(unittest.TestCase):
    def setup_fake(self, root, stack):
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs, quat_rotate_vector
        from scripts import run_navigation as nav
        from scripts.terminal_view import pose_snapshot, yaw_action

        scene = root / "scenes/test.scene_instance.json"
        scene.parent.mkdir()
        scene.write_text(json.dumps({"object_instances": [{"template_name": "target"}]}))
        dataset, mesh = root / "dataset.json", root / "mesh.navmesh"
        dataset.write_text("{}")
        mesh.write_bytes(b"CPU fixture")
        config = {"episode_id": "avoid_fixture", "scene": str(scene), "scene_id": "test",
            "scene_dataset_config": str(dataset), "navmesh": str(mesh),
            **{key+"_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
               for key, path in (("scene", scene), ("scene_dataset_config", dataset), ("navmesh", mesh))},
            "agent_height": 1.5, "agent_radius": .1, "semantic_sensor": True,
            "initial_instruction": "Go to A.", "initial_goal": "a", "start_position": [0, 0, 0],
            "start_rotation_xyzw": [0, 0, 0, 1], "goal_positions": {"a": [0, 0, -4]},
            "goal_metadata": {"a": {"template_id": "target", "source_instance_index": 0}},
            "events": [], "seed": 42, "resolution": [4, 4], "sensor_height": 1.5,
            "max_steps": 150, "success_radius_m": .3, "forward_step_m": .25, "turn_angle_degrees": 15,
            "terminal_alignment": {"enabled": True, "yaw_tolerance_degrees": 7.5},
            "terminal_visibility": {"enabled": True, "min_pixels": 4}}
        instances = []

        class Pathfinder:
            is_loaded = True
            navigable_area = 100
            num_islands = 1
            nav_mesh_settings = SimpleNamespace(agent_height=1.5, agent_radius=.1, include_static_objects=True)

            def seed(self, value):
                pass

            def load_nav_mesh(self, filename):
                return True

            def is_navigable(self, point):
                return True

            def snap_point(self, point):
                return np.asarray(point, dtype=np.float32)

            def try_step(self, start, end):
                return np.asarray(end, dtype=np.float32)

            def find_path(self, path):
                path.points = [np.asarray(path.requested_start), np.asarray(path.requested_end)]
                path.geodesic_distance = float(np.linalg.norm(path.points[1]-path.points[0]))
                return True

        class Simulator:
            def __init__(self, configuration):
                self.pathfinder = Pathfinder()
                self.initializations = 0
                instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def seed(self, value):
                pass

            def get_rigid_object_manager(self):
                return SimpleNamespace(get_object_handles=lambda: ["target"])

            def initialize_agent(self, index, state):
                self.initializations += 1
                self.state = SimpleNamespace(position=state.position.copy(), rotation=state.rotation)
                self.agent = SimpleNamespace(get_state=lambda: self.state)
                return self.agent

            def get_sensor_observations(self):
                return {"rgb": np.zeros((4, 4, 4), dtype=np.uint8), "depth": np.ones((4, 4), dtype=np.float32),
                        "semantic": np.zeros((4, 4), dtype=np.uint32), "collided": False}

            def step(self, action):
                if action == "move_forward":
                    self.state.position += quat_rotate_vector(self.state.rotation, np.array([0., 0., -.25]))
                else:
                    delta = math.radians(15 if action == "turn_left" else -15)
                    self.state.rotation *= quat_from_coeffs(np.array([0, math.sin(delta/2), 0, math.cos(delta/2)]))
                return self.get_sensor_observations()

        class Follower:
            def __init__(self, pf, agent, goal_radius, stop_key):
                self.agent, self.radius = agent, goal_radius

            def next_action_along(self, goal):
                state = self.agent.get_state()
                if np.linalg.norm(goal-state.position) <= self.radius:
                    return "STOP"
                action, _ = yaw_action(state, goal, 7.5)
                return "move_forward" if action == "STOP" else action

        def diagnose(sim, agent, target, threshold):
            report = {"evaluated": True, "method": "isolated_instance_semantic_relabel_v1",
                      "template_id": "target", "source_instance_index": 0, "instance_pixels": 16,
                      "min_pixels": threshold, "instance_visible": True, "labels_restored": True,
                      "pose_unchanged": True, "pose": pose_snapshot(agent)}
            return report, np.ones((4, 4), dtype=bool), np.zeros((4, 4, 3), dtype=np.uint8)

        stack.enter_context(patch.object(nav.habitat_sim, "Simulator", Simulator))
        stack.enter_context(patch.object(nav.habitat_sim, "PathFinder", Pathfinder))
        stack.enter_context(patch.object(nav.habitat_sim, "ShortestPath", SimpleNamespace))
        stack.enter_context(patch.object(nav.habitat_sim.nav, "GreedyGeodesicFollower", Follower))
        stack.enter_context(patch.object(nav, "make_configuration", lambda *args: None))
        stack.enter_context(patch.object(nav.terminal_view, "resolve_instances", lambda *args: {"a": {"world_center": [0, 0, -5]}}))
        stack.enter_context(patch.object(nav.terminal_view, "measure_visibility", diagnose))
        return nav, config, scene, instances, Pathfinder

    def test_three_conditions_keep_goal_and_ignore_reaches_but_fails_constraint(self):
        from scripts import avoid_region

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            nav, config, scene, instances, _ = self.setup_fake(root, stack)
            reference = root / "reference"
            nav.run(config, scene, 0, reference)
            prepared = root / "prepared"
            self.assertEqual(avoid_region.prepare(SimpleNamespace(reference_dir=reference, output_dir=prepared)), 0)
            args = SimpleNamespace(config=prepared/"config.json", output_dir=root/"experiment", gpu_id=0)
            self.assertEqual(avoid_region.run_experiment(args), 0)
            report = json.loads((args.output_dir/"summary.json").read_text())
            self.assertTrue(report["passed"])
            self.assertTrue(all(row["point_goal_success"] for row in report["conditions"]))
            self.assertEqual([row["dynamic_success"] for row in report["conditions"]], [True, True, False])
            self.assertGreater(report["conditions"][2]["violation_count"], 0)
            self.assertEqual([sim.initializations for sim in instances], [1, 1, 1, 1])
            # Old output and preparation must never be silently overwritten.
            with self.assertRaises(FileExistsError):
                avoid_region.run_experiment(args)
            with self.assertRaises(FileExistsError):
                avoid_region.prepare(SimpleNamespace(reference_dir=reference, output_dir=prepared))
            # Tampering with evaluator truth, even with an unchanged success flag,
            # must invalidate the independent audit.
            path = args.output_dir/"ignore/trajectory.json"
            rows = json.loads(path.read_text())
            rows[-1]["task_state"]["active_constraints"] = []
            path.write_text(json.dumps(rows))
            self.assertFalse(avoid_region.summarize(args.output_dir)["passed"])

    def test_roadmap_rejects_in_region_endpoint_and_has_no_unconstrained_fallback(self):
        from scripts.avoid_planner import plan_route
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            _, _, _, _, pathfinder_type = self.setup_fake(Path(temporary), stack)
            pf = pathfinder_type()
            with self.assertRaisesRegex(AvoidPlanningError, "endpoint"):
                plan_route(pf, np.array([0, 0, -2]), np.array([0, 0, -4]), REGION)
            # With no navigable guard samples, a blocked direct edge cannot be
            # replaced by an illegal shortcut.
            pf.snap_point = lambda point: np.full(3, float("nan"))
            with self.assertRaisesRegex(AvoidPlanningError, "No safe route"):
                plan_route(pf, np.array([0, 0, 0]), np.array([0, 0, -4]), REGION)

    def test_guard_rejects_unsafe_discrete_motion_even_if_planned_route_is_legal(self):
        import numpy as np
        from scripts.avoid_planner import AvoidFollower

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            nav, config, _, _, _ = self.setup_fake(Path(temporary), stack)
            sim = nav.habitat_sim.Simulator(None)
            state = nav.habitat_sim.AgentState()
            state.position = np.array([0, 0, -.5], dtype=np.float32)
            agent = sim.initialize_agent(0, state)
            follower = AvoidFollower(sim, agent, np.array([0, 0, -4]), [REGION], config)
            # Deliberately make the native movement filter jump through the
            # rectangle: checking both endpoints alone would miss this crossing.
            sim.pathfinder.try_step = lambda a, b: np.array([0, 0, -3])
            follower.follower = SimpleNamespace(next_action_along=lambda goal: "move_forward")
            with self.assertRaisesRegex(AvoidPlanningError, "Discrete forward"):
                follower.next_action_along(np.array([0, 0, -4]))

    def test_follow_planning_failure_retains_all_three_conditions(self):
        from scripts import avoid_region

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            nav, config, scene, _, _ = self.setup_fake(root, stack)
            reference = root/"reference"
            nav.run(config, scene, 0, reference)
            prepared = root/"prepared"
            avoid_region.prepare(SimpleNamespace(reference_dir=reference, output_dir=prepared))
            stack.enter_context(patch.object(nav, "AvoidFollower", side_effect=AvoidPlanningError("no safe route")))
            args = SimpleNamespace(config=prepared/"config.json", output_dir=root/"failed", gpu_id=0)
            self.assertEqual(avoid_region.run_experiment(args), 1)
            report = json.loads((args.output_dir/"summary.json").read_text())
            self.assertEqual(report["expected_episodes"], 3)
            self.assertEqual([item["passed"] for item in report["conditions"]], [True, False, True])
            self.assertTrue((args.output_dir/"ignore/metrics.json").is_file())

    def test_prepared_input_tamper_fails_before_simulator_or_output_creation(self):
        from scripts import avoid_region

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            nav, config, scene, instances, _ = self.setup_fake(root, stack)
            reference = root/"reference"
            nav.run(config, scene, 0, reference)
            prepared = root/"prepared"
            avoid_region.prepare(SimpleNamespace(reference_dir=reference, output_dir=prepared))
            args = SimpleNamespace(config=prepared/"config.json", output_dir=root/"experiment", gpu_id=0)
            args.config.write_text(args.config.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                avoid_region.run_experiment(args)
            self.assertEqual(len(instances), 1)
            self.assertFalse(args.output_dir.exists())
