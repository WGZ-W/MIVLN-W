"""Terminal controls and instance-isolated diagnostics, without GPU rendering."""

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

from scripts import terminal_view
from scripts.intervention_timing import load_json


class TerminalConfigTests(unittest.TestCase):
    def test_legacy_defaults_and_quantized_yaw_validation(self):
        terminal_view.validate({}, {})
        config = {"scene_dataset_config": "dataset.json", "turn_angle_degrees": 15, "resolution": [256, 256],
                  "semantic_sensor": True, "goal_metadata": {"a": {"template_id": "bed", "source_instance_index": 0}},
                  "terminal_alignment": {"enabled": True, "yaw_tolerance_degrees": 7.5},
                  "terminal_visibility": {"enabled": True, "min_pixels": 64}}
        terminal_view.validate(config, {"a": [0, 0, 0]})
        for key, value in (("terminal_alignment", {"enabled": True, "yaw_tolerance_degrees": 1}),
                           ("terminal_alignment", {"enabled": True, "max_turn_actions": 0}),
                           ("terminal_visibility", {"enabled": True, "min_pixels": 0}),
                           ("semantic_sensor", False), ("goal_metadata", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                terminal_view.validate({**config, key: value}, {"a": [0, 0, 0]})


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires native imports, not a GPU")
class TerminalNativeTests(unittest.TestCase):
    def test_turn_direction_wraparound_and_quantized_completion(self):
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs

        state = SimpleNamespace(position=np.zeros(3), rotation=quat_from_coeffs(np.array([0, 0, 0, 1])))
        self.assertEqual(terminal_view.yaw_action(state, [-1, 0, 0], 7.5)[0], "turn_left")
        self.assertEqual(terminal_view.yaw_action(state, [1, 0, 0], 7.5)[0], "turn_right")
        self.assertEqual(terminal_view.yaw_action(state, [0, 0, -1], 7.5)[0], "STOP")
        angle = math.radians(15)
        for _ in range(24):
            action, error = terminal_view.yaw_action(state, [0.2, 0, 2], 7.5)
            if action == "STOP":
                break
            delta = angle if action == "turn_left" else -angle
            state.rotation = state.rotation * quat_from_coeffs(np.array([0, math.sin(delta / 2), 0, math.cos(delta / 2)]))
        self.assertLessEqual(abs(error), 7.5 + 1e-5)
        np.testing.assert_array_equal(state.position, [0, 0, 0])

    def relabel_fixture(self, fail=False):
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs

        a, b = SimpleNamespace(semantic_id=27), SimpleNamespace(semantic_id=27)
        state = SimpleNamespace(position=np.zeros(3), rotation=quat_from_coeffs(np.array([0, 0, 0, 1])))
        agent = SimpleNamespace(get_state=lambda: state)
        buffer = np.zeros((4, 4), dtype=np.uint32)
        def observe():
            if fail and a.semantic_id != 27:
                raise RuntimeError("simulated renderer error")
            buffer[:] = 0
            buffer[:2, :2], buffer[2:, 2:] = a.semantic_id, b.semantic_id
            return {"semantic": buffer, "rgb": np.zeros((4, 4, 4), dtype=np.uint8)}
        manager = SimpleNamespace(get_object_handles=lambda: ["a", "b"], get_object_by_handle=lambda name: a if name == "a" else b)
        sim = SimpleNamespace(get_sensor_observations=observe, get_rigid_object_manager=lambda: manager)
        target = {"object": a, "object_id": 1, "template_id": "a", "source_instance_index": 0}
        return sim, agent, target, a, b

    def test_relabel_counts_one_instance_not_all_shared_category_pixels(self):
        sim, agent, target, a, b = self.relabel_fixture()
        report, mask, _ = terminal_view.measure_visibility(sim, agent, target, 4)
        self.assertEqual(report["instance_pixels"], 4)  # Eight original class-27 pixels, only four target pixels.
        self.assertTrue(report["instance_visible"])
        self.assertTrue(report["pose_unchanged"])
        self.assertTrue(report["labels_restored"])
        self.assertEqual((a.semantic_id, b.semantic_id), (27, 27))
        self.assertEqual(int(mask.sum()), 4)

    def test_relabel_restores_original_id_when_rendering_raises(self):
        sim, agent, target, a, b = self.relabel_fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError, "renderer error"):
            terminal_view.measure_visibility(sim, agent, target, 4)
        self.assertEqual((a.semantic_id, b.semantic_id), (27, 27))

    def setup_navigation(self, root, stack):
        import magnum as mn
        import numpy as np
        from habitat_sim.utils.common import quat_from_coeffs
        from scripts import run_navigation as navigation

        scene = root / "scenes/test.scene_instance.json"
        scene.parent.mkdir()
        scene.write_text(json.dumps({"object_instances": [{"template_name": "bed"}, {"template_name": "couch"}]}))
        dataset = root / "dataset.json"
        dataset.write_text("{}")
        mesh = root / "mesh.navmesh"
        mesh.write_bytes(b"fixture mesh")
        config = {"episode_id": "facing", "scene": str(scene), "scene_id": "test", "scene_dataset_config": str(dataset),
                  "navmesh": str(mesh), "navmesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                  "agent_height": 1.5, "agent_radius": .1, "semantic_sensor": True,
                  "initial_goal": "a", "start_position": [0, 0, 0], "start_rotation_xyzw": [0, 0, 0, 1],
                  "goal_positions": {"a": [2, 0, 0], "b": [-2, 0, 0]},
                  "goal_metadata": {"a": {"template_id": "bed", "source_instance_index": 0},
                                    "b": {"template_id": "couch", "source_instance_index": 1}},
                  "events": [{"event_id": "change", "trigger": {"type": "step", "value": 2},
                              "utterance": "Go to B.", "task_update": {"operation": "replace_goal", "old_goal": "a", "new_goal": "b"}}],
                  "seed": 42, "resolution": [4, 4], "sensor_height": 1.5, "max_steps": 60,
                  "success_radius_m": .3, "forward_step_m": .25, "turn_angle_degrees": 15,
                  "terminal_alignment": {"enabled": True, "yaw_tolerance_degrees": 7.5, "max_turn_actions": 24},
                  "terminal_visibility": {"enabled": True, "min_pixels": 4}}
        objects = {name: SimpleNamespace(object_id=index + 1, semantic_id=27,
            creation_attributes=SimpleNamespace(handle=f"{name}.object_config.json"),
            aabb=mn.Range3D(mn.Vector3(-.2, -.2, -.2), mn.Vector3(.2, .2, .2)),
            transformation=mn.Matrix4.translation(mn.Vector3(x, .5, 2)))
            for index, (name, x) in enumerate((("bed", 2), ("couch", -2)))}
        instances = []

        class Simulator:
            def __init__(self, native):
                instances.append(self)
                self.initializations = 0
                self.state = None
                self.agent = SimpleNamespace(get_state=lambda: self.state)
                self.pathfinder = SimpleNamespace(is_loaded=True, load_nav_mesh=lambda path: True,
                    seed=lambda seed: None, is_navigable=lambda p: True, navigable_area=20, num_islands=1,
                    nav_mesh_settings=SimpleNamespace(agent_height=1.5, agent_radius=.1, include_static_objects=True))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def seed(self, value):
                pass

            def get_rigid_object_manager(self):
                return SimpleNamespace(get_object_handles=lambda: list(objects), get_object_by_handle=lambda name: objects[name])

            def initialize_agent(self, index, state):
                self.initializations += 1
                self.state = SimpleNamespace(position=state.position.copy(), rotation=state.rotation)
                return self.agent

            def get_sensor_observations(self):
                semantic = np.zeros((4, 4), dtype=np.uint32)
                for name, x in (("bed", 2), ("couch", -2)):
                    _, error = terminal_view.yaw_action(self.state, [x, .5, 2], 7.5)
                    if abs(self.state.position[0] - x) <= .3 and abs(error) <= 7.5:
                        semantic[:2, :2] = objects[name].semantic_id
                return {"rgb": np.zeros((4, 4, 4), dtype=np.uint8), "depth": np.ones((4, 4), dtype=np.float32),
                        "semantic": semantic, "collided": False}

            def step(self, action):
                if action == "move_forward":
                    self.state.position[0] += self.agent.direction * .25
                else:
                    angle = math.radians(15 if action == "turn_left" else -15)
                    self.state.rotation *= quat_from_coeffs(np.array([0, math.sin(angle / 2), 0, math.cos(angle / 2)]))
                return self.get_sensor_observations()

        class Follower:
            def __init__(self, pathfinder, agent, goal_radius, stop_key):
                self.agent = agent

            def next_action_along(self, goal):
                remaining = goal[0] - self.agent.get_state().position[0]
                self.agent.direction = 1 if remaining > 0 else -1
                return "STOP" if abs(remaining) <= .3 else "move_forward"

        stack.enter_context(patch.object(navigation.habitat_sim, "Simulator", Simulator))
        stack.enter_context(patch.object(navigation.habitat_sim.nav, "GreedyGeodesicFollower", Follower))
        stack.enter_context(patch.object(navigation, "shortest_path", lambda pf, start, goal:
                            SimpleNamespace(geodesic_distance=float(np.linalg.norm(goal - start)), points=[start, goal])))
        return navigation, config, scene, objects, instances

    def test_turning_preserves_pair_prefix_and_ignore_truth_separation(self):
        from scripts.run_intervention import check_pair
        from scripts.run_ignore_intervention import check_negative
        from scripts.hssd_dataset import terminal_record

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            navigation, config, scene, objects, instances = self.setup_navigation(root, stack)
            config["events"][0]["trigger"]["value"] = 9  # Interrupt A's terminal turning, not just forward travel.
            control = deepcopy(config)
            control["events"] = []
            a = navigation.run(control, scene, 0, root / "control")
            b = navigation.run(config, scene, 0, root / "intervention")
            negative = navigation.run(config, scene, 0, root / "ignore", intervention_policy="ignore")
            self.assertTrue(check_pair(root, config["events"][0])["passed"])
            self.assertTrue(check_negative(root / "ignore", config, root / "control")["negative_check_passed"])
            self.assertTrue(a["point_goal_success"] and a["terminal_semantic_success"])
            self.assertTrue(b["point_goal_success"] and b["terminal_semantic_success"])
            self.assertFalse(negative["point_goal_success"])
            self.assertFalse(negative["terminal_instance_visible"])
            self.assertFalse(negative["terminal_semantic_success"])
            self.assertGreater(b["terminal_alignment_turns"], 0)
            self.assertEqual([sim.initializations for sim in instances], [1, 1, 1])
            self.assertTrue(all(obj.semantic_id == 27 for obj in objects.values()))
            rows = load_json(root / "intervention/trajectory.json")
            for index, row in enumerate(rows[1:], 1):
                if row["controller_phase"] == "terminal_align":
                    self.assertEqual(row["position"], rows[index - 1]["position"])
            self.assertTrue(terminal_record(config, b, rows, root / "intervention", "intervention")["passed"])
            (root / "intervention/terminal_instance_mask.npy").write_bytes(b"changed")
            self.assertFalse(terminal_record(config, b, rows, root / "intervention", "intervention")["passed"])

    def test_turn_limit_and_step_budget_cannot_forge_a_stop(self):
        for limited in (True, False):
            with self.subTest(limited=limited), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                navigation, config, scene, _, _ = self.setup_navigation(root, stack)
                config["events"] = []
                if limited:
                    config["terminal_alignment"]["max_turn_actions"] = 1
                else:
                    config["max_steps"] = 8
                metrics = navigation.run(config, scene, 0, root / "run")
                self.assertFalse(metrics["stopped"])
                self.assertFalse(metrics["point_goal_success"])
                self.assertFalse(metrics["terminal_alignment_complete"])
                self.assertEqual(metrics["termination"], "terminal_alignment_failed" if limited else "timeout")

    def test_wrong_source_instance_cannot_initialize_navigation(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            navigation, config, scene, _, instances = self.setup_navigation(root, stack)
            config["goal_metadata"]["a"]["source_instance_index"] = 1
            with self.assertRaisesRegex(ValueError, "source instance"):
                navigation.run(config, scene, 0, root / "run")
            self.assertEqual(instances[0].initializations, 0)

    def test_diagnostic_failure_does_not_rewrite_point_goal_success(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            navigation, config, scene, _, _ = self.setup_navigation(root, stack)
            config["events"] = []
            stack.enter_context(patch.object(terminal_view, "measure_visibility", side_effect=RuntimeError("diagnostic renderer failed")))
            result = navigation.run(config, scene, 0, root / "run")
            self.assertTrue(result["success"] and result["point_goal_success"])
            self.assertFalse(result["terminal_visibility_evaluated"])
            self.assertIsNone(result["terminal_semantic_success"])
            self.assertIn("diagnostic renderer failed", result["terminal_diagnostic_error"])
            self.assertFalse(load_json(root / "run/terminal_diagnostics.json")["evaluated"])

    def test_facing_derivation_preserves_frozen_tasks_and_original_files(self):
        from scripts import hssd_dataset
        import test_hssd_semantic as fixtures

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            helper = fixtures.SemanticPipelineTests()
            source, instances, execute = helper.setup_pipeline(Path(temporary), stack)
            stack.enter_context(patch.object(hssd_dataset, "execute_stage", execute))
            self.assertEqual(hssd_dataset.run(SimpleNamespace(output_dir=source, gpu_id=0)), 0)
            old_files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob('*') if path.is_file()}
            count = len(instances)
            destination = Path(temporary) / "derived"
            args = SimpleNamespace(source_dir=source, output_dir=destination, min_pixels=1)
            self.assertEqual(hssd_dataset.derive_facing(args), 0)
            self.assertEqual(count, len(instances))  # Derivation itself creates no simulator.
            manifest = load_json(destination / "manifest.json")
            hssd_dataset.validate_inputs(destination, manifest)
            for task in manifest["tasks"]:
                for key in ("base_config", "reference_config"):
                    old, new = load_json(source / task[key]), load_json(destination / task[key])
                    for field in ("start_position", "start_rotation_xyzw", "goal_positions", "goal_metadata", "seed", "events"):
                        self.assertEqual(old[field], new[field])
                    self.assertTrue(new["terminal_alignment"]["enabled"])
            self.assertEqual(old_files, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in source.rglob('*') if path.is_file()})
            with self.assertRaises(FileExistsError):
                hssd_dataset.derive_facing(args)


if __name__ == "__main__":
    unittest.main()
