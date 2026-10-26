"""Scene identity and navigation setup regressions without GPU rendering."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.navigation_scene import check_scene_match, navmesh_path, project_path


class SceneIdentityTests(unittest.TestCase):
    def test_legacy_reference_accepts_absolute_paths_and_explicit_sidecar(self):
        config = {"scene": "example.glb"}
        reference = {"scene": str(project_path("example.glb")),
                     "navmesh": str(project_path("example.navmesh"))}
        check_scene_match(config, reference)
        self.assertEqual(navmesh_path(config), project_path("example.navmesh"))

    def test_reference_rejects_different_scene_mesh_dimensions_or_hash(self):
        config = {"scene": "scene.json", "scene_dataset_config": "dataset.json", "scene_id": "test",
                  "navmesh": "furniture.navmesh", "agent_height": 1.5, "agent_radius": 0.1,
                  "navmesh_sha256": "a" * 64, "semantic_sensor": True}
        for key, value in (("scene", "other.json"), ("scene_id", "other"),
                           ("scene_dataset_config", "other-dataset.json"), ("navmesh", "other.navmesh"),
                           ("agent_height", 1.0), ("agent_radius", 0.2), ("semantic_sensor", False),
                           ("navmesh_sha256", "b" * 64)):
            with self.subTest(key=key):
                reference = {**config, key: value}
                with self.assertRaises(ValueError):
                    check_scene_match(config, reference)


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires Habitat imports, not a GPU")
class NavigationSetupTests(unittest.TestCase):
    def setUp(self):
        from scripts import run_navigation

        self.navigation = run_navigation
        self.config = {
            "episode_id": "test", "initial_goal": "a", "goal_positions": {"a": [2, 0, 0]},
            "start_position": [0, 0, 0], "start_rotation_xyzw": [0, 0, 0, 1], "events": [],
            "max_steps": 30, "success_radius_m": 0.3, "forward_step_m": 0.25,
            "turn_angle_degrees": 15, "sensor_height": 1.5, "resolution": [16, 16],
            "scene_dataset_config": "dataset.json", "scene_id": "test", "navmesh": "mesh.navmesh",
            "navmesh_sha256": "a" * 64, "agent_height": 1.5, "agent_radius": 0.1,
            "semantic_sensor": True,
        }

    def test_hssd_native_configuration_and_legacy_defaults(self):
        config = self.navigation.make_configuration(self.config, Path("test.scene_instance.json"), 0)
        self.assertEqual(config.sim_cfg.scene_id, "test")
        self.assertEqual(config.sim_cfg.scene_dataset_config_file, str(project_path("dataset.json")))
        agent = config.agents[0]
        self.assertEqual([sensor.uuid for sensor in agent.sensor_specifications], ["rgb", "depth", "semantic"])
        self.assertAlmostEqual(agent.height, 1.5)
        self.assertAlmostEqual(agent.radius, 0.1)
        self.assertEqual(set(agent.action_space), {"move_forward", "turn_left", "turn_right"})
        legacy = deepcopy(self.config)
        for key in ("scene_dataset_config", "scene_id", "semantic_sensor", "agent_height", "agent_radius"):
            legacy.pop(key)
        config = self.navigation.make_configuration(legacy, Path("legacy.glb"), 0)
        self.assertEqual(config.sim_cfg.scene_id, "legacy.glb")
        self.assertEqual([sensor.uuid for sensor in config.agents[0].sensor_specifications], ["rgb", "depth"])

    def test_hssd_requires_explicit_mesh_hash_and_dimensions(self):
        self.navigation.validate_config(self.config)
        for key in ("scene_id", "navmesh", "navmesh_sha256", "agent_height", "agent_radius"):
            config = deepcopy(self.config)
            config.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.navigation.validate_config(config)
        for key, value in (("scene_id", "../other"), ("agent_radius", -0.1),
                           ("semantic_sensor", "yes"), ("navmesh_sha256", "not-a-hash"),
                           ("scene_dataset_config", ""), ("navmesh", "")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.navigation.validate_config({**self.config, key: value})
        legacy = {**self.config, "scene": "test.scene_instance.json"}
        legacy.pop("scene_dataset_config")
        with self.assertRaises(ValueError):
            self.navigation.validate_config(legacy)

    def test_hssd_timing_preparation_cannot_resample_semantic_target(self):
        from scripts.intervention_timing import prepare

        config = deepcopy(self.config)
        config["goal_positions"]["b"] = [0, 0, 2]
        config["events"] = [{"event_id": "change", "trigger": {"type": "step", "value": 2},
                             "utterance": "Go to the couch instead.",
                             "task_update": {"operation": "replace_goal", "old_goal": "a", "new_goal": "b"}}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.json"
            base.write_text(json.dumps(config))
            args = SimpleNamespace(output_dir=root / "prepared", base_config=base, keep_target=False)
            with self.assertRaisesRegex(ValueError, "--keep-target"):
                prepare(args)
            self.assertFalse(args.output_dir.exists())

    def test_input_hash_and_scene_identity_checked_before_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scenes").mkdir()
            scene = root / "scenes/test.scene_instance.json"
            scene.write_text(json.dumps({"object_instances": [{"template_name": "bed"}]}))
            dataset = root / "dataset.json"
            dataset.write_text("{}")
            mesh = root / "mesh.navmesh"
            mesh.write_bytes(b"mesh-fixture")
            config = {**self.config, "scene_dataset_config": str(dataset), "navmesh": str(mesh),
                      "navmesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}
            record = self.navigation.scene_provenance(config, scene)
            self.assertEqual(record["expected_rigid_objects"], 1)
            self.assertEqual(record["navmesh_sha256"], config["navmesh_sha256"])
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.navigation.scene_provenance({**config, "navmesh_sha256": "b" * 64}, scene)
            with self.assertRaisesRegex(ValueError, "scene_instance"):
                self.navigation.scene_provenance({**config, "scene_id": "other"}, scene)
            mesh.unlink()
            with self.assertRaises(FileNotFoundError):
                self.navigation.scene_provenance(config, scene)

    def test_mesh_settings_and_load_failure_are_not_silently_ignored(self):
        calls = []
        settings = SimpleNamespace(agent_height=1.5, agent_radius=0.1, include_static_objects=True)
        pf = SimpleNamespace(is_loaded=True, nav_mesh_settings=settings, navigable_area=5.0, num_islands=1,
                             load_nav_mesh=lambda path: calls.append(path) or True)
        record = self.navigation.load_navigation_mesh(pf, self.config)
        self.assertEqual(calls, [str(project_path("mesh.navmesh"))])
        self.assertTrue(record["include_static_objects"])
        for key, value in (("agent_height", 1.0), ("agent_radius", 0.2), ("include_static_objects", False)):
            old = getattr(settings, key)
            setattr(settings, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.navigation.load_navigation_mesh(pf, self.config)
            setattr(settings, key, old)
        pf.load_nav_mesh = lambda path: False
        with self.assertRaises(RuntimeError):
            self.navigation.load_navigation_mesh(pf, self.config)

    def test_optional_semantic_frames_preserve_raw_ids(self):
        np = self.navigation.np
        observations = {"rgb": np.zeros((4, 4, 4), dtype=np.uint8),
                        "depth": np.ones((4, 4), dtype=np.float32),
                        "semantic": np.full((4, 4), 27, dtype=np.uint32)}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for folder in ("rgb", "depth", "depth_preview", "semantic"):
                (output / folder).mkdir()
            files = self.navigation.save_frame(output, 0, observations)
            np.testing.assert_array_equal(np.load(output / files["semantic"]), observations["semantic"])
            observations["semantic"] = np.zeros((2, 2), dtype=np.uint32)
            with self.assertRaises(RuntimeError):
                self.navigation.save_frame(output, 1, observations)
            observations.pop("semantic")
            with self.assertRaisesRegex(RuntimeError, "no observation"):
                self.navigation.save_frame(output, 2, observations, require_semantic=True)


if __name__ == "__main__":
    unittest.main()
