"""HSSD dependency and geometry checks; no simulator or GPU is created."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.check_hssd import inspect_assets, write_json


class HSSDAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "hssd-hab.scene_dataset_config.json"
        write_json(self.config, {
            "objects": {"paths": {".json": ["objects/*"]}},
            "scene_instances": {"default_attributes": {"semantic_scene_instance": "lexicon"}},
            "semantic_scene_descriptor_instances": {"lexicon": "semantics/lexicon.json"},
        })
        write_json(self.root / "scenes/test.scene_instance.json", {
            "stage_instance": {"template_name": "stages/test"},
            "object_instances": [{"template_name": "sample", "translation": [1, 0, 2]},
                                 {"template_name": "sample", "translation": [3, 0, 4]}],
        })
        write_json(self.root / "stages/test.stage_config.json", {"render_asset": "test.glb"})
        (self.root / "stages/test.glb").write_bytes(b"glTF" + b"fixture")
        self.template_path = self.root / "objects/a/sample.object_config.json"
        self.template = {"render_asset": "sample.glb", "semantic_id": 27}
        write_json(self.template_path, self.template)
        self.mesh = self.template_path.parent / "sample.glb"
        self.mesh.write_bytes(b"glTF" + b"fixture")
        write_json(self.root / "semantics/lexicon.json", {
            "classes": [{"id": 27, "name": "bed"}, {"id": 2, "name": "vase"}],
        })
        write_json(self.root / "semantics/scenes/test.semantic_config.json", {
            "region_annotations": [{"label": "bedroom"}],
        })

    def add_metadata(self, category):
        write_json(self.root / "metadata/objects.json", {"sample": {"name": "Test furniture"}})
        (self.root / "metadata/hssd_obj_semantics_condensed.csv").write_text(
            f"Object Hash,SEMANTIC CONDENSED\nsample,{category}\n", encoding="utf-8")

    def test_selected_dependencies_and_repeated_instances(self):
        self.add_metadata("bed")
        report, objects, regions, classes = inspect_assets(self.config, "test")
        self.assertTrue(report["passed"], report["issues"])
        self.assertEqual(report["resolved_object_instances"], 2)
        self.assertEqual([obj["source_instance_index"] for obj in objects], [0, 1])
        self.assertEqual([obj["category_id"] for obj in objects], [27, 27])
        self.assertEqual(report["object_category_counts"], {"bed": 2})
        self.assertEqual(report["category_label_disagreements"], [])
        self.assertEqual(regions[0]["label"], "bedroom")
        self.assertEqual(classes[27], "bed")
        self.assertIn(str(self.root / "semantics/lexicon.json"), report["input_sha256"])
        self.assertIn(str(self.root / "metadata/hssd_obj_semantics_condensed.csv"), report["input_sha256"])

    def test_metadata_disagreement_is_not_missing_asset(self):
        self.add_metadata("vase")
        report, objects, _, _ = inspect_assets(self.config, "test")
        self.assertTrue(report["passed"])
        self.assertEqual(report["category_label_disagreements"], [0, 1])
        self.assertEqual(objects[0]["category"], "vase")
        self.assertEqual(objects[0]["lexicon_category"], "bed")
        self.assertEqual(objects[0]["category_source"], "metadata_csv")

    def test_unlabeled_instances_are_reported_separately(self):
        self.template.pop("semantic_id")
        write_json(self.template_path, self.template)
        self.add_metadata("vase")
        report, objects, _, _ = inspect_assets(self.config, "test")
        self.assertTrue(report["passed"])
        self.assertEqual(report["unlabeled_template_instances"], 2)
        self.assertIsNone(objects[0]["category_id"])
        self.assertEqual(objects[0]["category"], "vase")

    def test_lfs_pointer_fails_asset_check(self):
        self.mesh.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
        report, _, _, _ = inspect_assets(self.config, "test")
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["issues"]), 1)
        self.assertIn("Git LFS pointer", report["issues"][0]["error"])

    def test_missing_collision_mesh_fails_asset_check(self):
        self.template["collision_asset"] = "missing.glb"
        write_json(self.template_path, self.template)
        report, _, _, _ = inspect_assets(self.config, "test")
        self.assertFalse(report["passed"])
        self.assertTrue(any(item["path"].endswith("missing.glb") for item in report["issues"]))

    def test_scene_identifier_cannot_escape_dataset(self):
        with self.assertRaises(ValueError):
            inspect_assets(self.config, "../elsewhere")


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires Habitat imports, but no GPU")
class HSSDNativeConfigurationTests(unittest.TestCase):
    def test_sensors_and_scene_dataset_configuration(self):
        from scripts.check_hssd import make_configuration

        args = SimpleNamespace(dataset_config=Path("dataset.json"), scene_id="102344280", gpu_id=0,
                               resolution=256, sensor_height=1.5, agent_height=1.5, agent_radius=0.1,
                               views=24)
        config = make_configuration(args)
        self.assertEqual(config.sim_cfg.scene_id, "102344280")
        self.assertEqual(config.sim_cfg.scene_dataset_config_file, str(args.dataset_config.resolve()))
        agent = config.agents[0]
        self.assertEqual([spec.uuid for spec in agent.sensor_specifications], ["rgb", "depth", "semantic"])
        self.assertAlmostEqual(agent.action_space["turn_right"].actuation.amount, 15)

    def test_object_export_uses_world_bounds_not_local_origin(self):
        import magnum as mn
        import numpy as np
        from scripts.check_hssd import export_objects

        obj = SimpleNamespace(
            object_id=314, semantic_id=27, translation=mn.Vector3(10, 0, 0),
            aabb=mn.Range3D(mn.Vector3(0, 0, 0), mn.Vector3(2, 1, 1)),
            transformation=mn.Matrix4.translation(mn.Vector3(10, 0, 0)) @ mn.Matrix4.rotation_y(mn.Deg(90)),
            creation_attributes=SimpleNamespace(handle="/assets/sample.object_config.json"),
        )
        manager = SimpleNamespace(get_object_handles=lambda: ["sample_0000"],
                                  get_object_by_handle=lambda handle: obj)
        sim = SimpleNamespace(get_rigid_object_manager=lambda: manager)
        records = export_objects(sim, {27: "bed"}, [{"template_id": "sample", "category": "vase",
                                                    "category_source": "metadata_csv"}])
        record = records[0]
        self.assertEqual(record["object_id"], 314)
        self.assertEqual(record["semantic_id"], 27)
        self.assertEqual(record["category"], "vase")
        np.testing.assert_allclose(record["world_aabb"]["min"], [10, 0, -2], atol=1e-6)
        np.testing.assert_allclose(record["world_aabb"]["max"], [11, 1, 0], atol=1e-6)
        self.assertNotEqual(record["world_aabb"]["center"], record["translation"])


if __name__ == "__main__":
    unittest.main()
