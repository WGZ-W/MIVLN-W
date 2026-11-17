"""Asset failures retain scene slots and do not imply rendering readiness."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import hssd_scene_inventory as inventory
from scripts.benchmark_manifest import read_json


class SceneInventoryTests(unittest.TestCase):
    def test_all_scenes_retained_after_a_checker_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            (root / "scenes").mkdir(parents=True)
            config = root / "dataset.json"
            config.write_text("{}")
            for scene in ("a", "b", "c"):
                (root / "scenes" / f"{scene}.scene_instance.json").write_text("{}")
            def inspect(config, scene):
                if scene == "b":
                    raise ValueError("Malformed scene")
                return ({"passed": True, "declared_object_instances": 2,
                         "resolved_object_instances": 2, "checked_files": 4, "issues": []}, [], [], {})
            output = Path(temp) / "audit"
            args = SimpleNamespace(dataset_config=config, output_dir=output)
            with patch.object(inventory, "annotation_issues", return_value=[]), redirect_stdout(io.StringIO()):
                self.assertEqual(inventory.run(args, inspect), 1)
            report = read_json(output / "summary.json")
            self.assertEqual(report["planned_scene_configurations"], 3)
            self.assertEqual(report["by_status"]["error"], 1)
            self.assertEqual(report["by_status"]["complete"], 2)
            self.assertFalse(report["rendering_verified"])
            self.assertTrue(report["all_checked"])

    def test_missing_required_annotations_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(len(inventory.annotation_issues(Path(temp), "scene")), 3)

    def test_interrupted_roster_is_not_complete(self):
        result = inventory.summarize([{"status": "complete"}, {"status": "not_checked"}], True)
        self.assertFalse(result["all_checked"])
        self.assertFalse(result["all_assets_complete"])

    def test_changed_scene_roster_cannot_pass(self):
        self.assertFalse(inventory.summarize([{"status": "complete"}], False)["all_assets_complete"])


if __name__ == "__main__":
    unittest.main()
