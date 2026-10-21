from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.debug_dataset import distance_to_polyline, run, summarize
from scripts.intervention_timing import load_json, write_json


class DebugDatasetTests(unittest.TestCase):
    def manifest(self):
        return {"tasks": [{"task_id": f"task_{index:03d}",
                           "base_config": f"configs/task_{index:03d}/base.json",
                           "reference_config": f"configs/task_{index:03d}/reference.json",
                           "output_subdir": f"tasks/task_{index:03d}"} for index in (1, 2)]}

    def test_path_distance_uses_segment_interior_not_just_endpoints(self):
        self.assertAlmostEqual(distance_to_polyline([5, 2, 0], [[0, 0, 0], [10, 0, 0]]), 2.0)
        self.assertAlmostEqual(distance_to_polyline([12, 0, 0], [[0, 0, 0], [10, 0, 0]]), 2.0)

    def test_path_distance_handles_repeated_points(self):
        self.assertAlmostEqual(distance_to_polyline([0, 3, 0], [[0, 0, 0], [0, 0, 0]]), 3.0)
        self.assertTrue(math.isinf(distance_to_polyline([0, 3, 0], [])))

    def test_missing_tasks_remain_in_pair_denominator(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            output = Path(temporary)
            write_json(output / "manifest.json", self.manifest())
            report = summarize(output)
            self.assertFalse(report["passed"])
            self.assertEqual(report["passed_tasks"], 0)
            self.assertEqual(report["total_tasks"], 2)
            self.assertEqual(report["total_pairs"], 6)
            self.assertEqual(report["failed_tasks"], ["task_001", "task_002"])
            self.assertEqual(len((output / "cases.csv").read_text().splitlines()), 7)

    def test_failed_reference_is_retained_and_next_task_is_attempted(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            output = Path(temporary)
            manifest = self.manifest()
            write_json(output / "manifest.json", manifest)
            for task in manifest["tasks"]:
                write_json(output / task["base_config"], {})
                write_json(output / task["reference_config"], {})

            def execute(command, log):
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("simulated log")
                return 1 if "task_001" in str(log) else 0

            with patch("scripts.debug_dataset.execute_stage", side_effect=execute) as stage:
                result = run(SimpleNamespace(output_dir=output, gpu_id=0))
            self.assertEqual(result, 1)
            self.assertEqual(stage.call_count, 4)
            status = load_json(output / "run_status.json")
            self.assertEqual([entry["stage"] for entry in status["task_001"]], ["reference"])
            self.assertEqual(len(status["task_002"]), 3)
            self.assertEqual(load_json(output / "summary.json")["total_pairs"], 6)
            # Existing logs and partial outputs must not be overwritten on a rerun.
            with patch("scripts.debug_dataset.execute_stage") as stage:
                with self.assertRaises(FileExistsError):
                    run(SimpleNamespace(output_dir=output, gpu_id=0))
                stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
