"""Batch invariants: fixed roster, all-or-nothing preflight and honest failures."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import avoid_dataset as dataset
from scripts import avoid_region
from scripts.hssd_targets import sha256
from scripts.intervention_timing import load_json, write_json


class AvoidStatisticsTests(unittest.TestCase):
    def test_invalid_reported_successes_do_not_enter_success_numerators(self):
        cases = [{"available": True, "validation_passed": False, "point_goal_success": True,
                  "dynamic_success": True, "path_length_m": 2, "violation_count": 0},
                 {"available": False, "validation_passed": False}]
        report = dataset.statistics(cases)
        self.assertEqual(report["planned"], 2)
        self.assertEqual(report["available"], 1)
        self.assertEqual(report["dynamic_success_rate"], 0)
        self.assertIsNone(report["mean_path_length_m"])
        self.assertEqual(report["path_length_m_count"], 0)


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires imports; every Simulator is a CPU fake")
class AvoidBatchTests(unittest.TestCase):
    def fixture(self, root, stack):
        """Two synthetic source tasks and reviewed fixtures, entirely in temp."""
        from tests.test_avoid_region import AvoidNavigationTests
        from tests.test_hssd_semantic import catalog_fixture

        nav, config, scene, instances, _ = AvoidNavigationTests().setup_fake(root, stack)
        config["goal_metadata"]["a"]["category"] = "counter"
        source = root/"source"
        records = []
        catalog, _ = catalog_fixture(root)
        write_json(root/"catalog.json", catalog)
        for index in (1, 2):
            task_id = f"task_{index:03d}"
            relative = f"tasks/{task_id}"
            nav.run(config, scene, 0, source/relative/"reference")
            base, reference = f"configs/{task_id}/base.json", f"configs/{task_id}/reference.json"
            write_json(source/base, config)
            write_json(source/reference, config)
            records.append({"task_id": task_id, "output_subdir": relative, "base_config": base,
                            "reference_config": reference, "base_config_sha256": sha256(source/base),
                            "reference_config_sha256": sha256(source/reference)})
        write_json(source/"manifest.json", {"schema_version": 1, "kind": "hssd_semantic_dataset",
            "task_count": 2, "tasks": records, "catalog_source": {
                "path": str(root/"catalog.json"), "sha256": sha256(root/"catalog.json")}})
        args = SimpleNamespace(source_dir=source, output_dir=root/"batch", gpu_id=0)

        def child(command, log):
            # Exercise the real single-case runner/audit inside these Simulator
            # patches, rather than launching an actual GPU child from a test.
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("CPU fake child\n")
            return avoid_region.run_experiment(SimpleNamespace(
                config=Path(command[command.index("--config")+1]),
                output_dir=Path(command[command.index("--output-dir")+1]), gpu_id=0))

        stack.enter_context(patch.object(dataset, "execute_stage", side_effect=child))
        return args, instances, child

    def test_full_batch_uses_raw_evidence_and_never_rewrites_child_summary(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, instances, _ = self.fixture(Path(temporary), stack)
            self.assertEqual(dataset.prepare(args), 0)
            manifest = load_json(args.output_dir/"manifest.json")
            ids = [load_json(args.output_dir/task["preparation_dir"]/"config.json")["episode_id"]
                   for task in manifest["tasks"]]
            self.assertEqual(len(set(ids)), 2)
            self.assertEqual(dataset.run(args), 0)
            report = load_json(args.output_dir/"summary.json")
            self.assertEqual(report["passed_tasks"], 2)
            self.assertEqual(report["expected_episodes"], 6)
            self.assertEqual(report["observed_episode_metrics"], 6)
            self.assertEqual(report["conditions"]["intervention"]["dynamic_successes"], 2)
            self.assertEqual(report["conditions"]["ignore"]["dynamic_successes"], 0)
            self.assertEqual(report["conditions"]["ignore"]["validation_passed"], 2)
            self.assertEqual(len(instances), 8)  # Two references + six fake episodes.
            cached = args.output_dir/"tasks/task_001/summary.json"
            write_json(cached, {"passed": False, "note": "Deliberately stale unit-test summary"})
            digest = sha256(cached)
            self.assertTrue(dataset.summarize(args.output_dir)["passed"])
            self.assertEqual(sha256(cached), digest)
            with self.assertRaises(FileExistsError):
                dataset.run(args)
            self.assertEqual(len(instances), 8)
            # Alter evaluator truth, leaving success flags untouched. The
            # independent batch audit must catch it and retain all six slots.
            path = args.output_dir/"tasks/task_001/ignore/trajectory.json"
            rows = load_json(path)
            rows[-1]["task_state"]["active_constraints"] = []
            write_json(path, rows)
            failed = dataset.summarize(args.output_dir)
            self.assertFalse(failed["passed"])
            self.assertEqual(failed["passed_tasks"], 1)
            self.assertEqual(failed["expected_episodes"], 6)
            self.assertEqual(failed["observed_episode_metrics"], 6)
            self.assertEqual(len((args.output_dir/"cases.csv").read_text().splitlines()), 7)

    def test_preparation_failure_is_not_resampled_or_removed_from_denominator(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack)
            original = avoid_region.prepare
            calls = []

            def fail_second(preparing):
                calls.append(preparing)
                if len(calls) == 2:
                    raise avoid_region.AvoidPreparationError("No safe candidate", [{"reason": "finite roadmap exhausted"}])
                return original(preparing)

            stack.enter_context(patch.object(avoid_region, "prepare", side_effect=fail_second))
            self.assertEqual(dataset.prepare(args), 1)
            manifest = load_json(args.output_dir/"manifest.json")
            self.assertEqual(len(calls), 2)
            self.assertEqual(manifest["prepared_tasks"], 1)
            self.assertEqual(len(manifest["tasks"]), 2)
            self.assertEqual(manifest["tasks"][1]["rejected_candidates"], [{"reason": "finite roadmap exhausted"}])
            self.assertEqual(dataset.run(args), 1)
            report = load_json(args.output_dir/"summary.json")
            self.assertEqual(report["prepared_episode_budget"], 3)
            self.assertEqual(report["expected_episodes"], 6)
            self.assertEqual(report["observed_episode_metrics"], 3)
            self.assertEqual(report["missing_episode_metrics"], 3)
            self.assertEqual(report["preparation_failed_tasks"], ["task_002"])
            self.assertEqual(report["conditions"]["intervention"]["planned"], 2)
            self.assertEqual(report["conditions"]["intervention"]["dynamic_success_rate"], .5)

    def test_all_inputs_preflight_before_first_child_and_no_preparation_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, instances, _ = self.fixture(Path(temporary), stack)
            dataset.prepare(args)
            frozen = sha256(args.output_dir/"manifest.json")
            with self.assertRaises(FileExistsError):
                dataset.prepare(args)
            self.assertEqual(sha256(args.output_dir/"manifest.json"), frozen)
            second = args.output_dir/"preparations/task_002/config.json"
            second.write_text(second.read_text()+"\n")
            with patch.object(dataset, "execute_stage") as execute:
                with self.assertRaisesRegex(ValueError, "Frozen avoid preparation changed"):
                    dataset.run(args)
                execute.assert_not_called()
            self.assertEqual(len(instances), 2)
            self.assertFalse((args.output_dir/"run_status.json").exists())

    def test_child_failure_retains_missing_cases_and_other_tasks(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, child = self.fixture(Path(temporary), stack)
            dataset.prepare(args)
            attempted = []

            def crash_second(command, log):
                attempted.append(command)
                return child(command, log) if len(attempted) == 1 else 7

            with patch.object(dataset, "execute_stage", side_effect=crash_second):
                self.assertEqual(dataset.run(args), 1)
            report = load_json(args.output_dir/"summary.json")
            self.assertEqual(len(attempted), 2)
            self.assertEqual(report["expected_episodes"], 6)
            self.assertEqual(report["observed_episode_metrics"], 3)
            self.assertEqual(report["passed_tasks"], 1)
            self.assertEqual(report["missing_episode_metrics"], 3)
            self.assertEqual(load_json(args.output_dir/"run_status.json")["task_002"]["returncode"], 7)

    def test_changed_source_reference_blocks_all_gpu_children(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack)
            dataset.prepare(args)
            metrics = args.source_dir/"tasks/task_002/reference/metrics.json"
            metrics.write_text(metrics.read_text()+"\n")
            with patch.object(dataset, "execute_stage") as execute:
                with self.assertRaisesRegex(ValueError, "Source reference changed"):
                    dataset.run(args)
                execute.assert_not_called()
            self.assertFalse((args.output_dir/"run_status.json").exists())
