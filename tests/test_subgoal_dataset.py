"""Subgoal batch acceptance: fixed denominators and raw-evidence replay, no GPU."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import subgoal, subgoal_dataset as dataset
from scripts.hssd_targets import sha256
from scripts.intervention_timing import load_json, write_json


class SubgoalBatchProtocolTests(unittest.TestCase):
    def test_trigger_uses_travel_not_action_count_and_precedes_action(self):
        rows = [{"distance_m": value} for value in (0, 0, .25, 0, .25, .25, .25, 0, 0)]
        self.assertEqual([subgoal.reference_trigger(rows, fraction) for fraction in (.25, .5, .75)], [2, 4, 5])
        for fraction in (True, 0, 1, -.1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                subgoal.reference_trigger(rows, fraction)
        with self.assertRaises(ValueError):
            subgoal.reference_trigger([{"distance_m": 0}]*5)
        with self.assertRaises(ValueError):
            subgoal.reference_trigger([{"distance_m": 0}, {"distance_m": 1}, {"distance_m": 0}])

    def test_timing_duplicates_or_unknown_names_are_not_silently_removed(self):
        for names in ([], ["early", "early"], ["unknown"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                dataset.timing_names(names)

    def test_success_flags_without_valid_evidence_do_not_increase_numerators(self):
        rows = [{"available": True, "validation_passed": False, "dynamic_success": True,
                 "root_goal_success": True, "path_length_m": 4, "completion_step": 7, "steps": 10},
                {"available": False, "validation_passed": False}]
        report = dataset.statistics(rows)
        self.assertEqual(report["planned"], 2)
        self.assertEqual(report["reported_dynamic_success_count"], 1)
        self.assertEqual(report["dynamic_success_rate"], 0)
        self.assertIsNone(report["mean_path_length_m"])
        self.assertEqual(report["completion_step_count"], 0)


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Native imports needed; every Simulator is a CPU fake")
class SubgoalBatchTests(unittest.TestCase):
    def fixture(self, root, stack, timings=None):
        """Two source tasks with real runner logic and isolated synthetic reviews."""
        from tests.test_subgoal import SubgoalNavigationTests
        from tests.test_hssd_semantic import catalog_fixture

        nav, single, scene, _, instances = SubgoalNavigationTests().fixture(root, stack)
        config = load_json(root/"base.json")
        source = root/"source"
        catalog, _ = catalog_fixture(root)
        write_json(root/"catalog.json", catalog)
        tasks = []
        for index in (1, 2):
            identity = f"task_{index:03d}"
            relative = f"tasks/{identity}"
            control = deepcopy(config)
            control["events"] = []
            nav.run(control, scene, 0, source/relative/"reference")
            base, reference = f"configs/{identity}/base.json", f"configs/{identity}/reference.json"
            write_json(source/base, config)
            write_json(source/reference, control)
            tasks.append({"task_id": identity, "output_subdir": relative,
                "base_config": base, "reference_config": reference,
                "base_config_sha256": sha256(source/base), "reference_config_sha256": sha256(source/reference)})
        write_json(source/"manifest.json", {"schema_version": 1, "kind": "hssd_semantic_dataset",
            "task_count": 2, "tasks": tasks,
            "catalog_source": {"path": str(root/"catalog.json"), "sha256": sha256(root/"catalog.json")}})
        args = SimpleNamespace(source_dir=source, output_dir=root/"batch", gpu_id=0,
                               timings=timings or list(dataset.TIMINGS))

        def child(command, log):
            # Keep the real prepare/run/audit contracts in one process under
            # the fake Simulator patches; never launch an actual GPU child.
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("CPU fake subgoal child\n")
            return subgoal.run_experiment(SimpleNamespace(
                config=Path(command[command.index("--config")+1]),
                output_dir=Path(command[command.index("--output-dir")+1]), gpu_id=0))

        stack.enter_context(patch.object(dataset, "execute_stage", side_effect=child))
        return args, instances, child

    def test_full_three_timing_batch_reaudits_raw_evidence_without_touching_children(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, instances, _ = self.fixture(Path(temporary), stack)
            self.assertEqual(dataset.prepare(args), 0)
            manifest = load_json(args.output_dir/"manifest.json")
            self.assertEqual(manifest["expected_episodes"], 18)
            configs = [load_json(args.output_dir/case["preparation_dir"]/"config.json") for case in manifest["cases"]]
            self.assertEqual(len({config["episode_id"] for config in configs}), 6)
            self.assertEqual(dataset.run(args), 0)
            report = load_json(args.output_dir/"summary.json")
            self.assertEqual(report["passed_cases"], 6)
            self.assertEqual(report["conditions"]["intervention"]["dynamic_success_count"], 6)
            self.assertEqual(report["conditions"]["ignore"]["dynamic_success_count"], 0)
            self.assertEqual(report["conditions"]["ignore"]["validation_passed"], 6)
            self.assertEqual(report["by_timing"]["late"]["intervention"]["planned"], 2)
            self.assertTrue(all(sim.initializations == 1 for sim in instances))
            cached = args.output_dir/"tasks/task_001/early/summary.json"
            write_json(cached, {"passed": False, "note": "intentionally stale cached summary"})
            before = {str(path): sha256(path) for path in (cached, args.output_dir/"summary.json")}
            self.assertTrue(dataset.check_batch(args.output_dir)[0]["passed"])
            self.assertEqual(before, {name: sha256(Path(name)) for name in before})
            self.assertEqual(len((args.output_dir/"cases.csv").read_text().splitlines()), 19)
            with self.assertRaises(FileExistsError):
                dataset.run(args)

    def test_preparation_failure_retains_all_slots_and_never_resamples(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack)
            original, attempted = subgoal.prepare, []

            def reject_second(preparing):
                attempted.append(preparing)
                if len(attempted) == 2:
                    raise ValueError("Synthetic unreachable B at this timing")
                return original(preparing)

            with patch.object(subgoal, "prepare", side_effect=reject_second):
                self.assertEqual(dataset.prepare(args), 1)
            self.assertEqual(len(attempted), 6)
            self.assertEqual(dataset.run(args), 1)
            report = load_json(args.output_dir/"summary.json")
            self.assertEqual(report["total_cases"], 6)
            self.assertEqual(report["prepared_cases"], 5)
            self.assertEqual(report["expected_episodes"], 18)
            self.assertEqual(report["missing_episode_metrics"], 3)
            self.assertEqual(report["preparation_failed_cases"], ["task_001_middle"])
            self.assertEqual(report["conditions"]["intervention"]["dynamic_success_rate"], 5/6)

    def test_all_case_hashes_preflight_before_any_gpu_child_and_prepare_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack)
            dataset.prepare(args)
            frozen = sha256(args.output_dir/"manifest.json")
            with self.assertRaises(FileExistsError):
                dataset.prepare(args)
            self.assertEqual(sha256(args.output_dir/"manifest.json"), frozen)
            final = args.output_dir/"preparations/task_002/late/config.json"
            final.write_text(final.read_text()+"\n")
            with patch.object(dataset, "execute_stage") as execute:
                with self.assertRaisesRegex(ValueError, "Frozen subgoal preparation changed"):
                    dataset.run(args)
                execute.assert_not_called()
            self.assertFalse((args.output_dir/"run_status.json").exists())

    def test_changed_source_reference_blocks_every_child(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack, ["early"])
            dataset.prepare(args)
            metrics = args.source_dir/"tasks/task_002/reference/metrics.json"
            metrics.write_text(metrics.read_text()+"\n")
            with patch.object(dataset, "execute_stage") as execute:
                with self.assertRaisesRegex(ValueError, "Source roster"):
                    dataset.run(args)
                execute.assert_not_called()

    def test_case_roster_and_output_paths_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack, ["early"])
            dataset.prepare(args)
            manifest = load_json(args.output_dir/"manifest.json")
            for key, value in (("task_id", "other"), ("output_subdir", "tasks/other"),
                               ("progress_fraction", .5), ("preparation_dir", "../escape")):
                changed = deepcopy(manifest)
                changed["cases"][0][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    dataset.validate_batch(args.output_dir, changed)

    def test_child_failure_does_not_stop_later_cases_or_reduce_denominator(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, child = self.fixture(Path(temporary), stack)
            dataset.prepare(args)
            calls = []

            def fail_second(command, log):
                calls.append(command)
                return 7 if len(calls) == 2 else child(command, log)

            with patch.object(dataset, "execute_stage", side_effect=fail_second):
                self.assertEqual(dataset.run(args), 1)
            report = load_json(args.output_dir/"summary.json")
            self.assertEqual(len(calls), 6)
            self.assertEqual(report["expected_episodes"], 18)
            self.assertEqual(report["passed_cases"], 5)
            self.assertEqual(report["missing_episode_metrics"], 3)

    def test_changed_completion_evidence_fails_even_if_metrics_and_summary_claim_success(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack, ["early"])
            dataset.prepare(args)
            dataset.run(args)
            folder = args.output_dir/"tasks/task_001/early"
            receipts = load_json(folder/"intervention/subgoal_events.json")
            receipts[0]["task_state_after"]["active_goal"] = "b"
            write_json(folder/"intervention/subgoal_events.json", receipts)
            report, rows = dataset.check_batch(args.output_dir)
            self.assertFalse(report["passed"])
            self.assertEqual(report["expected_episodes"], 6)
            self.assertEqual(report["passed_cases"], 1)
            self.assertEqual(report["conditions"]["intervention"]["dynamic_success_count"], 1)
            self.assertEqual(len(rows), 6)
            self.assertTrue(load_json(folder/"summary.json")["passed"])

    def test_swapped_valid_child_output_is_bound_to_its_original_batch_slot(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack, ["early"])
            dataset.prepare(args)
            dataset.run(args)
            first, second = (args.output_dir/f"tasks/task_{index:03d}/early" for index in (1, 2))
            shutil.rmtree(first)
            shutil.copytree(second, first)
            self.assertTrue(subgoal.check_experiment(first)["passed"])
            report, _ = dataset.check_batch(args.output_dir)
            self.assertFalse(report["passed"])
            self.assertIn("different batch case", report["cases"][0]["error"])

    def test_video_export_skips_existing_files_and_preserves_failed_slots(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack, ["early"])
            dataset.prepare(args)

            def export(folder):
                folder.mkdir(parents=True, exist_ok=True)
                (folder/"trajectory.mp4").write_bytes(b"synthetic video")

            with patch("scripts.export_video.export_video", side_effect=export) as mocked:
                self.assertEqual(dataset.videos(args), 0)
                self.assertEqual(mocked.call_count, 6)
                self.assertEqual(dataset.videos(args), 0)
                self.assertEqual(mocked.call_count, 6)
            (args.output_dir/"tasks/task_002/early/ignore/trajectory.mp4").unlink()
            with patch("scripts.export_video.export_video", side_effect=RuntimeError("synthetic encoder failure")):
                self.assertEqual(dataset.videos(args), 1)
            status = load_json(args.output_dir/"video_status.json")
            self.assertEqual(status["expected_videos"], 6)
            self.assertEqual(status["existing_videos"], 5)
            self.assertEqual(len(status["failed_exports"]), 1)

    def test_cpu_rehearsal_preserves_geometry_failures_without_claiming_visibility(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, instances, _ = self.fixture(Path(temporary), stack, ["early"])
            dataset.prepare(args)
            count = len(instances)
            with patch("scripts.subgoal_cpu.check_case", side_effect=[
                    {"geometry_passed": True, "error": None},
                    {"geometry_passed": False, "error": "synthetic budget failure"}]) as check:
                self.assertEqual(dataset.cpu_check(args.output_dir), 1)
                self.assertEqual(check.call_count, 2)
            report = load_json(args.output_dir/"cpu_navigation_check.json")
            self.assertEqual(report["geometry_passed_cases"], 1)
            self.assertEqual(report["total_cases"], 2)
            self.assertFalse(report["visibility_evaluated"])
            self.assertFalse(report["simulator_created"])
            self.assertEqual(len(instances), count)

    def test_nonzero_child_requires_full_raw_acceptance_and_explicit_native_projection_proof(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _ = self.fixture(Path(temporary), stack, ["early"])
            dataset.prepare(args)
            dataset.run(args)
            path = args.output_dir/"run_status.json"
            statuses = load_json(path)
            statuses["task_001_early"]["returncode"] = 1
            write_json(path, statuses)
            digest = sha256(path)
            # An arbitrary exit code 1 does not become success merely because
            # otherwise valid artifacts exist: explicit projection proof is needed.
            self.assertFalse(dataset.check_batch(args.output_dir)[0]["passed"])
            original = subgoal.check_experiment

            def with_proof(folder):
                report = original(folder)
                if folder.parent.name == "task_001":
                    report["conditions"][1]["native_projection_checks"] = [{"endpoint_error_m": 0.}]
                return report

            with patch.object(subgoal, "check_experiment", side_effect=with_proof):
                report, _ = dataset.check_batch(args.output_dir)
                self.assertTrue(report["passed"])
                self.assertEqual(report["recovered_audit_cases"], ["task_001_early"])
                self.assertEqual(report["cases"][0]["run_status"]["returncode"], 1)
            self.assertEqual(sha256(path), digest)
