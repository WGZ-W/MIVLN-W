"""Three-task V execution, reviewed-source binding and raw-evidence audits.

All image/instance approvals here belong to isolated synthetic fixtures. They
are never approvals or navigation evidence for the actual HSSD scene.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln import benchmark_navigation as navigation
from mivln.navigation_fixture import CONDITIONS, FreePlaneBackend, fixture_case, privileged_policy
from scripts import benchmark_navigation as cli
from scripts import language_annotations as language
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import sha256
from scripts.public_inputs import write_json


def case(family="replacement", timing="middle"):
    update = {"replacement": {"operation": "replace_goal", "old_goal": "a", "new_goal": "b"},
        "subgoal": {"operation": "insert_subgoal", "subgoal": "b", "resume_goal": "a"},
        "cancel_task": {"operation": "cancel_task"}}[family]
    return fixture_case({"navigation_protocol": navigation.PROTOCOL, "task_type": family, "timing": timing,
        "initial_goal": "a", "initial_instruction": "Go to the bed.", "goal_positions": {"a": [0, 0, -2], "b": [1, 0, -1]},
        "cancel_protocol": "cancel_task_v1", "events": [{"event_id": "private_event",
            "utterance": {"replacement": "Go to the sofa instead.", "subgoal": "First go to the sofa, then resume the bed.",
                          "cancel_task": "Cancel the task. Stop now."}[family],
            "trigger": {"type": "step", "value": 4}, "task_update": update}]})


def execute(config, condition="follow", callback=None, backend_type=FreePlaneBackend):
    backend = backend_type(config)
    return navigation.run_episode(config, backend, callback or privileged_policy(config, backend, condition))


def reviewed_fixture(root, approved=True):
    """Add complete, explicitly synthetic source-reference diagnostic files."""
    from tests.test_language_annotations import LanguageAnnotationTests
    from tests.test_language_navigation import source_fixture
    source = source_fixture(root)
    reference = source / "tasks/task_001/reference"
    config = language.read_json(source / "configs/task_001/base.json")
    rows = language.read_json(reference / "trajectory.json")
    for index, row in enumerate(rows):
        row.update(rotation_xyzw=[0., 0., 0., 1.], distance_m=.25 if 0 < index < len(rows)-1 else 0.,
            collided=False, policy_goal=config["initial_goal"], controller_phase="navigate",
            frames={name: name + ".fixture" for name in ("rgb", "depth", "semantic")})
    for name in ("rgb", "depth", "semantic"):
        (reference / (name + ".fixture")).write_bytes(b"Synthetic sensor payload, not HSSD.")
    files = []
    for name in ("terminal_instance_mask.npy", "terminal_instance_mask.png", "terminal_rgb.png"):
        (reference / name).write_bytes(b"Synthetic instance fixture, not actual visibility.")
        files.append({"path": name, "sha256": sha256(reference / name)})
    metadata = config["goal_metadata"][config["initial_goal"]]
    diagnostic = {"evaluated": True, "goal_id": config["initial_goal"], **metadata,
        "method": "isolated_instance_semantic_relabel_v1", "labels_restored": True, "pose_unchanged": True,
        "pose": {key: rows[-1][key] for key in ("position", "rotation_xyzw")}, "files": files,
        "instance_pixels": 4, "min_pixels": 4, "instance_visible": True}
    write_json(reference / "terminal_diagnostics.json", diagnostic)
    write_json(reference / "trajectory.json", rows)
    write_json(reference / "metrics.json", {"success": True, "stopped": True, "dynamic_success": True,
        "steps": len(rows)-1, "max_steps": config["max_steps"], "path_length_m": 1.75, "collision_count": 0,
        "final_goal": config["initial_goal"], "point_goal_success": True, "terminal_instance_pixels": 4,
        "terminal_instance_visible": True, "terminal_semantic_success": True, "terminal_alignment_complete": True,
        "terminal_yaw_error_degrees": 0., "terminal_alignment_turns": 0,
        "terminal_diagnostics_sha256": sha256(reference / "terminal_diagnostics.json")})
    draft = root / "draft"
    language.prepare(SimpleNamespace(source_dir=source, output_dir=draft))
    if not approved:
        return draft, source
    reviewed = LanguageAnnotationTests().approve(root, draft)
    return reviewed, source


class NavigationLoopTests(unittest.TestCase):
    def test_three_types_three_timings_five_conditions_and_measured_outcomes(self):
        for family in navigation.TASK_TYPES:
            for timing in ("early", "middle", "late"):
                for condition in CONDITIONS:
                    with self.subTest(family=family, timing=timing, condition=condition):
                        config = case(family, timing)
                        result = execute(config, condition)
                        metrics = result["metrics"]
                        expected = condition == "follow" or condition == "delayed" and family != "cancel_task"
                        self.assertTrue(metrics["evidence_valid"], metrics)
                        self.assertEqual(metrics["task_success"], expected, metrics)
                        self.assertEqual(len(result["events"]), 1)
                        k = config["events"][0]["trigger"]["value"]
                        self.assertEqual(result["events"][0]["delivered_step"], k)
                        self.assertEqual(result["events"][0]["pose_before"], result["events"][0]["pose_after"])
                        if condition == "invalid":
                            self.assertEqual(len(result["trajectory"])-1, k)
                            self.assertIsNone(result["policy_records"][-1]["action"])
                        if family == "cancel_task" and condition == "delayed":
                            self.assertEqual(metrics["cancellation"]["stop_delay_actions"], 1)
                            self.assertEqual(metrics["cancellation"]["post_cancel_path_length_m"], 0.)

    def test_policy_receives_only_causal_language_images_and_its_own_executed_history(self):
        config = case("subgoal")
        backend = FreePlaneBackend(config)
        oracle = privileged_policy(config, backend, "follow")
        def callback(request, assets):
            k, step = 4, request["step"]
            expected = [request["observation"]["rgb"], *[row["rgb"] for row in request["history"]["observations"]]]
            self.assertEqual(set(assets), {row["asset_id"] for row in expected})
            self.assertEqual(len(request["history"]["actions"]), step)
            self.assertEqual(request["current_intervention"] is not None, step == k)
            self.assertEqual(bool(request["message_history"]), step > k)
            if step < k:
                self.assertNotIn(config["events"][0]["utterance"], json.dumps(request))
            for forbidden in ("task_update", "goal_positions", "measurements", "task_type", "private_event", '"position"'):
                self.assertNotIn(forbidden, json.dumps(request))
            return oracle(request, assets)
        result = navigation.run_episode(config, backend, callback)
        self.assertTrue(result["metrics"]["task_success"])

    def test_subgoal_restoration_requires_measured_completion_and_preserves_pose(self):
        result = execute(case("subgoal"))
        completion = result["metrics"]["completions"]
        self.assertEqual(len(completion), 1)
        self.assertTrue(completion[0]["valid"])
        k = completion[0]["step"]
        self.assertEqual(result["trajectory"][k]["position"], result["trajectory"][k+1]["position"])
        self.assertEqual(result["final_task_state"]["active_goal"], "a")
        self.assertEqual(result["final_task_state"]["completed_subgoals"], ["b"])
        self.assertEqual(result["final_task_state"]["cancelled_goals"], [])

    def test_wrong_completion_executes_but_cannot_restore_truth_or_be_erased_by_later_success(self):
        config = case("subgoal")
        result = execute(config, "wrong")
        self.assertEqual(result["trajectory"][5]["action"], "COMPLETE_SUBGOAL")
        self.assertFalse(result["metrics"]["completions"][0]["valid"])
        self.assertTrue(result["metrics"]["completions"][1]["valid"])
        self.assertFalse(result["metrics"]["task_success"])
        self.assertTrue(result["metrics"]["terminal_goal_complete"])

    def test_early_stop_and_budget_exhaustion_fail_without_changing_planned_event(self):
        for family in navigation.TASK_TYPES:
            config = case(family)
            result = execute(config, callback=lambda request, assets: '{"action":"STOP"}')
            self.assertFalse(result["metrics"]["intervention_received"])
            self.assertFalse(result["metrics"]["task_success"])
            self.assertEqual(len(result["policy_records"]), 1)
            config["max_steps"] = 6
            result = execute(config, callback=lambda request, assets: '{"action":"move_forward"}')
            self.assertTrue(result["metrics"]["intervention_received"])
            self.assertFalse(result["metrics"]["task_success"])
            self.assertEqual(len(result["policy_records"]), 6)

    def test_callback_failure_preserves_input_and_does_not_execute_fallback(self):
        def fail(request, assets):
            if request["current_intervention"]:
                raise OSError("Synthetic unavailable policy")
            return '{"action":"move_forward"}'
        result = execute(case("cancel_task"), callback=fail)
        self.assertEqual(len(result["trajectory"]), 5)
        self.assertEqual(len(result["policy_records"]), 5)
        self.assertIsNotNone(result["policy_records"][-1]["request"]["current_intervention"])
        self.assertFalse(result["metrics"]["task_success"])
        self.assertIsNone(result["final_task_state"]["active_goal"])

    def test_backend_failure_keeps_issued_input_raw_response_and_invalid_evidence(self):
        class BrokenBackend(FreePlaneBackend):
            def act(self, action):
                raise RuntimeError("Synthetic execution failure")
        result = execute(case(), backend_type=BrokenBackend)
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertEqual(len(result["policy_records"]), 1)
        self.assertIn('"action"', result["policy_records"][0]["raw_response"])
        self.assertTrue(result["assets"])
        self.assertFalse(result["metrics"]["evidence_valid"])
        self.assertFalse(result["metrics"]["task_success"])

    def test_goal_tasks_require_facing_and_visibility_while_cancellation_does_not(self):
        class InvisibleBackend(FreePlaneBackend):
            def measurements(self):
                values = super().measurements()
                for value in values.values():
                    value["diagnostic"]["instance_visible"] = False
                return values
        for family in navigation.TASK_TYPES:
            result = execute(case(family), backend_type=InvisibleBackend)
            self.assertEqual(result["metrics"]["task_success"], family == "cancel_task")

    def test_duplicate_delivery_pose_reset_and_omitted_event_are_rejected(self):
        config = case()
        result = execute(config)
        for kind in ("duplicate", "reset", "omitted"):
            events = deepcopy(result["events"])
            if kind == "duplicate":
                events.append(deepcopy(events[0]))
            elif kind == "reset":
                events[0]["pose_after"]["position"][0] += 1
            else:
                events.clear()
            metrics = navigation.evaluate_episode(config, result["trajectory"], events)
            self.assertFalse(metrics["evidence_valid"], kind)
            self.assertFalse(metrics["task_success"], kind)


class NavigationEntryTests(unittest.TestCase):
    def fixture(self, root, run=False):
        reviewed, source = reviewed_fixture(root)
        prepared = root / "prepared"
        self.assertEqual(cli.prepare(SimpleNamespace(annotations_dir=reviewed, output_dir=prepared)), 0)
        if not run:
            return prepared, source
        output = root / "run"
        self.assertEqual(cli.cpu(SimpleNamespace(input_dir=prepared, output_dir=output, base_task="task_001", timing="middle")), 0)
        return output, prepared

    def test_prepare_binds_reviewed_text_original_geometry_and_common_external_timings(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            reviewed, source = reviewed_fixture(root)
            before = snapshot({"source": source, "reviewed": reviewed})
            output = root / "prepared"
            self.assertEqual(cli.prepare(SimpleNamespace(annotations_dir=reviewed, output_dir=output)), 0)
            manifest = cli.load_preparation(output)
            self.assertEqual(manifest["planned_cases"], 9)
            self.assertEqual(manifest["prepared_cases"], 9)
            configs = [language.read_json(output / row["config"]) for row in manifest["cases"]]
            self.assertTrue(all(config["execution_scope"] == "hssd_preparation_only" for config in configs))
            self.assertEqual({config["events"][0]["trigger"]["value"] for config in configs}, {2, 4, 6})
            self.assertTrue(all(config["goal_positions"]["target_000"] == [2, 0, 0] for config in configs))
            self.assertEqual(before, snapshot({"source": source, "reviewed": reviewed}))
            with self.assertRaises(FileExistsError):
                cli.prepare(SimpleNamespace(annotations_dir=reviewed, output_dir=output))

    def test_pending_language_cannot_enter_navigation_preparation(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            draft, _ = reviewed_fixture(root, approved=False)
            with self.assertRaisesRegex(ValueError, "approved"):
                cli.prepare(SimpleNamespace(annotations_dir=draft, output_dir=root / "bad"))
            self.assertFalse((root / "bad").exists())

    def test_failed_reference_keeps_all_nine_case_slots(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            reviewed, source = reviewed_fixture(root)
            (source / "tasks/task_001/reference/trajectory.json").unlink()
            output = root / "prepared"
            self.assertEqual(cli.prepare(SimpleNamespace(annotations_dir=reviewed, output_dir=output)), 1)
            manifest = cli.load_preparation(output)
            self.assertEqual(manifest["planned_cases"], 9)
            self.assertEqual(manifest["prepared_cases"], 0)
            self.assertEqual(len(manifest["cases"]), 9)
            run = root / "run"
            self.assertEqual(cli.cpu(SimpleNamespace(input_dir=output, output_dir=run, base_task="task_001", timing="middle")), 1)
            summary = language.read_json(run / "summary.json")
            self.assertEqual(summary["planned_episodes"], 15)
            self.assertTrue(all(not row["passed"] for row in summary["episodes"]))

    def test_prepared_language_or_geometry_cannot_be_changed_by_resealing(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            output, _ = self.fixture(root)
            manifest = language.read_json(output / "manifest.json")
            path = output / manifest["cases"][0]["config"]
            config = language.read_json(path)
            config["initial_instruction"] = "Another instruction."
            write_json(path, config)
            manifest["cases"][0]["config_sha256"] = sha256(path)
            write_json(output / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "reviewed origin"):
                cli.load_preparation(output)

    def test_cpu_and_read_only_audit_reconstruct_all_fifteen_slots_and_ignore_cached_metrics(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            output, _ = self.fixture(root, run=True)
            first = cli.check_run(output)
            self.assertTrue(first["passed"])
            self.assertEqual(first["planned_cases"], 3)
            self.assertEqual(first["planned_episodes"], 15)
            self.assertEqual(sum(row["task_success"] for row in first["episodes"]), 5)
            self.assertFalse(first["hssd_navigation_executed"])
            for path in output.glob('episodes/*/*/private/metrics.json'):
                write_json(path, {"task_success": True})
            before = snapshot({"run": output})
            audit = root / "audit"
            with patch('sys.argv', ['benchmark_navigation', 'audit', '--input-dir', str(output), '--output-dir', str(audit)]):
                self.assertEqual(cli.main(), 0)
            report = language.read_json(audit / "audit.json")
            self.assertTrue(report["source_files_unchanged"])
            self.assertEqual(before, snapshot({"run": output}))
            self.assertEqual(sum(row["task_success"] for row in report["episodes"]), 5)

    def test_missing_branch_retains_denominator_and_fails_acceptance(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            output, _ = self.fixture(root, run=True)
            shutil.rmtree(output / 'episodes/task_001_cancel_task_middle/ignore')
            report = cli.check_run(output)
            self.assertFalse(report["passed"])
            self.assertEqual(report["planned_episodes"], 15)
            self.assertEqual(len(report["episodes"]), 15)

    def test_future_message_modified_pixel_or_wrong_executed_action_cannot_pass_audit(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            output, _ = self.fixture(root, run=True)
            folder = output / 'episodes/task_001_replacement_middle/follow'
            targets = [('image', folder / 'public/assets'), ('message', folder / 'private/policy_records.jsonl'),
                       ('action', folder / 'private/trajectory.json')]
            for kind, target in targets:
                path = next(target.glob('*.png')) if kind == 'image' else target
                original = path.read_bytes()
                if kind == 'image':
                    path.write_bytes(b'Changed actual image')
                elif kind == 'message':
                    records = cli.lines(path)
                    records[0]['request']['current_intervention'] = {'delivered_step': 0, 'message': 'Future message.'}
                    cli.write_lines(path, records)
                else:
                    rows = language.read_json(path)
                    rows[1]['action'] = 'turn_left'
                    write_json(path, rows)
                self.assertFalse(cli.check_run(output)['passed'], kind)
                path.write_bytes(original)
            self.assertTrue(cli.check_run(output)['passed'])

    def test_planned_condition_cannot_be_removed_from_manifest(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            output, _ = self.fixture(root, run=True)
            manifest = language.read_json(output / 'manifest.json')
            manifest['planned_episodes'].pop()
            write_json(output / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'denominator'):
                cli.check_run(output)

    def test_unknown_timing_or_empty_selection_cannot_pass_as_an_empty_batch(self):
        manifest = {'cases': [{'source_task_id': 'task_001', 'timing': 'middle'}]}
        for timing in ('unknown', 'early'):
            with self.subTest(timing=timing), self.assertRaises(ValueError):
                cli.select_cases(manifest, 'all', timing)
        with self.assertRaises(ValueError):
            cli.select_cases({'cases': []}, 'all', 'all')

    def test_unrelated_negative_failure_cannot_replace_ignore_or_wrong_control(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            output, prepared = self.fixture(root, run=True)
            manifest = cli.load_preparation(prepared)
            source_case = next(row for row in manifest['cases']
                               if row['case_id'] == 'task_001_replacement_middle')
            config = fixture_case(language.read_json(prepared / source_case['config']))
            for condition in ('ignore', 'wrong'):
                # STOP at delivery fails replacement, but neither reaches A
                # (Ignore) nor executes invalid completion (Wrong).
                result = execute(config, callback=lambda request, assets:
                    '{"action":"STOP"}' if request['current_intervention'] else '{"action":"move_forward"}')
                folder = output / 'episodes/task_001_replacement_middle' / condition
                shutil.rmtree(folder)
                cli.write_episode(folder, config, result)
                report = cli.check_run(output)
                row = next(row for row in report['episodes']
                           if row['case_id'] == source_case['case_id'] and row['condition'] == condition)
                self.assertTrue(row['evidence_valid'])
                self.assertFalse(row['task_success'])
                self.assertFalse(row['passed'])

    def test_pose_or_visibility_measurements_cannot_be_forged_in_saved_logs(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            output, _ = self.fixture(root, run=True)
            path = output / 'episodes/task_001_subgoal_middle/follow/private/trajectory.json'
            original = path.read_bytes()
            for kind in ('pose', 'diagnostic'):
                rows = json.loads(original)
                if kind == 'pose':
                    rows[5]['position'][0] += .25
                else:
                    rows[-1]['measurements']['target_000']['diagnostic']['instance_visible'] = False
                write_json(path, rows)
                self.assertFalse(cli.check_run(output)['passed'], kind)
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
