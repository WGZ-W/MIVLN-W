"""Public model callbacks and three-condition navigation, without real GPUs/API."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

from mivln import TaskState
from mivln.navigation_updates import (apply_navigation_prediction, navigation_request,
                                     validate_navigation_candidates)
from mivln.task_update_prompts import encode, fingerprint
from scripts import language_navigation as experiment
from scripts import predict_task_updates as client, task_update as base
from scripts.hssd_targets import sha256


def catalog():
    return [{"goal_id": identity, "description": description, "aliases": [description]}
            for identity, description in (("a", "the metal bed"), ("b", "the dark sofa"), ("c", "the dining table"))]


def replace(old="a", new="b"):
    return encode({"status": "ok", "task_update": {"operation": "replace_goal", "old_goal": old, "new_goal": new}})


def source_fixture(root):
    """A small independently reviewed fixture, not evidence for actual HSSD data."""
    from tests.test_hssd_semantic import catalog_fixture
    scene = root / "scenes/test.scene_instance.json"
    scene.parent.mkdir()
    scene.write_text(encode({"object_instances": [{"template_name": f"template_{i}"} for i in range(3)]}))
    dataset, mesh = root / "dataset.json", root / "mesh.navmesh"
    dataset.write_text("{}")
    mesh.write_bytes(b"fixture mesh")
    config = {"episode_id": "task_001", "scene": str(scene), "scene_id": "test", "scene_dataset_config": str(dataset),
        "navmesh": str(mesh), "scene_sha256": sha256(scene), "scene_dataset_config_sha256": sha256(dataset),
        "navmesh_sha256": sha256(mesh), "agent_height": 1.5, "agent_radius": .1, "semantic_sensor": True,
        "initial_goal": "target_000", "initial_instruction": "Go to the visually distinct bed 0.",
        "start_position": [0, 0, 0], "start_rotation_xyzw": [0, 0, 0, 1], "seed": 42,
        "forward_step_m": .25, "turn_angle_degrees": 15, "success_radius_m": .3,
        "sensor_height": 1.5, "resolution": [4, 4], "max_steps": 60,
        "terminal_alignment": {"enabled": True, "yaw_tolerance_degrees": 7.5, "max_turn_actions": 24},
        "terminal_visibility": {"enabled": True, "min_pixels": 4}}
    reviewed, _ = catalog_fixture(root, config)
    path = root / "catalog.json"
    base.write_json(path, reviewed)
    config["catalog_source"] = {"path": str(path), "sha256": sha256(path)}
    config["goal_positions"] = {item["target_id"]: item["stop_position"] for item in reviewed["targets"][:2]}
    config["goal_metadata"] = {item["target_id"]: {"template_id": item["template_id"],
        "source_instance_index": item["source_instance_index"], "description": item["review"]["description"]}
        for item in reviewed["targets"][:2]}
    config["events"] = [{"event_id": "replace", "trigger": {"type": "step", "value": 1},
        "utterance": "Go to the visually distinct bed 1 instead.",
        "task_update": {"operation": "replace_goal", "old_goal": "target_000", "new_goal": "target_001"}}]
    source = root / "collection"
    for folder in (source / "configs/task_001", source / "tasks/task_001/reference", source / "tasks/task_001/timing/configs"):
        folder.mkdir(parents=True)
    base.write_json(source / "configs/task_001/base.json", config)
    reference = deepcopy(config)
    reference["events"] = []
    base.write_json(source / "configs/task_001/reference.json", reference)
    base.write_json(source / "tasks/task_001/reference/episode.json", reference)
    base.write_json(source / "tasks/task_001/reference/metrics.json", {"success": True, "stopped": True})
    rows = [{"step": None, "action": "INITIALIZE", "position": [0, 0, 0], "active_goal": "target_000", "distance_to_goal_m": 2}]
    for index in range(1, 9):
        x = min(index, 7) * .25
        rows.append({"step": index - 1, "action": "STOP" if index == 8 else "move_forward", "position": [x, 0, 0],
                     "active_goal": "target_000", "distance_to_goal_m": 2 - x})
    base.write_json(source / "tasks/task_001/reference/trajectory.json", rows)
    middle = deepcopy(config)
    middle["events"][0]["trigger"]["value"] = 4
    base.write_json(source / "tasks/task_001/timing/configs/middle.json", middle)
    task = {"task_id": "task_001", "output_subdir": "tasks/task_001", "base_config": "configs/task_001/base.json",
        "reference_config": "configs/task_001/reference.json"}
    task.update({key + "_sha256": sha256(source / task[key]) for key in ("base_config", "reference_config")})
    base.write_json(source / "manifest.json", {"schema_version": 1, "kind": "hssd_semantic_dataset", "task_count": 1,
        "catalog_source": config["catalog_source"], "tasks": [task]})
    return source


class NavigationBoundaryTests(unittest.TestCase):
    def test_public_request_uses_policy_state_without_event_or_geometry(self):
        request = navigation_request(22, "Go to the dark sofa instead.", catalog(), TaskState("a"))
        self.assertEqual(request["context"]["previous_state"]["active_goal"], "a")
        self.assertRegex(request["request_id"], r"^nav_[0-9a-f]{24}$")
        for private in ("task_update", "reference", "scene", "position", "goal_metadata", "new_goal"):
            self.assertNotIn(private, encode(request))

    def test_valid_wrong_target_really_changes_policy_without_reference_fallback(self):
        state = TaskState("a")
        request = navigation_request(2, "Go to the dark sofa instead.", catalog(), state)
        updated, record = apply_navigation_prediction(state, request, replace(new="c"))
        self.assertEqual(updated.active_goal, "c")
        self.assertEqual(state.active_goal, "a")
        self.assertEqual(record["outcome"], "applied")
        self.assertEqual(record["request_sha256"], fingerprint(request))

    def test_invalid_call_and_wrong_old_goal_leave_policy_unchanged(self):
        for raw, error in ((None, "offline"), ("not JSON", None), (replace(old="b", new="c"), None)):
            state = TaskState("a")
            request = navigation_request(2, "Go to B.", catalog(), state)
            updated, record = apply_navigation_prediction(state, request, raw, call_error=error)
            self.assertEqual(updated.to_dict(), state.to_dict())
            self.assertNotEqual(record["outcome"], "applied")

    def test_out_of_scope_operation_is_logged_not_converted_into_replacement(self):
        state = TaskState("a")
        request = navigation_request(2, "Go to B.", catalog(), state)
        raw = encode({"status": "ok", "task_update": {"operation": "insert_subgoal", "subgoal": "b", "resume_goal": "a"}})
        updated, record = apply_navigation_prediction(state, request, raw)
        self.assertTrue(record["schema_valid"])
        self.assertEqual(record["outcome"], "unsupported_navigation_operation")
        self.assertEqual(updated.to_dict(), state.to_dict())

    def test_candidate_catalog_rejects_coordinates_and_missing_goal_mapping(self):
        wrong = catalog()
        wrong[0]["position"] = [1, 2, 3]
        for values in (wrong, catalog()[:2]):
            with self.assertRaises(ValueError):
                validate_navigation_candidates(values, {"a", "b", "c"})


class LanguagePreparationTests(unittest.TestCase):
    def prepare(self, root):
        args = SimpleNamespace(source_dir=source_fixture(root), task_id="task_001", timing="middle", output_dir=root / "prepared")
        experiment.prepare(args)
        return args

    def test_preparation_is_cpu_only_and_preserves_reviewed_task(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            with patch.object(client, "urlopen") as http:
                args = self.prepare(root)
            http.assert_not_called()
            manifest, config = experiment.load_preparation(args.output_dir)
            self.assertEqual(config["events"][0]["trigger"]["value"], 4)
            self.assertEqual(config["goal_positions"], {"target_000": [2, 0, 0], "target_001": [-2, 0, 0]})
            self.assertEqual(manifest["conditions"], ["oracle", "model", "ignore"])
            self.assertNotIn("stop_position", encode(config["language_candidates"]))
            with self.assertRaises(FileExistsError):
                experiment.prepare(args)

    def test_preparation_rejects_modified_timing_reference_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = source_fixture(root)
            path = source / "tasks/task_001/timing/configs/middle.json"
            config = base.read_json(path)
            config["events"][0]["utterance"] = "A different task"
            base.write_json(path, config)
            with self.assertRaisesRegex(ValueError, "Timing changed"):
                experiment.prepare(SimpleNamespace(source_dir=source, task_id="task_001", timing="middle", output_dir=root / "bad"))
            self.assertFalse((root / "bad").exists())

    def test_prepared_catalog_cannot_be_changed_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            config = base.read_json(args.output_dir / "config.json")
            config["language_candidates"][0]["description"] = "A different object"
            base.write_json(args.output_dir / "config.json", config)
            manifest = base.read_json(args.output_dir / "manifest.json")
            manifest["config_sha256"] = sha256(args.output_dir / "config.json")
            base.write_json(args.output_dir / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "differs from the source"):
                experiment.load_preparation(args.output_dir)

    def test_changed_source_is_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            path = args.source_dir / "tasks/task_001/reference/trajectory.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                experiment.load_preparation(args.output_dir)


class LanguageClientTests(unittest.TestCase):
    def test_actual_http_body_and_raw_text_are_saved_without_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            controls = {"model": "mock", "base_url": "http://localhost:8000/v1", "temperature": 0.0,
                        "max_tokens": 512, "thinking_mode": "off", "timeout_seconds": 10.0}
            callback = experiment.ModelClient(controls, "Public instructions", output, "TASK_UPDATE_API_KEY")
            request = navigation_request(2, "Go to the dark sofa instead.", catalog(), TaskState("a"))
            captured = []
            def respond(http_request, **kwargs):
                captured.append(json.loads(http_request.data))
                return io.BytesIO(encode({"choices": [{"message": {"content": "```json\n{}\n```"}, "finish_reason": "stop"}]}).encode())
            with patch.object(client, "urlopen", side_effect=respond) as http:
                raw, _, error = callback(request)
            self.assertIsNone(error)
            self.assertEqual(raw, "```json\n{}\n```")
            self.assertEqual(http.call_count, 1)
            self.assertEqual(experiment.lines(output / "model_inputs.jsonl")[0]["payload"], captured[0])
            self.assertEqual(experiment.lines(output / "model_responses.jsonl")[0]["raw_prediction"], raw)

    def test_http_failure_retains_null_outcome_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            controls = {"model": "mock", "base_url": "http://localhost:8000/v1", "temperature": 0.0,
                        "max_tokens": 512, "thinking_mode": "off", "timeout_seconds": 10.0}
            callback = experiment.ModelClient(controls, "Public instructions", output, "TASK_UPDATE_API_KEY")
            request = navigation_request(2, "Go to B.", catalog(), TaskState("a"))
            with patch.object(client, "urlopen", side_effect=URLError("offline")) as http:
                raw, _, error = callback(request)
            self.assertIsNone(raw)
            self.assertIsNotNone(error)
            self.assertEqual(http.call_count, 1)
            self.assertEqual(len(experiment.lines(output / "model_responses.jsonl")), 1)


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires Habitat imports; renderer and HTTP are fake")
class ModelNavigationIntegrationTests(unittest.TestCase):
    def fixture(self, root, stack):
        from tests.test_terminal_view import TerminalNativeTests
        nav, config, scene, objects, instances = TerminalNativeTests().setup_navigation(root, stack)
        config["language_candidates"] = catalog()[:2]
        config["initial_instruction"] = "Go to the metal bed."
        config["events"][0]["utterance"] = "Go to the dark sofa instead."
        prepared = root / "prepared"
        prepared.mkdir()
        base.write_json(prepared / "manifest.json", {"unit_test_fixture": True})
        stack.enter_context(patch.object(experiment, "load_preparation", return_value=({}, config)))
        stack.enter_context(patch.object(experiment.subgoal, "pathfinder_for", return_value=SimpleNamespace()))
        stack.enter_context(patch.object(experiment.subgoal, "reviewed_centers", return_value={"a": [2, .5, 2], "b": [-2, .5, 2]}))
        args = SimpleNamespace(prepared_dir=prepared, output_dir=root / "run", gpu_id=0,
            model="mock", base_url="http://localhost:8000/v1", temperature=0.0, max_tokens=512,
            thinking_mode="off", timeout=10.0, api_key_env="TASK_UPDATE_API_KEY")
        return nav, config, scene, instances, args

    @staticmethod
    def response(raw):
        return io.BytesIO(encode({"model": "mock", "choices": [{"message": {"content": raw}, "finish_reason": "stop"}]}).encode())

    def test_three_branches_deliver_before_action_without_reset_or_label_leak(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            nav, config, _, instances, args = self.fixture(Path(temporary), stack)
            bodies = []
            def respond(request, **kwargs):
                bodies.append(json.loads(request.data))
                return self.response(replace())
            stack.enter_context(patch.object(client, "urlopen", side_effect=respond))
            self.assertEqual(experiment.run_experiment(args), 0)
            report = experiment.check_experiment(args.output_dir)
            self.assertTrue(report["passed"])
            self.assertTrue(report["integrity_passed"])
            self.assertEqual([r["dynamic_success"] for r in report["conditions"]], [True, True, False])
            self.assertEqual(len(bodies), 1)
            self.assertEqual([sim.initializations for sim in instances], [1, 1, 1])
            request = json.loads(bodies[0]["messages"][1]["content"])
            self.assertEqual(request["context"]["previous_state"]["active_goal"], "a")
            for private in ("goal_positions", "task_update", "template_id", "scene", "position", "new_goal"):
                self.assertNotIn(private, encode(request))
            expected = experiment.model_payload(base.read_json(args.output_dir / "run.json")["controls"],
                (args.output_dir / "model_system_prompt.txt").read_text(), request)
            self.assertEqual(expected, bodies[0])
            with self.assertRaises(FileExistsError):
                experiment.run_experiment(args)

    def test_invalid_prediction_is_validly_recorded_failure_not_oracle_fallback(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            _, _, _, _, args = self.fixture(Path(temporary), stack)
            stack.enter_context(patch.object(client, "urlopen", side_effect=lambda *a, **k: self.response("not JSON")))
            self.assertEqual(experiment.run_experiment(args), 1)
            report = experiment.check_experiment(args.output_dir)
            self.assertTrue(report["integrity_passed"])
            self.assertFalse(report["model_pipeline_success"])
            self.assertEqual(report["model_failure_stage"], "update")
            self.assertEqual(base.read_json(args.output_dir / "model/final_policy_state.json")["active_goal"], "a")
            self.assertEqual(base.read_json(args.output_dir / "model/final_task_state.json")["active_goal"], "b")
            self.assertEqual(base.read_json(args.output_dir / "model/replans.json"), [])

    def test_model_api_failure_preserves_denominator_and_continues_ignore(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            _, _, _, _, args = self.fixture(Path(temporary), stack)
            http = stack.enter_context(patch.object(client, "urlopen", side_effect=URLError("offline")))
            self.assertEqual(experiment.run_experiment(args), 1)
            report = experiment.check_experiment(args.output_dir)
            self.assertEqual(report["planned_episodes"], 3)
            self.assertTrue(report["integrity_passed"])
            self.assertEqual(report["model_failure_stage"], "model_call")
            self.assertTrue(report["conditions"][2]["ignore_expected_failure"])
            self.assertEqual(http.call_count, 1)

    def test_callback_cannot_mutate_the_request_used_to_validate_policy(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            nav, config, scene, _, _ = self.fixture(Path(temporary), stack)
            seen = []
            def callback(request):
                seen.append(deepcopy(request))
                request["context"]["previous_state"]["active_goal"] = "b"
                return replace(), {}, None
            result = nav.run(config, scene, 0, Path(temporary) / "direct", "model", callback)
            self.assertTrue(result["dynamic_success"])
            record = base.read_json(Path(temporary) / "direct/policy_updates.json")[0]
            self.assertEqual(record["request"], seen[0])

    def test_wrong_valid_target_is_planned_using_its_own_path(self):
        from tests.test_navigation_integration import NavigationIntegrationTests
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            fixture = NavigationIntegrationTests()
            fixture.setUp()
            fixture.simulator_patches(stack)
            from scripts import run_navigation as nav
            scene = root / "scene.glb"
            scene.write_bytes(b"unit test fixture")
            config = deepcopy(fixture.config)
            config["goal_positions"]["c"] = [4, 0, 0]
            config["language_candidates"] = catalog()
            result = nav.run(config, scene, 0, root / "wrong", "model", lambda request: (replace(new="c"), {}, None))
            self.assertFalse(result["dynamic_success"])
            self.assertEqual(result["policy_goal"], "c")
            plan = base.read_json(root / "wrong/replans.json")[0]
            self.assertEqual(plan["goal_id"], "c")
            self.assertEqual(plan["goal"], [4, 0, 0])
            self.assertAlmostEqual(plan["distance_m"], 3.5)

    def test_changed_actual_prompt_rejected_even_with_updated_journal_hash(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            _, _, _, _, args = self.fixture(Path(temporary), stack)
            stack.enter_context(patch.object(client, "urlopen", side_effect=lambda *a, **k: self.response(replace())))
            experiment.run_experiment(args)
            path = args.output_dir / "model_inputs.jsonl"
            values = experiment.lines(path)
            values[0]["payload"]["messages"][1]["content"] += " hidden reference"
            values[0]["payload_sha256"] = fingerprint(values[0]["payload"])
            base.write_jsonl(path, values)
            run = base.read_json(args.output_dir / "run.json")
            run["files_sha256"][path.name] = sha256(path)
            base.write_json(args.output_dir / "run.json", run)
            report = experiment.check_experiment(args.output_dir)
            self.assertFalse(report["integrity_passed"])
            self.assertIn("public policy-only", report["conditions"][1]["error"])

    def test_missing_model_results_keep_all_three_slots(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            _, _, _, _, args = self.fixture(Path(temporary), stack)
            stack.enter_context(patch.object(client, "urlopen", side_effect=lambda *a, **k: self.response(replace())))
            experiment.run_experiment(args)
            (args.output_dir / "model/metrics.json").unlink()
            report = experiment.check_experiment(args.output_dir)
            self.assertEqual(len(report["conditions"]), 3)
            self.assertFalse(report["conditions"][1]["integrity_passed"])
            self.assertFalse(report["passed"])

    def test_terminal_mask_tamper_rejected_after_updating_evidence_hashes(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            _, _, _, _, args = self.fixture(Path(temporary), stack)
            stack.enter_context(patch.object(client, "urlopen", side_effect=lambda *a, **k: self.response(replace())))
            self.assertEqual(experiment.run_experiment(args), 0)
            folder = args.output_dir / "oracle"
            np.save(folder / "terminal_instance_mask.npy", np.zeros((4, 4), dtype=np.bool_))
            diagnostic = base.read_json(folder / "terminal_diagnostics.json")
            for record in diagnostic["files"]:
                record["sha256"] = sha256(folder / record["path"])
            base.write_json(folder / "terminal_diagnostics.json", diagnostic)
            metrics = base.read_json(folder / "metrics.json")
            metrics["terminal_diagnostics_sha256"] = sha256(folder / "terminal_diagnostics.json")
            base.write_json(folder / "metrics.json", metrics)
            report = experiment.check_experiment(args.output_dir)
            self.assertFalse(report["integrity_passed"])
            self.assertIn("Instance pixels", report["conditions"][0]["error"])


if __name__ == "__main__":
    unittest.main()
