"""Versioned representative preparation, orchestration, and fake native loops.

Source audits and transport are synthetic in the orchestration tests; original
file/protocol hashes, request construction and independent selection remain real.
Native-import tests execute the actual navigation loop with a fake renderer.
"""

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
from urllib.error import URLError

from mivln import TaskState
from scripts import language_navigation as single, language_navigation_candidates as five
from scripts import language_navigation_dataset as batch, language_navigation_robustness as offline
from scripts import language_navigation_representatives as experiment, task_update as base
from tests.test_language_navigation_candidates import five_collection, model_evidence, physical_records
from tests.test_language_navigation_dataset import args_for, fake_audit, fake_run


def five_destination_fixture(root):
    source = five_collection(root)
    original = base.read_json(source / "manifest.json")
    for index in range(3, 6):
        identity = f"task_{index:03d}"
        shutil.copytree(source / "configs/task_001", source / f"configs/{identity}")
        shutil.copytree(source / "tasks/task_001", source / f"tasks/{identity}")
        original["tasks"].append({"task_id": identity, "base_config": f"configs/{identity}/base.json",
            "reference_config": f"configs/{identity}/reference.json", "output_subdir": f"tasks/{identity}"})
    catalog = base.read_json(root / "catalog.json")
    targets = {t["target_id"]: t for t in catalog["targets"]}
    for index, task in enumerate(original["tasks"]):
        old, new = f"target_{(index + 1) % 5:03d}", f"target_{index:03d}"
        config = base.read_json(source / task["base_config"])
        config.update(initial_goal=old, initial_instruction="Go to " + targets[old]["review"]["description"] + ".")
        config["goal_positions"] = {g: targets[g]["stop_position"] for g in (old, new)}
        config["goal_metadata"] = {g: {"template_id": targets[g]["template_id"],
            "source_instance_index": targets[g]["source_instance_index"], "category": targets[g]["category"],
            "description": targets[g]["review"]["description"]} for g in (old, new)}
        config["events"][0].update(utterance="Go to " + targets[new]["review"]["description"] + " instead.",
            task_update={"operation": "replace_goal", "old_goal": old, "new_goal": new})
        base.write_json(source / task["base_config"], config)
        reference = deepcopy(config)
        reference["events"] = []
        base.write_json(source / task["reference_config"], reference)
        directory = source / task["output_subdir"]
        base.write_json(directory / "reference/episode.json", reference)
        trajectory = base.read_json(directory / "reference/trajectory.json")
        for row in trajectory:
            row["active_goal"] = old
        base.write_json(directory / "reference/trajectory.json", trajectory)
        config["events"][0]["trigger"]["value"] = 4
        base.write_json(directory / "timing/configs/middle.json", config)
        for key in ("base_config", "reference_config"):
            task[key + "_sha256"] = single.sha256(source / task[key])
    original["task_count"] = 5
    base.write_json(source / "manifest.json", original)
    return source, targets


class RepresentativeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.contexts.enter_context(redirect_stdout(io.StringIO()))
        self.contexts.enter_context(patch.object(single, "check_experiment", side_effect=fake_audit))
        self.contexts.enter_context(patch.object(experiment, "geometry", return_value={"candidate_count": 5}))
        source, targets = five_destination_fixture(self.root)
        args = args_for(self.root, source)
        args.output_dir = self.root / "baseline"
        batch.prepare(args)
        self.populate(args.output_dir)
        self.five = self.root / "five"
        five.prepare(SimpleNamespace(baseline_dir=args.output_dir, output_dir=self.five))
        self.populate(self.five / "batch")
        wording = {"schema_version": 1, "kind": "hssd_explicit_replacement_wordings_v1", "language": "en",
            "provenance": "Synthetic test fixture", "templates": {
                "paraphrase_1": "Change the destination to {goal}.",
                "paraphrase_2": "Cancel the old destination. Go to {goal}."},
            "goals": {g: {"source_description": t["review"]["description"],
                "paraphrase_1": t["review"]["description"].replace("visually distinct", "uniquely marked"),
                "paraphrase_2": t["review"]["description"].replace("visually distinct", "individually identified")}
                for g, t in targets.items()}}
        path = self.root / "wordings.json"
        base.write_json(path, wording)
        self.prepared, self.offline_run = self.root / "prepared", self.root / "offline"
        offline.prepare(SimpleNamespace(source_dir=self.five, wordings=path, output_dir=self.prepared))
        self.records = base.read_json(self.prepared / "private/records.json")
        self.expected = {r["request"]["request_id"]: r["expected_prediction"] for r in self.records}
        with patch.object(offline.client, "complete_payload", side_effect=self.transport):
            offline.predict(SimpleNamespace(requests_dir=self.prepared / "public", output_dir=self.offline_run, api_key_env="TEST_KEY"))
        self.output = self.root / "representatives"

    def populate(self, output):
        manifest = base.read_json(output / "manifest.json")
        for task in manifest["tasks"]:
            directory, prepared = batch.result_dir(output, task), output / task["preparation_dir"]
            fake_run(directory, prepared, manifest["controls"])
            physical_records(directory)
            model_evidence(directory, prepared, manifest["controls"])

    def transport(self, endpoint, payload, key, timeout):
        request = base.strict_json(payload["messages"][-1]["content"])
        return single.encode(self.expected[request["request_id"]]), {"finish_reason": "stop"}

    def prepare(self):
        experiment.prepare(SimpleNamespace(prepared_dir=self.prepared, offline_run=self.offline_run, output_dir=self.output))
        return experiment.load_experiment(self.output)

    def fake_report(self, root, manifest, task, config):
        if not (root / task["output_subdir"] / "run.json").exists():
            raise FileNotFoundError("Not run")
        report = fake_audit(root / task["output_subdir"])
        after = TaskState(config["initial_goal"])
        after.apply_update(config["events"][0]["task_update"])
        report.update(task_id=task["task_id"], gpu_id=2,
            prediction={"status": "ok", "task_update": config["events"][0]["task_update"]}, model_state=after.to_dict())
        return report

    def launcher(self, codes=None):
        calls = []
        def launch(command, log):
            identity = command[command.index("--task-id") + 1]
            calls.append(identity)
            manifest = base.read_json(self.output / "manifest.json")
            task = next(t for t in manifest["tasks"] if t["task_id"] == identity)
            shutil.copytree(task["baseline_run_dir"], self.output / task["output_subdir"])
            return (codes or {}).get(identity, 0)
        return calls, launch

    def run_batch(self, launch, audit=None):
        with patch.object(batch, "execute_stage", side_effect=launch), \
                patch.object(experiment, "audit_task", side_effect=audit or self.fake_report):
            return experiment.run(SimpleNamespace(output_dir=self.output, gpu_id=2, api_key_env="TEST_KEY"))

    def test_selection_uses_target_coverage_and_lexical_task_id(self):
        records = deepcopy(self.records)
        extra = deepcopy(next(r for r in records if r["wording"] == experiment.WORDING and r["order"] == experiment.ORDER))
        extra.update(task_id="task_000", correct=False)
        records.append(extra)
        chosen = experiment.select_records(list(reversed(records)))
        self.assertEqual(len(chosen), 5)
        self.assertEqual(chosen[0]["task_id"], "task_000")
        self.assertEqual({r["expected_prediction"]["task_update"]["new_goal"] for r in chosen}, {f"target_{i:03d}" for i in range(5)})

    def test_only_identity_message_and_candidate_order_change(self):
        manifest, configs, _ = self.prepare()
        self.assertEqual(manifest["planned_episodes"], 15)
        self.assertEqual(manifest["planned_model_calls"], 5)
        for task in manifest["tasks"]:
            _, old = single.load_preparation(Path(task["baseline_prepared_dir"]))
            new = configs[task["task_id"]]
            for key in set(old) - {"episode_id", "events", "language_candidates"}:
                self.assertEqual(new[key], old[key])
            self.assertEqual(new["events"][0]["task_update"], old["events"][0]["task_update"])
            self.assertEqual(new["events"][0]["trigger"], old["events"][0]["trigger"])
            self.assertEqual(new["language_candidates"], list(reversed(old["language_candidates"])))
        previews = single.lines(self.output / "request_previews.jsonl")
        selected = experiment.select_records(self.records)
        for request, record in zip(previews, selected):
            self.assertEqual(experiment.semantic_request(request), experiment.semantic_request(record["request"]))
            self.assertNotEqual(request["request_id"], record["request"]["request_id"])

    def test_preparation_never_calls_model_or_changes_offline_files(self):
        before = {p.name: single.sha256(p) for p in self.offline_run.iterdir()}
        with patch.object(single.client, "complete_payload") as model:
            self.prepare()
        model.assert_not_called()
        self.assertEqual(before, {p.name: single.sha256(p) for p in self.offline_run.iterdir()})

    def test_pre_run_summary_retains_all_fifteen_slots(self):
        self.prepare()
        report = experiment.summarize(self.output)
        self.assertEqual(report["audited_episodes"], 0)
        self.assertEqual(report["model_failure_stages"]["not_run"], 5)
        self.assertTrue(report["results_provisional"])
        self.assertEqual(len(single.lines(self.output / "cases.jsonl")), 15)

    def test_config_semantic_drift_rejected_even_after_resigning(self):
        manifest, _, _ = self.prepare()
        task = manifest["tasks"][0]
        path = self.output / task["config_path"]
        value = base.read_json(path)
        value["events"][0]["trigger"]["value"] += 1
        base.write_json(path, value)
        task["config_sha256"] = single.sha256(path)
        base.write_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "changed more"):
            experiment.load_experiment(self.output)

    def test_selection_or_denominator_cannot_be_shrunk(self):
        manifest, _, _ = self.prepare()
        manifest["tasks"] = manifest["tasks"][:-1]
        base.write_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "selection"):
            experiment.load_experiment(self.output)

    def test_offline_response_tamper_blocks_new_launch(self):
        self.prepare()
        path = self.offline_run / "responses.jsonl"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with patch.object(batch, "execute_stage") as child:
            with self.assertRaisesRegex(ValueError, "Offline source"):
                experiment.run(SimpleNamespace(output_dir=self.output, gpu_id=2, api_key_env="KEY"))
        child.assert_not_called()

    def test_nonzero_child_continues_remaining_destinations(self):
        manifest, _, _ = self.prepare()
        calls, launch = self.launcher({manifest["tasks"][0]["task_id"]: 1})
        self.assertEqual(self.run_batch(launch), 0)
        self.assertEqual(calls, [t["task_id"] for t in manifest["tasks"]])
        self.assertEqual(base.read_json(self.output / "run_status.json")[0]["returncode"], 1)
        report = base.read_json(self.output / "summary.json")
        self.assertEqual(report["audited_episodes"], 15)
        self.assertEqual(report["offline_online_same_count"], 5)

    def test_model_failure_keeps_fixed_denominator_and_offline_disagreement(self):
        manifest, _, _ = self.prepare()
        calls, launch = self.launcher()
        def wrong(root, plan, task, config):
            report = self.fake_report(root, plan, task, config)
            if task["task_id"] == manifest["tasks"][0]["task_id"]:
                report.update(model_update_correct=False, model_pipeline_success=False, model_failure_stage="update", passed=False)
                report["prediction"] = {"status": "abstain", "reason": "unknown_goal"}
                report["model_state"] = TaskState(config["initial_goal"]).to_dict()
                report["conditions"][1]["joint_success"] = False
            return report
        self.assertEqual(self.run_batch(launch, wrong), 1)
        report = base.read_json(self.output / "summary.json")
        self.assertEqual(report["model_update_correct_count"], 4)
        self.assertEqual(report["model_update_correct_rate"], .8)
        self.assertEqual(report["offline_online_same_count"], 4)
        self.assertEqual(len(single.lines(self.output / "cases.jsonl")), 15)

    def test_interrupt_preserves_pending_slots_and_blocks_silent_retry(self):
        self.prepare()
        def interrupt(*args):
            raise KeyboardInterrupt
        self.assertEqual(self.run_batch(interrupt), 1)
        statuses = base.read_json(self.output / "run_status.json")
        self.assertEqual([s["status"] for s in statuses], ["interrupted", "pending", "pending", "pending", "pending"])
        self.assertEqual(base.read_json(self.output / "summary.json")["model_failure_stages"]["not_run"], 4)
        with self.assertRaises(FileExistsError):
            self.run_batch(interrupt)

    def test_occupied_outputs_block_before_first_gpu_child(self):
        self.prepare()
        (self.output / "tasks/task_005").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.run_batch(lambda *args: self.fail("must not launch"))
        self.assertFalse((self.output / "batch_run.json").exists())

    def test_cross_batch_changed_physical_control_is_not_model_success(self):
        manifest, _, _ = self.prepare()
        calls, normal = self.launcher()
        def changed(command, log):
            result = normal(command, log)
            if len(calls) == 1:
                path = self.output / manifest["tasks"][0]["output_subdir"] / "oracle/trajectory.json"
                rows = base.read_json(path)
                rows[0]["position"][0] += .1
                base.write_json(path, rows)
            return result
        self.assertEqual(self.run_batch(changed), 1)
        report = base.read_json(self.output / "summary.json")
        self.assertFalse(report["comparison_complete"])
        self.assertEqual(report["oracle_ignore_behavior_matched_tasks"], 4)

    def test_video_outputs_are_new_and_source_directories_untouched(self):
        self.prepare()
        with patch.object(batch, "export_video", return_value=Path("fixture.mp4")) as export:
            self.assertEqual(experiment.videos(SimpleNamespace(output_dir=self.output, fps=10.0, task_id=None)), 0)
        self.assertEqual(export.call_count, 15)
        self.assertTrue(all(self.output / "videos" in call.kwargs["output"].parents for call in export.call_args_list))


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires native imports; renderer and model are fake")
class RepresentativeLoopTests(unittest.TestCase):
    def fixture(self, root, stack):
        import magnum as mn
        from tests.test_language_navigation import catalog
        from tests.test_terminal_view import TerminalNativeTests
        nav, config, scene, objects, instances = TerminalNativeTests().setup_navigation(root, stack)
        # A third executable stop makes the wrong-target test a LEGAL update.
        # Replacing A with A is schema-invalid and cannot exercise replanning.
        objects["table"] = SimpleNamespace(object_id=3, semantic_id=27,
            creation_attributes=SimpleNamespace(handle="table.object_config.json"),
            aabb=objects["bed"].aabb, transformation=mn.Matrix4.translation(mn.Vector3(4, .5, 2)))
        contents = base.read_json(scene)
        contents["object_instances"].append({"template_name": "table"})
        base.write_json(scene, contents)
        config["goal_positions"]["c"] = [4, 0, 0]
        config["goal_metadata"]["c"] = {"template_id": "table", "source_instance_index": 2}
        config["initial_instruction"] = "Go to the metal bed."
        config["language_candidates"] = list(reversed(catalog()))
        stack.enter_context(patch.object(single.subgoal, "pathfinder_for", return_value=SimpleNamespace()))
        stack.enter_context(patch.object(single.subgoal, "reviewed_centers", return_value={
            "a": [2, .5, 2], "b": [-2, .5, 2], "c": [4, .5, 2]}))
        config["events"][0]["utterance"] = "Cancel the old destination. Go to the dark-colored couch."
        output = root / "representatives"
        output.mkdir()
        task = {"task_id": "fixture", "output_subdir": "tasks/fixture", "reference_dir": str(root)}
        manifest = {"controls": {"model": "mock", "base_url": "http://localhost:8000/v1", "temperature": 0.0,
            "max_tokens": 512, "timeout_seconds": 10.0, "thinking_mode": "off"}, "tasks": [task]}
        base.write_json(output / "manifest.json", manifest)
        stack.enter_context(patch.object(experiment, "load_experiment", return_value=(manifest, {"fixture": config}, {})))
        stack.enter_context(patch.object(experiment, "geometry", return_value={}))
        return config, instances, task, manifest, SimpleNamespace(output_dir=output, task_id="fixture", gpu_id=0, api_key_env="KEY")

    def test_reversed_catalog_and_paraphrase_reach_real_navigation_callback(self):
        from tests.test_language_navigation import replace
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            config, instances, task, manifest, args = self.fixture(Path(temporary), stack)
            seen = []
            def response(endpoint, payload, key, timeout):
                seen.append(base.strict_json(payload["messages"][-1]["content"]))
                return replace(), {}
            stack.enter_context(patch.object(single.client, "complete_payload", side_effect=response))
            self.assertEqual(experiment.run_task(args), 0)
            self.assertEqual(len(seen), 1)
            self.assertEqual([c["goal_id"] for c in seen[0]["candidate_goals"]], ["c", "b", "a"])
            self.assertEqual(seen[0]["message"], config["events"][0]["utterance"])
            self.assertEqual([sim.initializations for sim in instances], [1, 1, 1])
            self.assertTrue(experiment.audit_task(args.output_dir, manifest, task, config)["passed"])

    def test_invalid_online_output_is_not_replaced_by_offline_answer(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            config, _, task, manifest, args = self.fixture(Path(temporary), stack)
            stack.enter_context(patch.object(single.client, "complete_payload", return_value=("not JSON", {})))
            self.assertEqual(experiment.run_task(args), 1)
            report = experiment.audit_task(args.output_dir, manifest, task, config)
            self.assertTrue(report["integrity_passed"])
            self.assertEqual(report["model_failure_stage"], "update")
            self.assertEqual(base.read_json(args.output_dir / "tasks/fixture/model/final_policy_state.json")["active_goal"], "a")

    def test_api_failure_keeps_three_conditions_and_records_call_stage(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            config, _, task, manifest, args = self.fixture(Path(temporary), stack)
            call = stack.enter_context(patch.object(single.client, "complete_payload", side_effect=URLError("offline")))
            self.assertEqual(experiment.run_task(args), 1)
            report = experiment.audit_task(args.output_dir, manifest, task, config)
            self.assertEqual(report["planned_episodes"], 3)
            self.assertTrue(report["integrity_passed"])
            self.assertEqual(report["model_failure_stage"], "model_call")
            self.assertTrue(report["conditions"][2]["ignore_expected_failure"])
            self.assertEqual(call.call_count, 1)

    def test_valid_wrong_online_goal_is_executed_without_reference_repair(self):
        from tests.test_language_navigation import replace
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            config, _, task, manifest, args = self.fixture(Path(temporary), stack)
            stack.enter_context(patch.object(single.client, "complete_payload", return_value=(replace(new="c"), {})))
            self.assertEqual(experiment.run_task(args), 1)
            report = experiment.audit_task(args.output_dir, manifest, task, config)
            self.assertTrue(report["integrity_passed"])
            self.assertFalse(report["model_update_correct"])
            self.assertEqual(report["model_state"]["active_goal"], "c")
            replan = base.read_json(args.output_dir / "tasks/fixture/model/replans.json")[0]
            self.assertEqual(replan["goal_id"], "c")


if __name__ == "__main__":
    unittest.main()
