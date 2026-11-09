"""Five-candidate derivation and paired controls, without real GPU/API calls.

Native audits are explicitly synthetic stubs here. Source/review binding, actual
public requests, response replay, state application and file hashes remain real;
these fixtures must never be interpreted as model or HSSD performance evidence.
"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln import TaskState
from mivln.navigation_updates import apply_navigation_prediction, navigation_request
from mivln.task_update_prompts import encode, fingerprint, system_prompt
from scripts import hssd_targets, language_navigation as single
from scripts import language_navigation_candidates as experiment, language_navigation_dataset as batch
from scripts import task_update as base
from tests.test_language_navigation_dataset import collection, args_for, fake_audit, fake_run


def five_collection(root):
    """Extend the small review fixture with two more synthetic reviewed targets."""
    source = collection(root, count=2)
    path = root / "catalog.json"
    catalog = base.read_json(path)
    evidence_path = root / "previews/preview.json"
    evidence = base.read_json(evidence_path)
    for index in (3, 4):
        target = deepcopy(catalog["targets"][0])
        identity = f"target_{index:03d}"
        target.update(target_id=identity, source_instance_index=index, template_id=f"template_{index}",
                      stop_position=[2 + index * 3, 0, 0])
        target["review"]["description"] = f"the visually distinct bed {index}"
        catalog["targets"].append(target)
        folder = root / "previews" / identity
        folder.mkdir()
        files = []
        for name in ("rgb.png", "depth.npy", "semantic.npy", "category_mask.png"):
            attachment = folder / name
            attachment.write_bytes(f"Synthetic {identity}/{name}".encode())
            files.append({"path": str(attachment), "sha256": hssd_targets.sha256(attachment)})
        evidence["targets"][identity] = {"files": files, "same_category_pixels": 10}
    evidence["candidate_identity"] = hssd_targets.candidate_identity(catalog)
    base.write_json(evidence_path, evidence)
    for target in catalog["targets"]:
        target["review"]["preview"]["sha256"] = hssd_targets.sha256(evidence_path)
    base.write_json(path, catalog)
    manifest = base.read_json(source / "manifest.json")
    catalog_source = {"path": str(path), "sha256": hssd_targets.sha256(path)}
    for candidate in source.rglob("*.json"):
        if candidate.name == "manifest.json":
            continue
        value = base.read_json(candidate)
        if isinstance(value, dict) and "catalog_source" in value:
            value["catalog_source"] = catalog_source
            base.write_json(candidate, value)
    manifest["catalog_source"] = catalog_source
    for task in manifest["tasks"]:
        for key in ("base_config", "reference_config"):
            task[key + "_sha256"] = hssd_targets.sha256(source / task[key])
    base.write_json(source / "manifest.json", manifest)
    return source


def model_evidence(directory, prepared, controls, goal=None):
    """Use real model-boundary serialization/replay with clearly synthetic text."""
    _, config = single.load_preparation(prepared)
    event = config["events"][0]
    state = TaskState(config["initial_goal"])
    request = navigation_request(event["trigger"]["value"], event["utterance"], config["language_candidates"], state)
    update = deepcopy(event["task_update"])
    if goal:
        update["new_goal"] = goal
    raw = encode({"status": "ok", "task_update": update})
    payload = single.model_payload(controls, system_prompt(base.INSTRUCTIONS), request)
    base.write_jsonl(directory / "model_inputs.jsonl", [{"request_id": request["request_id"],
        "request_sha256": fingerprint(request), "payload": payload, "payload_sha256": fingerprint(payload)}])
    base.write_jsonl(directory / "model_responses.jsonl", [{"request_id": request["request_id"],
        "raw_prediction": raw, "metadata": {}, "error": None}])
    _, record = apply_navigation_prediction(state, request, raw)
    record.update(event_id=event["event_id"], delivered_step=event["trigger"]["value"], callback_seconds=.25)
    base.write_json(directory / "model/policy_updates.json", [record])


def physical_records(directory):
    """Opaque sensor bytes; the production native audit is stubbed in this file."""
    for condition in single.CONDITIONS:
        branch = directory / condition
        row = {"step": None, "action": "INITIALIZE", "position": [0, 0, 0],
               "rotation_xyzw": [0, 0, 0, 1], "collided": False, "distance_m": 0,
               "frames": {kind: "frame.bin" for kind in ("rgb", "depth", "semantic")}}
        base.write_json(branch / "trajectory.json", [row])


class CandidateExpansionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.audit = contexts.enter_context(patch.object(single, "check_experiment", side_effect=fake_audit))
        contexts.enter_context(redirect_stdout(io.StringIO()))
        self.source = five_collection(self.root)
        args = args_for(self.root, self.source)
        self.baseline = self.root / "baseline"
        args.output_dir = self.baseline
        batch.prepare(args)
        self.original = base.read_json(self.baseline / "manifest.json")
        for task in self.original["tasks"]:
            folder = batch.result_dir(self.baseline, task)
            prepared = self.baseline / task["preparation_dir"]
            fake_run(folder, prepared, self.original["controls"])
            physical_records(folder)
            model_evidence(folder, prepared, self.original["controls"])
        self.output = self.root / "five"

    def prepare(self):
        experiment.prepare(SimpleNamespace(baseline_dir=self.baseline, output_dir=self.output))
        return experiment.validate(self.output)

    def new_runs(self, wrong=False):
        _, inner = self.prepare()
        for task in inner["tasks"]:
            directory = batch.result_dir(self.output / "batch", task)
            prepared = self.output / "batch" / task["preparation_dir"]
            fake_run(directory, prepared, inner["controls"])
            physical_records(directory)
            model_evidence(directory, prepared, inner["controls"], "target_004" if wrong else None)
        if wrong:
            self.audit.side_effect = lambda path: fake_audit(path, correct=self.output not in path.parents)
        return inner

    def test_expansion_changes_only_executor_maps_and_public_candidates(self):
        _, inner = self.prepare()
        for original_task, task in zip(self.original["tasks"], inner["tasks"]):
            _, old = single.load_preparation(self.baseline / original_task["preparation_dir"])
            _, new = single.load_preparation(self.output / "batch" / task["preparation_dir"])
            self.assertEqual(len(new["language_candidates"]), 5)
            for key in set(old) - {"goal_positions", "goal_metadata", "language_candidates"}:
                self.assertEqual(new[key], old[key], key)
            for key in ("goal_positions", "goal_metadata"):
                for identity, value in old[key].items():
                    self.assertEqual(new[key][identity], value)
            self.assertEqual([c["goal_id"] for c in new["language_candidates"]], sorted(new["goal_positions"]))
        self.assertEqual(inner["controls"], self.original["controls"])
        self.assertEqual(inner["reused_tasks"], 0)
        self.assertEqual(inner["planned_episodes"], 6)
        self.assertEqual(inner["new_model_calls"], 2)

    def test_reference_copies_are_unmodified_trigger_evidence(self):
        self.prepare()
        for task in self.original["tasks"]:
            info, _ = single.load_preparation(self.baseline / task["preparation_dir"])
            for name in experiment.REFERENCE_FILES:
                original = Path(info["reference_dir"]) / name
                copied = self.output / "collection/tasks" / task["task_id"] / "reference" / name
                self.assertEqual(original.read_bytes(), copied.read_bytes())
            copied_episode = base.read_json(self.output / "collection/tasks" / task["task_id"] / "reference/episode.json")
            self.assertEqual(len(copied_episode["goal_positions"]), 2)
        self.assertIn("trigger selection only", base.read_json(self.output / "collection/manifest.json")["derivation"]["reference_usage"])

    def test_preparation_has_no_api_calls_and_does_not_modify_baseline(self):
        before = experiment.frozen_inputs(self.baseline, self.original)
        with patch.object(single.client, "complete_payload") as call:
            self.prepare()
        call.assert_not_called()
        self.assertEqual(experiment.frozen_inputs(self.baseline, self.original), before)
        for request in single.lines(self.output / "request_previews.jsonl"):
            self.assertEqual(len(request["candidate_goals"]), 5)
            for private in ("position", "template_id", "task_update", "reference", "scene", "navmesh"):
                self.assertNotIn(private, encode(request))

    def test_honest_baseline_model_failure_is_not_filtered_out(self):
        self.audit.side_effect = lambda path: fake_audit(path, correct=False)
        self.prepare()
        summary = experiment.summarize(self.output)
        self.assertEqual(summary["baseline_update_correct_count"], 0)
        self.assertEqual(summary["task_count"], 2)

    def test_incomplete_baseline_rejected_before_output_creation(self):
        self.audit.side_effect = lambda path: fake_audit(path, integrity=False)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_candidate_count_must_be_five_before_output_creation(self):
        with patch.object(experiment, "validate_catalog", return_value=[]):
            with self.assertRaisesRegex(ValueError, "exactly five"):
                self.prepare()
        self.assertFalse(self.output.exists())

    def test_output_never_overwritten(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_changed_source_baseline_blocks_future_run(self):
        self.prepare()
        path = batch.result_dir(self.baseline, self.original["tasks"][0]) / "model/frame.bin"
        path.write_bytes(b"tampered baseline observation")
        with patch.object(batch, "run") as launch:
            with self.assertRaisesRegex(ValueError, "Frozen baseline"):
                experiment.run(SimpleNamespace(output_dir=self.output, gpu_id=2, api_key_env="KEY"))
        launch.assert_not_called()

    def test_semantic_drift_is_rejected_even_after_resigning_inner_hashes(self):
        self.prepare()
        path = self.output / "collection/tasks/task_001/timing/configs/middle.json"
        value = base.read_json(path)
        value["events"][0]["utterance"] = "A different request"
        base.write_json(path, value)
        with self.assertRaisesRegex(ValueError, "beyond candidate maps"):
            experiment.validate(self.output)

    def test_existing_goal_position_must_match_review(self):
        config = base.read_json(self.source / "configs/task_001/base.json")
        config["goal_positions"][config["initial_goal"]][0] += 10
        targets = hssd_targets.validate_catalog(base.read_json(self.root / "catalog.json"))
        with self.assertRaisesRegex(ValueError, "reviewed"):
            experiment.expand_config(config, targets)

    def test_pre_run_summary_keeps_fixed_slots_and_provisional_flag(self):
        self.prepare()
        report = experiment.summarize(self.output)
        self.assertEqual(report["expanded_update_correct_count"], 0)
        self.assertEqual(report["planned_episodes"], 6)
        self.assertTrue(report["results_provisional"])
        self.assertFalse(report["comparison_complete"])
        self.assertEqual(len(single.lines(self.output / "comparison.jsonl")), 2)
        self.assertEqual(len(single.lines(self.output / "batch/cases.jsonl")), 6)
        self.assertTrue(all(p["oracle_behavior_matched"] is None for p in report["pairs"]))

    def test_new_runs_are_compared_to_raw_baseline_not_cached_summary(self):
        self.new_runs()
        base.write_json(self.baseline / "summary.json", {"model_update_correct_count": -999})
        report = experiment.summarize(self.output)
        self.assertEqual(report["baseline_update_correct_count"], 2)
        self.assertEqual(report["expanded_pipeline_success_count"], 2)
        self.assertTrue(report["comparison_complete"])
        self.assertTrue(report["passed"])

    def test_valid_wrong_distractor_is_applied_and_retained_as_failure(self):
        self.new_runs(wrong=True)
        report = experiment.summarize(self.output)
        self.assertEqual(report["selected_distractor_count"], 2)
        self.assertEqual(report["expanded_pipeline_success_count"], 0)
        self.assertTrue(report["comparison_complete"])
        self.assertFalse(report["passed"])
        for pair in report["pairs"]:
            self.assertEqual(pair["predicted_goal"], "target_004")
            path = self.output / "batch/tasks" / pair["task_id"] / "model/policy_updates.json"
            record = base.read_json(path)[0]
            self.assertEqual(record["after_state"]["active_goal"], "target_004")
            self.assertNotEqual(record["after_state"]["active_goal"], pair["expected_goal"])

    def test_changed_oracle_behavior_invalidates_controlled_comparison(self):
        inner = self.new_runs()
        folder = batch.result_dir(self.output / "batch", inner["tasks"][0])
        path = folder / "oracle/trajectory.json"
        rows = base.read_json(path)
        rows[0]["position"][0] = .1
        base.write_json(path, rows)
        report = experiment.summarize(self.output)
        self.assertEqual(report["oracle_ignore_behavior_matched_tasks"], 1)
        self.assertFalse(report["comparison_complete"])
        self.assertFalse(report["passed"])

    def test_cross_batch_pixels_are_reported_separately_from_physical_controls(self):
        inner = self.new_runs()
        folder = batch.result_dir(self.output / "batch", inner["tasks"][0])
        (folder / "oracle/frame.bin").write_bytes(b"Different GPU rendering fixture")
        report = experiment.summarize(self.output)
        self.assertTrue(report["pairs"][0]["oracle_behavior_matched"])
        self.assertFalse(report["pairs"][0]["oracle_observations_matched"])
        self.assertTrue(report["comparison_complete"])

    def test_baseline_evidence_roster_cannot_be_shrunk(self):
        self.prepare()
        path = self.output / "manifest.json"
        manifest = base.read_json(path)
        name = next(iter(manifest["baseline_inputs_sha256"]))
        del manifest["baseline_inputs_sha256"][name]
        base.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "Frozen baseline"):
            experiment.validate(self.output)

    def test_unreachable_candidate_blocks_run_without_dropping_it(self):
        self.prepare()
        native = SimpleNamespace(validate_config=lambda config: None,
                                 scene_provenance=lambda config, path: None)
        pathfinder = SimpleNamespace(is_navigable=lambda point: True)
        with patch.dict(sys.modules, {"scripts.run_navigation": native}), \
                patch.object(single.subgoal, "pathfinder_for", return_value=pathfinder), \
                patch.object(single.subgoal, "distance_between", return_value=None), \
                patch.object(batch, "run") as launch:
            with self.assertRaisesRegex(ValueError, "unreachable"):
                experiment.run(SimpleNamespace(output_dir=self.output, gpu_id=2, api_key_env="KEY"))
        launch.assert_not_called()
        self.assertEqual(len(single.lines(self.output / "request_previews.jsonl")[0]["candidate_goals"]), 5)


if __name__ == "__main__":
    unittest.main()
