"""Frozen offline wording/order evaluation with synthetic source audits/HTTP.

Temporary scene evidence is not native navigation evidence. Real serialization,
request isolation, raw-response auditing and transactional state updates are
exercised while native audits and model transport are explicit test stubs.
"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

from mivln.task_update_prompts import encode, fingerprint
from scripts import language_navigation as single, language_navigation_candidates as five
from scripts import language_navigation_dataset as batch, language_navigation_robustness as experiment
from scripts import task_update as base
from tests.test_language_navigation_candidates import five_collection, model_evidence, physical_records
from tests.test_language_navigation_dataset import args_for, fake_audit, fake_run


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(redirect_stdout(io.StringIO()))
        self.audit = contexts.enter_context(patch.object(single, "check_experiment", side_effect=fake_audit))
        source = five_collection(self.root)
        args = args_for(self.root, source)
        args.output_dir = self.root / "baseline"
        batch.prepare(args)
        baseline = base.read_json(args.output_dir / "manifest.json")
        self.populate(args.output_dir, baseline)
        self.source = self.root / "five"
        five.prepare(SimpleNamespace(baseline_dir=args.output_dir, output_dir=self.source))
        self.inner = base.read_json(self.source / "batch/manifest.json")
        self.populate(self.source / "batch", self.inner)
        self.wordings = self.root / "wordings.json"
        _, config = single.load_preparation(self.source / "batch" / self.inner["tasks"][0]["preparation_dir"])
        wordings = {"schema_version": 1, "kind": "hssd_explicit_replacement_wordings_v1", "language": "en",
            "provenance": "Synthetic test wording fixture", "templates": {
                "paraphrase_1": "Change destination to {goal}; cancel the previous one.",
                "paraphrase_2": "The old destination is cancelled. Go to {goal}."},
            "goals": {c["goal_id"]: {"source_description": c["description"],
                "paraphrase_1": c["description"].replace("visually distinct", "uniquely marked"),
                "paraphrase_2": c["description"].replace("visually distinct", "individually identified")}
                for c in config["language_candidates"]}}
        base.write_json(self.wordings, wordings)
        self.prepared = self.root / "prepared"
        self.run_dir = self.root / "predictions"
        self.scored = self.root / "scored"

    def populate(self, output, manifest):
        for task in manifest["tasks"]:
            directory = batch.result_dir(output, task)
            prepared = output / task["preparation_dir"]
            fake_run(directory, prepared, manifest["controls"])
            physical_records(directory)
            model_evidence(directory, prepared, manifest["controls"])

    def prepare(self):
        experiment.prepare(SimpleNamespace(source_dir=self.source, wordings=self.wordings, output_dir=self.prepared))
        return experiment.load_prepared(self.prepared)

    def predict(self, transform=None, public=None):
        records = base.read_json(self.prepared / "private/records.json")
        expected = {r["request"]["request_id"]: r["expected_prediction"] for r in records}
        calls = []
        def complete(endpoint, payload, key, timeout):
            request = base.strict_json(payload["messages"][-1]["content"])
            calls.append(request)
            prediction = deepcopy(expected[request["request_id"]])
            if transform:
                result = transform(len(calls), prediction, request)
                if isinstance(result, str):
                    return result, {}
                prediction = result
            return encode(prediction), {"finish_reason": "stop", "reported_model": "qwen3.5-9b"}
        with patch.object(experiment.client, "complete_payload", side_effect=complete):
            code = experiment.predict(SimpleNamespace(requests_dir=public or self.prepared / "public",
                output_dir=self.run_dir, api_key_env="TEST_KEY"))
        return code, calls

    def score(self):
        return experiment.score(SimpleNamespace(prepared_dir=self.prepared, run_dir=self.run_dir, output_dir=self.scored))

    def resign(self):
        path = self.run_dir / "sealed_run.json"
        seal = base.read_json(path)
        seal["files_sha256"] = {name: single.sha256(self.run_dir / name) for name in experiment.RUN_FILES}
        base.write_json(path, seal)

    def test_complete_factorial_plan_changes_only_wording_and_order(self):
        manifest, records = self.prepare()
        self.assertEqual(manifest["planned_records"], 12)
        self.assertEqual(len({r["request"]["request_id"] for r in records}), 12)
        for task in self.inner["tasks"]:
            group = [r for r in records if r["task_id"] == task["task_id"]]
            self.assertEqual(len(group), 6)
            for wording in experiment.WORDINGS:
                first, second = [r for r in group if r["wording"] == wording]
                self.assertEqual(first["request"]["candidate_goals"], list(reversed(second["request"]["candidate_goals"])))
                self.assertEqual(first["request"]["context"], second["request"]["context"])
                self.assertEqual(first["expected_prediction"], second["expected_prediction"])
                self.assertEqual(first["request"]["message"], second["request"]["message"])
            self.assertEqual(len({r["request"]["message"] for r in group}), 3)
            self.assertTrue(all(r["request"]["mode"] == "state" for r in group))

    def test_preparation_never_calls_model_and_public_contains_no_references(self):
        with patch.object(experiment.client, "complete_payload") as call:
            self.prepare()
        call.assert_not_called()
        self.assertEqual({p.name for p in (self.prepared / "public").iterdir()},
                         {"manifest.json", "instructions.txt", "requests.jsonl", "request_index.json"})
        for request in single.lines(self.prepared / "public/requests.jsonl"):
            for key in ("expected_prediction", "expected_state", "task_id", "position", "scene", "navmesh", "template_id"):
                self.assertNotIn(key, encode(request))
        preview = base.read_json(self.prepared / "prompt_plan/run.json")
        self.assertTrue(preview["prepare_only"])
        self.assertFalse((self.prepared / "prompt_plan/predictions.jsonl").exists())

    def test_predict_reads_only_public_export_and_uses_frozen_controls(self):
        self.prepare()
        public = self.root / "copied_public"
        shutil.copytree(self.prepared / "public", public)
        with patch.object(experiment, "source_tasks", side_effect=AssertionError("Private source opened")), \
                patch.object(experiment, "load_prepared", side_effect=AssertionError("Private plan opened")):
            code, calls = self.predict(public=public)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 12)
        run = base.read_json(self.run_dir / "run.json")
        for key, value in self.inner["controls"].items():
            self.assertEqual(run[key], value)
        self.assertEqual(run["prompt_variant"], "baseline")

    def test_perfect_stub_scores_full_denominators_and_groups(self):
        self.prepare()
        self.predict()
        self.assertEqual(self.score(), 0)
        report = base.read_json(self.scored / "summary.json")
        self.assertEqual(report["statistics"]["correct_count"], 12)
        self.assertEqual(report["order_pairs"]["all_correct_count"], 6)
        self.assertEqual(report["wording_groups"]["all_correct_count"], 4)
        self.assertEqual(report["task_groups"]["all_correct_count"], 2)
        self.assertEqual(len(single.lines(self.scored / "cases.jsonl")), 12)

    def test_consistent_wrong_distractor_never_counts_as_correct(self):
        self.prepare()
        def wrong(index, prediction, request):
            prediction["task_update"]["new_goal"] = "target_004"
            return prediction
        self.predict(wrong)
        self.assertEqual(self.score(), 1)
        report = base.read_json(self.scored / "summary.json")
        self.assertEqual(report["statistics"]["correct_count"], 0)
        self.assertEqual(report["statistics"]["distractor_selection_count"], 12)
        self.assertEqual(report["order_pairs"]["consistent_count"], 6)
        self.assertEqual(report["order_pairs"]["all_correct_count"], 0)
        self.assertTrue(all(r["after_state"]["active_goal"] == "target_004" for r in single.lines(self.scored / "cases.jsonl")))

    def test_http_failure_and_malformed_text_keep_all_slots(self):
        self.prepare()
        def fail(index, prediction, request):
            if index == 1:
                raise URLError("offline fixture")
            return "```json\n{}\n```" if index == 2 else prediction
        code, calls = self.predict(fail)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 12)
        self.assertEqual(self.score(), 1)
        report = base.read_json(self.scored / "summary.json")
        self.assertEqual(report["statistics"]["planned"], 12)
        self.assertEqual(report["statistics"]["correct_count"], 10)
        self.assertEqual(report["statistics"]["schema_invalid_count"], 2)
        self.assertEqual(report["statistics"]["call_failure_count"], 1)

    def test_interruption_retains_null_tail_and_cannot_gain_consistency(self):
        self.prepare()
        def interrupt(*args):
            raise KeyboardInterrupt
        code, calls = self.predict(interrupt)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.score(), 1)
        report = base.read_json(self.scored / "summary.json")
        self.assertEqual(report["statistics"]["planned"], 12)
        self.assertEqual(report["order_pairs"]["consistent_count"], 0)
        self.assertEqual(report["actual_attempted"], 1)
        self.assertTrue(report["interrupted"])

    def test_failed_old_goal_does_not_carry_into_another_trial(self):
        self.prepare()
        def wrong_old(index, prediction, request):
            if index == 1:
                prediction["task_update"]["old_goal"] = "target_003"
            return prediction
        self.predict(wrong_old)
        self.score()
        rows = single.lines(self.scored / "cases.jsonl")
        self.assertEqual(rows[0]["outcome"], "application_failed")
        self.assertEqual(rows[0]["after_state"]["active_goal"], "target_000")
        self.assertTrue(rows[1]["correct"])

    def test_source_or_wordings_drift_blocks_score_before_output(self):
        self.prepare()
        value = base.read_json(self.wordings)
        value["goals"]["target_001"]["paraphrase_1"] = "a different target"
        base.write_json(self.wordings, value)
        with self.assertRaisesRegex(ValueError, "wordings changed"):
            experiment.load_prepared(self.prepared)

    def test_private_reference_tamper_is_caught_after_resigning(self):
        self.prepare()
        path = self.prepared / "private/records.json"
        rows = base.read_json(path)
        rows[0]["expected_prediction"]["task_update"]["new_goal"] = "target_004"
        base.write_json(path, rows)
        manifest = base.read_json(self.prepared / "manifest.json")
        manifest["private_records_sha256"] = single.sha256(path)
        base.write_json(self.prepared / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "independently rebuilt"):
            experiment.load_prepared(self.prepared)

    def test_changed_public_request_is_caught_after_resigning(self):
        self.prepare()
        path = self.prepared / "public/requests.jsonl"
        rows = single.lines(path)
        rows[0]["candidate_goals"][0]["aliases"].append("a secret extra alias")
        base.write_jsonl(path, rows)
        public = base.read_json(self.prepared / "public/manifest.json")
        index = {r["request_id"]: fingerprint(r) for r in rows}
        base.write_json(self.prepared / "public/request_index.json", index)
        public.update(requests_sha256=single.sha256(path),
                      request_index_sha256=single.sha256(self.prepared / "public/request_index.json"))
        base.write_json(self.prepared / "public/manifest.json", public)
        manifest = base.read_json(self.prepared / "manifest.json")
        manifest["public_manifest_sha256"] = single.sha256(self.prepared / "public/manifest.json")
        base.write_json(self.prepared / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "roster"):
            experiment.load_prepared(self.prepared)

    def test_raw_response_tamper_rejected_without_overwriting_scores(self):
        self.prepare()
        self.predict()
        path = self.run_dir / "responses.jsonl"
        rows = single.lines(path)
        rows[0]["finish_reason"] = "length"
        base.write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "Sealed raw"):
            self.score()
        self.assertFalse(self.scored.exists())

    def test_actual_payload_drift_rejected_even_after_resigning(self):
        self.prepare()
        self.predict()
        path = self.run_dir / "model_inputs.jsonl"
        rows = single.lines(path)
        rows[0]["payload"]["messages"][1]["content"] = "A different request"
        rows[0]["payload_sha256"] = fingerprint(rows[0]["payload"])
        base.write_jsonl(path, rows)
        run = base.read_json(self.run_dir / "run.json")
        run["model_inputs_sha256"] = single.sha256(path)
        base.write_json(self.run_dir / "run.json", run)
        self.resign()
        with self.assertRaisesRegex(ValueError, "Actual model input"):
            self.score()

    def test_duplicate_or_missing_predictions_cannot_shrink_denominator(self):
        self.prepare()
        self.predict()
        path = self.run_dir / "predictions.jsonl"
        rows = single.lines(path)
        base.write_jsonl(path, rows[:-1])
        run = base.read_json(self.run_dir / "run.json")
        run["predictions_sha256"] = single.sha256(path)
        base.write_json(self.run_dir / "run.json", run)
        self.resign()
        with self.assertRaisesRegex(ValueError, "entire request roster"):
            self.score()

    def test_wrong_review_description_rejected_before_preparation(self):
        value = base.read_json(self.wordings)
        value["goals"]["target_001"]["source_description"] = "unreviewed description"
        base.write_json(self.wordings, value)
        with self.assertRaisesRegex(ValueError, "reviewed description"):
            self.prepare()
        self.assertFalse(self.prepared.exists())

    def test_honest_source_model_failures_do_not_filter_tasks(self):
        self.audit.side_effect = lambda path: fake_audit(path, correct=False)
        manifest, _ = self.prepare()
        self.assertEqual(manifest["planned_records"], 12)

    def test_missing_source_evidence_is_rejected_before_preparation(self):
        self.audit.side_effect = lambda path: fake_audit(path, integrity=False)
        with self.assertRaisesRegex(ValueError, "complete evidence"):
            self.prepare()
        self.assertFalse(self.prepared.exists())

    def test_preparation_prediction_and_score_never_overwrite(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.predict()
        with self.assertRaises(FileExistsError):
            self.predict()
        self.score()
        with self.assertRaises(FileExistsError):
            self.score()


if __name__ == "__main__":
    unittest.main()
