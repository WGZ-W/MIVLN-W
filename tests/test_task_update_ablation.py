"""Controlled prompt inputs, provenance and paired errors without a real model."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

from mivln.task_update_prompts import VARIANTS, encode, fingerprint, system_prompt, user_prompt
from scripts import predict_task_updates as inference
from scripts import task_update as base
from scripts import task_update_ablation as ablation
from scripts import task_update_pairs as pairs


class PromptAblationTests(unittest.TestCase):
    def prepare(self, root):
        config = base.read_json(pairs.DEFAULT_CONFIG)
        config["families"] = [config["families"][0], config["families"][8]]
        source = root / "source.json"
        base.write_json(source, config)
        public = root / "public"
        pairs.export(SimpleNamespace(dataset=source, split="dev", mode="all", output_dir=public))
        return SimpleNamespace(requests_dir=public, model="mock-model", base_url="http://localhost:8000/v1",
            temperature=0.0, max_tokens=512, timeout=120.0, thinking_mode="off",
            api_key_env="TASK_UPDATE_API_KEY", output_dir=root / "preview")

    @staticmethod
    def response(request, **kwargs):
        # Deliberately constant incorrect output, not an actual model result.
        content = encode({"status": "abstain", "reason": "missing_context"})
        return io.BytesIO(encode({"choices": [{"message": {"content": content},
                                             "finish_reason": "stop"}]}).encode("utf-8"))

    def runs(self, root, args, response=None):
        folders = {}
        for variant in VARIANTS:
            folder = root / variant
            child = SimpleNamespace(**{**vars(args), "output_dir": folder, "prompt_variant": variant})
            with patch.object(inference, "urlopen", side_effect=response or self.response):
                inference.generate(child)
            folders[variant] = folder
        return SimpleNamespace(requests_dir=args.requests_dir, baseline_run=folders["baseline"],
            history_run=folders["history_timeline"], refusal_run=folders["refusal_check"], output_dir=root / "compare")

    def test_baseline_exact_legacy_bytes_and_history_changes_only_history(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            _, _, requests, instructions = inference.load_public_requests(args.requests_dir)
            suffix = ("\n本次调用只输出 prediction 对应的 JSON 对象。外层 request_id、request_sha256、"
                      "prediction 包装由客户端生成，不要在模型回答中输出外层包装。\n")
            self.assertEqual(system_prompt(instructions), instructions + suffix)
            for request in requests.values():
                original = deepcopy(request)
                self.assertEqual(user_prompt(request), json.dumps(request, ensure_ascii=False, allow_nan=False))
                timeline = user_prompt(request, "history_timeline")
                self.assertEqual(system_prompt(instructions, "history_timeline"), system_prompt(instructions))
                if request["mode"] != "history":
                    self.assertEqual(timeline, user_prompt(request))
                else:
                    self.assertTrue(timeline.endswith(encode(request["message"])))
                    self.assertIn(encode(request["context"]["initial_instruction"]), timeline)
                    for index, message in enumerate(request["context"]["message_history"], 1):
                        self.assertIn(f"{index}. {encode(message)}", timeline)
                    public = json.loads(timeline.splitlines()[1])
                    self.assertEqual(public, {key: value for key, value in request.items() if key not in ("message", "context")})
                    self.assertNotIn("active_goal", timeline)
                self.assertEqual(request, original)

    def test_refusal_has_identical_generic_instructions_and_unchanged_user_json(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            _, _, requests, instructions = inference.load_public_requests(args.requests_dir)
            extra = system_prompt(instructions, "refusal_check").replace(instructions, "", 1)
            for request in requests.values():
                self.assertEqual(user_prompt(request, "refusal_check"), user_prompt(request))
                for item in request["candidate_goals"]:
                    self.assertNotIn(item["goal_id"], extra)
            for reason in ("missing_context", "unknown_goal", "ambiguous_region"):
                self.assertIn(reason, extra)

    def test_prepare_makes_no_calls_or_predictions_and_counts_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            with patch.object(inference, "urlopen") as transport:
                self.assertEqual(ablation.prepare(args), 0)
            transport.assert_not_called()
            for variant, changed in zip(VARIANTS, (0, 24, 72)):
                folder = args.output_dir / variant
                run = base.read_json(folder / "run.json")
                self.assertEqual(run["changed_request_count"], changed)
                self.assertEqual(run["attempted"], 0)
                self.assertFalse((folder / "predictions.jsonl").exists())
                self.assertFalse((folder / "responses.jsonl").exists())
                self.assertEqual(len(ablation.lines(folder / "model_inputs.jsonl")), 72)

    def test_logged_payload_is_actual_http_body_for_every_condition(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            bodies = []
            def capture(request, **kwargs):
                bodies.append(json.loads(request.data))
                if len(bodies) == 73:
                    # Same latest input, same semantic answer, different raw
                    # response: the stability control must expose this change.
                    content = " " + encode({"status": "abstain", "reason": "missing_context"})
                    return io.BytesIO(encode({"choices": [{"message": {"content": content}}]}).encode())
                return self.response(request, **kwargs)
            comparison = self.runs(root, args, capture)
            for position, folder in enumerate((comparison.baseline_run, comparison.history_run, comparison.refusal_run)):
                logged = ablation.lines(folder / "model_inputs.jsonl")
                self.assertEqual([row["payload"] for row in logged], bodies[position * 72:(position + 1) * 72])
                self.assertTrue(all(row["payload_sha256"] == fingerprint(row["payload"]) for row in logged))
                self.assertTrue(all(len(row["payload"]["messages"]) == 2 for row in logged))
            self.assertEqual(ablation.compare(comparison), 0)
            report = base.read_json(comparison.output_dir / "comparison.json")
            self.assertEqual(report["vs_baseline"]["history_timeline"]["unchanged_input_count"], 48)
            self.assertEqual(report["vs_baseline"]["history_timeline"]["raw_response_changes_on_unchanged_inputs"], 1)
            self.assertEqual(report["vs_baseline"]["refusal_check"]["unchanged_input_count"], 0)
            self.assertEqual(report["vs_baseline"]["history_timeline"]["matched"]["planned"], 72)
            self.assertEqual(report["conditions"]["baseline"]["errors_by_mode"]["state"]["false_refusal_count"], 12)

    def test_compare_accepts_explicit_legacy_baseline_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            comparison = self.runs(root, self.prepare(root))
            run = base.read_json(comparison.baseline_run / "run.json")
            for key in ("prompt_protocol", "prompt_variant", "model_inputs_sha256"):
                del run[key]
            base.write_json(comparison.baseline_run / "run.json", run)
            (comparison.baseline_run / "model_inputs.jsonl").unlink()
            self.assertEqual(ablation.compare(comparison), 0)
            report = base.read_json(comparison.output_dir / "comparison.json")
            self.assertTrue(report["conditions"]["baseline"]["legacy_baseline_reconstruction"])

    def test_compare_rejects_generation_control_change_before_output(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            comparison = self.runs(root, self.prepare(root))
            path = comparison.refusal_run / "run.json"
            run = base.read_json(path)
            run["timeout_seconds"] = 121.0
            base.write_json(path, run)
            with self.assertRaisesRegex(ValueError, "controls must match"):
                ablation.compare(comparison)
            self.assertFalse(comparison.output_dir.exists())

    def test_compare_rejects_tampered_payload_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            comparison = self.runs(root, self.prepare(root))
            folder = comparison.history_run
            rows = ablation.lines(folder / "model_inputs.jsonl")
            rows[0]["payload"]["messages"][1]["content"] += " hidden answer"
            rows[0]["payload_sha256"] = fingerprint(rows[0]["payload"])
            base.write_jsonl(folder / "model_inputs.jsonl", rows)
            run = base.read_json(folder / "run.json")
            run["model_inputs_sha256"] = inference.sha256(folder / "model_inputs.jsonl")
            base.write_json(folder / "run.json", run)
            with self.assertRaisesRegex(ValueError, "input bodies differ"):
                ablation.compare(comparison)
            self.assertFalse(comparison.output_dir.exists())

    def test_preview_cannot_be_used_as_a_completed_run(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            ablation.prepare(args)
            manifest, index, requests, instructions = inference.load_public_requests(args.requests_dir)
            with self.assertRaisesRegex(ValueError, "preview"):
                ablation.audit_run(args.output_dir / "baseline", "baseline", manifest, index, requests, instructions)

    def test_new_baseline_cannot_drop_its_input_log_and_claim_legacy(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            comparison = self.runs(root, args)
            (comparison.baseline_run / "model_inputs.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "input bodies differ"):
                ablation.compare(comparison)
            self.assertFalse(comparison.output_dir.exists())

    def test_invalid_answers_and_transport_failures_keep_full_denominator(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            calls = 0
            def damaged(request, **kwargs):
                nonlocal calls
                calls += 1
                if calls % 72 == 1:
                    raise URLError("offline test")
                return io.BytesIO(encode({"choices": [{"message": {"content": "not JSON"}}]}).encode())
            comparison = self.runs(root, args, damaged)
            self.assertEqual(ablation.compare(comparison), 1)
            report = base.read_json(comparison.output_dir / "comparison.json")
            for variant in VARIANTS:
                self.assertEqual(report["conditions"][variant]["call_failures"], 1)
                self.assertEqual(report["conditions"][variant]["score_return_code"], 1)
                scored = ablation.lines(comparison.output_dir / (variant + "_score") / "cases.jsonl")
                self.assertEqual(len(scored), 72)
            self.assertEqual(report["vs_baseline"]["history_timeline"]["matched"]["both_wrong"], 72)

    def test_goal_confusion_and_refusal_execution_diagnostics(self):
        # Use the unchanged paired scorer to establish eligibility and fields;
        # diagnosis must distinguish wrong memory from failed JSON parsing.
        from mivln.task_update_eval import evaluate_record
        from mivln.task_update_pairs import annotate_decision, build_plan
        config = base.read_json(pairs.DEFAULT_CONFIG)
        dataset, records = build_plan(config, ["history"])
        record = next(row for row in records if row["pair_metadata"]["requires_context"])
        prediction = deepcopy(record["case"]["reference"]["prediction"])
        key = "old_goal" if "old_goal" in prediction["task_update"] else "resume_goal"
        prediction["task_update"][key] = record["case"]["reference"]["initial_goal"]
        confused = annotate_decision(record, evaluate_record(dataset, record, prediction))
        invalid = annotate_decision(record, evaluate_record(dataset, record, "not JSON"))
        ambiguous = next(row for row in records if row["case"]["group_id"] == "ambiguous_region")
        region = next(item["region_id"] for item in dataset["candidate_regions"]
                      if item["region_id"] != ambiguous["before"].active_constraints[0]["region_id"])
        guessed = annotate_decision(ambiguous, evaluate_record(dataset, ambiguous, {"status": "ok", "task_update": {
            "operation": "add_avoid_region", "region_id": region}}))
        metrics = ablation.error_metrics([confused, invalid, guessed], {row["request_id"]: row for row in records})
        self.assertEqual(metrics["reference_goal_wrong_count"], 1)
        self.assertEqual(metrics["initial_goal_confusion_count"], 1)
        self.assertEqual(metrics["invalid_or_missing_count"], 1)
        self.assertEqual(metrics["refusal_execution_count"], 1)
        self.assertEqual(metrics["refusal_applied_count"], 1)


if __name__ == "__main__":
    unittest.main()
