"""Inference boundary and raw-output accounting without a server or GPU."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from mivln.rule_task_updater import predict_rule
from scripts import predict_task_updates as inference
from scripts import task_update as evaluation


class PredictionGenerationTests(unittest.TestCase):
    def prepare(self, root):
        dataset = evaluation.read_json(evaluation.DEFAULT_DATASET)
        identities = {"replace_zh_implicit", "avoid_zh_keep_old_constraint",
                      "insert_zh_keep_constraint", "ambiguous_sofa"}
        dataset["cases"] = [case for case in dataset["cases"] if case["case_id"] in identities]
        source = root / "dataset.json"
        evaluation.write_json(source, dataset)
        public = root / "public"
        self.assertEqual(evaluation.export(SimpleNamespace(dataset=source, mode="all", output_dir=public)), 0)
        return SimpleNamespace(requests_dir=public, base_url="http://localhost:8000/v1", model="mock-model",
            api_key_env="TASK_UPDATE_API_KEY", timeout=10.0, temperature=0.0, max_tokens=512,
            output_dir=root / "inference")

    def response(self, request, **kwargs):
        """A local transport stub returns rules for testing, never claimed as LLM output."""
        payload = json.loads(request.data)
        public = json.loads(payload["messages"][1]["content"])
        content = json.dumps(predict_rule(public), ensure_ascii=False)
        return io.BytesIO(json.dumps({"model": "mock-model", "choices": [
            {"message": {"content": content}, "finish_reason": "stop"}]}).encode())

    def score(self, root, args):
        return evaluation.score(SimpleNamespace(requests_dir=args.requests_dir,
            predictions=args.output_dir / "predictions.jsonl", updater_name="transport_stub",
            output_dir=root / "scored"))

    def test_public_only_stateless_calls_and_raw_predictions_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            with patch.dict("os.environ", {"TASK_UPDATE_API_KEY": "test-key"}), \
                    patch.object(inference, "urlopen", side_effect=self.response) as transport:
                self.assertEqual(inference.generate(args), 0)
            self.assertEqual(transport.call_count, 12)
            for call in transport.call_args_list:
                request = call.args[0]
                self.assertEqual(request.full_url, "http://localhost:8000/v1/chat/completions")
                self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
                payload = json.loads(request.data)
                self.assertEqual(len(payload["messages"]), 2)
                self.assertNotIn("chat_template_kwargs", payload)
                public = json.loads(payload["messages"][1]["content"])
                if public["mode"] == "latest":
                    self.assertEqual(public["context"], {})
                self.assertNotIn('"reference"', str(payload))
                self.assertNotIn("bounds_xz", str(payload))
            rows = [json.loads(line) for line in (args.output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(isinstance(row["prediction"], str) for row in rows))
            self.assertEqual(self.score(root, args), 0)
            report = evaluation.read_json(root / "scored/summary.json")
            self.assertEqual(report["by_mode"]["state"]["correct_count"], 4)
            run = evaluation.read_json(args.output_dir / "run.json")
            self.assertEqual(run["received"], 12)
            self.assertNotIn("test-key", (args.output_dir / "run.json").read_text())

    def test_thinking_mode_is_an_explicit_boolean_and_recorded(self):
        for mode, expected in (("off", False), ("on", True)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
                args = self.prepare(Path(temporary))
                args.thinking_mode = mode
                with patch.object(inference, "urlopen", side_effect=self.response) as transport:
                    self.assertEqual(inference.generate(args), 0)
                for call in transport.call_args_list:
                    payload = json.loads(call.args[0].data)
                    self.assertIs(payload["chat_template_kwargs"]["enable_thinking"], expected)
                run = evaluation.read_json(args.output_dir / "run.json")
                self.assertEqual(run["thinking_mode"], mode)
                self.assertIs(run["chat_template_kwargs"]["enable_thinking"], expected)

    def test_generation_does_not_open_source_dataset(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            manifest = evaluation.read_json(args.requests_dir / "manifest.json")
            manifest["dataset_path"] = str(Path(temporary) / "unavailable_evaluator_data.json")
            evaluation.write_json(args.requests_dir / "manifest.json", manifest)
            with patch.object(inference, "urlopen", side_effect=self.response):
                self.assertEqual(inference.generate(args), 0)

    def test_failed_http_and_invalid_raw_json_keep_slots_and_are_scored(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            count = 0

            def respond(request, **kwargs):
                nonlocal count
                count += 1
                if count == 1:
                    raise HTTPError(request.full_url, 503, "failure", {}, None)
                if count == 2:
                    return io.BytesIO(json.dumps({"choices": [{"message": {
                        "content": "```json\n{}\n```"}, "finish_reason": "length"}]}).encode())
                return self.response(request, **kwargs)

            with patch.object(inference, "urlopen", side_effect=respond):
                self.assertEqual(inference.generate(args), 1)
            rows = [json.loads(line) for line in (args.output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertIsNone(rows[0]["prediction"])
            self.assertEqual(rows[1]["prediction"], "```json\n{}\n```")
            self.assertEqual(self.score(root, args), 1)
            report = evaluation.read_json(root / "scored/summary.json")
            self.assertEqual(report["statistics"]["invalid_or_missing_count"], 2)
            self.assertEqual(report["statistics"]["planned"], 12)

    def test_interruption_retains_unattempted_slots(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            with patch.object(inference, "urlopen", side_effect=KeyboardInterrupt) as transport:
                self.assertEqual(inference.generate(args), 1)
            self.assertEqual(transport.call_count, 1)
            run = evaluation.read_json(args.output_dir / "run.json")
            self.assertTrue(run["interrupted"])
            self.assertEqual(run["failed"], 12)
            rows = (args.output_dir / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 12)

    def test_frozen_input_tamper_fails_before_output_or_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            requests = args.requests_dir / "requests.jsonl"
            requests.write_text(requests.read_text() + "\n")
            with patch.object(inference, "urlopen") as transport, self.assertRaises(ValueError):
                inference.generate(args)
            transport.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_output_reuse_rejected_before_calls(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            args.output_dir.mkdir()
            with patch.object(inference, "urlopen") as transport, self.assertRaises(FileExistsError):
                inference.generate(args)
            transport.assert_not_called()

    def test_invalid_parameters_and_credential_urls_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            args = self.prepare(Path(temporary))
            args.thinking_mode = "default"
            for name, value in (("timeout", float("nan")), ("temperature", -1), ("max_tokens", 0),
                                ("base_url", "http://user:secret@localhost/v1"), ("thinking_mode", "invalid")):
                previous = getattr(args, name)
                setattr(args, name, value)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    inference.generate(args)
                setattr(args, name, previous)
            self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
