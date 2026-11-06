"""Sequential state ownership, public feedback and exact replay without a GPU."""

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

from mivln.task_state import TaskState
from mivln.task_update import public_state, strict_json
from mivln.task_update_prompts import encode, fingerprint
from mivln.task_update_rollout import (FEEDBACK_PREFIX, iter_rollout, predict_rules,
                                      render_user, step, validate_dialogue)
from mivln.task_update_rollout_eval import (public_dialogue, reference_steps,
                                           score_records, summarize, validate_dataset)
from scripts import predict_task_updates as client
from scripts import task_update as base
from scripts import task_update_rollout as cli


def dataset(small=True):
    value = base.read_json(cli.DEFAULT_DATASET)
    if small:
        # Both trajectory templates and both languages; unrelated labels never
        # enter inference. Each dialogue has six updates and one completion.
        value["dialogues"] = [value["dialogues"][i] for i in (0, 6, 12, 18)]
    return validate_dataset(value)


class SequentialCoreTests(unittest.TestCase):
    def setUp(self):
        self.dataset = dataset()
        self.public = [public_dialogue(self.dataset, case) for case in self.dataset["dialogues"]]
        self.registry = self.dataset["region_registry"]

    def rules(self, request):
        return encode(predict_rules(request, self.registry)), {}, None

    def test_design_balanced_explicit_labels_and_opaque_public_boundary(self):
        full = dataset(False)
        self.assertEqual(len(full["dialogues"]), 24)
        self.assertEqual(sum(len(reference_steps(full, case)) for case in full["dialogues"]), 168)
        for case in full["dialogues"]:
            public = public_dialogue(full, case)
            self.assertRegex(public["dialogue_id"], r"^d[0-9a-f]{24}$")
            self.assertEqual(sum(e["type"] == "message" for e in public["events"]), 6)
            self.assertEqual(sum(e["type"] == "complete_subgoal" for e in public["events"]), 1)
            for private in ("reference_initial_goal", "prediction", "group_id", "case_id", "bounds_xz", "floor_y"):
                self.assertNotIn(private, encode(public))
            self.assertTrue(all(e.get("prediction", {"status": "ok"})["status"] == "ok" for e in case["events"]))

    def test_rules_complete_both_modes_and_restore_without_navigation(self):
        trace = list(iter_rollout(self.public, ["history", "state"], self.registry, self.rules))
        summary = summarize(score_records(self.dataset, trace, ["history", "state"]))
        self.assertEqual(len(trace), 56)
        self.assertEqual(summary["paired_dialogues"]["all_steps_correct"]["both_correct"], 4)
        for mode in ("history", "state"):
            stats = summary["by_mode"][mode]
            self.assertEqual(stats["step_correct_count"], 24)
            self.assertEqual(stats["completion_correct_count"], 4)
            self.assertEqual(stats["completion_restoration_correct_count"], 4)
            self.assertEqual(stats["all_steps_correct_count"], 4)
            self.assertEqual(stats["carried_error_count"], 0)
            self.assertEqual(len(stats["by_turn"]), 6)

    def test_wrong_but_applicable_prediction_is_next_state_not_gold(self):
        seen = []
        def wrong(request):
            seen.append(deepcopy(request))
            if request["request_id"].endswith("e01"):
                return encode({"status": "ok", "task_update": {"operation": "replace_goal", "old_goal": "island", "new_goal": "table"}}), {}, None
            return self.rules(request)
        trace = list(iter_rollout(self.public[:1], ["state"], self.registry, wrong))
        self.assertEqual(trace[0]["outcome"], "applied")
        self.assertEqual(seen[1]["context"]["previous_state"]["active_goal"], "table")
        self.assertEqual(seen[2]["context"]["previous_state"]["active_goal"], "table")
        self.assertEqual(trace[2]["prediction"]["task_update"]["resume_goal"], "table")
        self.assertEqual(trace[3]["after_state"]["active_goal"], "table")
        one = deepcopy(self.dataset)
        one["dialogues"] = one["dialogues"][:1]
        rows = score_records(one, trace, ["state"])
        self.assertEqual(rows[0]["error_origin"], "new_error")
        self.assertEqual(rows[1]["error_origin"], "continued_error")
        self.assertFalse(rows[-1]["state_exact"])
        self.assertTrue(rows[-1]["field_correct"]["active_goal"])

    def test_invalid_or_abstained_update_holds_actual_state(self):
        for raw in ("not JSON", encode({"status": "abstain", "reason": "invalid_context"})):
            with self.subTest(raw=raw):
                def failed(request):
                    return (raw, {}, None) if request["request_id"].endswith("e01") else self.rules(request)
                trace = list(iter_rollout(self.public[:1], ["state"], self.registry, failed))
                self.assertEqual(trace[0]["before_state"], trace[0]["after_state"])
                self.assertEqual(trace[1]["request"]["context"]["previous_state"]["active_goal"], "island")
                self.assertEqual(len(trace), 7)

    def test_history_contains_actual_feedback_and_failure_not_hidden_state(self):
        def failed(request):
            if request["request_id"].endswith("e01"):
                return "invalid", {}, None
            return self.rules(request)
        trace = list(iter_rollout(self.public[:1], ["history"], self.registry, failed))
        request = trace[1]["request"]
        history = request["context"]["message_history"]
        self.assertEqual(len(history), 2)
        feedback = strict_json(history[1][len(FEEDBACK_PREFIX):])
        self.assertEqual(feedback, {"type": "update_feedback", "prediction": None, "outcome": "prediction_invalid"})
        self.assertEqual(predict_rules(request, self.registry)["task_update"]["operation"], "add_avoid_region")
        self.assertNotIn("previous_state", encode(request))
        self.assertNotIn("bounds_xz", encode(request))
        text = render_user(request, "timeline")
        self.assertIn("是否生效以随后执行反馈为准", text)
        self.assertTrue(text.endswith(encode(request["message"])))

    def test_completion_requires_matching_subgoal_and_never_repairs_divergence(self):
        state = TaskState("island")
        after, result = step(state, {"type": "complete_subgoal", "goal_id": "bathtub"}, None, None, self.registry)
        self.assertEqual(public_state(state), public_state(after))
        self.assertEqual(result["outcome"], "completion_failed")
        state.apply_update({"operation": "insert_subgoal", "subgoal": "bathtub", "resume_goal": "island"})
        after, result = step(state, {"type": "complete_subgoal", "goal_id": "bathtub"}, None, None, self.registry)
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(after.active_goal, "island")
        self.assertEqual(after.completed_subgoals, ["bathtub"])
        self.assertEqual(state.active_goal, "bathtub")

    def test_branch_reset_only_at_dialogue_boundary_and_no_completion_calls(self):
        seen = []
        def record(request):
            seen.append(deepcopy(request))
            return self.rules(request)
        list(iter_rollout(self.public, ["history", "state"], self.registry, record))
        self.assertEqual(len(seen), 48)
        for request in seen:
            if request["request_id"].endswith("e01"):
                if request["mode"] == "history":
                    self.assertEqual(request["context"]["message_history"], [])
                else:
                    self.assertEqual(request["context"]["previous_state"]["cancelled_goals"], [])

    def test_reference_mutation_cannot_become_a_public_request(self):
        source = deepcopy(self.dataset)
        before = public_dialogue(source, source["dialogues"][0])
        source["dialogues"][0]["events"][0]["prediction"]["task_update"]["new_goal"] = "table"
        self.assertEqual(public_dialogue(source, source["dialogues"][0]), before)
        forbidden = deepcopy(before)
        forbidden["events"][0]["prediction"] = {"status": "ok"}
        with self.assertRaises(ValueError):
            validate_dialogue(forbidden)

    def test_reference_validation_rejects_bad_completion_duplicate_and_refusal(self):
        mutations = []
        value = deepcopy(self.dataset)
        value["dialogues"][0]["events"][3]["goal_id"] = "bed"
        mutations.append(value)
        value = deepcopy(self.dataset)
        value["dialogues"].append(deepcopy(value["dialogues"][0]))
        mutations.append(value)
        value = deepcopy(self.dataset)
        value["dialogues"][0]["events"][0]["prediction"] = {"status": "abstain", "reason": "missing_context"}
        mutations.append(value)
        for value in mutations:
            with self.assertRaises(ValueError):
                validate_dataset(value)


class SequentialEntryTests(unittest.TestCase):
    def prepare(self, root):
        source = root / "source.json"
        base.write_json(source, dataset())
        public = root / "public"
        self.assertEqual(cli.export(SimpleNamespace(dataset=source, mode="all", output_dir=public)), 0)
        return SimpleNamespace(requests_dir=public, output_dir=root / "run", model="mock-model",
            base_url="http://localhost:8000/v1", temperature=0.0, max_tokens=512, timeout=120.0,
            thinking_mode="off", history_format="timeline", api_key_env="TASK_UPDATE_API_KEY")

    def scoring(self, root, args):
        return SimpleNamespace(requests_dir=args.requests_dir, run_dir=args.output_dir, output_dir=root / "score")

    def test_rule_cli_roundtrip_and_reject_output_reuse(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            with patch.object(client, "urlopen") as http:
                self.assertEqual(cli.generate(args, rule=True), 0)
            http.assert_not_called()
            self.assertEqual(cli.score(self.scoring(root, args)), 0)
            summary = base.read_json(root / "score/summary.json")
            self.assertEqual(summary["statistics"]["planned_updates"], 48)
            self.assertEqual(summary["statistics"]["step_correct_count"], 48)
            self.assertTrue(summary["full_causal_replay_verified"])
            with self.assertRaises(FileExistsError):
                cli.generate(args, rule=True)

    def test_inference_never_opens_source_labels(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            (root / "source.json").unlink()
            self.assertEqual(cli.generate(args, rule=True), 0)

    def test_http_bodies_match_recorded_feedback_aware_inputs(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            bodies = []
            def response(request, **kwargs):
                bodies.append(json.loads(request.data))
                return io.BytesIO(encode({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode())
            with patch.object(client, "urlopen", side_effect=response):
                self.assertEqual(cli.generate(args), 0)
            inputs = cli.lines(args.output_dir / "model_inputs.jsonl")
            self.assertEqual([row["payload"] for row in inputs], bodies)
            self.assertEqual(len(bodies), 48)
            self.assertTrue(all(len(body["messages"]) == 2 for body in bodies))
            for body in bodies:
                for private in ("reference_initial_goal", "expected_after_state", "bounds_xz", "floor_y"):
                    self.assertNotIn(private, encode(body))
            self.assertTrue(all(row["payload_sha256"] == fingerprint(row["payload"]) for row in inputs))
            self.assertEqual(cli.score(self.scoring(root, args)), 1)
            self.assertEqual(base.read_json(root / "score/summary.json")["statistics"]["schema_invalid_count"], 48)

    def test_interruption_retains_all_future_slots_and_error_states(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            calls = 0
            def interrupted(*unused, **kwargs):
                nonlocal calls
                calls += 1
                raise KeyboardInterrupt()
            with patch.object(client, "urlopen", side_effect=interrupted):
                self.assertEqual(cli.generate(args), 1)
            self.assertEqual(calls, 1)
            run = base.read_json(args.output_dir / "run.json")
            self.assertEqual((run["attempted"], run["received"], run["failed"]), (1, 0, 48))
            self.assertTrue(all(row["prediction"] is None for row in cli.lines(args.output_dir / "predictions.jsonl")))
            self.assertEqual(len(cli.lines(args.output_dir / "trace.jsonl")), 56)
            self.assertEqual(cli.score(self.scoring(root, args)), 1)
            self.assertEqual(base.read_json(root / "score/summary.json")["statistics"]["planned_updates"], 48)

    def test_transport_failure_continues_next_call_without_state_reset(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            with patch.object(client, "urlopen", side_effect=URLError("offline")) as http:
                self.assertEqual(cli.generate(args), 1)
            self.assertEqual(http.call_count, 48)
            self.assertEqual(cli.score(self.scoring(root, args)), 1)

    def test_score_rejects_state_reset_even_if_trace_file_hash_updated(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            cli.generate(args, rule=True)
            trace = cli.lines(args.output_dir / "trace.jsonl")
            trace[1]["before_state"]["active_goal"] = "table"
            base.write_jsonl(args.output_dir / "trace.jsonl", trace)
            run = base.read_json(args.output_dir / "run.json")
            run["files_sha256"]["trace.jsonl"] = client.sha256(args.output_dir / "trace.jsonl")
            base.write_json(args.output_dir / "run.json", run)
            with self.assertRaisesRegex(ValueError, "transitions differ"):
                cli.score(self.scoring(root, args))
            self.assertFalse((root / "score").exists())

    def test_score_rejects_dynamic_prompt_tamper_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            cli.generate(args, rule=True)
            inputs = cli.lines(args.output_dir / "model_inputs.jsonl")
            inputs[1]["payload"]["messages"][1]["content"] += " hidden correct state"
            inputs[1]["payload_sha256"] = fingerprint(inputs[1]["payload"])
            base.write_jsonl(args.output_dir / "model_inputs.jsonl", inputs)
            run = base.read_json(args.output_dir / "run.json")
            run["files_sha256"]["model_inputs.jsonl"] = client.sha256(args.output_dir / "model_inputs.jsonl")
            base.write_json(args.output_dir / "run.json", run)
            with self.assertRaisesRegex(ValueError, "Dynamic request/body changed"):
                cli.score(self.scoring(root, args))
            self.assertFalse((root / "score").exists())

    def test_score_rejects_modified_reference_source_before_output(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            cli.generate(args, rule=True)
            with (root / "source.json").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Reference dataset changed"):
                cli.score(self.scoring(root, args))
            self.assertFalse((root / "score").exists())


if __name__ == "__main__":
    unittest.main()
