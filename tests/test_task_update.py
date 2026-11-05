"""Language-update boundaries, state retention and honest offline scoring."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from mivln.rule_task_updater import initial_goal, mentions, predict_rule
from mivln.task_update import (apply_prediction, normalize_prediction,
                              public_state, strict_json, validate_request)
from mivln.task_update_eval import (evaluate_record, prepare_records,
                                   summarize_records, validate_dataset)
from scripts import task_update as cli


def fixture_dataset():
    """Manual reference labels come from the checked-in development fixtures."""
    return validate_dataset(cli.read_json(cli.DEFAULT_DATASET))


class TaskUpdateContractTests(unittest.TestCase):
    def setUp(self):
        self.dataset = fixture_dataset()
        self.record = prepare_records(self.dataset, ["state"])[0]
        self.request = self.record["request"]

    def test_modes_have_strict_distinct_context_and_no_privileged_coordinates(self):
        records = prepare_records(self.dataset, ["latest", "history", "state"])
        first = records[:3]
        self.assertEqual(first[0]["request"]["context"], {})
        self.assertEqual(set(first[1]["request"]["context"]), {"initial_instruction", "message_history"})
        self.assertEqual(set(first[2]["request"]["context"]), {"previous_state"})
        constrained = next(record for record in records if record["request_id"] == "insert_zh_keep_constraint__state")
        self.assertEqual(constrained["request"]["context"]["previous_state"]["active_constraints"], ["region_red"])
        for record in records:
            encoded = str(record["request"])
            for private in ("bounds_xz", "floor_y", "region_registry", "reference", "context_updates", "expected_after"):
                self.assertNotIn(private, encoded)

    def test_public_reference_answers_cannot_change_generated_policy_context(self):
        original = prepare_records(self.dataset, ["state"])
        changed = deepcopy(self.dataset)
        case = next(item for item in changed["cases"] if item["case_id"] == "avoid_zh_keep_goal")
        case["reference"]["initial_goal"] = "bed"
        altered = prepare_records(changed, ["state"])
        self.assertEqual([item["request"] for item in original], [item["request"] for item in altered])
        record = next(item for item in altered if item["case"]["case_id"] == "avoid_zh_keep_goal")
        row = evaluate_record(changed, record, predict_rule(record["request"]))
        self.assertFalse(row["context_correct"])
        self.assertFalse(row["correct"])
        self.assertEqual(row["before_state"]["active_goal"], "island")

    def test_request_extras_and_hidden_geometry_are_rejected(self):
        variants = []
        extra = deepcopy(self.request)
        extra["reference"] = {"new_goal": "bed"}
        variants.append(extra)
        geometry = deepcopy(self.request)
        geometry["candidate_regions"][0]["bounds_xz"] = [[0, 0], [1, 1]]
        variants.append(geometry)
        geometry_state = deepcopy(self.request)
        geometry_state["context"]["previous_state"]["active_constraints"] = [{"region_id": "region_red", "floor_y": 0}]
        variants.append(geometry_state)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_request(value)

    def test_strict_json_rejects_duplicate_keys_nonfinite_values_and_fences(self):
        for text in ('{"status":"ok","status":"abstain"}', '{"value":NaN}', '{"value":Infinity}',
                     '{"value":1e999}', '{"value":-1e999}', '```json\n{}\n```'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_json(text)

    def test_prediction_schema_rejects_unknown_ids_geometry_and_extra_state(self):
        for value in (
            {"status": "ok", "task_update": {"operation": "replace_goal", "old_goal": "island", "new_goal": "unknown"}},
            {"status": "ok", "task_update": {"operation": "add_avoid_region", "region_id": "region_red", "bounds_xz": [[0, 0], [1, 1]]}},
            {"status": "ok", "task_update": {"operation": "replace_goal", "old_goal": "island", "new_goal": "bed"}, "state": {}},
            {"status": "abstain", "reason": "invented_reason"},
            {"status": "ok", "task_update": {"operation": []}},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_prediction(value, self.request)

    def test_replacement_preserves_constraints_and_subgoal_insertion_preserves_resume(self):
        records = prepare_records(self.dataset, ["state"])
        for identity in ("replace_zh_keep_constraint", "avoid_zh_keep_old_constraint", "insert_zh_keep_constraint"):
            record = next(item for item in records if item["case"]["case_id"] == identity)
            before = record["before"].to_dict()
            prediction = predict_rule(record["request"])
            updated = apply_prediction(record["before"], prediction, record["request"], self.dataset["region_registry"])
            self.assertEqual(record["before"].to_dict(), before)
            self.assertEqual(public_state(updated), public_state(record["expected_after"]))
            self.assertIn("region_red", public_state(updated)["active_constraints"])
            if identity.startswith("insert"):
                self.assertEqual(updated.active_goal, "table")
                self.assertEqual(updated.suspended_goals, ["island"])
                self.assertEqual(updated.cancelled_goals, [])

    def test_guard_failure_and_abstention_never_modify_input_state(self):
        state = self.record["before"]
        before = state.to_dict()
        bad = {"status": "ok", "task_update": {"operation": "replace_goal", "old_goal": "bed", "new_goal": "table"}}
        with self.assertRaises(ValueError):
            apply_prediction(state, bad, self.request, self.dataset["region_registry"])
        self.assertEqual(state.to_dict(), before)
        abstained = apply_prediction(state, {"status": "abstain", "reason": "missing_context"},
                                     self.request, self.dataset["region_registry"])
        self.assertEqual(abstained.to_dict(), before)
        self.assertIsNot(abstained, state)

    def test_region_registry_lookup_is_private_and_identity_checked(self):
        value = {"status": "ok", "task_update": {"operation": "add_avoid_region", "region_id": "region_red"}}
        normalized = normalize_prediction(value, self.request)
        self.assertEqual(normalized, value)
        registry = deepcopy(self.dataset["region_registry"])
        registry["region_red"]["region_id"] = "other"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            apply_prediction(self.record["before"], value, self.request, registry)


class TaskUpdateRuleTests(unittest.TestCase):
    def setUp(self):
        self.dataset = fixture_dataset()

    def test_long_alias_resolves_specific_sofa_but_generic_alias_stays_ambiguous(self):
        goals = self.dataset["candidate_goals"]
        self.assertEqual(mentions("深色的沙发", goals, "goal_id"), {"sofa_dark"})
        self.assertEqual(mentions("沙发", goals, "goal_id"), {"sofa_dark", "sofa_light"})
        self.assertEqual(mentions("bedroom", goals, "goal_id"), set())
        self.assertEqual(initial_goal("Go to the metal bed.", goals), "bed")

    def test_bilingual_fixtures_all_three_operations_and_safe_abstentions(self):
        rows = []
        for record in prepare_records(self.dataset, ["history", "state"]):
            prediction = predict_rule(record["request"])
            row = evaluate_record(self.dataset, record, prediction)
            self.assertTrue(row["correct"], row)
            rows.append(row)
        self.assertEqual({row["expected_kind"] for row in rows},
                         {"replace_goal", "add_avoid_region", "insert_subgoal", "abstain"})

    def test_latest_can_handle_explicit_references_but_cannot_invent_original_task(self):
        records = {record["case"]["case_id"]: record for record in prepare_records(self.dataset, ["latest"])}
        explicit = records["insert_zh_explicit_resume"]
        self.assertTrue(evaluate_record(self.dataset, explicit, predict_rule(explicit["request"]))["correct"])
        implicit = records["insert_zh_implicit_resume"]
        self.assertEqual(predict_rule(implicit["request"]), {"status": "abstain", "reason": "missing_context"})
        avoiding = records["avoid_zh_keep_goal"]
        self.assertTrue(evaluate_record(self.dataset, avoiding, predict_rule(avoiding["request"]))["correct"])

    def test_history_reconstructs_changed_goal_and_state_uses_policy_previous_goal(self):
        for mode in ("history", "state"):
            record = next(item for item in prepare_records(self.dataset, [mode]) if item["case"]["case_id"] == "replace_zh_prior_change")
            self.assertEqual(predict_rule(record["request"])["task_update"],
                             {"operation": "replace_goal", "old_goal": "bed", "new_goal": "table"})

    def test_compound_or_unhandled_messages_abstain_instead_of_guessing(self):
        record = prepare_records(self.dataset, ["state"])[0]
        for message in ("改去深色沙发，同时避开红色区域。", "先去深色沙发。", "暂停任务。"):
            request = deepcopy(record["request"])
            request["message"] = message
            self.assertEqual(predict_rule(request), {"status": "abstain", "reason": "unsupported_message"})

    def test_initial_context_failure_remains_in_every_mode_denominator(self):
        data = deepcopy(self.dataset)
        data["cases"] = [data["cases"][0]]
        data["cases"][0]["initial_instruction"] = "去一个未说明的位置。"
        validate_dataset(data)
        records = prepare_records(data, ["latest", "history", "state"])
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["request"] is None for record in records))
        rows = [evaluate_record(data, record, None) for record in records]
        self.assertEqual(summarize_records(rows)["statistics"]["invalid_or_missing_count"], 3)

    def test_wrong_labels_cannot_be_hidden_by_retaining_unchanged_fields(self):
        record = next(item for item in prepare_records(self.dataset, ["state"]) if item["case"]["case_id"] == "insert_zh_keep_constraint")
        row = evaluate_record(self.dataset, record, {"status": "abstain", "reason": "missing_context"})
        self.assertTrue(row["requirements_preserved"])
        self.assertFalse(row["prediction_exact"])
        self.assertFalse(row["state_exact"])
        self.assertFalse(row["correct"])


class TaskUpdateEntryTests(unittest.TestCase):
    def fixture(self, root):
        data = fixture_dataset()
        keep = {"replace_zh_implicit", "avoid_zh_keep_old_constraint", "insert_zh_keep_constraint", "ambiguous_sofa"}
        data["cases"] = [case for case in data["cases"] if case["case_id"] in keep]
        path = root/"dataset.json"
        cli.write_json(path, data)
        return path

    def run_rules(self, root):
        source = self.fixture(root)
        args = SimpleNamespace(dataset=source, mode="all", output_dir=root/"rules")
        self.assertEqual(cli.run(args), 0)
        return args

    def test_run_export_and_external_import_roundtrip_uses_exact_request_hashes(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.run_rules(root)
            exported = SimpleNamespace(dataset=args.dataset, mode="all", output_dir=root/"public")
            self.assertEqual(cli.export(exported), 0)
            self.assertFalse((exported.output_dir/"cases.jsonl").exists())
            requests = (exported.output_dir/"requests.jsonl").read_text()
            self.assertNotIn('"reference"', requests)
            self.assertNotIn("bounds_xz", requests)
            scoring = SimpleNamespace(requests_dir=exported.output_dir, predictions=args.output_dir/"predictions.jsonl",
                                      updater_name="imported_rules", output_dir=root/"scored")
            self.assertEqual(cli.score(scoring), 0)
            original, scored = (cli.read_json(path/"summary.json") for path in (args.output_dir, scoring.output_dir))
            self.assertEqual(original["statistics"], scored["statistics"])
            self.assertEqual(original["by_mode"], scored["by_mode"])
            with self.assertRaises(FileExistsError):
                cli.run(args)
            with self.assertRaises(FileExistsError):
                cli.score(scoring)

    def test_missing_invalid_and_wrong_hash_predictions_keep_original_case_slots(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.run_rules(root)
            values = [strict_json(line) for line in (args.output_dir/"predictions.jsonl").read_text().splitlines()]
            values = values[1:]  # First planned response is missing.
            values[0]["prediction"] = '{"status":"ok","status":"abstain"}'
            values[1]["request_sha256"] = "0"*64
            cli.write_jsonl(root/"bad.jsonl", values)
            scoring = SimpleNamespace(requests_dir=args.output_dir, predictions=root/"bad.jsonl",
                                      updater_name="bad_external", output_dir=root/"scored")
            self.assertEqual(cli.score(scoring), 1)
            report = cli.read_json(scoring.output_dir/"summary.json")
            self.assertEqual(report["expected_records"], 12)
            self.assertEqual(report["statistics"]["invalid_or_missing_count"], 3)
            self.assertEqual(len((scoring.output_dir/"cases.jsonl").read_text().splitlines()), 12)

    def test_duplicate_or_unknown_prediction_ids_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.run_rules(root)
            values = [strict_json(line) for line in (args.output_dir/"predictions.jsonl").read_text().splitlines()]
            for suffix, bad in (("duplicate", [values[0], values[0]]),
                                ("unknown", [{**values[0], "request_id": "unknown"}])):
                cli.write_jsonl(root/f"{suffix}.jsonl", bad)
                scoring = SimpleNamespace(requests_dir=args.output_dir, predictions=root/f"{suffix}.jsonl",
                                          updater_name="bad", output_dir=root/suffix)
                with self.assertRaises(ValueError):
                    cli.score(scoring)
                self.assertFalse(scoring.output_dir.exists())

    def test_changed_dataset_or_public_requests_preflight_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.run_rules(root)
            path = args.output_dir/"requests.jsonl"
            path.write_text(path.read_text()+"\n")
            scoring = SimpleNamespace(requests_dir=args.output_dir, predictions=args.output_dir/"predictions.jsonl",
                                      updater_name="bad", output_dir=root/"scored")
            with self.assertRaisesRegex(ValueError, "Frozen public input changed"):
                cli.score(scoring)
            self.assertFalse(scoring.output_dir.exists())

    def test_reference_validation_rejects_bad_updates_and_duplicate_cases(self):
        data = fixture_dataset()
        bad = deepcopy(data)
        bad["cases"].append(deepcopy(bad["cases"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate case"):
            validate_dataset(bad)
        bad = deepcopy(data)
        bad["cases"][0]["reference"]["prediction"]["task_update"]["old_goal"] = "bed"
        with self.assertRaises(ValueError):
            validate_dataset(bad)
