"""Information boundaries and matched-pair accounting, without model inference."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln.task_update_eval import evaluate_record
from mivln.task_update_pairs import annotate_decision, build_plan, summarize_pairs, validate_config
from scripts import task_update as base
from scripts import task_update_pairs as pairs
from scripts.predict_task_updates import load_public_requests


def source():
    return base.read_json(pairs.DEFAULT_CONFIG)


def small_source():
    config = source()
    config["families"] = [config["families"][0], config["families"][9]]
    return config


def score(dataset, record, prediction):
    return annotate_decision(record, evaluate_record(dataset, record, prediction))


class PairedDesignTests(unittest.TestCase):
    def test_fixed_roster_balance_determinism_and_family_split_isolation(self):
        config = source()
        dataset, records = build_plan(config, ["latest", "history", "state"])
        self.assertEqual(len(dataset["cases"]), 144)
        self.assertEqual(len(records), 432)
        self.assertEqual(records, build_plan(config, ["latest", "history", "state"])[1])
        self.assertTrue(all(record["context_error"] is None for record in records))
        families = {}
        first = Counter()
        region_first = Counter()
        for record in records:
            case, meta, request = record["case"], record["pair_metadata"], record["request"]
            self.assertNotIn(case["group_id"], request["request_id"])
            self.assertRegex(request["request_id"], r"^r[0-9a-f]{24}__(latest|history|state)$")
            families.setdefault(meta["family_id"], set()).add(meta["split"])
            first[request["candidate_goals"][0]["goal_id"]] += 1
            region_first[request["candidate_regions"][0]["region_id"]] += 1
            for private in ("family_id", "split", "context_id", "requires_context", "reference", "bounds_xz", "floor_y"):
                self.assertNotIn(private, str(request))
        self.assertEqual(set(first.values()), {72})
        self.assertEqual(set(region_first.values()), {216})
        self.assertTrue(all(len(splits) == 1 for splits in families.values()))
        dev, dev_records = build_plan(config, ["latest", "history", "state"], "dev")
        test, test_records = build_plan(config, ["latest", "history", "state"], "test")
        self.assertEqual((len(dev_records), len(test_records)), (216, 216))
        for selected in (dev_records, test_records):
            first = Counter(record["request"]["candidate_goals"][0]["goal_id"] for record in selected)
            self.assertEqual(set(first.values()), {36})
        self.assertTrue({case["group_id"] for case in dev["cases"]}.isdisjoint({case["group_id"] for case in test["cases"]}))
        self.assertEqual({record["request_id"] for record in records},
                         {record["request_id"] for record in dev_records + test_records})

    def test_context_controls_keep_public_message_catalogs_and_order_controls_keep_context(self):
        _, records = build_plan(source(), ["latest", "history", "state"])
        contexts, orders = defaultdict(list), defaultdict(list)
        for record in records:
            meta = record["pair_metadata"]
            orders[(meta["family_id"], record["case"]["language"], meta["context_id"], record["mode"])].append(record)
            if meta["requires_context"]:
                contexts[(meta["family_id"], record["case"]["language"], meta["order"], record["mode"])].append(record)
        for group in contexts.values():
            self.assertEqual(len(group), 3)
            projected = []
            for record in group:
                public = deepcopy(record["request"])
                public.pop("request_id")
                public.pop("context")
                projected.append(public)
                self.assertNotEqual(record["case"]["reference"]["initial_goal"], record["before"].active_goal)
                self.assertEqual(len(record["before"].active_constraints), 1)
            self.assertTrue(all(value == projected[0] for value in projected))
        for group in orders.values():
            first, second = [deepcopy(record["request"]) for record in group]
            self.assertEqual(first["candidate_goals"], list(reversed(second["candidate_goals"])))
            self.assertEqual(first["candidate_regions"], list(reversed(second["candidate_regions"])))
            for public in (first, second):
                for key in ("request_id", "candidate_goals", "candidate_regions"):
                    public.pop(key)
            self.assertEqual(first, second)

    def test_config_rejects_label_inconsistency_duplicate_families_and_bad_placeholders(self):
        changes = []
        duplicate = source()
        duplicate["families"].append(deepcopy(duplicate["families"][0]))
        changes.append(duplicate)
        invalid = source()
        invalid["families"][0]["target_id"] = "bed"
        changes.append(invalid)
        invalid = source()
        invalid["families"][0]["requires_context"] = False
        changes.append(invalid)
        for placeholder in ("{reference}", "{active_goal!r}", "{active_goal:>20}"):
            invalid = source()
            invalid["families"][6]["messages"]["zh"] = placeholder
            changes.append(invalid)
        for config in changes:
            with self.assertRaises(ValueError):
                validate_config(config)


class PairedDecisionTests(unittest.TestCase):
    def setUp(self):
        self.dataset, self.records = build_plan(small_source(), ["latest", "history", "state"])

    def test_lucky_hidden_goal_match_is_not_a_supported_decision(self):
        record = next(record for record in self.records if record["mode"] == "latest" and record["pair_metadata"]["requires_context"])
        guessed = score(self.dataset, record, record["case"]["reference"]["prediction"])
        self.assertTrue(guessed["correct"])
        self.assertTrue(guessed["unsupported_guess_hit"])
        self.assertFalse(guessed["decision_correct"])
        refused = score(self.dataset, record, {"status": "abstain", "reason": "missing_context"})
        self.assertFalse(refused["correct"])
        self.assertTrue(refused["decision_correct"])
        self.assertTrue(refused["decision_state_exact"])
        wrong_reason = score(self.dataset, record, {"status": "abstain", "reason": "unknown_goal"})
        self.assertFalse(wrong_reason["decision_correct"])

    def test_legal_ambiguity_execution_and_false_refusal_are_failures(self):
        ambiguous = next(record for record in self.records if record["case"]["group_id"] == "ambiguous_region")
        region = next(item["region_id"] for item in self.dataset["candidate_regions"]
                      if item["region_id"] != ambiguous["before"].active_constraints[0]["region_id"])
        guessed = score(self.dataset, ambiguous, {"status": "ok", "task_update": {"operation": "add_avoid_region", "region_id": region}})
        self.assertTrue(guessed["application_valid"])
        self.assertTrue(guessed["ambiguity_execution"])
        self.assertFalse(guessed["decision_correct"])
        actionable = next(record for record in self.records if record["mode"] == "state" and record["pair_metadata"]["requires_context"])
        refused = score(self.dataset, actionable, {"status": "abstain", "reason": "missing_context"})
        self.assertFalse(refused["decision_correct"])

    def test_constant_reference_guesses_and_order_agreement_do_not_prove_correctness(self):
        rows = []
        for record in self.records:
            prediction = deepcopy(record["case"]["reference"]["prediction"])
            if record["pair_metadata"]["requires_context"]:
                prediction["task_update"]["resume_goal"] = "island"
            rows.append(score(self.dataset, record, prediction))
        report = summarize_pairs(self.records, rows)
        for mode in ("latest", "history", "state"):
            contexts = report["context_groups_by_mode"][mode]
            self.assertEqual(contexts["planned_groups"], 4)
            self.assertEqual(contexts["constant_reference_guess_count"], 4)
            self.assertEqual(contexts["all_decisions_correct_count"], 0)
        state_orders = report["order_groups_by_mode"]["state"]
        self.assertEqual(state_orders["prediction_agreement_count"], state_orders["planned_groups"])
        self.assertLess(state_orders["both_decisions_correct_count"], state_orders["planned_groups"])

    def test_context_failure_preserves_every_case_mode_and_pair_denominator(self):
        with patch("mivln.task_update_eval.build_context", side_effect=ValueError("context unavailable")):
            dataset, records = build_plan(small_source(), ["latest", "history", "state"])
        rows = [score(dataset, record, None) for record in records]
        report = summarize_pairs(records, rows)
        self.assertEqual(report["statistics"]["planned"], 72)
        self.assertEqual(report["statistics"]["decision_correct_count"], 0)
        self.assertTrue(all(record["request"] is None for record in records))
        self.assertEqual(report["order_groups_by_mode"]["state"]["planned_groups"], 12)

    def test_one_order_variant_error_breaks_its_matched_order_and_context_groups(self):
        rows = []
        changed = False
        for record in self.records:
            prediction = record["case"]["reference"]["prediction"]
            if record["mode"] == "latest" and record["pair_metadata"]["requires_context"]:
                prediction = {"status": "abstain", "reason": "missing_context"}
            if not changed and record["mode"] == "state" and record["pair_metadata"]["requires_context"]:
                prediction = {"status": "abstain", "reason": "missing_context"}
                changed = True
            rows.append(score(self.dataset, record, prediction))
        report = summarize_pairs(self.records, rows)
        orders = report["order_groups_by_mode"]["state"]
        self.assertEqual(orders["prediction_agreement_count"], orders["planned_groups"] - 1)
        self.assertEqual(orders["both_decisions_correct_count"], orders["planned_groups"] - 1)
        contexts = report["context_groups_by_mode"]["state"]
        self.assertEqual(contexts["all_decisions_correct_count"], contexts["planned_groups"] - 1)


class PairedEntryTests(unittest.TestCase):
    def prepare(self, root):
        config = root / "config.json"
        base.write_json(config, small_source())
        args = SimpleNamespace(dataset=config, split="all", mode="all", output_dir=root / "rules")
        self.assertEqual(pairs.run(args), 0)
        return args

    def test_rule_export_model_client_inputs_and_score_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            public = root / "public"
            self.assertEqual(pairs.export(SimpleNamespace(**{**vars(args), "output_dir": public})), 0)
            _, index, requests, _ = load_public_requests(public)
            self.assertEqual(len(index), 72)
            self.assertEqual(len(requests), 72)
            self.assertFalse((public / "cases.jsonl").exists())
            scoring = SimpleNamespace(requests_dir=public, predictions=args.output_dir / "predictions.jsonl",
                                      updater_name="imported_rules", output_dir=root / "scored")
            self.assertEqual(pairs.score(scoring), 0)
            original, imported = [base.read_json(folder / "summary.json") for folder in (args.output_dir, scoring.output_dir)]
            self.assertEqual(original["paired"], imported["paired"])
            self.assertEqual(original["statistics"], imported["statistics"])
            self.assertEqual(len((scoring.output_dir / "decision_cases.csv").read_text().splitlines()), 73)
            with self.assertRaises(FileExistsError):
                pairs.score(scoring)

    def test_missing_invalid_and_mismatched_responses_keep_pairs(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            args = self.prepare(root)
            predictions = [base.strict_json(line) for line in (args.output_dir / "predictions.jsonl").read_text().splitlines()][1:]
            predictions[0]["prediction"] = "```json\n{}\n```"
            predictions[1]["request_sha256"] = "0" * 64
            path = root / "invalid.jsonl"
            base.write_jsonl(path, predictions)
            scoring = SimpleNamespace(requests_dir=args.output_dir, predictions=path,
                                      updater_name="invalid", output_dir=root / "scored")
            self.assertEqual(pairs.score(scoring), 1)
            report = base.read_json(scoring.output_dir / "summary.json")
            self.assertEqual(report["statistics"]["planned"], 72)
            self.assertEqual(report["statistics"]["invalid_or_missing_count"], 3)
            self.assertEqual(report["paired"]["statistics"]["planned"], 72)
            self.assertEqual(report["paired"]["order_groups_by_mode"]["state"]["planned_groups"], 12)

    def test_changed_config_split_and_unknown_ids_fail_before_output(self):
        for change in ("config", "split", "response"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
                root = Path(temporary)
                args = self.prepare(root)
                if change == "config":
                    args.dataset.write_text(args.dataset.read_text() + "\n")
                elif change == "split":
                    manifest = base.read_json(args.output_dir / "manifest.json")
                    manifest["paired_protocol"]["split"] = "dev"
                    base.write_json(args.output_dir / "manifest.json", manifest)
                else:
                    path = args.output_dir / "predictions.jsonl"
                    predictions = [base.strict_json(line) for line in path.read_text().splitlines()]
                    predictions[0]["request_id"] = "unknown"
                    base.write_jsonl(path, predictions)
                scoring = SimpleNamespace(requests_dir=args.output_dir, predictions=args.output_dir / "predictions.jsonl",
                                          updater_name="invalid", output_dir=root / "scored")
                with self.assertRaises(ValueError):
                    pairs.score(scoring)
                self.assertFalse(scoring.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
