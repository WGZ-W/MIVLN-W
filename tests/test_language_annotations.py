"""Language curation workflow tests on isolated, synthetic reviewed fixtures."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln import language_annotations as model
from scripts import language_annotations as cli
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import sha256


class LanguageAnnotationTests(unittest.TestCase):
    def fixture(self, root):
        from tests.test_language_navigation import source_fixture
        source = source_fixture(root)
        output = root / "draft"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.prepare(SimpleNamespace(source_dir=source, output_dir=output)), 0)
        return output

    def decisions(self, root, bundle, name="decisions.json", subset=None, reviewer="Bob", decision="approve"):
        """Explicit synthetic decisions for tests only; never actual annotations."""
        dataset = cli.load_bundle(bundle)
        records = model.replay(dataset)
        template = model.review_template(dataset)
        for row in template["decisions"]:
            if subset is not None and row["annotation_id"] not in subset:
                continue
            row.update(decision=decision, reviewer=reviewer, note="Synthetic fixture review; not a real human/evidence claim.",
                       checks={key: True for key in model.CHECKS},
                       checked_evidence_ids=records[row["annotation_id"]]["required_evidence_ids"][:])
        path = root / name
        cli.write_json(path, template)
        return path

    def approve(self, root, source, name="approved", subset=None, reviewer="Bob"):
        path = self.decisions(root, source, name + ".json", subset, reviewer)
        output = root / name
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.review(SimpleNamespace(input_dir=source, decisions=path, output_dir=output)), 0)
        return output

    def edit_sheet(self, root, source, identity="task_001_initial", text="去那张金属床旁停下。", author="Alice", name="sheet.csv"):
        with (source / "annotation_sheet.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["annotation_id"] == identity:
                row.update(language="zh-CN", text=text, author=author, note="Synthetic wording revision.")
        path = root / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=model.SHEET_COLUMNS)
            writer.writeheader(); writer.writerows(rows)
        return path

    def test_prepare_has_fixed_roster_three_types_and_no_inherited_language_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = self.fixture(root)
            dataset = cli.load_bundle(bundle)
            report = model.summary(dataset)
            self.assertEqual(report["base_tasks"], 1)
            self.assertEqual(report["planned_annotations"], 4)
            self.assertEqual(report["planned_task_cases"], 3)
            self.assertEqual(report["review_status"]["pending"], 4)
            self.assertEqual({r["task_type"] for r in dataset["records"]}, {"initial", *model.TASK_TYPES})
            self.assertTrue(all(r["author"]["kind"] == "template" for r in dataset["records"]))
            self.assertTrue((bundle / "review_index.md").is_file())

    def test_edits_are_new_versions_and_do_not_change_labels_or_old_authority(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); original = self.fixture(root)
            before = snapshot({"original": original})
            sheet = self.edit_sheet(root, original)
            output = root / "revised"
            self.assertEqual(cli.revise(SimpleNamespace(input_dir=original, sheet=sheet, output_dir=output)), 0)
            self.assertEqual(before, snapshot({"original": original}))
            data = cli.load_bundle(output)
            self.assertEqual(data["revision_number"], 1)
            row = data["records"][0]
            self.assertEqual(row["language"], "zh-CN")
            self.assertEqual(row["origin"], "template_derived")
            self.assertEqual(row["human_contributors"], ["Alice"])
            self.assertEqual(row["semantic_label"], cli.load_bundle(original)["records"][0]["semantic_label"])

    def test_review_must_be_explicit_with_checks_and_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = self.fixture(root)
            for kind in ("blank", "check_missing", "evidence_missing", "placeholder"):
                path = self.decisions(root, source, kind + ".json")
                value = cli.read_json(path)
                if kind == "blank":
                    value = model.review_template(cli.load_bundle(source))
                elif kind == "check_missing":
                    value["decisions"][0]["checks"]["natural_language"] = False
                elif kind == "evidence_missing":
                    value["decisions"][0]["checked_evidence_ids"] = []
                else:
                    value["decisions"][0]["reviewer"] = "YOUR_REVIEWER_NAME"
                cli.write_json(path, value)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    cli.review(SimpleNamespace(input_dir=source, decisions=path, output_dir=root / ("bad_" + kind)))
                self.assertFalse((root / ("bad_" + kind)).exists())

    def test_initial_must_be_approved_before_intervention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = self.fixture(root)
            path = self.decisions(root, source, subset={"task_001_replacement"})
            with self.assertRaises(ValueError):
                cli.review(SimpleNamespace(input_dir=source, decisions=path, output_dir=root / "bad"))

    def test_independent_reviewer_normalization_prevents_self_approval(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); source = self.fixture(root)
            sheet = self.edit_sheet(root, source)
            revised = root / "revised"
            cli.revise(SimpleNamespace(input_dir=source, sheet=sheet, output_dir=revised))
            for reviewer in (" Alice ", "ＡＬＩＣＥ"):
                path = self.decisions(root, revised, reviewer=reviewer, subset={"task_001_initial"})
                with self.assertRaises(ValueError):
                    cli.review(SimpleNamespace(input_dir=revised, decisions=path, output_dir=root / "self"))
            approved = self.approve(root, revised, subset={"task_001_initial"})
            self.assertEqual(model.summary(cli.load_bundle(approved))["review_status"]["approved"], 1)

    def test_initial_edit_invalidates_all_dependent_approvals_and_old_review_file(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); source = self.fixture(root)
            approved = self.approve(root, source)
            stale = self.decisions(root, approved, "stale.json")
            sheet = self.edit_sheet(root, approved, author="Carol")
            revised = root / "revised"
            cli.revise(SimpleNamespace(input_dir=approved, sheet=sheet, output_dir=revised))
            report = model.summary(cli.load_bundle(revised))
            self.assertEqual(report["review_status"]["pending"], 4)
            self.assertEqual(report["language_ready_cases"], 0)
            self.assertEqual(model.summary(cli.load_bundle(approved))["review_status"]["approved"], 4)
            with self.assertRaises(ValueError):
                cli.review(SimpleNamespace(input_dir=revised, decisions=stale, output_dir=root / "stale"))

    def test_rejecting_initial_context_blocks_prior_approved_dependents(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); source = self.approve(root, self.fixture(root))
            path = self.decisions(root, source, "reject.json", {"task_001_initial"}, decision="needs_revision")
            output = root / "rejected"
            cli.review(SimpleNamespace(input_dir=source, decisions=path, output_dir=output))
            report = model.summary(cli.load_bundle(output))
            self.assertEqual(report["review_status"]["needs_revision"], 1)
            self.assertEqual(report["review_status"]["pending"], 3)
            self.assertEqual(report["language_ready_cases"], 0)

    def test_manual_status_or_label_changes_fail_even_with_a_recomputed_file_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = self.fixture(root)
            data = cli.read_json(source / "annotations.json")
            data["records"][0]["review"]["status"] = "approved"
            cli.write_json(source / "annotations.json", data)
            manifest = cli.read_json(source / "manifest.json")
            manifest["annotations_sha256"] = sha256(source / "annotations.json")
            cli.write_json(source / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                cli.load_bundle(source)

    def test_origin_geometry_or_instance_metadata_cannot_be_rebound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = self.fixture(root)
            data = cli.read_json(source / "annotations.json")
            data["base_tasks"][0]["targets"][data["base_tasks"][0]["initial_goal"]]["stop_position"][0] += 1
            cli.write_json(source / "annotations.json", data)
            seal = cli.read_json(source / "manifest.json")
            seal["annotations_sha256"] = sha256(source / "annotations.json")
            cli.write_json(source / "manifest.json", seal)
            with self.assertRaises(ValueError):
                cli.load_bundle(source)

    def test_csv_cannot_drop_rows_change_bindings_or_hide_authorship(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = self.fixture(root)
            for kind in ("drop", "binding", "no_author", "coordinate"):
                sheet = self.edit_sheet(root, source, name=kind+".csv",
                    text="Go to target_000." if kind == "coordinate" else "去那张床旁停下。")
                with sheet.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                if kind == "drop":
                    rows.pop()
                elif kind == "binding":
                    rows[0]["base_task_id"] = "another_task"
                elif kind == "no_author":
                    rows[0]["author"] = ""
                with sheet.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=model.SHEET_COLUMNS); writer.writeheader(); writer.writerows(rows)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    cli.revise(SimpleNamespace(input_dir=source, sheet=sheet, output_dir=root / ("bad_" + kind)))

    def test_changed_image_evidence_prevents_new_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = self.fixture(root)
            dataset = cli.load_bundle(source)
            base = dataset["base_tasks"][0]
            target = base["targets"][base["initial_goal"]]
            Path(target["rgb"]["path"]).write_bytes(b"different evidence")
            with self.assertRaises(ValueError):
                cli.load_bundle(source)

    def test_export_is_blocked_without_silently_dropping_pending_cases(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); source = self.fixture(root)
            output = root / "export"
            before = snapshot({"source": source})
            self.assertEqual(cli.export(SimpleNamespace(input_dir=source, output_dir=output)), 1)
            self.assertEqual(before, snapshot({"source": source}))
            self.assertFalse((output / "public").exists())
            report = cli.read_json(output / "export_report.json")
            self.assertEqual(report["planned_task_cases"], 3)
            self.assertEqual(len((output / "private/planned_cases.jsonl").read_text().splitlines()), 3)

    def test_one_ready_case_does_not_publish_an_unannounced_subset(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); source = self.approve(root, self.fixture(root), subset={"task_001_initial", "task_001_replacement"})
            report = model.summary(cli.load_bundle(source))
            self.assertEqual(report["language_ready_cases"], 1)
            self.assertFalse(report["all_language_approved"])
            output = root / "partial"
            self.assertEqual(cli.export(SimpleNamespace(input_dir=source, output_dir=output)), 1)
            self.assertFalse((output / "public").exists())

    def test_complete_synthetic_review_exports_only_texts_and_opaque_ids(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); source = self.approve(root, self.fixture(root))
            output = root / "export"
            self.assertEqual(cli.export(SimpleNamespace(input_dir=source, output_dir=output)), 0)
            rows = [json.loads(line) for line in (output / "public/language.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(set(row) == {"sample_id", "initial_instruction", "initial_language", "intervention_message", "intervention_language"} for row in rows))
            text = (output / "public/language.jsonl").read_text()
            for forbidden in ("target_000", "task_001", "semantic_label", "task_type", "Bob", str(root)):
                self.assertNotIn(forbidden, text)
            private = [json.loads(line) for line in (output / "private/tasks.jsonl").read_text().splitlines()]
            self.assertEqual(len({row["leakage_group"] for row in private}), 1)
            self.assertTrue(all(row["split"] is None and not row["new_language_navigation_executed"] for row in private))
            with self.assertRaises(FileExistsError):
                cli.export(SimpleNamespace(input_dir=source, output_dir=output))

    def test_parent_authority_is_immutable_and_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); source = self.fixture(root)
            approved = self.approve(root, source)
            manifest = cli.read_json(source / "manifest.json"); manifest["changed"] = True
            cli.write_json(source / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                cli.load_bundle(approved)

    def test_revision_cannot_rewrite_parent_events_even_if_the_new_history_replays(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            original = self.fixture(root)
            approved = self.approve(root, original)
            revised = root / "revised"
            sheet = self.edit_sheet(root, approved, author="Carol")
            cli.revise(SimpleNamespace(input_dir=approved, sheet=sheet, output_dir=revised))
            data = cli.read_json(revised / "annotations.json")
            # A changed historical note still produces internally valid records
            # after a replay, but must fail the append-only parent-prefix check.
            data["events"][0]["payload"]["note"] = "Rewritten historical claim."
            records = {row["annotation_id"]: row for row in model.seed_records(data["base_tasks"])}
            for event in data["events"]:
                model.apply_event(records, event)
            data["records"] = list(records.values())
            model.replay(data)
            cli.write_json(revised / "annotations.json", data)
            seal = cli.read_json(revised / "manifest.json")
            seal["annotations_sha256"] = sha256(revised / "annotations.json")
            cli.write_json(revised / "manifest.json", seal)
            with self.assertRaisesRegex(ValueError, "append events"):
                cli.load_bundle(revised)

    def test_review_checks_content_and_context_hashes_not_only_revision_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = self.fixture(root)
            for field in ("content_sha256", "context_sha256"):
                path = self.decisions(root, original)
                value = cli.read_json(path)
                value["decisions"][1][field] = "0" * 64
                cli.write_json(path, value)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "stale"):
                    cli.review(SimpleNamespace(input_dir=original, decisions=path, output_dir=root / field))
                self.assertFalse((root / field).exists())

    def test_audit_cli_is_read_only_and_does_not_mean_language_approval(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            original = self.fixture(root)
            before = snapshot({"bundle": original})
            output = root / "audit"
            argv = ["language_annotations", "audit", "--input-dir", str(original), "--output-dir", str(output)]
            with patch("sys.argv", argv):
                self.assertEqual(cli.main(), 0)
            self.assertEqual(before, snapshot({"bundle": original}))
            report = cli.read_json(output / "audit.json")
            self.assertTrue(report["audit_passed"])
            self.assertTrue(report["authority_files_unchanged"])
            self.assertEqual(report["language_status"]["review_status"]["approved"], 0)


if __name__ == "__main__":
    unittest.main()
