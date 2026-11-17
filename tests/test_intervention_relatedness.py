"""Regression checks for full-roster classification and private provenance."""

import unittest

from scripts import intervention_relatedness as related


class RelatednessTests(unittest.TestCase):
    def candidates(self):
        return [{"case_id": str(index), "base_task_id": f"scene/base_{index}", "scene_id": "scene",
                 "initial_instruction": "Go to A", "intervention_instruction": "Go to B"}
                for index in range(3)]

    def rows(self):
        rows = related.worksheet(self.candidates())
        rows[0].update(decision="approve", relatedness_level="strong", reviewer="WGZ",
                       relatedness_rationale="Same functional area; verified in evidence",
                       semantic_relation="Related furniture", spatial_relation="Same local area")
        return rows

    def test_full_denominator_with_pending_cases_and_no_release_claim(self):
        candidates = self.candidates()
        rows = related.review_rows(candidates, self.rows())
        report = related.report(candidates, rows)
        self.assertEqual(report["planned_candidates"], 3)
        self.assertEqual(report["by_decision"]["pending"], 2)
        self.assertEqual(report["approved_by_level"]["strong"], 1)
        self.assertFalse(report["release_ready"])

    def test_drop_duplicate_and_source_edit_are_rejected(self):
        candidates = self.candidates()
        rows = self.rows()
        with self.assertRaises(ValueError):
            related.review_rows(candidates, rows[:-1])
        with self.assertRaises(ValueError):
            related.review_rows(candidates, [rows[0], rows[0], rows[2]])
        candidates[0]["initial_instruction"] = "Changed language"
        with self.assertRaises(ValueError):
            related.review_rows(candidates, rows)

    def test_approval_requires_level_and_separate_relation_evidence(self):
        for key, value in (("relatedness_level", "unassigned"), ("semantic_relation", ""),
                           ("spatial_relation", ""), ("relatedness_rationale", "")):
            rows = self.rows()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                related.review_rows(self.candidates(), rows)

    def test_rejection_stays_in_roster_without_forced_classification(self):
        rows = related.worksheet(self.candidates())
        rows[0].update(decision="reject", reviewer="WGZ", relatedness_rationale="Ambiguous relation")
        result = related.review_rows(self.candidates(), rows)
        self.assertEqual(result[0]["relatedness_level"], "unassigned")
        self.assertEqual(len(result), 3)

    def test_blank_templates_and_missing_cells_are_not_reviews(self):
        with self.assertRaises(ValueError):
            related.review_rows(self.candidates(), related.worksheet(self.candidates()))
        rows = self.rows()
        rows[0]["spatial_relation"] = None
        with self.assertRaises(ValueError):
            related.review_rows(self.candidates(), rows)


if __name__ == "__main__":
    unittest.main()
