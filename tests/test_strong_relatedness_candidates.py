"""Sampling guards: same category is a proposal, distance is only one factor."""

import unittest
from pathlib import Path
import tempfile

from scripts.strong_relatedness_candidates import difficulty, relation_hint, image_link


class StrongCandidateTests(unittest.TestCase):
    def test_image_link_is_relative_and_escapes_spaces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "previews" / "target image.png"
            image.parent.mkdir()
            image.write_bytes(b"fixture")
            self.assertEqual(image_link(image, root / "package"), "../previews/target%20image.png")
            with self.assertRaises(ValueError):
                image_link(root / "missing.png", root / "package")

    def test_self_replacement_is_not_a_proposal(self):
        target = {"target_id": "a", "category": "bed"}
        self.assertIsNone(relation_hint(target, target))

    def test_distinct_same_category_and_functional_pairs(self):
        self.assertEqual(relation_hint({"target_id": "a", "category": "bed"},
                                       {"target_id": "b", "category": "bed"}),
                         "same_category_instance_correction")
        self.assertIsNotNone(relation_hint({"target_id": "a", "category": "table"},
                                          {"target_id": "b", "category": "bench"}))
        self.assertIsNone(relation_hint({"target_id": "a", "category": "bed"},
                                       {"target_id": "b", "category": "counter"}))

    def test_both_matching_tolerances_are_required(self):
        self.assertTrue(difficulty(9, 10, 2, .25)["distance_match"])
        self.assertFalse(difficulty(12, 10, 1, .25)["distance_match"])
        self.assertFalse(difficulty(2, 1, 2, .25)["distance_match"])

    def test_unreachable_and_zero_baseline(self):
        for value in (None, float("inf"), -1):
            self.assertFalse(difficulty(value, 10, 2, .25)["distance_match"])
        self.assertFalse(difficulty(1, 0, 2, .25)["distance_match"])
        self.assertTrue(difficulty(0, 0, 2, .25)["distance_match"])


if __name__ == "__main__":
    unittest.main()
