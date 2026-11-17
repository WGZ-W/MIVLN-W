"""Full-denominator inventory guards without Habitat or geometric rendering."""

import unittest

from scripts.hssd_reachability import coverage, inventory


class ReachabilityTests(unittest.TestCase):
    def source(self, index, template="a", category="unknown"):
        return {"source_instance_index": index, "template_id": template, "category": category}

    def runtime(self, template="a"):
        return {"template_id": template, "object_id": 9}

    def test_unlabeled_instances_are_still_sampled(self):
        rows = inventory([self.source(0)], [self.runtime()], lambda obj: {"anchor_distance_m": .2})
        self.assertEqual(rows[0]["status"], "reachable")
        self.assertEqual(coverage(rows)["reachable_labeled_instances"], 0)

    def test_repeated_template_does_not_guess_instance_identity(self):
        rows = inventory([self.source(0), self.source(1)], [self.runtime(), self.runtime()],
                         lambda obj: self.fail("Ambiguous instances must not be sampled"))
        self.assertEqual(coverage(rows)["by_status"]["mapping_ambiguous"], 2)
        self.assertEqual(len(rows), 2)

    def test_no_sample_is_not_global_unreachability(self):
        rows = inventory([self.source(0)], [self.runtime()], lambda obj: None)
        self.assertEqual(rows[0]["status"], "no_stopping_point_found")
        self.assertIsNone(coverage(rows)["global_unreachable_count"])

    def test_missing_runtime_and_sampler_failure_preserve_denominator(self):
        def broken(obj):
            raise ValueError("Bad AABB")
        rows = inventory([self.source(0), self.source(1, "b")], [self.runtime()], broken)
        self.assertEqual([row["status"] for row in rows], ["error", "runtime_missing"])
        self.assertEqual(coverage(rows)["planned_instances"], 2)

    def test_duplicate_source_indices_rejected(self):
        with self.assertRaises(ValueError):
            inventory([self.source(0), self.source(0)], [], lambda obj: None)


if __name__ == "__main__":
    unittest.main()
