"""The full reachable roster is retained without copying old target approvals."""

import unittest

from scripts.hssd_reachable_previews import make_targets


class ReachablePreviewTests(unittest.TestCase):
    def fixture(self):
        return ([{"source_instance_index": 2, "template_id": "a", "category": "unknown",
                  "status": "reachable", "runtime_object_id": 8,
                  "geometry": {"stop_position": [1, 0, 0], "anchor_distance_m": .5}}],
                [{"source_instance_index": 2, "template_id": "a"}],
                [{"object_id": 8, "template_id": "a", "semantic_id": 0,
                  "world_aabb": {"center": [0, 0, 0]}}])

    def test_unknown_target_gets_preview_but_no_approval(self):
        result = make_targets(*self.fixture())
        self.assertEqual(result[0]["target_id"], "target_002")
        self.assertEqual(result[0]["semantic_id"], 0)
        self.assertEqual(result[0]["review"]["status"], "pending")

    def test_template_mismatch_is_rejected(self):
        rows, sources, objects = self.fixture()
        sources[0]["template_id"] = "different"
        with self.assertRaises(ValueError):
            make_targets(rows, sources, objects)

    def test_unreachable_rows_are_not_added_to_target_roster(self):
        rows, sources, objects = self.fixture()
        rows.append({"source_instance_index": 3, "status": "mapping_ambiguous"})
        self.assertEqual(len(make_targets(rows, sources, objects)), 1)


if __name__ == "__main__":
    unittest.main()
