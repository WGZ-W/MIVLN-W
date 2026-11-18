"""Check that both pilot variants share a prefix but retain distinct obligations."""

from pathlib import Path
import unittest

from scripts.benchmark_manifest import read_json
from scripts.strong_relatedness_pilot import compare_distances, make_case


class StrongRelatednessPilotTests(unittest.TestCase):
    def test_drafts_share_a_and_delivery_but_subgoal_resumes_a(self):
        config = read_json(Path(__file__).resolve().parents[1] / "configs/simulator/hssd_navigation_001.json")
        pair = {"pair_id": "test_pair", "initial_goal": "A", "inserted_or_replacement_goal": "B",
                "initial_description_draft": "the island with drawers",
                "updated_description_draft": "the counter with the cooktop"}
        points = {"A": [0., 0., 1.], "B": [1., 0., 0.]}
        replacement = make_case(config, pair, "replacement", "same_base", points, 22)
        subgoal = make_case(config, pair, "subgoal", "same_base", points, 22)
        one, two = replacement["events"][0], subgoal["events"][0]
        self.assertEqual(one["trigger"], {"type": "step", "value": 22})
        self.assertEqual(one["trigger"], two["trigger"])
        self.assertEqual(replacement["start_position"], subgoal["start_position"])
        self.assertEqual(one["task_update"], {"operation": "replace_goal", "old_goal": "A", "new_goal": "B"})
        self.assertEqual(two["task_update"], {"operation": "insert_subgoal", "subgoal": "B", "resume_goal": "A"})
        self.assertEqual(subgoal["language_status"], "pending_human_review")

    def test_close_insertion_distance_cannot_hide_costly_resumption(self):
        new = {"p_to_b_m": 8.0, "b_to_a_m": 3.5, "via_b_m": 11.5}
        old = {"p_to_b_m": 8.0, "b_to_a_m": 1.0, "via_b_m": 9.0}
        self.assertEqual(compare_distances("replacement", new, old)["status"], "matched")
        self.assertEqual(compare_distances("subgoal", new, old)["status"], "distance_mismatch")
        self.assertEqual(compare_distances("subgoal", new, None)["status"], "no_shared_prefix_comparator")
