"""Verify that execution branches cannot inflate collection coverage."""

from copy import deepcopy
import unittest

from scripts.benchmark_collection import inventory


class CollectionTests(unittest.TestCase):
    def fixture(self):
        config = {"start_position": [0, 0, 0], "start_rotation_xyzw": [0, 0, 0, 1], "initial_goal": "A"}
        common = {"scene_id": "scene_1", "scene_sha256": "hash", "scene_group": "scene_1@hash",
                  "base_task_id": "scene_1@hash/base", "relatedness_level": "weak", "config": config}
        cases = [{**common, "case_id": kind, "task_type": kind} for kind in ("replacement", "subgoal")]
        episodes = [{"episode_id": f"{case['case_id']}/{condition}", "case_id": case["case_id"],
                     "condition": condition, "accepted": True, "task_success": condition == "oracle"}
                    for case in cases for condition in ("oracle", "ignore")]
        return cases, episodes

    def test_two_variants_and_four_runs_retain_one_base_task(self):
        bases, result = inventory(*self.fixture())
        self.assertEqual(len(bases), 1)
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["audited_episodes"], 4)
        self.assertEqual(result["quota_by_scene"]["scene_1@hash"]["subgoal"]["weak"],
                         {"count": 1, "target": 15, "shortfall": 14})
        self.assertEqual(bases[0]["split"], "unassigned")

    def test_missing_and_invalid_negative_controls_rejected(self):
        cases, episodes = self.fixture()
        with self.assertRaisesRegex(ValueError, "Missing execution evidence"):
            inventory(cases, episodes[:-1])
        episodes[-1]["task_success"] = True
        with self.assertRaisesRegex(ValueError, "planned outcome"):
            inventory(cases, episodes)

    def test_same_base_id_cannot_group_different_starts(self):
        cases, episodes = self.fixture()
        cases[1]["config"] = deepcopy(cases[1]["config"])
        cases[1]["config"]["start_position"] = [1, 0, 0]
        with self.assertRaisesRegex(ValueError, "different initial geometry"):
            inventory(cases, episodes)
