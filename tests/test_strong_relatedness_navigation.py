"""Regression checks for exact approval and action-before-delivery boundaries."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_manifest import sha256
from scripts.intervention_relatedness import digest
from scripts.public_inputs import write_json
from scripts.strong_relatedness_navigation import check_review, validate_prefix


class PilotNavigationTests(unittest.TestCase):
    def test_duplicate_approval_or_changed_draft_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cases.json"
            cases = [{"case_id": str(index), "config": {"instruction": str(index)}} for index in range(6)]
            write_json(source, cases)
            review = {"kind": "strong_relatedness_pilot_target_grounding_review_v1",
                      "schema_version": 1, "review_scope": "initial_and_intervention_instruction_target_reference",
                      "reviewer": "WGZ", "source_cases_sha256": sha256(source),
                      "approved_target_reference_cases": 6, "planned_cases": 6,
                      "decisions": [{"case_id": row["case_id"], "draft_sha256": digest(row["config"]),
                                     "decision": "approve"} for row in cases]}
            check_review(review, source, cases)
            duplicated = deepcopy(review)
            duplicated["decisions"][-1] = duplicated["decisions"][0]
            with self.assertRaisesRegex(ValueError, "All six exact"):
                check_review(duplicated, source, cases)
            cases[-1]["config"]["instruction"] = "new text"
            write_json(source, cases)
            review["source_cases_sha256"] = sha256(source)
            with self.assertRaisesRegex(ValueError, "All six exact"):
                check_review(review, source, cases)

    def test_prefix_includes_delivery_rgb_and_excludes_delivery_action(self):
        pose = {"position": [1, 0, 0], "rotation_xyzw": [0, 0, 0, 1]}
        prefix = {"actions": ["move_forward"], "pose_before_delivery": pose}
        trajectory = [{"step": step, "action": action, "collided": False, "distance_m": 0, **pose}
                      for step, action in ((None, "INITIALIZE"), (0, "move_forward"), (1, "turn_right"))]
        records = [{"request": {"observation": {"rgb": {"sha256": value}}}} for value in ("initial", "delivery")]
        result = {"trajectory": trajectory, "policy_records": records}
        physical, images = validate_prefix(prefix, 1, result)
        self.assertEqual([row["action"] for row in physical], ["INITIALIZE", "move_forward"])
        self.assertEqual(images, ["initial", "delivery"])
        trajectory[1] = {**trajectory[1], "position": [0, 0, 0]}
        with self.assertRaisesRegex(ValueError, "pre-action intervention pose"):
            validate_prefix(prefix, 1, result)

    def test_prefix_without_executed_delivery_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "did not execute"):
            validate_prefix({"actions": ["move_forward"]}, 1,
                            {"trajectory": [{}, {}], "policy_records": [{}, {}]})


if __name__ == "__main__":
    unittest.main()
