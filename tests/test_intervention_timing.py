from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

from scripts.intervention_timing import select_triggers, summarize, write_json


class InterventionTimingTests(unittest.TestCase):
    def trajectory(self):
        rows = [{"step": None, "action": "INITIALIZE", "position": [0, 0, 0],
                 "distance_to_goal_m": 10.0}]
        rows.extend({"step": step, "action": "move_forward", "position": [step + 1, 0, 0],
                     "distance_to_goal_m": 9.0 - step} for step in range(10))
        rows.append({"step": 10, "action": "STOP", "position": [10, 0, 0],
                     "distance_to_goal_m": 0.0})
        return rows

    def test_trigger_step_uses_state_before_the_next_action(self):
        cases = select_triggers(self.trajectory())
        self.assertEqual([case["trigger_step"] for case in cases], [3, 5, 8])
        self.assertEqual([case["position"] for case in cases], [[3, 0, 0], [5, 0, 0], [8, 0, 0]])
        for case, progress in zip(cases, (0.3, 0.5, 0.8)):
            self.assertAlmostEqual(case["actual_progress"], progress)

    def test_turning_does_not_advance_distance_based_trigger(self):
        rows = self.trajectory()
        turn = deepcopy(rows[1])
        turn["action"] = "turn_left"
        rows.insert(2, turn)
        for step, row in enumerate(rows[1:]):
            row["step"] = step
        self.assertEqual([case["trigger_step"] for case in select_triggers(rows)], [4, 6, 9])

    def test_incomplete_or_discontinuous_reference_is_rejected(self):
        rows = self.trajectory()
        with self.assertRaises(ValueError):
            select_triggers(rows[:-1])
        rows[3]["step"] = 99
        with self.assertRaises(ValueError):
            select_triggers(rows)

    def test_short_reference_cannot_reuse_one_step_for_multiple_timings(self):
        rows = [self.trajectory()[0],
                {"step": 0, "action": "move_forward", "position": [9, 0, 0], "distance_to_goal_m": 1.0},
                {"step": 1, "action": "move_forward", "position": [10, 0, 0], "distance_to_goal_m": 0.0},
                {"step": 2, "action": "STOP", "position": [10, 0, 0], "distance_to_goal_m": 0.0}]
        with self.assertRaises(ValueError):
            select_triggers(rows)

    def test_missing_results_cannot_pass_summary(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            output = Path(temporary)
            write_json(output / "manifest.json", {
                "target_b": {"position": [1, 0, 1]},
                "cases": [{"name": "early", "target_progress": 0.25, "actual_progress": 0.3,
                           "trigger_step": 3, "config": "configs/early.json", "output_subdir": "early"}],
            })
            report = summarize(output)
            self.assertFalse(report["passed"])
            self.assertEqual(report["passed_cases"], 0)
            self.assertIsNotNone(report["cases"][0]["error"])
            self.assertTrue((output / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
