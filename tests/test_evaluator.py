import unittest

from mivln import EpisodeEvaluator, TaskState


class EvaluatorTests(unittest.TestCase):
    def evaluate_navigation(self, **overrides):
        values = {
            "task_state": TaskState(active_goal="bedroom", cancelled_goals=["kitchen"]),
            "stopped": True,
            "goal_distances_m": {"bedroom": 0.2, "kitchen": 5.0},
            "success_radius_m": 0.3,
            "intervention_step": 5,
            "first_compliant_action_step": 5,
            "path_length": 3.25,
            "steps": 12,
            "max_steps": 20,
        }
        values.update(overrides)
        return EpisodeEvaluator().evaluate_navigation(**values)

    def test_navigation_stop_at_measured_active_goal_succeeds(self):
        result = self.evaluate_navigation()
        self.assertTrue(result.dynamic_success)
        self.assertEqual(result.stopped_goal, "bedroom")

    def test_navigation_stop_at_old_goal_fails_despite_active_goal_label(self):
        result = self.evaluate_navigation(goal_distances_m={"bedroom": 5.0, "kitchen": 0.2})
        self.assertFalse(result.dynamic_success)
        self.assertFalse(result.goal_update_compliance)
        self.assertEqual(result.stopped_goal, "kitchen")

    def test_navigation_reaching_goal_without_stop_fails(self):
        result = self.evaluate_navigation(stopped=False)
        self.assertFalse(result.dynamic_success)
        self.assertIsNone(result.stopped_goal)

    def test_navigation_unreachable_or_invalid_active_goal_fails(self):
        for distance in (None, float("inf"), float("nan"), -0.1):
            with self.subTest(distance=distance):
                result = self.evaluate_navigation(goal_distances_m={"bedroom": distance})
                self.assertFalse(result.dynamic_success)

    def test_navigation_over_budget_fails(self):
        self.assertFalse(self.evaluate_navigation(steps=21).dynamic_success)

    def test_updated_goal_success(self) -> None:
        state = TaskState(active_goal="bedroom", cancelled_goals=["kitchen"])

        metrics = EpisodeEvaluator().evaluate(
            task_state=state,
            stopped_goal="bedroom",
            intervention_step=5,
            first_compliant_action_step=7,
            path_length=3.25,
            steps=12,
            max_steps=20,
        )

        self.assertTrue(metrics.dynamic_success)
        self.assertTrue(metrics.goal_update_compliance)
        self.assertEqual(metrics.response_latency_steps, 2)

    def test_stopping_at_cancelled_goal_fails(self) -> None:
        state = TaskState(active_goal="bedroom", cancelled_goals=["kitchen"])

        metrics = EpisodeEvaluator().evaluate(
            task_state=state,
            stopped_goal="kitchen",
            intervention_step=5,
            first_compliant_action_step=None,
            path_length=3.25,
            steps=12,
            max_steps=20,
        )

        self.assertFalse(metrics.dynamic_success)
        self.assertFalse(metrics.goal_update_compliance)
        self.assertIsNone(metrics.response_latency_steps)


if __name__ == "__main__":
    unittest.main()

