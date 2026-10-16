import unittest

from mivln import EpisodeEvaluator, TaskState


class EvaluatorTests(unittest.TestCase):
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

