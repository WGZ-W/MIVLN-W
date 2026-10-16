import unittest

from mivln import TaskState


class TaskStateTests(unittest.TestCase):
    def test_replace_goal_cancels_previous_goal(self) -> None:
        state = TaskState(active_goal="kitchen")

        state.apply_update(
            {
                "operation": "replace_goal",
                "old_goal": "kitchen",
                "new_goal": "bedroom",
            }
        )

        self.assertEqual(state.active_goal, "bedroom")
        self.assertEqual(state.cancelled_goals, ["kitchen"])

    def test_replace_wrong_active_goal_is_rejected(self) -> None:
        state = TaskState(active_goal="kitchen")

        with self.assertRaises(ValueError):
            state.apply_update(
                {
                    "operation": "replace_goal",
                    "old_goal": "living_room",
                    "new_goal": "bedroom",
                }
            )

    def test_unknown_operation_is_rejected(self) -> None:
        state = TaskState(active_goal="kitchen")

        with self.assertRaises(ValueError):
            state.apply_update({"operation": "teleport", "new_goal": "bedroom"})


if __name__ == "__main__":
    unittest.main()

