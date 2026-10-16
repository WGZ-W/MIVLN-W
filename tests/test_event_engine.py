import unittest

from mivln import EventEngine, InterventionEvent, TaskState


class EventEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event = InterventionEvent.from_dict(
            {
                "event_id": "replace_goal",
                "trigger": {"type": "step", "value": 5},
                "utterance": "Instead, go to the bedroom.",
                "task_update": {
                    "operation": "replace_goal",
                    "old_goal": "kitchen",
                    "new_goal": "bedroom",
                },
            }
        )

    def test_event_is_delivered_at_trigger_step_once(self) -> None:
        state = TaskState(active_goal="kitchen")
        engine = EventEngine([self.event])

        self.assertEqual(engine.step(4, state), [])
        self.assertEqual(state.active_goal, "kitchen")
        self.assertEqual(engine.step(5, state), [self.event])
        self.assertEqual(state.active_goal, "bedroom")
        self.assertEqual(engine.step(6, state), [])

    def test_duplicate_event_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EventEngine([self.event, self.event])


if __name__ == "__main__":
    unittest.main()

