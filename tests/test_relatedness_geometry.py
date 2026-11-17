"""Delivery indexing checks with a fake replay; never launches Habitat."""

from copy import deepcopy
import unittest

from scripts.relatedness_geometry import measure_prefix, pose_matches


class FakeReplay:
    def __init__(self):
        self.x = 0

    def pose(self):
        return {"position": [self.x, 0, 0], "rotation_xyzw": [0, 0, 0, 1]}

    def act(self, action):
        self.x += 1

    def distance(self, position):
        return float(position[0] - self.x)


class GeometryTests(unittest.TestCase):
    def fixture(self, step=2):
        definition = {"event_id": "event", "trigger": {"type": "step", "value": step},
                      "task_update": {"new_goal": "B"}}
        config = {"events": [definition], "goal_positions": {"B": [10, 0, 0]}}
        trajectory = [{"position": [index, 0, 0], "rotation_xyzw": [0, 0, 0, 1],
                       "action": "move_forward"} for index in range(4)]
        pose = {key: trajectory[step][key] for key in ("position", "rotation_xyzw")}
        events = [{"event_id": "event", "delivered_step": step,
                   "task_update": definition["task_update"], "pose_before": pose, "pose_after": deepcopy(pose)}]
        return config, trajectory, events

    def test_distance_is_before_delivery_action(self):
        result = measure_prefix(*self.fixture(), FakeReplay())
        self.assertEqual(result["prefix_actions_replayed"], 2)
        self.assertEqual(result["updated_goal_geodesic_distance_m"], 8.0)
        self.assertIsNone(result["reference_route_overlap"])

    def test_zero_step_has_no_prefix_actions(self):
        result = measure_prefix(*self.fixture(0), FakeReplay())
        self.assertEqual(result["updated_goal_geodesic_distance_m"], 10.0)

    def test_replay_divergence_and_pose_reset_are_rejected(self):
        config, trajectory, events = self.fixture()
        trajectory[1]["position"][0] = 100
        with self.assertRaises(ValueError):
            measure_prefix(config, trajectory, events, FakeReplay())
        config, trajectory, events = self.fixture()
        events[0]["pose_after"]["position"][0] = 100
        with self.assertRaises(ValueError):
            measure_prefix(config, trajectory, events, FakeReplay())

    def test_nonfinite_distance_is_rejected(self):
        replay = FakeReplay()
        replay.distance = lambda position: float("nan")
        with self.assertRaises(ValueError):
            measure_prefix(*self.fixture(), replay)

    def test_quaternion_sign_is_equivalent(self):
        self.assertTrue(pose_matches({"position": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]},
                                     {"position": [0, 0, 0], "rotation_xyzw": [0, 0, 0, -1]}))


if __name__ == "__main__":
    unittest.main()
