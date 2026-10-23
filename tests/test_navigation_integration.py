"""Exercise the real episode loop against a CPU-only fake simulator."""

from copy import deepcopy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_intervention import check_pair


if importlib.util.find_spec("habitat_sim") is not None:
    from scripts import run_navigation as navigation
else:
    navigation = None


@unittest.skipIf(navigation is None, "Requires Habitat imports; runs without rendering or a GPU")
class NavigationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "episode_id": "test_pair", "initial_goal": "a",
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "goal_positions": {"a": [2.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]},
            "events": [{"event_id": "return", "trigger": {"type": "step", "value": 2},
                        "utterance": "Return to B.",
                        "task_update": {"operation": "replace_goal", "old_goal": "a", "new_goal": "b"}}],
            "success_radius_m": 0.3, "max_steps": 30, "seed": 42,
            "forward_step_m": 0.25, "turn_angle_degrees": 15.0, "sensor_height": 1.5,
        }

    def simulator_patches(self, stack, ignore_new_goal=False):
        np = navigation.np
        instances = []

        class FakeAgent:
            def get_state(self):
                return self.state

        class FakeSimulator:
            def __init__(self, config):
                instances.append(self)
                self.agent = FakeAgent()
                self.initializations = 0
                self.pathfinder = SimpleNamespace(
                    is_loaded=True, seed=lambda value: None, is_navigable=lambda position: True,
                )

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def seed(self, value):
                pass

            def initialize_agent(self, index, state):
                self.initializations += 1
                self.agent.state = SimpleNamespace(position=state.position.copy(), rotation=state.rotation)
                return self.agent

            def get_sensor_observations(self):
                return {}

            def step(self, action):
                self.agent.state.position[0] += self.agent.direction * 0.25
                return {"collided": False}

        class FakeFollower:
            def __init__(self, pathfinder, agent, goal_radius, stop_key):
                self.agent = agent
                self.radius = goal_radius

            def next_action_along(self, goal):
                target = 2.0 if ignore_new_goal else goal[0]
                remaining = target - self.agent.state.position[0]
                if abs(remaining) <= self.radius:
                    return "STOP"
                self.agent.direction = 1 if remaining > 0 else -1
                return "move_forward"

        def fake_path(pathfinder, start, goal):
            return SimpleNamespace(geodesic_distance=float(np.linalg.norm(goal - start)),
                                   points=[start.copy(), goal.copy()])

        def fake_pose(agent):
            return {"position": agent.state.position.tolist(), "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}

        for target, replacement in (
            ("make_configuration", lambda *args: None),
            ("shortest_path", fake_path), ("pose", fake_pose),
            ("save_frame", lambda *args: {}),
        ):
            stack.enter_context(patch.object(navigation, target, replacement))
        stack.enter_context(patch.object(navigation.habitat_sim, "Simulator", FakeSimulator))
        stack.enter_context(patch.object(navigation.habitat_sim.nav, "GreedyGeodesicFollower", FakeFollower))
        return instances

    def test_paired_episode_switches_before_action_without_reset(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            scene = output / "scene.glb"
            scene.touch()
            instances = self.simulator_patches(stack)
            control = deepcopy(self.config)
            control["events"] = []
            navigation.run(control, scene, 0, output / "control")
            result = navigation.run(self.config, scene, 0, output / "intervention")
            summary = check_pair(output, self.config["events"][0])
            self.assertTrue(summary["passed"], summary["checks"])
            self.assertEqual([sim.initializations for sim in instances], [1, 1])
            self.assertEqual(result["intervention_step"], 2)
            self.assertEqual(result["response_latency_steps"], 0)
            self.assertEqual(result["stopped_goal"], "b")
            self.assertIsNone(result["spl"])
            self.assertIsNotNone(result["post_intervention_spl"])
            # A deliberate pose reset must invalidate the recorded pair.
            path = output / "intervention" / "events.json"
            events = json.loads(path.read_text())
            events[0]["pose_after"]["position"] = [0.0, 0.0, 0.0]
            path.write_text(json.dumps(events))
            self.assertFalse(check_pair(output, self.config["events"][0])["passed"])

    def test_oracle_ignoring_intervention_fails_at_cancelled_goal(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            scene = output / "scene.glb"
            scene.touch()
            self.simulator_patches(stack, ignore_new_goal=True)
            result = navigation.run(self.config, scene, 0, output / "ignored")
            self.assertFalse(result["dynamic_success"])
            self.assertEqual(result["final_goal"], "b")
            self.assertEqual(result["stopped_goal"], "a")

    def test_ignore_policy_updates_truth_but_keeps_original_physical_trajectory(self):
        from scripts.run_ignore_intervention import check_negative

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            scene = output / "scene.glb"
            scene.touch()
            instances = self.simulator_patches(stack)
            config = deepcopy(self.config)
            config.update({"scene": str(scene), "resolution": [256, 256]})
            control = deepcopy(config)
            control["events"] = []
            navigation.run(control, scene, 0, output / "control")
            metrics = navigation.run(config, scene, 0, output / "ignored", intervention_policy="ignore")
            report = check_negative(output / "ignored", config, output / "control")
            self.assertTrue(report["negative_check_passed"], report["checks"])
            self.assertEqual([sim.initializations for sim in instances], [1, 1])
            self.assertFalse(metrics["dynamic_success"])
            self.assertEqual(metrics["policy_goal"], "a")
            self.assertEqual(metrics["final_goal"], "b")
            self.assertEqual(metrics["stopped_goal"], "a")
            self.assertEqual(metrics["termination"], "stopped_at_cancelled_goal")
            self.assertEqual(metrics["post_intervention_spl"], 0.0)
            self.assertIsNone(metrics["first_compliant_action_step"])
            self.assertIsNone(metrics["response_latency_steps"])
            self.assertEqual(metrics["ignored_event_count"], 1)
            truth = json.loads((output / "ignored/final_task_state.json").read_text())
            policy = json.loads((output / "ignored/final_policy_state.json").read_text())
            self.assertEqual(truth["cancelled_goals"], ["a"])
            self.assertEqual(policy["cancelled_goals"], [])
            # An erroneous success flag must fail negative-control validation.
            metrics["dynamic_success"] = True
            (output / "ignored/metrics.json").write_text(json.dumps(metrics))
            self.assertFalse(check_negative(output / "ignored", config)["negative_check_passed"])

    def test_negative_control_rejects_an_event_that_was_never_delivered(self):
        from scripts.run_ignore_intervention import check_negative

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            scene = output / "scene.glb"
            scene.touch()
            self.simulator_patches(stack)
            config = deepcopy(self.config)
            config["events"][0]["trigger"]["value"] = 29
            navigation.run(config, scene, 0, output / "ignored", intervention_policy="ignore")
            report = check_negative(output / "ignored", config)
            self.assertFalse(report["negative_check_passed"])
            self.assertFalse(report["checks"]["event_delivered_once"])

    def test_negative_control_rejects_timeout_even_when_dynamic_success_is_false(self):
        from scripts.run_ignore_intervention import check_negative

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            scene = output / "scene.glb"
            scene.touch()
            self.simulator_patches(stack)
            config = deepcopy(self.config)
            config["max_steps"] = 3
            metrics = navigation.run(config, scene, 0, output / "ignored", intervention_policy="ignore")
            self.assertFalse(metrics["dynamic_success"])
            self.assertEqual(metrics["delivered_event_count"], 1)
            report = check_negative(output / "ignored", config)
            self.assertFalse(report["negative_check_passed"])
            self.assertFalse(report["checks"]["failure_is_cancelled_goal_not_timeout"])

    def test_timeout_does_not_count_as_success(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            scene = output / "scene.glb"
            scene.touch()
            self.simulator_patches(stack)
            config = deepcopy(self.config)
            config["events"] = []
            config["max_steps"] = 1
            result = navigation.run(config, scene, 0, output / "timeout")
            self.assertFalse(result["dynamic_success"])
            self.assertFalse(result["stopped"])
            self.assertEqual(result["termination"], "timeout")

    def test_invalid_trigger_and_same_goal_are_rejected(self):
        for trigger in (True, 30):
            config = deepcopy(self.config)
            config["events"][0]["trigger"]["value"] = trigger
            with self.assertRaises(ValueError):
                navigation.validate_config(config)
        config = deepcopy(self.config)
        config["events"][0]["task_update"]["new_goal"] = "a"
        with self.assertRaises(ValueError):
            navigation.validate_config(config)

    def test_timing_batch_preserves_controls_and_summarizes_three_pairs(self):
        from scripts.intervention_timing import load_json, run_cases, write_json

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            scene = output / "scene.glb"
            scene.touch()
            instances = self.simulator_patches(stack)
            cases = []
            for name, fraction, trigger in (("early", 0.25, 1), ("middle", 0.5, 2), ("late", 0.75, 3)):
                config = deepcopy(self.config)
                config["goal_positions"]["b"] = [-2.0, 0.0, 0.0]
                config["events"][0]["trigger"]["value"] = trigger
                relative = f"configs/{name}.json"
                write_json(output / relative, config)
                cases.append({"name": name, "target_progress": fraction, "actual_progress": fraction,
                              "trigger_step": trigger, "config": relative, "output_subdir": name})
            write_json(output / "manifest.json", {"cases": cases, "target_b": {"position": [-2, 0, 0]}})

            def run_child(command, cwd):
                config = load_json(Path(command[command.index("--config") + 1]))
                case_dir = Path(command[command.index("--output-dir") + 1])
                control = deepcopy(config)
                control["events"] = []
                navigation.run(control, scene, 0, case_dir / "control")
                navigation.run(config, scene, 0, case_dir / "intervention")
                return SimpleNamespace(returncode=0)

            from contextlib import redirect_stdout
            import io

            stack.enter_context(patch("scripts.intervention_timing.subprocess.run", run_child))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = run_cases(SimpleNamespace(output_dir=output, gpu_id=0, no_video=True))
            summary = load_json(output / "summary.json")
            self.assertEqual(result, 0)
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["controls_identical"])
            self.assertEqual(summary["passed_cases"], 3)
            self.assertEqual([sim.initializations for sim in instances], [1] * 6)
            # Running the batch again must not overwrite any completed case.
            with self.assertRaises(FileExistsError):
                run_cases(SimpleNamespace(output_dir=output, gpu_id=0, no_video=True))

    def test_debug_collection_keeps_fixed_target_through_all_stages(self):
        from contextlib import redirect_stdout
        import io
        from scripts.debug_dataset import run as run_collection
        from scripts.intervention_timing import load_json, prepare, run_cases, write_json

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            scene = output / "scene.glb"
            scene.touch()
            scene.with_suffix(".navmesh").touch()
            instances = self.simulator_patches(stack)
            np = navigation.np

            class FakePathFinder:
                def load_nav_mesh(self, path):
                    return True

                def is_navigable(self, position):
                    return True

                def find_path(self, path):
                    path.geodesic_distance = float(np.linalg.norm(path.requested_end - path.requested_start))
                    return True

                def get_random_navigable_point(self):
                    raise AssertionError("A prepared target must never be resampled")

            stack.enter_context(patch.object(navigation.habitat_sim, "PathFinder", FakePathFinder))
            stack.enter_context(patch.object(navigation.habitat_sim, "ShortestPath", SimpleNamespace))
            config = deepcopy(self.config)
            config.update({"scene": str(scene), "resolution": [256, 256]})
            config["goal_positions"]["b"] = [-2.0, 0.0, 0.0]
            task = {"task_id": "task_001", "base_config": "configs/base.json",
                    "reference_config": "configs/reference.json", "output_subdir": "tasks/task_001"}
            reference = deepcopy(config)
            reference["events"] = []
            write_json(output / task["base_config"], config)
            write_json(output / task["reference_config"], reference)
            write_json(output / "manifest.json", {"tasks": [task]})

            def value(command, key):
                return Path(command[command.index(key) + 1])

            def execute(command, log):
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("CPU-only integration test")
                if command[3] == "scripts.run_navigation":
                    navigation.run(load_json(value(command, "--config")), scene, 0, value(command, "--output-dir"))
                elif "prepare" in command:
                    prepare(SimpleNamespace(output_dir=value(command, "--output-dir"),
                        base_config=value(command, "--base-config"), reference_dir=value(command, "--reference-dir"),
                        seed=123, max_attempts=1, min_separation=2.0, min_route_distance=1.0,
                        max_distance=20.0, keep_target=True))
                else:
                    return run_cases(SimpleNamespace(output_dir=value(command, "--output-dir"), gpu_id=0, no_video=True))
                return 0

            def run_pair(command, cwd):
                variant = load_json(value(command, "--config"))
                control = deepcopy(variant)
                control["events"] = []
                destination = value(command, "--output-dir")
                navigation.run(control, scene, 0, destination / "control")
                navigation.run(variant, scene, 0, destination / "intervention")
                return SimpleNamespace(returncode=0)

            stack.enter_context(patch("scripts.debug_dataset.execute_stage", execute))
            stack.enter_context(patch("scripts.intervention_timing.subprocess.run", run_pair))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result = run_collection(SimpleNamespace(output_dir=output, gpu_id=0))
            report = load_json(output / "summary.json")
            self.assertEqual(result, 0)
            self.assertTrue(report["passed"])
            self.assertTrue(report["tasks"][0]["fixed_task_parameters"])
            self.assertEqual(report["passed_pairs"], 3)
            self.assertEqual(len(instances), 7)
            manifest = load_json(output / "tasks/task_001/timing/manifest.json")
            self.assertEqual(manifest["target_b"]["position"], [-2.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
