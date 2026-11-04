"""Temporary-task semantics and measured B-then-A replay, with no GPU Simulator."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln import EpisodeEvaluator, EventEngine, InterventionEvent, TaskState
from mivln.subgoals import COMPLETE_SUBGOAL, completion_is_valid, sequence_satisfied
from scripts import subgoal
from scripts.hssd_targets import sha256
from scripts.intervention_timing import load_json, write_json


class SubgoalSemanticsTests(unittest.TestCase):
    def test_displacement_check_rejects_wrong_distance_and_nonmoving_action_teleport(self):
        previous = {"position": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]}
        for action, recorded in (("move_forward", .1), ("STOP", .25), ("turn_left", .25),
                                 (COMPLETE_SUBGOAL, .25)):
            row = {"position": [0, 0, -.25], "distance_m": recorded, "action": action, "step": 1}
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, "displacement is inconsistent"):
                subgoal.check_action_displacement(None, {"forward_step_m": .25}, previous, row)

    @unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Native Agent only, no Simulator")
    def test_large_native_projection_requires_reproduced_endpoint_without_simulator(self):
        import habitat_sim
        import math
        import magnum as mn

        # Regression for the real task_004/middle corner projection: the native
        # filter can move 0.28174 m after a nominal 0.25 m forward command.
        previous = {"position": [-4.979033946990967, .15372097492218018, -1.5820796489715576],
                    "rotation_xyzw": [0., .7777823209762573, 0., .6285337805747986]}
        endpoint = [-5.2234649658203125, .15372097492218018, -1.441969871520996]
        row = {"position": endpoint, "distance_m": math.dist(previous["position"], endpoint),
               "action": "move_forward", "step": 64}
        # The movement-filter return is independent of the claimed endpoint;
        # mutating the claim below must fail even with a self-consistent length.
        pathfinder = SimpleNamespace(try_step=lambda start, attempted: mn.Vector3(endpoint))
        with patch.object(habitat_sim, "Simulator", side_effect=AssertionError("No GPU Simulator allowed")):
            proof = subgoal.check_action_displacement(pathfinder, {"forward_step_m": .25}, previous, row)
            self.assertEqual(proof["endpoint_error_m"], 0.)
            self.assertFalse(proof["simulator_created"])
            changed = deepcopy(row)
            changed["position"][0] -= .02
            changed["distance_m"] = math.dist(previous["position"], changed["position"])
            with self.assertRaisesRegex(ValueError, "differs from native"):
                subgoal.check_action_displacement(pathfinder, {"forward_step_m": .25}, previous, changed)

    def test_insert_and_complete_preserve_original_goal_constraints_and_history(self):
        state = TaskState("a", active_constraints=[{"region_id": "existing"}], cancelled_goals=["old"])
        state.apply_update({"operation": "insert_subgoal", "subgoal": "b", "resume_goal": "a"})
        self.assertEqual(state.active_goal, "b")
        self.assertEqual(state.suspended_goals, ["a"])
        self.assertEqual(state.cancelled_goals, ["old"])
        self.assertEqual(state.active_constraints, [{"region_id": "existing"}])
        state.complete_subgoal("b")
        self.assertEqual(state.active_goal, "a")
        self.assertEqual(state.completed_subgoals, ["b"])
        self.assertEqual(state.suspended_goals, [])
        self.assertEqual(state.cancelled_goals, ["old"])

    def test_invalid_insert_and_wrong_or_repeated_completion_are_rejected(self):
        for update in ({"subgoal": "a"}, {"subgoal": ""}, {"subgoal": "b", "resume_goal": "wrong"}):
            state = TaskState("a")
            with self.subTest(update=update), self.assertRaises(ValueError):
                state.apply_update({"operation": "insert_subgoal", **update})
            self.assertEqual(state.active_goal, "a")
            self.assertEqual(state.suspended_goals, [])
        state = TaskState("a")
        with self.assertRaises(ValueError):
            state.complete_subgoal("a")
        state.apply_update({"operation": "insert_subgoal", "subgoal": "b"})
        with self.assertRaises(ValueError):
            state.apply_update({"operation": "insert_subgoal", "subgoal": "c"})
        with self.assertRaises(ValueError):
            state.complete_subgoal("a")
        state.complete_subgoal("b")
        with self.assertRaises(ValueError):
            state.complete_subgoal("b")

    def test_insertion_delivered_once_before_action(self):
        state = TaskState("a")
        engine = EventEngine([InterventionEvent("insert", 2, "B then A", {"operation": "insert_subgoal", "subgoal": "b"})])
        self.assertEqual(engine.step(1, state), [])
        self.assertEqual(len(engine.step(2, state)), 1)
        self.assertEqual(state.suspended_goals, ["a"])
        self.assertEqual(engine.step(3, state), [])
        self.assertEqual(state.suspended_goals, ["a"])

    def test_completion_requires_measured_distance_yaw_and_valid_visible_diagnostic(self):
        report = {"evaluated": True, "instance_visible": True, "labels_restored": True, "pose_unchanged": True}
        self.assertTrue(completion_is_valid(.2, .3, 7., 7.5, report))
        for distance, yaw, changed in ((None, 0, report), (.4, 0, report), (.2, 8, report),
                (float("nan"), 0, report), (.2, 0, {**report, "evaluated": False}),
                (.2, 0, {**report, "instance_visible": False}), (.2, 0, {**report, "labels_restored": False})):
            with self.subTest(distance=distance, yaw=yaw, report=changed):
                self.assertFalse(completion_is_valid(distance, .3, yaw, 7.5, changed))

    def test_goal_arrival_without_sequence_or_final_stop_cannot_succeed(self):
        evaluator = EpisodeEvaluator()
        state = TaskState("a")
        state.apply_update({"operation": "insert_subgoal", "subgoal": "b"})
        values = dict(task_state=state, stopped=True, goal_distances_m={"a": 4., "b": .1},
            success_radius_m=.3, intervention_step=2, first_compliant_action_step=2, path_length=4,
            steps=10, max_steps=20, constraint_violation_count=0)
        with self.assertRaisesRegex(ValueError, "measured sequence"):
            evaluator.evaluate_navigation(**values)
        self.assertFalse(evaluator.evaluate_navigation(**values, subgoal_sequence_satisfied=False).dynamic_success)
        # Even a claimed true sequence cannot pass with a pending original goal.
        self.assertFalse(evaluator.evaluate_navigation(**values, subgoal_sequence_satisfied=True).dynamic_success)
        state.complete_subgoal("b")
        values["goal_distances_m"] = {"a": .1, "b": 4}
        self.assertTrue(evaluator.evaluate_navigation(**values, subgoal_sequence_satisfied=True).dynamic_success)
        self.assertFalse(evaluator.evaluate_navigation(**{**values, "stopped": False}, subgoal_sequence_satisfied=True).dynamic_success)
        self.assertFalse(evaluator.evaluate_navigation(**{**values, "steps": 21}, subgoal_sequence_satisfied=True).dynamic_success)
        self.assertFalse(sequence_satisfied("a", "b", state, []))
        self.assertTrue(sequence_satisfied("a", "b", state, [{"goal_id": "b", "valid": True}]))


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires native imports; all rendering is fake")
class SubgoalNavigationTests(unittest.TestCase):
    def fixture(self, root, stack):
        from tests.test_terminal_view import TerminalNativeTests

        nav, base, scene, objects, instances = TerminalNativeTests().setup_navigation(root, stack)
        base["max_steps"] = 100
        base["initial_instruction"] = "Go to A."
        for goal, category in (("a", "bed"), ("b", "couch")):
            base["goal_metadata"][goal].update(category=category, description=f"the {category}")
        source = root/"base.json"
        write_json(source, base)
        reference = root/"reference"
        control = deepcopy(base)
        control["events"] = []
        nav.run(control, scene, 0, reference)
        # Synthetic scene/approval geometry is isolated from the real catalog.
        # The native identity relabel diagnostic itself runs against fake objects.
        centers = {"a": [2., .5, 2.], "b": [-2., .5, 2.]}
        stack.enter_context(patch.object(subgoal, "reviewed_centers", side_effect=lambda config, goals: {goal: centers[goal] for goal in goals}))
        stack.enter_context(patch.object(subgoal, "pathfinder_for", return_value=SimpleNamespace(is_navigable=lambda point: True)))
        preparing = SimpleNamespace(source_config=source, reference_dir=reference, subgoal="b", output_dir=root/"prepared")
        subgoal.prepare(preparing)
        args = SimpleNamespace(config=preparing.output_dir/"config.json", output_dir=root/"experiment", gpu_id=0)
        return nav, args, scene, objects, instances

    def test_full_three_condition_entry_measures_b_restores_a_and_ignore_keeps_a(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            nav, args, _, _, instances = self.fixture(Path(temporary), stack)
            self.assertEqual(subgoal.run_experiment(args), 0)
            report = load_json(args.output_dir/"summary.json")
            self.assertTrue(report["passed"])
            self.assertEqual([row["root_goal_success"] for row in report["conditions"]], [True]*3)
            self.assertEqual([row["dynamic_success"] for row in report["conditions"]], [True, True, False])
            self.assertEqual([sim.initializations for sim in instances], [1]*4)
            rows = load_json(args.output_dir/"intervention/trajectory.json")
            complete = [index for index, row in enumerate(rows) if row["action"] == COMPLETE_SUBGOAL]
            self.assertEqual(len(complete), 1)
            index = complete[0]
            self.assertEqual(rows[index]["position"], rows[index-1]["position"])
            self.assertEqual(rows[index]["rotation_xyzw"], rows[index-1]["rotation_xyzw"])
            self.assertLess(index, len(rows)-1)
            self.assertEqual(rows[index-1]["active_goal"], "b")
            self.assertEqual(rows[index]["active_goal"], "a")
            self.assertEqual(rows[-1]["action"], "STOP")
            self.assertEqual(load_json(args.output_dir/"intervention/final_task_state.json")["cancelled_goals"], [])
            ignored = load_json(args.output_dir/"ignore/metrics.json")
            self.assertTrue(ignored["root_goal_success"])
            self.assertFalse(ignored["point_goal_success"])
            self.assertEqual(ignored["termination"], "subgoal_sequence_incomplete")
            self.assertIsNone(ignored["response_latency_steps"])
            before = sha256(args.output_dir/"summary.json")
            self.assertTrue(subgoal.check_experiment(args.output_dir)["passed"])
            self.assertEqual(sha256(args.output_dir/"summary.json"), before)
            with self.assertRaises(FileExistsError):
                subgoal.run_experiment(args)

    def test_failed_or_invisible_b_evidence_never_advances_evaluator_truth(self):
        for mode in ("error", "invisible", "pose_error"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
                import numpy as np

                nav, args, _, _, _ = self.fixture(Path(temporary), stack)
                original = nav.terminal_view.measure_visibility

                def broken(sim, agent, target, minimum):
                    if target["template_id"] == "couch" and agent.get_state().position[0] < -1:
                        if mode == "pose_error":
                            agent.get_state().position[0] -= .25
                            raise RuntimeError("Synthetic diagnostic moved the agent")
                        if mode == "error":
                            raise RuntimeError("Synthetic failed B diagnostic")
                        report, mask, rgb = original(sim, agent, target, minimum)
                        report.update(instance_visible=False, instance_pixels=0)
                        return report, np.zeros_like(mask), rgb
                    return original(sim, agent, target, minimum)

                stack.enter_context(patch.object(nav.terminal_view, "measure_visibility", side_effect=broken))
                self.assertEqual(subgoal.run_experiment(args), 1)
                truth = load_json(args.output_dir/"intervention/final_task_state.json")
                belief = load_json(args.output_dir/"intervention/final_policy_state.json")
                self.assertEqual(truth["active_goal"], "b")
                self.assertEqual(truth["suspended_goals"], ["a"])
                self.assertEqual(truth["completed_subgoals"], [])
                self.assertEqual(belief["active_goal"], "a")
                self.assertEqual(belief["completed_subgoals"], ["b"])
                metrics = load_json(args.output_dir/"intervention/metrics.json")
                self.assertTrue(metrics["root_goal_success"])
                self.assertFalse(metrics["dynamic_success"])
                self.assertFalse(metrics["subgoal_completed"])
                self.assertTrue((args.output_dir/"ignore/metrics.json").is_file())
                if mode == "pose_error":
                    receipt = load_json(args.output_dir/"intervention/subgoal_events.json")[0]
                    self.assertNotEqual(receipt["pose_before"], receipt["pose_after"])

    def test_budget_exhausted_immediately_after_b_completion_is_not_success(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            nav, args, scene, _, _ = self.fixture(Path(temporary), stack)
            self.assertEqual(subgoal.run_experiment(args), 0)
            completion = load_json(args.output_dir/"intervention/subgoal_events.json")[0]["step"]
            config = load_json(args.config)
            config["max_steps"] = completion+1
            metrics = nav.run(config, scene, 0, Path(temporary)/"timed_out")
            self.assertTrue(metrics["subgoal_completed"])
            self.assertTrue(metrics["resumed_original_goal"])
            self.assertFalse(metrics["stopped"])
            self.assertFalse(metrics["root_goal_success"])
            self.assertFalse(metrics["dynamic_success"])
            self.assertEqual(metrics["termination"], "timeout")

    def test_completing_b_then_stopping_without_returning_to_a_fails(self):
        """An actual final STOP at B cannot replace returning to the restored A."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            nav, args, scene, _, instances = self.fixture(Path(temporary), stack)
            original = nav.habitat_sim.nav.GreedyGeodesicFollower

            class StopAfterB(original):
                def next_action_along(self, goal):
                    if goal[0] > 0 and self.agent.get_state().position[0] < -1:
                        return "STOP"
                    return super().next_action_along(goal)

            stack.enter_context(patch.object(nav.habitat_sim.nav, "GreedyGeodesicFollower", StopAfterB))
            metrics = nav.run(load_json(args.config), scene, 0, Path(temporary)/"only_b")
            self.assertTrue(metrics["subgoal_completed"])
            self.assertTrue(metrics["resumed_original_goal"])
            self.assertTrue(metrics["stopped"])
            self.assertFalse(metrics["root_goal_success"])
            self.assertFalse(metrics["sequence_success"])
            self.assertFalse(metrics["dynamic_success"])
            self.assertEqual(metrics["termination"], "stopped_outside_goal")
            self.assertTrue(all(sim.initializations == 1 for sim in instances))

    def test_completion_claim_or_mask_tamper_is_rejected_by_raw_evidence_audit(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            _, args, _, _, _ = self.fixture(Path(temporary), stack)
            self.assertEqual(subgoal.run_experiment(args), 0)
            path = args.output_dir/"intervention/subgoal_events.json"
            receipts = load_json(path)
            original = deepcopy(receipts)
            receipts[0]["distance_m"] += 1
            write_json(path, receipts)
            self.assertFalse(subgoal.check_experiment(args.output_dir)["passed"])
            write_json(path, original)
            trajectory_path = args.output_dir/"intervention/trajectory.json"
            rows = load_json(trajectory_path)
            original_rows = deepcopy(rows)
            next(row for row in rows if row["action"] == COMPLETE_SUBGOAL)["action"] = "STOP"
            write_json(trajectory_path, rows)
            self.assertFalse(subgoal.check_experiment(args.output_dir)["passed"])
            write_json(trajectory_path, original_rows)
            mask = args.output_dir/"intervention"/receipts[0]["diagnostic_dir"]/"terminal_instance_mask.npy"
            mask.write_bytes(b"tampered stage mask")
            self.assertFalse(subgoal.check_experiment(args.output_dir)["passed"])

    def test_protocol_required_and_config_tamper_fails_before_new_simulator(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            nav, args, _, _, instances = self.fixture(Path(temporary), stack)
            config = load_json(args.config)
            config.pop("subgoal_protocol")
            with self.assertRaisesRegex(ValueError, "explicit subgoal_protocol"):
                nav.validate_config(config)
            args.config.write_text(args.config.read_text()+"\n")
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                subgoal.run_experiment(args)
            self.assertEqual(len(instances), 1)
            self.assertFalse(args.output_dir.exists())
