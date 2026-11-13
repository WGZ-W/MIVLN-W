"""Cancel-task semantics, pose-based scoring and versioned V delivery; no GPU."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln import EventEngine, InterventionEvent, TaskState
from mivln.cancel_task import (CancellationTaskState, CancellationVisualSession,
    evaluate_cancellation, validate_config)
from mivln.visual_frames import RGBFrame
from mivln.visual_inputs import VisualInputSession
from mivln.visual_messages import parse_action
from scripts import cancel_task as cli
from scripts.benchmark_audit import snapshot


def evidence(actions, received=True):
    """Actual deterministic line poses, with cancellation before action 2."""
    config = cli.fixture_config()
    point = [0., 0., 0.]
    rotation = [0., 0., 0., 1.]
    trajectory = [{"step": None, "action": "INITIALIZE", "position": point[:], "rotation_xyzw": rotation[:],
                   "distance_m": 0., "collided": False, "delivered_event_ids": []}]
    events = []
    for k, action in enumerate(actions):
        before = {"position": point[:], "rotation_xyzw": rotation[:]}
        if received and k == 2:
            events.append({"event_id": "private_cancel", "delivered_step": 2, "task_update": {"operation": "cancel_task"},
                           "pose_before": deepcopy(before), "pose_after": deepcopy(before)})
        if action == "move_forward":
            point[2] -= .25
        trajectory.append({"step": k, "action": action, "position": point[:], "rotation_xyzw": rotation[:],
            "distance_m": .25 if action == "move_forward" else 0., "collided": False,
            "delivered_event_ids": ["private_cancel"] if received and k == 2 else []})
    return config, trajectory, events


class CancellationStateTests(unittest.TestCase):
    def test_cancel_removes_unfinished_goals_but_preserves_completed_history_and_constraints(self):
        state = CancellationTaskState("b", suspended_goals=["a"], completed_subgoals=["done"],
                                      active_constraints=[{"region_id": "existing"}])
        state.apply_update({"operation": "cancel_task"})
        self.assertIsNone(state.active_goal)
        self.assertEqual(state.status, "cancelled")
        self.assertEqual(state.cancelled_goals, ["b", "a"])
        self.assertEqual(state.suspended_goals, [])
        self.assertEqual(state.completed_subgoals, ["done"])
        self.assertEqual(state.active_constraints, [{"region_id": "existing"}])
        for update in ({"operation": "cancel_task"}, {"operation": "replace_goal", "new_goal": "c"}):
            with self.assertRaises(ValueError):
                state.apply_update(update)
        with self.assertRaises(ValueError):
            state.complete_subgoal("b")

    def test_malformed_cancel_is_transactional_and_legacy_state_is_unchanged(self):
        state = CancellationTaskState("a")
        before = state.to_dict()
        for update in ({"operation": "cancel_task", "new_goal": "STOP"}, None):
            with self.assertRaises(ValueError):
                state.apply_update(update)
            self.assertEqual(state.to_dict(), before)
        with self.assertRaises(ValueError):
            TaskState("a").apply_update({"operation": "cancel_task"})

    def test_event_engine_delivers_once_before_action_without_auto_stopping(self):
        event = InterventionEvent.from_dict(cli.fixture_config()["events"][0])
        engine = EventEngine([event])
        state = CancellationTaskState("a")
        self.assertEqual(engine.step(1, state), [])
        self.assertEqual(len(engine.step(2, state)), 1)
        self.assertEqual(state.status, "cancelled")
        self.assertEqual(engine.step(2, state), [])
        self.assertEqual(engine.step(3, state), [])


class CancellationEvaluationTests(unittest.TestCase):
    def test_immediate_stop_succeeds_far_from_initial_object_without_visibility(self):
        config, rows, events = evidence(["move_forward", "move_forward", "STOP"])
        config.pop("goal_positions")
        result = evaluate_cancellation(config, rows, events)
        self.assertTrue(result["evidence_valid"])
        self.assertTrue(result["cancellation_success"])
        self.assertEqual(result["stop_delay_actions"], 0)
        self.assertEqual(result["post_cancel_action_count"], 1)
        self.assertEqual(result["post_cancel_path_length_m"], 0.)
        self.assertIsNone(result["task_state"]["active_goal"])

    def test_ignore_reaching_original_goal_cannot_replace_cancel_compliance(self):
        result = evaluate_cancellation(*evidence(["move_forward"]*4 + ["STOP"]))
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["cancellation_success"])
        self.assertEqual(result["stop_delay_actions"], 2)
        self.assertEqual(result["post_cancel_path_length_m"], .5)

    def test_delayed_turn_stop_fails_even_without_translation(self):
        result = evaluate_cancellation(*evidence(["move_forward", "move_forward", "turn_left", "STOP"]))
        self.assertFalse(result["cancellation_success"])
        self.assertEqual(result["stop_delay_actions"], 1)
        self.assertEqual(result["post_cancel_path_length_m"], 0.)

    def test_net_displacement_cannot_hide_leaving_and_returning(self):
        config, rows, events = evidence(["move_forward"]*4 + ["STOP"])
        rows[4]["position"] = rows[2]["position"][:]
        rows[5]["position"] = rows[2]["position"][:]
        result = evaluate_cancellation(config, rows, events)
        self.assertTrue(result["evidence_valid"])
        self.assertEqual(result["post_cancel_net_displacement_m"], 0.)
        self.assertEqual(result["post_cancel_path_length_m"], .5)
        self.assertFalse(result["cancellation_success"])

    def test_early_stop_no_stop_and_over_budget_remain_failures(self):
        early = evaluate_cancellation(*evidence(["STOP"], received=False))
        self.assertTrue(early["evidence_valid"])
        self.assertFalse(early["cancellation_received"])
        self.assertFalse(early["cancellation_success"])
        config, rows, events = evidence(["move_forward"]*8)
        no_stop = evaluate_cancellation(config, rows, events)
        self.assertEqual(no_stop["failure_reason"], "no_stop")
        config["max_steps"] = 3
        result = evaluate_cancellation(config, rows, events)
        self.assertFalse(result["budget_satisfied"])
        self.assertFalse(result["cancellation_success"])

    def test_missed_duplicated_or_reset_event_cannot_be_accepted(self):
        original = evidence(["move_forward", "move_forward", "STOP"])
        for kind in ("missing", "duplicate", "reset", "wrong_step", "bad_per_action_id"):
            config, rows, events = deepcopy(original)
            if kind == "missing":
                events.clear()
            elif kind == "duplicate":
                events.append(deepcopy(events[0]))
            elif kind == "reset":
                events[0]["pose_after"]["position"][0] += 1
            elif kind == "wrong_step":
                events[0]["delivered_step"] = 1
            else:
                rows[1]["delivered_event_ids"] = ["private_cancel"]
            with self.subTest(kind=kind):
                result = evaluate_cancellation(config, rows, events)
                self.assertFalse(result["evidence_valid"])
                self.assertFalse(result["cancellation_success"])

    def test_stop_teleport_rotation_or_invalid_distance_is_bad_evidence(self):
        for kind in ("teleport", "rotation", "nan", "missing_trajectory", "after_stop"):
            config, rows, events = evidence(["move_forward", "move_forward", "STOP"])
            if kind == "teleport":
                rows[-1]["position"][0] += 1
                rows[-1]["distance_m"] = 1
            elif kind == "rotation":
                rows[-1]["rotation_xyzw"] = [0., 1., 0., 0.]
            elif kind == "nan":
                rows[-1]["distance_m"] = float("nan")
            elif kind == "missing_trajectory":
                rows.clear()
            else:
                extra = deepcopy(rows[-1])
                extra["step"] = 3
                rows.append(extra)
            with self.subTest(kind=kind):
                result = evaluate_cancellation(config, rows, events)
                self.assertFalse(result["evidence_valid"])
                self.assertFalse(result["cancellation_success"])

    def test_completion_claim_is_not_a_cancel_stop(self):
        result = evaluate_cancellation(*evidence(["move_forward", "move_forward", "COMPLETE_SUBGOAL", "STOP"]))
        self.assertEqual(result["failure_reason"], "invalid_subgoal_completion")
        self.assertFalse(result["cancellation_success"])

    def test_invalid_configuration_is_rejected_before_execution(self):
        for change in ({"max_steps": True}, {"cancel_protocol": "legacy"}, {"subgoal_protocol": {"enabled": True}}):
            with self.assertRaises(ValueError):
                validate_config({**cli.fixture_config(), **change})


class CancellationVisualTests(unittest.TestCase):
    def test_v_receives_message_and_chooses_action_without_cancel_state_or_opcode(self):
        config = cli.fixture_config()
        owner = CancellationVisualSession(config["initial_instruction"],
            {"width": 8, "height": 8, "fx": 4, "fy": 4, "cx": 3.5, "cy": 3.5}, max_steps=8, session_id="0"*32)
        owner.initialize(RGBFrame.collect(owner.session_id, 0, 8, 8, b"\x00"*192))
        event = deepcopy(config["events"][0]); event["trigger"]["value"] = 0
        owner.deliver(event)
        request = owner.request()
        self.assertEqual(request["current_intervention"]["message"], event["utterance"])
        for secret in ("cancel_task", "private_cancel", "active_goal", "goal_positions"):
            self.assertNotIn(secret, json.dumps(request))
        # Ignoring is allowed to execute; the evaluator, not the input builder,
        # determines that this response violates the cancellation task.
        action = parse_action('{"action":"move_forward"}', request)
        owner.advance(action, RGBFrame.collect(owner.session_id, 1, 8, 8, b"\x01"*192))
        self.assertEqual(owner.step, 1)

    def test_legacy_visual_session_keeps_its_frozen_operation_scope(self):
        owner = VisualInputSession("Go to A.", {"width": 8, "height": 8, "fx": 4, "fy": 4, "cx": 3.5, "cy": 3.5})
        owner.initialize(RGBFrame.collect(owner.session_id, 0, 8, 8, b"\x00"*192))
        event = deepcopy(cli.fixture_config()["events"][0]); event["trigger"]["value"] = 0
        with self.assertRaises(ValueError):
            owner.deliver(event)


class CancellationEntryTests(unittest.TestCase):
    def test_three_condition_fixture_audits_raw_evidence_without_rewriting_sources(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp) / "fixture"
            self.assertEqual(cli.demo(SimpleNamespace(output_dir=root)), 0)
            before = snapshot({"fixture": root})
            report = cli.check_fixture(root)
            self.assertEqual(before, snapshot({"fixture": root}))
            self.assertTrue(report["passed"])
            follow = next(row for row in report["conditions"] if row["condition"] == "follow")
            ignore = next(row for row in report["conditions"] if row["condition"] == "ignore")
            self.assertTrue(follow["task_success"])
            self.assertFalse(follow["root_point_goal_success"])
            self.assertFalse(ignore["task_success"])
            self.assertTrue(ignore["root_point_goal_success"])
            cli.write_json(root / "summary.json", {"passed": False, "intentionally_stale": True})
            self.assertTrue(cli.check_fixture(root)["passed"])

    def test_forged_raw_response_or_public_message_is_detected(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp) / "fixture"
            cli.demo(SimpleNamespace(output_dir=root))
            path = root / "follow/policy_records.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[-1]["raw_response"] = '{"action":"move_forward"}'
            cli.write_lines(path, records)
            self.assertFalse(cli.check_fixture(root)["passed"])

    def source(self, root):
        from tests.test_language_navigation import source_fixture
        source = source_fixture(root)
        path = source / "tasks/task_001/reference/trajectory.json"
        rows = json.loads(path.read_text())
        for index, row in enumerate(rows):
            row["distance_m"] = 0 if index == 0 else math.dist(rows[index-1]["position"], row["position"])
        cli.write_json(path, rows)
        cli.write_json(source / "tasks/task_001/reference/terminal_diagnostics.json", {"fixture_only": True})
        return source

    def test_preparation_freezes_three_timings_and_preserves_failed_slots(self):
        from scripts import hssd_dataset
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            source = self.source(root)
            # This input fixture has no rendered payloads. Only the existing
            # image evidence checker is mocked; timing and bindings are real.
            with patch.object(hssd_dataset, "check_episode_records"), patch.object(hssd_dataset, "terminal_record", return_value={"passed": True}):
                args = SimpleNamespace(source_dir=source, output_dir=root / "prepared")
                before = snapshot({"source": source})
                self.assertEqual(cli.prepare(args), 0)
                self.assertEqual(before, snapshot({"source": source}))
                report = json.loads((args.output_dir / "manifest.json").read_text())
                self.assertEqual(report["planned_cases"], 3)
                self.assertEqual(report["expected_episodes"], 9)
                for case in report["cases"]:
                    config = json.loads((args.output_dir / case["config"]).read_text())
                    self.assertEqual(config["events"][0]["task_update"], {"operation": "cancel_task"})
                    self.assertEqual(len(config["goal_positions"]), 1)
                    self.assertEqual(case["language_review_status"], "not_recorded")
                args.output_dir = root / "duplicate_timings"
                with patch("scripts.subgoal.reference_trigger", return_value=1):
                    self.assertEqual(cli.prepare(args), 1)
                duplicate = json.loads((args.output_dir / "manifest.json").read_text())
                self.assertEqual(duplicate["planned_cases"], 3)
                self.assertEqual(duplicate["prepared_cases"], 0)
                (source / "tasks/task_001/reference/trajectory.json").unlink()
                args.output_dir = root / "failed"
                self.assertEqual(cli.prepare(args), 1)
                failed = json.loads((args.output_dir / "manifest.json").read_text())
                self.assertEqual(failed["planned_cases"], 3)
                self.assertEqual(failed["prepared_cases"], 0)
                self.assertEqual(len(failed["cases"]), 3)


if __name__ == "__main__":
    unittest.main()
