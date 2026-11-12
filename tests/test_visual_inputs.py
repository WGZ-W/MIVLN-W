"""Causal RGB navigation inputs, action outputs and actual marker-byte checks."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln.visual_frames import RGBFrame, mark_frame, validate_png
from mivln.visual_inputs import VisualInputSession, call_policy
from mivln.visual_messages import action_schema, parse_action, request_schema, seal_request, validate_action, validate_request
from scripts import visual_inputs as cli


CAMERA = {"width": 8, "height": 8, "fx": 4, "fy": 4, "cx": 3.5, "cy": 3.5}


def session(profile="V_RGB_v1", **kwargs):
    result = VisualInputSession("Go to the wooden island.", CAMERA, observation_profile=profile,
                                session_id="0"*32, **kwargs)
    result.initialize(frame(result, 0))
    return result


def frame(owner, step, pixels=None, identity=None):
    return RGBFrame.collect(owner.session_id, step, 8, 8, pixels if pixels is not None else bytes([step, 40, 60]) * 64,
                            identity if identity is not None else f"{step+1:032x}")


def event(step=1, operation="replace_goal"):
    return {"event_id": "PRIVATE_identity", "trigger": {"type": "step", "value": step},
            "utterance": "Go to the dark sofa instead.", "task_update": {"operation": operation,
            "new_goal": "PRIVATE_B", "region": {"region_id": "PRIVATE_R", "bounds_xz": [[0, 0], [1, 1]]}}}


def advance(owner, action="move_forward", collided=False):
    request = owner.request()
    envelope = parse_action(json.dumps({"action": action}), request)
    after = None if action == "STOP" else frame(owner, owner.step+1,
        owner._frames[owner.step].pixels if action == "COMPLETE_SUBGOAL" else None)
    owner.advance(envelope, after, collided=collided)
    return request, envelope


class VisualHistoryTests(unittest.TestCase):
    def test_request_contains_four_core_inputs_and_no_symbolic_answers(self):
        owner = session()
        initial = owner.request()
        self.assertEqual(initial["initial_instruction"], "Go to the wooden island.")
        self.assertEqual(initial["history"], {"observations": [], "actions": []})
        self.assertIsNone(initial["current_intervention"])
        data = owner.assets_for(initial["request_id"])
        resource = initial["observation"]["rgb"]
        self.assertEqual(hashlib.sha256(data[resource["asset_id"]]).hexdigest(), resource["sha256"])
        validate_png(data[resource["asset_id"]], 8, 8)
        for name in ("goal_positions", "previous_state", "pose", "reference", "task_update", "candidate_goals"):
            self.assertNotIn(name, json.dumps(initial))

    def test_new_intervention_moves_to_history_exactly_after_its_action(self):
        owner = session()
        advance(owner, collided=True)
        owner.deliver(event())
        current = owner.request()
        self.assertEqual(current["current_intervention"]["delivered_step"], 1)
        self.assertEqual(current["message_history"], [])
        self.assertEqual(owner.request(), current)
        self.assertTrue(current["observation"]["collided"])
        advance(owner, "turn_left")
        after = owner.request()
        self.assertIsNone(after["current_intervention"])
        self.assertEqual(after["message_history"], [current["current_intervention"]])
        self.assertEqual([row["step"] for row in after["history"]["actions"]], [0, 1])

    def test_historical_frames_use_fixed_stride_window_and_actual_action_alignment(self):
        owner = session(max_history_frames=2, history_stride=2)
        old = {}
        for k in range(5):
            request, _ = advance(owner, "turn_right" if k % 2 else "move_forward")
            old[k] = request["observation"]["rgb"]
        request = owner.request()
        self.assertEqual([row["step"] for row in request["history"]["observations"]], [2, 4])
        self.assertEqual([row["rgb"] for row in request["history"]["observations"]], [old[2], old[4]])
        self.assertEqual(len(request["history"]["actions"]), 5)
        self.assertEqual(set(owner.assets_for(request["request_id"])), {
            request["observation"]["rgb"]["asset_id"], old[2]["asset_id"], old[4]["asset_id"]})

    def test_zero_history_frames_does_not_erase_actual_action_or_message_history(self):
        owner = session(max_history_frames=0)
        advance(owner)
        owner.deliver(event())
        advance(owner)
        request = owner.request()
        self.assertEqual(request["history"]["observations"], [])
        self.assertEqual(len(request["history"]["actions"]), 2)
        self.assertEqual(len(request["message_history"]), 1)

    def test_private_update_labels_and_event_ids_do_not_change_model_input(self):
        views = []
        for correct_goal in ("PRIVATE_B", "PRIVATE_C"):
            owner = session()
            advance(owner)
            private = event()
            private["event_id"] = "private_" + correct_goal
            private["task_update"]["new_goal"] = correct_goal
            private["reference"] = {"goal": correct_goal, "future_trigger": 99}
            owner.deliver(private)
            views.append(owner.request())
        self.assertEqual(views[0], views[1])

    def test_wrong_branch_future_frame_or_asset_reuse_does_not_advance(self):
        owner = session()
        request = owner.request()
        action = parse_action('{"action":"move_forward"}', request)
        for bad in (replace(frame(owner, 1), session_id="f"*32), frame(owner, 2),
                    replace(frame(owner, 1), asset_id=owner._frames[0].asset_id)):
            with self.assertRaises(ValueError):
                owner.advance(action, bad)
            self.assertEqual(owner.step, 0)
        owner.advance(action, frame(owner, 1))

    def test_event_is_due_once_and_cannot_retroactively_change_a_chosen_action(self):
        owner = session()
        with self.assertRaises(ValueError):
            owner.deliver(event())
        advance(owner)
        current = owner.request()
        with self.assertRaises(ValueError):
            owner.deliver(event())
        self.assertEqual(owner.request(), current)
        another = session()
        advance(another)
        another.deliver(event())
        with self.assertRaises(ValueError):
            another.deliver(event())

    def test_nested_privileged_fields_and_forged_past_timestamps_are_rejected(self):
        owner = session()
        advance(owner)
        original = owner.request()
        for target in ("request", "rgb", "action", "camera", "history"):
            value = deepcopy(original)
            location = {"request": value, "rgb": value["observation"]["rgb"],
                "action": value["history"]["actions"][0], "camera": value["camera"], "history": value["history"]}[target]
            location["reference"] = "PRIVATE_B"
            with self.subTest(target=target), self.assertRaises(ValueError):
                seal_request(value)
        value = deepcopy(original)
        value["history"]["observations"][0]["step"] = 1
        with self.assertRaises(ValueError):
            seal_request(value)
        value = deepcopy(original)
        value["current_intervention"] = {"delivered_step": 2, "message": "future"}
        with self.assertRaises(ValueError):
            seal_request(value)


class VisualActionTests(unittest.TestCase):
    def test_bound_action_cannot_be_reused_for_a_later_slot_or_other_session(self):
        owner = session()
        request, action = advance(owner)
        next_request = owner.request()
        with self.assertRaises(ValueError):
            validate_action(action, next_request)
        other = deepcopy(action)
        other["session_id"] = "1"*32
        with self.assertRaises(ValueError):
            validate_action(other, request)
        with self.assertRaises(ValueError):
            owner.assets_for(request["request_id"])

    def test_stop_and_budget_close_the_session_without_extra_choices(self):
        owner = session()
        advance(owner, "STOP")
        self.assertEqual(owner.step, 1)
        self.assertTrue(owner.closed)
        with self.assertRaises(ValueError):
            owner.request()
        limited = session(max_steps=1)
        advance(limited)
        self.assertTrue(limited.closed)
        with self.assertRaises(ValueError):
            limited.request()

    def test_subgoal_completion_counts_without_rgb_advance_or_truth_feedback(self):
        owner = session(max_steps=1)
        request = owner.request()
        action = parse_action('{"action":"COMPLETE_SUBGOAL"}', request)
        with self.assertRaises(ValueError):
            owner.advance(action, frame(owner, 1))
        owner.advance(action, frame(owner, 1, owner._frames[0].pixels))
        self.assertEqual(owner.step, 1)
        self.assertTrue(owner.closed)
        self.assertFalse(hasattr(owner, "task_state"))

    def test_invalid_model_outputs_are_not_repaired_or_replaced_by_oracle_actions(self):
        for raw in ('{"action":"teleport"}', '{"action":"STOP","new_goal":"b"}',
                    '{"action":"STOP","action":"move_forward"}', '```json\n{"action":"STOP"}\n```', {"action": "STOP"}):
            owner = session()
            record = call_policy(owner, lambda request, assets: raw)
            self.assertIsNone(record["action"])
            self.assertTrue(record["error"])
            self.assertTrue(owner.closed)
            self.assertEqual(owner.step, 0)
            if isinstance(raw, str):
                self.assertEqual(record["raw_response"], raw)

    def test_callback_gets_owned_snapshot_and_only_permitted_png_bytes(self):
        owner = session()
        def callback(request, assets):
            self.assertEqual(len(assets), 1)
            request["initial_instruction"] = "tampered"
            assets.clear()
            return '{"action":"turn_left"}'
        record = call_policy(owner, callback)
        self.assertIsNone(record["error"])
        self.assertEqual(record["request"]["initial_instruction"], owner.initial_instruction)
        self.assertEqual(len(owner.assets_for(record["request"]["request_id"])), 1)

    def test_callback_error_is_recorded_once_without_retry(self):
        owner = session()
        with patch("builtins.print"):
            callback = unittest.mock.Mock(side_effect=OSError("offline"))
            record = call_policy(owner, callback)
        callback.assert_called_once()
        self.assertIsNone(record["action"])
        self.assertIn("offline", record["error"])
        self.assertTrue(owner.closed)


class MarkerVisibilityTests(unittest.TestCase):
    def marked(self, owner, indices=range(16)):
        return mark_frame(owner._frames[owner.step], "PRIVATE_R", indices, asset_id="f"*32)

    def test_historical_hssd_style_unmarked_avoid_message_is_not_visual_eligibility(self):
        for profile in ("V_RGB_v1", "V_RGB_MARKED_v1"):
            owner = session(profile)
            advance(owner)
            with self.assertRaises(ValueError):
                owner.deliver(event(operation="add_avoid_region"))
            self.assertFalse(owner._delivered)

    def test_marker_reaches_actual_callback_png_and_only_private_evidence_has_geometry_identity(self):
        owner = session("V_RGB_MARKED_v1")
        advance(owner)
        marked, proof = self.marked(owner)
        original = owner._frames[1]
        owner.deliver(event(operation="add_avoid_region"), marked_frame=marked, marker_evidence=proof)
        request = owner.request()
        resource = request["observation"]["rgb"]
        self.assertEqual(resource, marked.resource())
        assets = owner.assets_for(request["request_id"])
        pixels = validate_png(assets[resource["asset_id"]], 8, 8)
        self.assertEqual(pixels[:48], bytes([255, 0, 0])*16)
        self.assertNotIn(original.asset_id, assets)
        self.assertNotIn("PRIVATE_R", json.dumps(request))
        self.assertNotIn("bounds_xz", json.dumps(request))
        self.assertFalse(owner.marker_audits[0]["world_projection_verified"])

    def test_marker_proof_rejects_tampering_invisible_pixels_wrong_region_or_cross_step(self):
        variants = ("bytes", "count", "identity", "step", "wrong_hash", "color")
        for kind in variants:
            owner = session("V_RGB_MARKED_v1")
            advance(owner)
            marked, proof = self.marked(owner, range(8) if kind == "count" else range(16))
            if kind == "bytes":
                marked = replace(marked, pixels=bytes([0, 0, 0])*64)
            elif kind == "identity":
                proof = replace(proof, region_id="other")
            elif kind == "step":
                marked = replace(marked, step=2)
            elif kind == "wrong_hash":
                proof = replace(proof, marked_sha256="0"*64)
            elif kind == "color":
                marked, proof = mark_frame(owner._frames[owner.step], "PRIVATE_R", range(16), (0, 0, 255), asset_id="f"*32)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                owner.deliver(event(operation="add_avoid_region"), marked_frame=marked, marker_evidence=proof)
            self.assertFalse(owner._delivered)

    def test_png_checks_metadata_corruption_dimensions_and_non_rgb_bytes(self):
        owner = session()
        data = owner._frames[0].png()
        self.assertEqual(validate_png(data, 8, 8), owner._frames[0].pixels)
        for bad, width, height in ((data+b"metadata", 8, 8), (data[:-1], 8, 8),
                                  (data, 7, 8), (b"private labels", 8, 8)):
            with self.assertRaises(ValueError):
                validate_png(bad, width, height)


class VisualExportTests(unittest.TestCase):
    def test_schema_snapshots(self):
        root = Path(__file__).resolve().parents[1] / "schemas"
        self.assertEqual(json.loads((root / "benchmark_visual_request_v1.schema.json").read_text()), request_schema())
        self.assertEqual(json.loads((root / "benchmark_visual_action_v1.schema.json").read_text()), action_schema())

    def test_cpu_demo_checks_actual_assets_actions_and_history(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp) / "demo"
            self.assertEqual(cli.demo(SimpleNamespace(output_dir=output)), 0)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["planned_cases"], 3)
            self.assertEqual(report["requests"], 11)
            self.assertFalse(report["task_success_evaluated"])
            self.assertFalse(report["hssd_rendered"])
            stats = cli.validate_files(output / "public/requests.jsonl", output / "public/assets", output / "public/actions.jsonl")
            self.assertEqual(stats["sessions"], 3)
            self.assertTrue(stats["actions_checked"])
            with self.assertRaises(FileExistsError):
                cli.demo(SimpleNamespace(output_dir=output))
            asset = next((output / "public/assets").glob("*.png"))
            asset.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                cli.validate_files(output / "public/requests.jsonl", output / "public/assets")

    def test_demo_failed_policy_keeps_three_planned_cases_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp) / "demo"
            with patch.object(cli, "scripted_policy", return_value="invalid"):
                self.assertEqual(cli.demo(SimpleNamespace(output_dir=output)), 1)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["planned_cases"], 3)
            self.assertEqual(report["completed_cases"], 0)
            self.assertEqual(report["actions"], 0)
            self.assertEqual(len((output / "private/cases.jsonl").read_text().splitlines()), 3)
            self.assertEqual(cli.validate_files(output / "public/requests.jsonl", output / "public/assets")["sessions"], 3)

    def test_independent_validator_rejects_rewritten_historical_images_and_messages(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp) / "demo"
            cli.demo(SimpleNamespace(output_dir=output))
            original = [json.loads(line) for line in (output / "public/requests.jsonl").read_text().splitlines()]
            path = output / "altered.jsonl"
            for kind in ("image", "message", "action"):
                requests = deepcopy(original)
                if kind == "image":
                    requests[1]["history"]["observations"][0]["rgb"] = requests[2]["observation"]["rgb"]
                elif kind == "message":
                    requests[2]["message_history"][0]["message"] = "A rewritten past instruction."
                else:
                    requests[2]["history"]["actions"][0]["action"] = "turn_right"
                index = 1 if kind == "image" else 2
                requests[index] = seal_request(requests[index])
                cli.write_lines(path, requests)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    cli.validate_files(path, output / "public/assets")


if __name__ == "__main__":
    unittest.main()
