"""Public input contracts: leakage, provenance, timing and read-only export."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln.navigation_updates import apply_navigation_prediction, navigation_request
from mivln.public_inputs import PublicInputSession, update_request
from mivln.public_messages import message_schema, validate_message
from mivln.task_state import TaskState
from mivln.task_update_prompts import fingerprint
from scripts import public_inputs as cli
from scripts.benchmark_manifest import sha256


def fixture():
    config = {"max_steps": 10, "initial_instruction": "Go to the bed.", "initial_goal": "a",
        "start_position": [0, 0, 0], "start_rotation_xyzw": [0, 0, 0, 1],
        "goal_positions": {"a": [2, 0, 0], "b": [-2, 0, 0], "c": [0, 0, 3]},
        "forward_step_m": .25, "turn_angle_degrees": 15, "success_radius_m": .3, "agent_radius": .1,
        "reference": {"answer": "b"}, "events": [{"utterance": "SECRET FUTURE"}],
        "source_config": "/private/answer.json", "terminal_diagnostics": {"instance_pixels": 999}}
    goals = [{"goal_id": identity, "description": name, "aliases": [name]}
             for identity, name in (("a", "bed"), ("b", "sofa"), ("c", "table"))]
    event = {"event_id": "private_replace_b", "trigger": {"type": "step", "value": 1},
        "utterance": "Go to the sofa instead.", "task_update": {"operation": "replace_goal", "new_goal": "b"}}
    return config, goals, event


def seal(message):
    message["message_id"] = "pub_" + fingerprint({key: value for key, value in message.items() if key != "message_id"})[:24]
    return message


class PublicInputContractTests(unittest.TestCase):
    def session(self, track="U", mode="state"):
        config, goals, event = fixture()
        options = {"mode": mode}
        if track == "U" and mode == "state":
            options["context_source"] = "provided_initial_goal_then_policy_rollout"
        session = PublicInputSession(track, 10, session_id="0" * 32, **options)
        init = session.initialize(config, goals, **({"map_ref": {"asset_id": "1"*32, "sha256": "a"*64}} if track == "G" else {}))
        return session, init, event

    def test_private_answers_future_events_and_paths_do_not_change_u_view(self):
        config, goals, event = fixture()
        outputs = []
        for changed in (False, True):
            candidate = deepcopy(config)
            current = deepcopy(event)
            if changed:
                candidate.update(initial_goal="b", reference={"answer": "c"}, source_config="/different/private",
                                 goal_positions={"c": [99, 88, 77]}, terminal_diagnostics={"instance_pixels": 0})
                candidate["events"] = [{"utterance": "DIFFERENT FUTURE", "trigger": 999}]
                current.update(event_id="completely_different_private_id", task_update={"operation": "insert_subgoal"})
            session = PublicInputSession("U", 10, session_id="0"*32,
                context_source="provided_initial_goal_then_policy_rollout")
            init = session.initialize(candidate, goals)
            session.advance("move_forward")
            outputs.append([init, session.intervention(current, policy_state=TaskState("c"))])
        self.assertEqual(outputs[0], outputs[1])
        encoded = json.dumps(outputs[0])
        for secret in ("SECRET FUTURE", "/private", "task_update", "goal_positions", "instance_pixels"):
            self.assertNotIn(secret, encoded)

    def test_existing_callback_body_remains_identical(self):
        session, _, event = self.session()
        session.advance("move_forward")
        message = session.intervention(event, policy_state=TaskState("c"))
        self.assertEqual(message["payload"]["request"], navigation_request(1, event["utterance"], fixture()[1], TaskState("c")))

    def test_wrong_predictions_abstention_and_call_failure_are_not_corrected(self):
        goals = fixture()[1]
        state = TaskState("a")
        first = update_request(1, "Go to the sofa instead.", goals, [], "state", state)
        wrong = json.dumps({"status": "ok", "task_update": {"operation": "replace_goal", "old_goal": "a", "new_goal": "c"}})
        state, _ = apply_navigation_prediction(state, first, wrong)
        for raw, error in ((json.dumps({"status": "abstain", "reason": "missing_context"}), None),
                           (None, "HTTP unavailable"), ("invalid JSON", None)):
            request = update_request(2, "Continue.", goals, [], "state", state)
            self.assertEqual(request["context"]["previous_state"]["active_goal"], "c")
            state, _ = apply_navigation_prediction(state, request, raw, call_error=error)
            self.assertEqual(update_request(3, "Continue.", goals, [], "state", state)["context"]["previous_state"]["active_goal"], "c")
        with self.assertRaises(ValueError):
            update_request(1, "Continue.", goals, [], "state")

    def test_modes_do_not_gain_other_context_from_initialization(self):
        for mode in ("latest", "history", "state"):
            session, init, event = self.session(mode=mode)
            self.assertNotIn("initial_instruction", init["payload"])
            session.advance("move_forward")
            request = session.intervention(event, policy_state=TaskState("a"))["payload"]["request"]
            expected = {"latest": set(), "history": {"initial_instruction", "message_history"}, "state": {"previous_state"}}
            self.assertEqual(set(request["context"]), expected[mode])
            if mode == "history":
                self.assertEqual(request["context"]["message_history"], [])

    def test_future_past_duplicate_and_after_stop_delivery_are_rejected(self):
        session, _, event = self.session()
        with self.assertRaises(ValueError):
            session.intervention(event, policy_state=TaskState("a"))
        session.advance("move_forward")
        past = deepcopy(event)
        past["trigger"]["value"] = 0
        with self.assertRaises(ValueError):
            session.intervention(past, policy_state=TaskState("a"))
        session.intervention(event, policy_state=TaskState("a"))
        with self.assertRaises(ValueError):
            session.intervention(event, policy_state=TaskState("a"))
        stopped, _, event = self.session()
        stopped.advance("STOP")
        with self.assertRaises(ValueError):
            stopped.intervention(event, policy_state=TaskState("a"))

    def test_budget_counts_completion_and_rejects_more_actions(self):
        session = PublicInputSession("U", 1, mode="latest")
        config, goals, _ = fixture()
        config["max_steps"] = 1
        session.initialize(config, goals)
        session.advance("COMPLETE_SUBGOAL")
        self.assertTrue(session.closed)
        self.assertEqual(session.step, 1)
        with self.assertRaises(ValueError):
            session.advance("STOP")

    def test_g_geometry_announcements_do_not_read_private_operation(self):
        session, init, event = self.session("G")
        observation = {"session_id": session.session_id, "step": 1,
            "pose": {"position": [0, 0, -.25], "rotation_xyzw": [0, 0, 0, 1]}, "collided": False}
        session.advance("move_forward", observation)
        packet = {"session_id": session.session_id, "step": 1, "regions": [{"region": {
            "region_id": "r", "type": "rectangle_xz", "bounds_xz": [[0, 0], [1, 1]], "floor_y": 0, "floor_tolerance_m": .5},
            "description": "marked region", "aliases": ["marked region"]}]}
        message = session.intervention(event, region_packet=packet)
        self.assertEqual(init["payload"]["goal_geometry"][0]["stop_position"], [2, 0, 0])
        self.assertEqual(message["payload"]["regions"][0]["region"]["bounds_xz"], [[0, 0], [1, 1]])
        self.assertNotIn("operation", json.dumps(message))

    def test_g_rejects_cross_session_future_or_extra_observation_fields(self):
        session, _, event = self.session("G")
        valid = {"session_id": session.session_id, "step": 1,
                 "pose": {"position": [0, 0, -.25], "rotation_xyzw": [0, 0, 0, 1]}, "collided": False}
        for patch_value in ({"session_id": "2"*32}, {"step": 2}, {"step": True},
                            {"rgb": {"path": "future/frame.png"}}, {"sequence_success": True}):
            with self.subTest(patch=patch_value), self.assertRaises(ValueError):
                session.advance("move_forward", {**valid, **patch_value})
            self.assertEqual(session.step, 0)
        session.advance("move_forward", valid)
        for packet in ({"session_id": "2"*32, "step": 1, "regions": []},
                       {"session_id": session.session_id, "step": 2, "regions": []}):
            with self.assertRaises(ValueError):
                session.intervention(event, region_packet=packet)
        session.intervention(event, region_packet={"session_id": session.session_id, "step": 1, "regions": []})

    def test_unauthorized_nested_fields_cannot_be_hidden_in_public_messages(self):
        session, init, event = self.session()
        session.advance("move_forward")
        message = session.intervention(event, policy_state=TaskState("a"))
        for target in ("payload", "request", "context", "candidate", "state"):
            mutated = deepcopy(message)
            request = mutated["payload"]["request"]
            location = {"payload": mutated["payload"], "request": request, "context": request["context"],
                "candidate": request["candidate_goals"][0], "state": request["context"]["previous_state"]}[target]
            location["reference"] = {"new_goal": "b"}
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_message(seal(mutated))
        for goal_id in ("../answer", "/private/source", "target:answer"):
            mutated = deepcopy(init)
            mutated["payload"]["candidate_goals"][0]["goal_id"] = goal_id
            with self.assertRaises(ValueError):
                validate_message(seal(mutated))

    def test_u_rejects_geometry_and_v_is_not_silently_implemented(self):
        session, _, event = self.session()
        with self.assertRaises(ValueError):
            session.advance("move_forward", {"pose": [0, 0, 0]})
        session.advance("move_forward")
        with self.assertRaises(ValueError):
            session.intervention(event, policy_state=TaskState("a"), region_packet={"regions": []})
        with self.assertRaises(ValueError):
            PublicInputSession("V", 10)
        with self.assertRaises(ValueError):
            PublicInputSession("U", 10)

    def test_bad_region_geometry_and_nested_truth_cannot_consume_delivery(self):
        session, _, event = self.session("G")
        session.advance("turn_left", {"session_id": session.session_id, "step": 1,
            "pose": {"position": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]}, "collided": False})
        entry = {"region": {"region_id": "r", "type": "rectangle_xz", "bounds_xz": [[0, 0], [1, 1]],
                 "floor_y": 0, "floor_tolerance_m": .5}, "description": "marked", "aliases": ["marked"]}
        for kind in ("inverted_bounds", "duplicate", "truth_in_region", "truth_in_aliases"):
            entries = [deepcopy(entry)]
            if kind == "inverted_bounds":
                entries[0]["region"]["bounds_xz"] = [[1, 1], [0, 0]]
            elif kind == "duplicate":
                entries.append(deepcopy(entry))
            elif kind == "truth_in_region":
                entries[0]["region"]["operation"] = "add_avoid_region"
            else:
                entries[0]["aliases"] = [{"reference": "answer"}]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                session.intervention(event, region_packet={"session_id": session.session_id, "step": 1, "regions": entries})
        session.intervention(event, region_packet={"session_id": session.session_id, "step": 1, "regions": [entry]})

    def test_map_resource_cannot_embed_private_file_paths(self):
        config, goals, _ = fixture()
        session = PublicInputSession("G", 10)
        for resource in ({"asset_id": "/private/navmesh", "sha256": "a"*64},
                         {"asset_id": "1"*32, "sha256": "a"*64, "path": "/private/navmesh"}):
            with self.assertRaises(ValueError):
                session.initialize(config, goals, map_ref=resource)
            self.assertFalse(session.initialized)
        session.initialize(config, goals, map_ref={"asset_id": "1"*32, "sha256": "a"*64})

    def test_geometry_checks_finite_positions_unit_rotations_and_boundaries(self):
        _, init, _ = self.session("G")
        for value in (float("nan"), float("inf"), True, "0", 10**1000):
            mutated = deepcopy(init)
            mutated["payload"]["pose"]["position"][0] = value
            with self.subTest(value=str(value)[:15]), self.assertRaises(ValueError):
                validate_message(seal(mutated))
        mutated = deepcopy(init)
        mutated["payload"]["pose"]["rotation_xyzw"] = [0, 0, 0, 0]
        with self.assertRaises(ValueError):
            validate_message(seal(mutated))

    def test_returned_views_and_host_mutations_cannot_change_session_state(self):
        config, goals, event = fixture()
        session = PublicInputSession("U", 10, context_source="model_rollout")
        init = session.initialize(config, goals)
        goals[0]["description"] = "changed host description"
        init["payload"]["candidate_goals"][0]["description"] = "changed output"
        config["initial_instruction"] = "changed host language"
        session.advance("turn_left")
        state = TaskState("a")
        message = session.intervention(event, policy_state=state)
        message["payload"]["request"]["context"]["previous_state"]["active_goal"] = "b"
        self.assertEqual(state.active_goal, "a")
        self.assertEqual(session._goals[0]["description"], "bed")


class PublicInputExportTests(unittest.TestCase):
    def test_checked_in_schema_matches_the_runtime_contract(self):
        path = Path(__file__).resolve().parents[1] / "schemas/benchmark_public_messages_v1.schema.json"
        self.assertEqual(json.loads(path.read_text()), message_schema())

    def source(self, root):
        from tests.test_language_navigation import source_fixture
        from scripts.intervention_timing import select_triggers
        source = source_fixture(root)
        config = json.loads((source / "configs/task_001/base.json").read_text())
        rows = json.loads((source / "tasks/task_001/reference/trajectory.json").read_text())
        for trigger in select_triggers(rows):
            variant = deepcopy(config)
            variant["events"][0]["trigger"]["value"] = trigger["trigger_step"]
            cli.write_json(source / f"tasks/task_001/timing/configs/{trigger['name']}.json", variant)
        return source

    def test_export_is_readonly_no_model_calls_and_keeps_private_bindings_separate(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            source = self.source(root)
            before = {str(path): sha256(path) for path in root.rglob("*") if path.is_file()}
            args = SimpleNamespace(source_dir=source, output_dir=root / "export")
            with patch("scripts.predict_task_updates.urlopen") as http:
                self.assertEqual(cli.export_replacement(args), 0)
            http.assert_not_called()
            for name, digest in before.items():
                self.assertEqual(sha256(name), digest)
            report = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(report["exported_requests"], 3)
            public = args.output_dir / "public"
            self.assertEqual(len((public / "messages.jsonl").read_text().splitlines()), 6)
            self.assertEqual(json.loads((public / "schema.json").read_text()), message_schema())
            for line in (public / "messages.jsonl").read_text().splitlines():
                validate_message(json.loads(line))
            text = "".join(path.read_text() for path in public.iterdir())
            self.assertNotIn(str(source), text)
            self.assertNotIn("case_id", (public / "messages.jsonl").read_text())
            self.assertTrue((args.output_dir / "private/case_index.jsonl").exists())
            with self.assertRaises(FileExistsError):
                cli.export_replacement(args)

    def test_missing_case_aborts_export_and_output_cannot_overlap_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root)
            with self.assertRaises(ValueError):
                cli.fresh_output(source / "public", source)
            (source / "tasks/task_001/timing/configs/late.json").unlink()
            args = SimpleNamespace(source_dir=source, output_dir=root / "export")
            with self.assertRaises(OSError):
                cli.export_replacement(args)
            self.assertFalse(args.output_dir.exists())

    def test_demo_is_only_synthetic_messages_not_navigation_evidence(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp) / "demo"
            self.assertEqual(cli.geometry_demo(SimpleNamespace(output_dir=output)), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["navigation_executed"])
            self.assertFalse(summary["map_resource_served"])
            self.assertEqual(summary["messages"], 3)
            for row in (output / "public/messages.jsonl").read_text().splitlines():
                validate_message(json.loads(row))


if __name__ == "__main__":
    unittest.main()
