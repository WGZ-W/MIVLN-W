"""CPU-only public schema export, frozen U request export, and minimal G demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mivln.public_inputs import PublicInputSession
from mivln.public_messages import CONTRACT, message_schema, validate_message
from mivln.task_state import TaskState
from mivln.task_update import strict_json
from scripts.benchmark_manifest import PROJECT_ROOT, read_json, sha256, within


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def fresh_output(path, source=None):
    path = path.resolve()
    if source is not None:
        source = source.resolve()
        if path == source or source in path.parents or path in source.parents:
            raise ValueError("Public export must not overlap its private source")
    if path.exists():
        raise FileExistsError("Use a fresh output directory")
    return path


def export_replacement(args):
    """Export 3 frozen U requests per base, without running any policy.

    Each independent case has a declared, provided initial symbolic goal. The
    observed reference prefix only advances the delivery clock; it is private
    and never enters the updater request. No model trajectory is fabricated.
    All source checks finish before writing; invalid sources fail the complete
    export rather than reporting a filtered subset as a complete batch.
    """
    from scripts import hssd_dataset as source_checker
    from scripts.intervention_timing import check_reference, select_triggers
    from scripts.language_navigation import reviewed_candidates
    from mivln.navigation_updates import navigation_request

    source = args.source_dir.resolve()
    output = fresh_output(args.output_dir, source)
    consumed = {}

    def load(path):
        # Record bytes before reading, then compare again at the end. This is
        # a consumed-input check, not B0's full 9 GB evidence-tree comparison.
        path = Path(path).resolve()
        digest = sha256(path)
        if str(path) in consumed and consumed[str(path)] != digest:
            raise ValueError("A previously consumed source input changed")
        consumed[str(path)] = digest
        return read_json(path)

    manifest = load(source / "manifest.json")
    source_checker.validate_inputs(source, manifest)
    messages, requests, cases = [], [], []
    identities = [item["task_id"] for item in manifest["tasks"]]
    if len(identities) != len(set(identities)) or len(identities) != manifest["task_count"]:
        raise ValueError("Ambiguous canonical roster")
    for task in manifest["tasks"]:
        base = load(within(source, task["base_config"]))
        load(within(source, task["reference_config"]))
        # The reviewed catalog is an external dependency and is bound too.
        load(Path(base["catalog_source"]["path"]))
        destination = within(source, task["output_subdir"])
        reference = destination / "reference"
        for name in ("episode.json", "trajectory.json", "metrics.json"):
            load(reference / name)
        rows = check_reference(base, reference)
        triggers = {item["name"]: item["trigger_step"] for item in select_triggers(rows)}
        for timing in ("early", "middle", "late"):
            config_path = within(source, str((destination / f"timing/configs/{timing}.json").relative_to(source)))
            config = load(config_path)
            source_checker.same_task(base, config)
            event = config["events"][0]
            if (len(config["events"]) != 1 or event["trigger"]["value"] != triggers[timing]
                    or event["task_update"] != base["events"][0]["task_update"]
                    or event["utterance"] != base["events"][0]["utterance"]):
                raise ValueError("Frozen timing/update differs from the canonical task")
            goals = reviewed_candidates(config)
            session = PublicInputSession("U", config["max_steps"],
                context_source="provided_initial_goal_then_policy_rollout")
            messages.append(session.initialize(config, goals))
            for row in rows[1:triggers[timing]+1]:
                session.advance(row["action"])
            state = TaskState(config["initial_goal"])
            message = session.intervention(event, policy_state=state)
            request = message["payload"]["request"]
            if request != navigation_request(triggers[timing], event["utterance"], goals, state):
                raise ValueError("Export changed the existing navigation callback body")
            messages.append(message)
            requests.append(request)
            cases.append({"session_id": session.session_id, "source_task_id": task["task_id"],
                "timing": timing, "source_config": str(config_path), "config_sha256": consumed[str(config_path)],
                "request_id": request["request_id"], "trigger_step": triggers[timing]})
    expected = manifest["task_count"] * 3
    if len(requests) != expected:
        raise ValueError("Fixed export denominator changed")
    after = {name: sha256(name) for name in consumed}
    if after != consumed:
        raise ValueError("Consumed source files changed during public-view construction")
    output.mkdir(parents=True)
    write_json(output / "public/schema.json", message_schema())
    write_lines(output / "public/messages.jsonl", messages)
    write_lines(output / "public/requests.jsonl", requests)
    public = {"contract": CONTRACT, "track": "U", "mode": "state",
        "context_source": "provided_initial_goal_then_policy_rollout", "sessions": expected,
        "messages": len(messages), "requests": expected,
        "files": {name: sha256(output / "public" / name) for name in ("schema.json", "messages.jsonl", "requests.jsonl")}}
    write_json(output / "public/manifest.json", public)
    write_lines(output / "private/case_index.jsonl", cases)
    write_json(output / "private/source_bindings.json", {"before": consumed, "after": after, "unchanged": True})
    write_json(output / "private/code_bindings.json", {str(PROJECT_ROOT / name): sha256(PROJECT_ROOT / name)
        for name in ("mivln/public_inputs.py", "mivln/public_messages.py", "scripts/public_inputs.py",
                     "mivln/navigation_updates.py", "mivln/task_update.py", "scripts/language_navigation.py")})
    report = {"passed": True, "kind": "cpu_public_view_export", "expected_requests": expected,
        "exported_requests": len(requests), "source_inputs_unchanged": True,
        "legacy_callback_requests_equal": True, "model_called": False, "navigation_executed": False,
        "runtime_isolation_implemented": False, "private_sources": str(source),
        "notes": ["Independent fixed-context requests, not model-rollout success",
                  "Public messages contain reviewed free text; semantic content review is still required",
                  "Only public/ is a candidate public bundle; root/private files are evaluator-only",
                  "requests.jsonl bodies match callbacks; this is not a predict_task_updates batch manifest"]}
    write_json(output / "summary.json", report)
    print(f"U export: {len(requests)}/{expected} callback bodies equal; consumed inputs unchanged")
    return 0


def geometry_demo(args):
    """Synthetic messages only; coordinates are not HSSD feasibility evidence."""
    output = fresh_output(args.output_dir)
    config = {"initial_instruction": "去木质岛台。", "max_steps": 10, "start_position": [0, 0, 0],
        "start_rotation_xyzw": [0, 0, 0, 1], "goal_positions": {"a": [2, 0, 0], "b": [-2, 0, 0]},
        "forward_step_m": .25, "turn_angle_degrees": 15, "success_radius_m": .3, "agent_radius": .1,
        "events": [{"task_update": {"operation": "private_answer"}}], "reference": "private"}
    goals = [{"goal_id": identity, "description": description, "aliases": [description]}
             for identity, description in (("a", "木质岛台"), ("b", "深色沙发"))]
    session = PublicInputSession("G", 10)
    messages = [session.initialize(config, goals, map_ref={"asset_id": "1"*32, "sha256": "a"*64})]
    messages.append(session.advance("move_forward", {"session_id": session.session_id, "step": 1,
        "pose": {"position": [0, 0, -.25], "rotation_xyzw": [0, 0, 0, 1]}, "collided": False}))
    event = {"event_id": "private_avoid_case", "trigger": {"type": "step", "value": 1},
        "utterance": "继续去原目标，但不要经过标记区域。", "task_update": {"operation": "add_avoid_region"}}
    packet = {"session_id": session.session_id, "step": 1, "regions": [{
        "region": {"region_id": "r", "type": "rectangle_xz", "bounds_xz": [[0, -2], [1, -1]],
                   "floor_y": 0, "floor_tolerance_m": .5}, "description": "标记的地面区域", "aliases": ["标记区域"]}]}
    messages.append(session.intervention(event, region_packet=packet))
    output.mkdir(parents=True)
    write_json(output / "public/schema.json", message_schema())
    write_lines(output / "public/messages.jsonl", messages)
    write_json(output / "summary.json", {"passed": True, "kind": "synthetic_geometry_message_demo", "messages": 3,
        "map_resource_served": False, "model_called": False, "navigation_executed": False,
        "notes": ["Synthetic asset ID and digest only; no map is distributed", "Not HSSD or navigation performance evidence"]})
    print("G demo: initialize, observation, intervention validated; no navigation/model execution")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("schema", "demo", "export-replacement"):
        child = subparsers.add_parser(command)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "export-replacement":
            child.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "outputs/hssd_dataset_facing_001")
    child = subparsers.add_parser("validate")
    child.add_argument("--messages", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "export-replacement":
            return export_replacement(args)
        if args.command == "demo":
            return geometry_demo(args)
        if args.command == "schema":
            output = fresh_output(args.output_dir)
            output.mkdir(parents=True)
            write_json(output / "schema.json", message_schema())
            print(f"Schema: {output / 'schema.json'}")
        else:
            count = 0
            for line in args.messages.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    validate_message(strict_json(line))
                    count += 1
            if not count:
                raise ValueError("Empty message file")
            print(f"Validated {count} message bodies; this does not replay delivery or navigation")
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"Public input contract failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
