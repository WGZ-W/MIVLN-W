"""CPU-only V schemas, scripted RGB contract demos, and request/asset checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from mivln.task_update import strict_json
from mivln.visual_frames import RGBFrame, mark_frame, validate_png
from mivln.visual_inputs import VisualInputSession, call_policy
from mivln.visual_messages import CONTRACT, action_schema, request_schema, validate_action, validate_request
from scripts.public_inputs import fresh_output, write_json, write_lines


def scripted_policy(request, assets):
    """Known demo phrases/actions only, with no access to private task labels.

    This is an interface fixture, not a navigation or language-understanding
    baseline. It receives genuine PNG bytes but does not interpret the image.
    """
    for resource in [request["observation"]["rgb"], *[row["rgb"] for row in request["history"]["observations"]]]:
        if hashlib.sha256(assets[resource["asset_id"]]).hexdigest() != resource["sha256"]:
            raise ValueError("Callback asset bytes differ from the public request")
    messages = request["message_history"] + ([request["current_intervention"]] if request["current_intervention"] else [])
    text = " ".join(item["message"] for item in messages)
    step = request["step"]
    if not messages:
        action = "move_forward"
    elif "先去" in text:
        action = {1: "turn_left", 2: "COMPLETE_SUBGOAL", 3: "move_forward"}.get(step, "STOP")
    else:
        action = ("turn_right" if "红色" in text else "turn_left") if step == 1 else "STOP"
    return json.dumps({"action": action})


def synthetic_pixels(step, width=32, height=32):
    """Deterministic artificial floor/upper-image colors, not rendered HSSD."""
    return bytes(value for y in range(height) for x in range(width)
                 for value in ((40 + step * 5, 80, 120) if y < height // 2 else (110, 100 + (x % 8), 75)))


def demo(args):
    output = fresh_output(args.output_dir)
    requests, actions, records, cases, marker_logs = [], [], [], [], []
    public_assets, private_assets = {}, {}
    definitions = (("replacement", "取消原来的目的地，改去沙发。", {"operation": "replace_goal", "new_goal": "private_B"}),
        ("avoid", "继续去原目标，但避开红色标记的地面区域。", {"operation": "add_avoid_region", "region": {"region_id": "private_R"}}),
        ("subgoal", "先去沙发，然后返回原来的岛台。", {"operation": "insert_subgoal", "subgoal": "private_B"}))
    for name, message, private_update in definitions:
        session = VisualInputSession("去木质岛台。", {"width": 32, "height": 32, "fx": 16, "fy": 16, "cx": 15.5, "cy": 15.5},
            max_steps=8, observation_profile="V_RGB_MARKED_v1", max_history_frames=2)
        session.initialize(RGBFrame.collect(session.session_id, 0, 32, 32, synthetic_pixels(0)))
        event = {"event_id": f"private_{name}", "trigger": {"type": "step", "value": 1},
                 "utterance": message, "task_update": private_update}
        while not session.closed:
            if session.step == 1:
                if name == "avoid":
                    original = session._frames[session.step]
                    marked, evidence = mark_frame(original, "private_R", [y*32+x for y in range(20, 28) for x in range(4, 12)])
                    private_assets[original.asset_id] = original.png()
                    session.deliver(event, marked_frame=marked, marker_evidence=evidence)
                else:
                    session.deliver(event)
            # Preserve the input bytes even if the callback fails and closes the
            # session. Failed trials must remain auditable, not lose their RGB.
            pending = session.request()
            public_assets.update(session.assets_for(pending["request_id"]))
            record = call_policy(session, scripted_policy)
            records.append(record)
            requests.append(record["request"])
            # Keep exactly the bytes served for this request, including historic
            # assets, without copying the collector's unpublished raw frames.
            if record["error"]:
                break
            action = record["action"]
            actions.append(action)
            frame = None
            if action["action"] != "STOP":
                pixels = (session._frames[session.step].pixels if action["action"] == "COMPLETE_SUBGOAL"
                          else synthetic_pixels(session.step + 1))
                frame = RGBFrame.collect(session.session_id, session.step + 1, 32, 32, pixels)
            session.advance(action, frame)
        cases.append({"case": name, "session_id": session.session_id, "interface_completed": session.closed and not session.failure_reason,
                      "counted_actions": session.step, "task_success_evaluated": False, "error": session.failure_reason})
        marker_logs.extend({"session_id": session.session_id, **row} for row in session.marker_audits)
    output.mkdir(parents=True)
    write_json(output / "public/request_schema.json", request_schema())
    write_json(output / "public/action_schema.json", action_schema())
    write_lines(output / "public/requests.jsonl", requests)
    write_lines(output / "public/actions.jsonl", actions)
    for identity, data in public_assets.items():
        path = output / "public/assets" / f"{identity}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for identity, data in private_assets.items():
        path = output / "private/original_frames" / f"{identity}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_lines(output / "private/policy_records.jsonl", records)
    write_lines(output / "private/cases.jsonl", cases)
    write_lines(output / "private/marker_evidence.jsonl", marker_logs)
    report = {"passed": len(cases) == 3 and all(row["interface_completed"] for row in cases),
        "kind": "cpu_visual_contract_demo", "contract": CONTRACT, "planned_cases": 3,
        "completed_cases": sum(row["interface_completed"] for row in cases), "requests": len(requests),
        "actions": len(actions), "published_rgb_assets": len(public_assets),
        "task_success_evaluated": False, "hssd_rendered": False, "real_model_called": False,
        "world_projection_verified": False, "process_isolation_implemented": False,
        "notes": ["Scripted policy and synthetic pixels; not navigation performance",
                  "Marker proof binds actual RGB bytes, not a world-floor projection",
                  "A full public bundle is an offline artifact, not an online Agent filesystem"]}
    write_json(output / "summary.json", report)
    print(f"V CPU contracts: {report['completed_cases']}/3 scenarios; {len(requests)} requests, {len(actions)} actions")
    return 0 if report["passed"] else 1


def validate_files(requests_path, assets_dir, actions_path=None):
    """Independently check exported bytes and optional complete action traces."""
    requests = [validate_request(strict_json(line)) for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not requests:
        raise ValueError("Empty V request file")
    identities = [item["request_id"] for item in requests]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate V request IDs")
    by_id = {item["request_id"]: item for item in requests}
    action_map = None
    if actions_path is not None:
        action_map = {}
        for line in actions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            action = strict_json(line)
            identity = action["request_id"]
            if identity not in by_id or identity in action_map:
                raise ValueError("Unknown or duplicate action request binding")
            action_map[identity] = validate_action(action, by_id[identity])
        if set(action_map) != set(by_id):
            raise ValueError("Complete action trace is missing a response")
    previous, observed_frames = {}, {}
    for request in requests:
        session = request["session_id"]
        prior = previous.get(session)
        if request["step"] != (prior["step"] + 1 if prior else 0):
            raise ValueError("Request trace has skipped, reordered or cross-session steps")
        if prior:
            for field in ("initial_instruction", "camera", "observation_profile", "history_policy", "actions"):
                if request[field] != prior[field]:
                    raise ValueError("Frozen V input configuration changed")
            if {k: v for k, v in request["limits"].items() if k != "remaining_actions"} != {k: v for k, v in prior["limits"].items() if k != "remaining_actions"}:
                raise ValueError("Action limits changed during execution")
            if request["history"]["actions"][:-1] != prior["history"]["actions"]:
                raise ValueError("Historical executed actions were rewritten")
            messages = prior["message_history"] + ([prior["current_intervention"]] if prior["current_intervention"] else [])
            if request["message_history"] != messages:
                raise ValueError("Previously delivered human messages were rewritten")
            if action_map:
                action = action_map[prior["request_id"]]
                if action["action"] != request["history"]["actions"][-1]["action"] or action["action"] == "STOP":
                    raise ValueError("Next input does not follow the actual accepted action")
        past = observed_frames.setdefault(session, {})
        for row in request["history"]["observations"]:
            if past.get(row["step"]) != row["rgb"]:
                raise ValueError("Historical RGB was not the image actually supplied at that step")
        past[request["step"]] = request["observation"]["rgb"]
        for resource in [request["observation"]["rgb"], *[row["rgb"] for row in request["history"]["observations"]]]:
            path = assets_dir / f"{resource['asset_id']}.png"
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing or linked RGB asset")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != resource["sha256"]:
                raise ValueError("Missing, linked or changed RGB asset")
            validate_png(data, resource["width"], resource["height"])
        previous[session] = request
    if action_map:
        for request in previous.values():
            if action_map[request["request_id"]]["action"] != "STOP" and request["step"] + 1 < request["limits"]["max_steps"]:
                raise ValueError("Successful demo trace ended before STOP or budget exhaustion")
    return {"requests": len(requests), "sessions": len(previous), "actions_checked": action_map is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("schema", "demo"):
        child = sub.add_parser(command)
        child.add_argument("--output-dir", type=Path, required=True)
    child = sub.add_parser("validate")
    child.add_argument("--requests", type=Path, required=True)
    child.add_argument("--assets-dir", type=Path, required=True)
    child.add_argument("--actions", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            return demo(args)
        if args.command == "schema":
            output = fresh_output(args.output_dir)
            output.mkdir(parents=True)
            write_json(output / "request_schema.json", request_schema())
            write_json(output / "action_schema.json", action_schema())
            print(f"V schemas: {output}")
        else:
            print(validate_files(args.requests, args.assets_dir, args.actions))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"V contract failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
