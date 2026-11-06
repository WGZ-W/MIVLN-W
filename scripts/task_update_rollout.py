"""Export, run rules, predict and score feedback-aware sequential task updates.

Only predict contacts an explicitly configured model service. Public dialogues
freeze future messages, not future structured requests: those depend on each
branch's own earlier predictions. Score reconstructs that dependency in full.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError

from mivln.regions import validate_region
from mivln.task_update import candidates
from mivln.task_update_prompts import encode, fingerprint, system_prompt
from mivln.task_update_rollout import (MODES, PROTOCOL, iter_rollout, predict_rules,
                                      render_user, validate_dialogue)
from scripts import predict_task_updates as client
from scripts import task_update as base


DEFAULT_DATASET = base.ROOT / "configs/task_update_rollouts_v1.json"
PUBLIC_FILES = ("dialogues.jsonl", "executor_regions.json", "instructions.txt")
RUN_FILES = ("predictions.jsonl", "responses.jsonl", "model_inputs.jsonl", "trace.jsonl", "system_prompt.txt")
INSTRUCTIONS = base.INSTRUCTIONS + """

本实验是连续明确更新。此前用户消息是请求，是否实际生效以执行反馈为准。
history 的历史记录包含用户消息和执行反馈；state 提供本分支实际更新后的 previous_state。
执行反馈 prediction 是此前实际尝试的输出；outcome=applied 才应用该更新。
call_failed、prediction_invalid、application_failed、abstained 均不改变状态，不能把这些请求当成成功。
外部合成 complete_subgoal 通知不需要模型输出；completed 表示已恢复挂起目标，completion_failed 不改变状态。
这些通知只用于文本状态机实验，不表示真实导航已完成。只输出当前消息对应的三类更新之一。
"""


def lines(path):
    return [base.strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def modes_for(value):
    return list(MODES) if value == "all" else [value]


def export(args):
    # Labels are read ONLY on the export/evaluation side. The inference loader
    # below deliberately has no dependency on this module or dataset_path.
    from mivln.task_update_rollout_eval import public_dialogue, validate_dataset
    dataset = validate_dataset(base.read_json(args.dataset))
    dialogues = [public_dialogue(dataset, case) for case in dataset["dialogues"]]
    modes = modes_for(args.mode)
    output = base.fresh_output(args.output_dir)
    base.write_jsonl(output / "dialogues.jsonl", dialogues)
    base.write_json(output / "executor_regions.json", dataset["region_registry"])
    (output / "instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")
    updates = sum(event["type"] == "message" for dialogue in dialogues for event in dialogue["events"])
    completions = sum(event["type"] == "complete_subgoal" for dialogue in dialogues for event in dialogue["events"])
    manifest = {"schema_version": 1, "kind": "public_task_update_rollouts", "protocol": PROTOCOL,
        "dataset_path": str(args.dataset.resolve()), "dataset_sha256": client.sha256(args.dataset),
        "files_sha256": {name: client.sha256(output / name) for name in PUBLIC_FILES},
        "modes": modes, "dialogue_ids": [dialogue["dialogue_id"] for dialogue in dialogues],
        "planned_updates": updates * len(modes), "planned_completion_events": completions * len(modes)}
    base.write_json(output / "manifest.json", manifest)
    print(f"Exported {len(dialogues)} dialogues x {len(modes)} modes; {manifest['planned_updates']} updates; {manifest['planned_completion_events']} synthetic completions")
    return 0


def load_public(folder):
    manifest = base.read_json(folder / "manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "public_task_update_rollouts"
            or manifest.get("protocol") != PROTOCOL):
        raise ValueError("Unsupported public rollout export")
    if not manifest["modes"] or len(set(manifest["modes"])) != len(manifest["modes"]) or any(mode not in MODES for mode in manifest["modes"]):
        raise ValueError("Invalid frozen rollout modes")
    if set(manifest["files_sha256"]) != set(PUBLIC_FILES):
        raise ValueError("Public file roster changed")
    for name in PUBLIC_FILES:
        if client.sha256(folder / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Frozen public input changed: {name}")
    dialogues = [validate_dialogue(value) for value in lines(folder / "dialogues.jsonl")]
    identities = [dialogue["dialogue_id"] for dialogue in dialogues]
    if not identities or len(set(identities)) != len(identities) or identities != manifest["dialogue_ids"]:
        raise ValueError("Public dialogue roster changed")
    updates = sum(event["type"] == "message" for dialogue in dialogues for event in dialogue["events"])
    completions = sum(event["type"] == "complete_subgoal" for dialogue in dialogues for event in dialogue["events"])
    if (manifest["planned_updates"] != updates * len(manifest["modes"])
            or manifest["planned_completion_events"] != completions * len(manifest["modes"])):
        raise ValueError("Planned denominator changed")
    registry = base.read_json(folder / "executor_regions.json")
    if not isinstance(registry, dict):
        raise ValueError("Executor region registry must be an object")
    for dialogue in dialogues:
        if set(registry) != candidates(dialogue["candidate_regions"], "region_id"):
            raise ValueError("Executor registry differs from public region IDs")
    for identity, region in registry.items():
        if validate_region(region)["region_id"] != identity:
            raise ValueError("Executor region identity changed")
    instructions = (folder / "instructions.txt").read_text(encoding="utf-8")
    if not instructions.strip():
        raise ValueError("Instructions must be nonempty")
    return manifest, dialogues, registry, instructions


def payload_for(run, system, request):
    payload = client.payload_for(run["model"], system, request, run["temperature"], run["max_tokens"], run["thinking_mode"])
    payload["messages"][1]["content"] = render_user(request, run["history_format"])
    return payload


def generate(args, rule=False):
    """Generate all slots in causal order; later states cannot be pre-exported."""
    if (args.history_format not in ("timeline", "json") or not args.model.strip()
            or not math.isfinite(args.temperature) or args.temperature < 0
            or not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_tokens <= 0
            or args.thinking_mode not in ("off", "on", "default")):
        raise ValueError("Invalid model/generation controls")
    endpoint = client.endpoint_for(args.base_url)
    public = args.requests_dir.resolve()
    manifest, dialogues, registry, instructions = load_public(public)
    system = system_prompt(instructions)
    output = base.fresh_output(args.output_dir)
    run = {"kind": "rule_task_update_rollout" if rule else "chat_task_update_rollout", "protocol": PROTOCOL,
        "started_at": datetime.now(timezone.utc).isoformat(), "requests_dir": str(public),
        "public_manifest_sha256": client.sha256(public / "manifest.json"), "modes": manifest["modes"],
        "model": args.model, "base_url": args.base_url, "temperature": args.temperature, "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout, "thinking_mode": args.thinking_mode, "history_format": args.history_format,
        "system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "planned_updates": manifest["planned_updates"], "planned_completion_events": manifest["planned_completion_events"],
        "attempted": 0, "received": 0, "failed": 0, "interrupted": False,
        "notes": "Policy state carried without gold reset; history includes own execution feedback; synthetic completion only"}
    (output / "system_prompt.txt").write_text(system, encoding="utf-8")
    api_key = os.environ.get(args.api_key_env, "")
    # Keep output streams open across the whole run and flush each event. On
    # KeyboardInterrupt the remaining slots are explicit null failures, rather
    # than quietly shortening trajectories or resetting the next policy state.
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as predictions, \
            (output / "responses.jsonl").open("x", encoding="utf-8") as responses, \
            (output / "model_inputs.jsonl").open("x", encoding="utf-8") as inputs, \
            (output / "trace.jsonl").open("x", encoding="utf-8") as trace:
        def predict(request):
            body = payload_for(run, system, request)
            identity, digest = request["request_id"], fingerprint(request)
            inputs.write(encode({"request_id": identity, "request_sha256": digest,
                                 "payload": body, "payload_sha256": fingerprint(body)}) + "\n")
            inputs.flush()
            raw, metadata, error = None, {}, None
            if run["interrupted"]:
                error = "Not attempted after interruption"
            else:
                run["attempted"] += 1
                try:
                    if rule:
                        raw = encode(predict_rules(request, registry))
                        metadata = {"source": "public_feedback_rules"}
                    else:
                        raw, metadata = client.complete_payload(endpoint, body, api_key, args.timeout)
                except KeyboardInterrupt:
                    run["interrupted"] = True
                    error = "Interrupted during model call"
                except HTTPError as failure:
                    error = f"HTTP error {failure.code}"
                    if failure.fp is not None:
                        failure.close()
                except (URLError, OSError, ValueError, KeyError, TypeError, IndexError) as failure:
                    error = f"Model call failed: {type(failure).__name__}"
            run["received" if error is None else "failed"] += 1
            predictions.write(encode({"request_id": identity, "request_sha256": digest, "prediction": raw}) + "\n")
            responses.write(encode({"request_id": identity, "metadata": metadata, "error": error}) + "\n")
            predictions.flush()
            responses.flush()
            print(f"[{run['received'] + run['failed']}/{run['planned_updates']}] {identity}: {'received' if error is None else error}", flush=True)
            return raw, metadata, error
        for record in iter_rollout(dialogues, manifest["modes"], registry, predict):
            trace.write(encode(record) + "\n")
            trace.flush()
    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    run["files_sha256"] = {name: client.sha256(output / name) for name in RUN_FILES}
    base.write_json(output / "run.json", run)
    return 1 if run["failed"] else 0


def score(args):
    """Reconstruct requests, transport bodies and states before opening output."""
    from mivln.task_update_rollout_eval import public_dialogue, score_records, summarize, validate_dataset
    public, folder = args.requests_dir.resolve(), args.run_dir.resolve()
    manifest, dialogues, registry, instructions = load_public(public)
    source = Path(manifest["dataset_path"])
    if client.sha256(source) != manifest["dataset_sha256"]:
        raise ValueError("Reference dataset changed after export")
    dataset = validate_dataset(base.read_json(source))
    if ([public_dialogue(dataset, case) for case in dataset["dialogues"]] != dialogues
            or dataset["region_registry"] != registry):
        raise ValueError("Public export differs from the frozen source")
    run = base.read_json(folder / "run.json")
    if (run.get("kind") not in ("rule_task_update_rollout", "chat_task_update_rollout")
            or run.get("protocol") != PROTOCOL or run.get("modes") != manifest["modes"]
            or run["public_manifest_sha256"] != client.sha256(public / "manifest.json")
            or run["planned_updates"] != manifest["planned_updates"]
            or run["planned_completion_events"] != manifest["planned_completion_events"]
            or set(run["files_sha256"]) != set(RUN_FILES)):
        raise ValueError("Run protocol/roster differs from the frozen export")
    for name in RUN_FILES:
        if client.sha256(folder / name) != run["files_sha256"][name]:
            raise ValueError(f"Run artifact changed: {name}")
    system = system_prompt(instructions)
    if ((folder / "system_prompt.txt").read_text(encoding="utf-8") != system
            or hashlib.sha256(system.encode("utf-8")).hexdigest() != run["system_prompt_sha256"]):
        raise ValueError("Run system prompt differs from the frozen instructions")
    predictions, responses, inputs = [lines(folder / name) for name in RUN_FILES[:3]]
    expected_ids = [f"{dialogue['dialogue_id']}__{mode}__e{index:02d}"
                    for dialogue in dialogues for mode in manifest["modes"]
                    for index, event in enumerate(dialogue["events"], 1) if event["type"] == "message"]
    if any([row["request_id"] for row in group] != expected_ids for group in (predictions, responses, inputs)):
        raise ValueError("All raw slots must retain the complete ordered request roster")
    received = sum(row["prediction"] is not None for row in predictions)
    if (run["received"] != received or run["failed"] != len(expected_ids) - received
            or not received <= run["attempted"] <= len(expected_ids)):
        raise ValueError("Raw slots disagree with run counters")
    iterator = iter(zip(predictions, responses, inputs))
    def replay(request):
        prediction, response, model_input = next(iterator)
        digest = fingerprint(request)
        body = payload_for(run, system, request)
        if (prediction["request_sha256"] != digest or model_input != {"request_id": request["request_id"],
                "request_sha256": digest, "payload": body, "payload_sha256": fingerprint(body)}):
            raise ValueError("Dynamic request/body changed; previous state or history was reset")
        raw, error = prediction["prediction"], response["error"]
        if ((raw is None) != (error is not None) or (raw is not None and (not isinstance(raw, str) or not raw.strip()))):
            raise ValueError("Raw prediction and call outcome disagree")
        return raw, response["metadata"], error
    trace = list(iter_rollout(dialogues, manifest["modes"], registry, replay))
    if trace != lines(folder / "trace.jsonl"):
        raise ValueError("Recorded transitions differ from raw prediction replay")
    rows = score_records(dataset, trace, manifest["modes"])
    report = summarize(rows)
    report.update({"protocol": PROTOCOL, "generator": run["kind"], "model": run["model"],
        "run_dir": str(folder), "predictions_sha256": client.sha256(folder / "predictions.jsonl"),
        "dataset_sha256": manifest["dataset_sha256"], "call_failures": run["failed"],
        "history_format": run["history_format"], "full_causal_replay_verified": True,
        "code_sha256": {name: client.sha256(base.ROOT / name) for name in (
            "mivln/task_update_rollout.py", "mivln/task_update_rollout_eval.py", "scripts/task_update_rollout.py",
            "mivln/task_update.py", "mivln/task_state.py", "mivln/regions.py", "mivln/rule_task_updater.py",
            "mivln/task_update_prompts.py", "scripts/predict_task_updates.py")}})
    output = base.fresh_output(args.output_dir)
    base.write_json(output / "summary.json", report)
    base.write_jsonl(output / "cases.jsonl", rows)
    fields = ("record_id", "dialogue_id", "mode", "event_index", "turn_index", "is_update", "prediction_exact",
              "before_state_exact", "state_exact", "step_correct", "error_origin", "outcome", "error")
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    base.write_jsonl(output / "trajectories.jsonl", report["trajectories"])
    for mode, stats in report["by_mode"].items():
        print(f"{mode}: {stats['step_correct_count']}/{stats['planned_updates']} exact updates/states; "
              f"{stats['all_steps_correct_count']}/{stats['planned_dialogues']} entire dialogues; "
              f"carried errors: {stats['carried_error_count']}")
    return 1 if run["failed"] or any(row["error"] for row in rows) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporting = commands.add_parser("export", help="CPU: freeze clear-update dialogues and private executor fixtures")
    exporting.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    exporting.add_argument("--mode", choices=("all",) + MODES, default="all")
    for name in ("run", "predict"):
        command = commands.add_parser(name, help="CPU rule sanity baseline" if name == "run" else "Call a configured model, carrying its own state")
        command.add_argument("--requests-dir", type=Path, required=True)
        command.add_argument("--model", default="public-feedback-rules" if name == "run" else None, required=name == "predict")
        command.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
        command.add_argument("--temperature", type=float, default=0.0)
        command.add_argument("--max-tokens", type=int, default=512)
        command.add_argument("--timeout", type=float, default=120.0)
        command.add_argument("--thinking-mode", choices=("off", "on", "default"), default="off")
        command.add_argument("--history-format", choices=("timeline", "json"), default="timeline")
        command.add_argument("--api-key-env", default="TASK_UPDATE_API_KEY")
    scoring = commands.add_parser("score", help="CPU: independently replay actual states and evaluate full chains")
    scoring.add_argument("--requests-dir", type=Path, required=True)
    scoring.add_argument("--run-dir", type=Path, required=True)
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command in ("run", "predict"):
        return generate(args, rule=args.command == "run")
    return {"export": export, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
