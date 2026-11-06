"""Prepare public prompt conditions or audit/score three completed model runs.

Both commands are CPU-only. Real model calls use predict_task_updates with an
explicit prompt variant. Comparison reconstructs every intended HTTP body and
rescoring uses the same frozen paired source for all conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

from mivln.task_update import strict_json
from mivln.task_update_prompts import PROTOCOL, VARIANTS, fingerprint, system_prompt
from scripts import predict_task_updates as inference
from scripts import task_update as base
from scripts import task_update_pairs as pairs


CONTROL_FIELDS = ("model", "base_url", "temperature", "max_tokens", "timeout_seconds",
                  "thinking_mode", "chat_template_kwargs")


def lines(path):
    return [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(args):
    """Freeze all conditions without loading references or contacting a model."""
    _, index, _, _ = inference.load_public_requests(args.requests_dir.resolve())
    output = base.fresh_output(args.output_dir)
    for variant in VARIANTS:
        child = SimpleNamespace(**{**vars(args), "prompt_variant": variant, "prepare_only": True,
                                   "output_dir": output / variant})
        if inference.generate(child):
            raise ValueError("Public context failures must be resolved before an ablation")
    base.write_json(output / "plan.json", {"protocol": PROTOCOL, "conditions": list(VARIANTS),
        "planned_per_condition": len(index), "model_calls": 0,
        "notes": "Input previews only; no predictions or scores generated"})
    return 0


def audit_run(folder, variant, manifest, index, requests, instructions):
    """Bind provenance, raw outputs and actual bodies to one public-input roster.

    Historical baseline runs predate the body log. They are accepted only as
    explicitly marked legacy reconstructions with the original prompt/hash
    fields. New runs cannot silently discard their body log and downgrade.
    """
    run = base.read_json(folder / "run.json")
    if run.get("kind") != "chat_task_update_predictions" or run.get("prepare_only", False):
        raise ValueError("A prompt preview is not a completed prediction run")
    if run.get("prompt_variant", "baseline") != variant:
        raise ValueError("Model run has the wrong prompt condition")
    for name, key in (("requests.jsonl", "requests_sha256"), ("request_index.json", "request_index_sha256")):
        if run[key] != manifest[key]:
            raise ValueError(f"Conditions must use identical frozen {name}")
    expected_system = system_prompt(instructions, variant)
    if ((folder / "system_prompt.txt").read_text(encoding="utf-8") != expected_system
            or run["system_prompt_sha256"] != hashlib.sha256(expected_system.encode("utf-8")).hexdigest()
            or run["instructions_sha256"] != hashlib.sha256(instructions.encode("utf-8")).hexdigest()):
        raise ValueError("Actual system prompt changed from the versioned condition")
    if run["predictions_sha256"] != inference.sha256(folder / "predictions.jsonl"):
        raise ValueError("Predictions changed after generation")
    predictions, responses = lines(folder / "predictions.jsonl"), lines(folder / "responses.jsonl")
    if ([row["request_id"] for row in predictions] != list(index)
            or [row["request_id"] for row in responses] != list(index)
            or run["planned_records"] != len(index)):
        raise ValueError("Completed generation must retain the entire request roster")
    for prediction, response in zip(predictions, responses):
        if prediction["request_sha256"] != index[prediction["request_id"]]:
            raise ValueError("Prediction is bound to a different request")
        value = prediction["prediction"]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("Generator provenance requires raw output strings or null failures")
        if (value is None) != (response.get("error") is not None):
            raise ValueError("Raw prediction and call outcome disagree")
    received = sum(row["prediction"] is not None for row in predictions)
    if (run["received"] != received or run["failed"] != len(index) - received
            or not received <= run["attempted"] <= len(index)):
        raise ValueError("Generation counters disagree with the raw records")
    expected_inputs = []
    for identity, digest in index.items():
        request = requests.get(identity)
        payload = (inference.payload_for(run["model"], expected_system, request, run["temperature"],
                   run["max_tokens"], run["thinking_mode"], variant) if request is not None else None)
        expected_inputs.append({"request_id": identity, "request_sha256": digest,
            "payload": payload, "payload_sha256": fingerprint(payload) if payload is not None else None})
    legacy = (variant == "baseline" and not any(key in run for key in
              ("prompt_protocol", "prompt_variant", "model_inputs_sha256")))
    if not legacy:
        if (run.get("prompt_protocol") != PROTOCOL or not run.get("model_inputs_sha256")
                or not (folder / "model_inputs.jsonl").is_file()
                or inference.sha256(folder / "model_inputs.jsonl") != run["model_inputs_sha256"]
                or lines(folder / "model_inputs.jsonl") != expected_inputs):
            raise ValueError("Actual model input bodies differ from the controlled condition")
    return {"run": run, "raw": {row["request_id"]: row["prediction"] for row in predictions},
            "inputs": expected_inputs, "legacy_reconstruction": legacy, "folder": folder}


def matched_counts(baseline, condition):
    """Paired wins/losses include every record, not just valid model answers."""
    if [row["request_id"] for row in baseline] != [row["request_id"] for row in condition]:
        raise ValueError("Comparison changed the matched request roster")
    return {"planned": len(baseline),
        "gained": sum(not old["decision_correct"] and new["decision_correct"] for old, new in zip(baseline, condition)),
        "lost": sum(old["decision_correct"] and not new["decision_correct"] for old, new in zip(baseline, condition)),
        "both_correct": sum(old["decision_correct"] and new["decision_correct"] for old, new in zip(baseline, condition)),
        "both_wrong": sum(not old["decision_correct"] and not new["decision_correct"] for old, new in zip(baseline, condition))}


def error_metrics(rows, records):
    """Diagnose errors AFTER scoring; these private facts never enter prompts.

    Attempted execution uses schema-valid status=ok answers; applied execution
    additionally passed the transactional adapter. Malformed answers remain in
    the invalid/missing count and in all planned denominators. Goal confusion is
    counted only for schema-valid, correct-operation answers on actionable
    cases, so missing JSON fields are not mislabeled as a wrong remembered goal.
    """
    result = {key: 0 for key in ("planned", "actionable_required", "false_refusal_count",
        "reference_goal_required", "reference_goal_wrong_count", "initial_goal_confusion_count",
        "refusal_required", "refusal_execution_count", "refusal_applied_count", "invalid_or_missing_count")}
    for row in rows:
        result["planned"] += 1
        prediction = row["prediction"] if isinstance(row["prediction"], dict) else {}
        actionable = row["decision_category"] == "actionable"
        result["actionable_required" if actionable else "refusal_required"] += 1
        result["invalid_or_missing_count"] += not row["schema_valid"]
        result["false_refusal_count"] += actionable and prediction.get("status") == "abstain"
        attempted = not actionable and prediction.get("status") == "ok"
        result["refusal_execution_count"] += attempted
        result["refusal_applied_count"] += attempted and row["application_valid"]
        expected = row["expected_decision"].get("task_update", {})
        field = next((key for key in ("old_goal", "resume_goal") if key in expected), None)
        if not actionable or field is None:
            continue
        result["reference_goal_required"] += 1
        if row["schema_valid"] and row["operation_correct"]:
            actual = prediction.get("task_update", {}).get(field)
            wrong = actual != expected[field]
            result["reference_goal_wrong_count"] += wrong
            result["initial_goal_confusion_count"] += wrong and actual == records[row["request_id"]]["case"]["reference"]["initial_goal"]
    return result


def compare(args):
    public = args.requests_dir.resolve()
    manifest, index, requests, instructions = inference.load_public_requests(public)
    protocol = manifest.get("paired_protocol", {})
    if protocol.get("name") != "task_update_context_pairs_v1" or protocol.get("split") not in pairs.SPLITS:
        raise ValueError("Prompt ablation comparison requires frozen paired inputs")
    # Validate evaluator inputs and ALL generation folders before creating any
    # comparison output. Model generation itself never reads this source file.
    _, records, _ = base.load_export(public, lambda path, modes: pairs.load_plan(path, modes, protocol["split"]))
    record_index = {record["request_id"]: record for record in records}
    folders = {"baseline": args.baseline_run, "history_timeline": args.history_run, "refusal_check": args.refusal_run}
    audits = {variant: audit_run(folder.resolve(), variant, manifest, index, requests, instructions)
              for variant, folder in folders.items()}
    controls = {key: audits["baseline"]["run"][key] for key in CONTROL_FIELDS}
    for audit in audits.values():
        if any(audit["run"][key] != controls[key] for key in CONTROL_FIELDS):
            raise ValueError("Model/service and generation controls must match across conditions")
    output = base.fresh_output(args.output_dir)
    reports, rows = {}, {}
    for variant in VARIANTS:
        audit = audits[variant]
        scoring = SimpleNamespace(requests_dir=public, predictions=audit["folder"] / "predictions.jsonl",
            updater_name=controls["model"] + "-" + variant, output_dir=output / (variant + "_score"))
        score_code = pairs.score(scoring)
        report = base.read_json(scoring.output_dir / "summary.json")
        reports[variant] = {"run_dir": str(audit["folder"]), "legacy_baseline_reconstruction": audit["legacy_reconstruction"],
            "score_return_code": score_code, "call_failures": audit["run"]["failed"],
            "task_by_mode": report["by_mode"], "decision_by_mode": report["paired"]["by_mode"],
            "by_mode_category": report["paired"]["by_mode_category"],
            "context_groups_by_mode": report["paired"]["context_groups_by_mode"],
            "order_groups_by_mode": report["paired"]["order_groups_by_mode"]}
        rows[variant] = lines(scoring.output_dir / "cases.jsonl")
        reports[variant]["errors_by_mode"] = {mode: error_metrics(
            [row for row in rows[variant] if row["mode"] == mode], record_index) for mode in manifest["modes"]}
    deltas = {}
    for variant in VARIANTS[1:]:
        old, new = rows["baseline"], rows[variant]
        unchanged = [first["request_id"] for first, second in zip(audits["baseline"]["inputs"], audits[variant]["inputs"])
                     if first["payload"] == second["payload"]]
        deltas[variant] = {"matched": matched_counts(old, new),
            "by_mode": {mode: matched_counts([row for row in old if row["mode"] == mode],
                                            [row for row in new if row["mode"] == mode]) for mode in manifest["modes"]},
            "by_category": {category: matched_counts([row for row in old if row["decision_category"] == category],
                                                    [row for row in new if row["decision_category"] == category])
                            for category in sorted({row["decision_category"] for row in old})},
            "unchanged_input_count": len(unchanged),
            "error_count_changes_by_mode": {mode: {
                key: value - reports["baseline"]["errors_by_mode"][mode][key]
                for key, value in reports[variant]["errors_by_mode"][mode].items()}
                for mode in manifest["modes"]},
            "raw_response_changes_on_unchanged_inputs": sum(audits["baseline"]["raw"][identity] != audits[variant]["raw"][identity]
                                                           for identity in unchanged)}
    scorer_files = ("mivln/task_update.py", "mivln/task_state.py", "mivln/regions.py", "mivln/task_update_eval.py",
                    "mivln/task_update_pairs.py", "scripts/task_update.py", "scripts/task_update_pairs.py")
    report = {"kind": "task_update_prompt_ablation_comparison", "protocol": PROTOCOL,
        "paired_split": protocol["split"], "controls": controls, "planned_per_condition": len(index),
        "requests_sha256": manifest["requests_sha256"], "dataset_sha256": manifest["dataset_sha256"],
        "scorer_files_sha256": {name: inference.sha256(base.ROOT / name) for name in scorer_files},
        "conditions": reports, "vs_baseline": deltas,
        "completed_without_call_errors": all(audit["run"]["failed"] == 0 for audit in audits.values()),
        "limitations": ["Client controls/model alias do not prove identical server weights or GPU settings",
            "Legacy baseline bodies are reconstructed, not previously recorded HTTP evidence",
            "Identical inputs may produce different outputs; inspect unchanged-input controls",
            "Observed test is a regression set; new claims need new frozen validation groups"]}
    base.write_json(output / "comparison.json", report)
    with (output / "comparison_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("request_id", "mode", "decision_category") + VARIANTS
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for position, row in enumerate(rows["baseline"]):
            writer.writerow({**{key: row[key] for key in fields[:3]},
                             **{variant: rows[variant][position]["decision_correct"] for variant in VARIANTS}})
    print(f"Comparison: {output / 'comparison.json'}")
    return 0 if report["completed_without_call_errors"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("prepare", help="CPU: freeze three prompt previews, no inference")
    preview.add_argument("--model", required=True)
    preview.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    preview.add_argument("--temperature", type=float, default=0.0)
    preview.add_argument("--max-tokens", type=int, default=512)
    preview.add_argument("--timeout", type=float, default=120.0)
    preview.add_argument("--thinking-mode", choices=("default", "off", "on"), default="off")
    preview.add_argument("--api-key-env", default="TASK_UPDATE_API_KEY")
    comparison = commands.add_parser("compare", help="CPU: audit and rescore three model batches")
    comparison.add_argument("--baseline-run", type=Path, required=True)
    comparison.add_argument("--history-run", type=Path, required=True)
    comparison.add_argument("--refusal-run", type=Path, required=True)
    for command in commands.choices.values():
        command.add_argument("--requests-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return {"prepare": prepare, "compare": compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
