"""Run/export/score independent language task updates without a simulator or API.

``run`` uses an explicitly labelled rule baseline. ``export`` writes only public
requests and an answer schema. ``score`` imports a model's JSONL responses and
retains every planned slot, including missing or invalid responses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from mivln.rule_task_updater import predict_rule
from mivln.task_update import MODES, exact_keys, strict_json
from mivln.task_update_eval import (evaluate_record, prepare_records,
                                    summarize_records, validate_dataset)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT/"configs/task_update_cases_v1.json"
BUILDER = "public_language_rules_v1"
INSTRUCTIONS = """任务：根据请求中的当前消息、候选描述及允许的上下文，输出一次任务更新。
latest 模式只读当前消息；history 模式还可读初始指令与此前消息；state 模式可读旧策略状态。
引用只能使用候选目标或区域的标识。不得猜测未知目标、区域坐标或缺失的原任务。
输出必须是单个 JSON 对象，不添加 Markdown、解释、额外字段或非有限数字。

可执行更新的三种格式：
{"status":"ok","task_update":{"operation":"replace_goal","old_goal":"候选目标标识","new_goal":"候选目标标识"}}
{"status":"ok","task_update":{"operation":"add_avoid_region","region_id":"候选区域标识"}}
{"status":"ok","task_update":{"operation":"insert_subgoal","subgoal":"候选目标标识","resume_goal":"候选目标标识"}}

无法明确解释时：
{"status":"abstain","reason":"missing_context"}
reason 允许：missing_context、unknown_goal、ambiguous_goal、unknown_region、ambiguous_region、unsupported_message、invalid_context。

将每条模型输出包装为一行：
{"request_id":"复制请求标识","request_sha256":"请求的规范 JSON 哈希","prediction":模型输出对象或原始 JSON 字符串}
请求哈希清单在 request_index.json。只把 requests.jsonl、此说明和响应格式交给模型；参考答案不在这些文件中。
"""


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return strict_json(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_jsonl(path, values):
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


def fresh_output(path):
    """Reject reuse rather than silently mix models, inputs or evaluation runs."""
    output = path.resolve()
    if output.exists():
        raise FileExistsError("Use a new task-update output directory")
    output.mkdir(parents=True)
    return output


def modes_for(name):
    return list(MODES) if name == "all" else [name]


def load_plan(dataset_path, modes):
    dataset = validate_dataset(read_json(dataset_path))
    return dataset, prepare_records(dataset, modes)


def export_files(output, dataset_path, modes, records):
    """Freeze public request views; do not copy references or region geometry."""
    requests = [record["request"] for record in records if record["request"] is not None]
    write_jsonl(output/"requests.jsonl", requests)
    index = {record["request_id"]: record["request_sha256"] for record in records}
    write_json(output/"request_index.json", index)
    (output/"instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")
    manifest = {"schema_version": 1, "kind": "conditional_task_update_requests",
        "dataset_path": str(dataset_path.resolve()), "dataset_sha256": file_sha256(dataset_path),
        "context_builder": BUILDER, "modes": modes, "planned_records": len(records),
        "request_ids": list(index), "requests_sha256": file_sha256(output/"requests.jsonl"),
        "request_index_sha256": file_sha256(output/"request_index.json"),
        "context_errors": {record["request_id"]: record["context_error"] for record in records if record["context_error"]}}
    write_json(output/"manifest.json", manifest)


def write_results(output, records, rows, updater, provenance):
    if [row["request_id"] for row in rows] != [record["request_id"] for record in records]:
        raise ValueError("Evaluation changed the planned request roster")
    report = summarize_records(rows)
    report.update({"updater": updater, "context_builder": BUILDER, "source": provenance,
                   "completed_without_input_errors": not any(row["error"] for row in rows)})
    write_jsonl(output/"cases.jsonl", rows)
    scalar_fields = ("request_id", "case_id", "group_id", "language", "mode", "expected_kind", "expected_actionable",
        "context_correct", "schema_valid", "application_valid", "status_correct", "operation_correct",
        "arguments_correct", "prediction_exact", "state_exact", "requirements_preserved", "correct", "error")
    with (output/"cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in rows)
    write_json(output/"summary.json", report)
    for mode, stats in report["by_mode"].items():
        print(f"{mode}: {stats['correct_count']}/{stats['planned']} exact updates and states; "
              f"invalid/missing: {stats['invalid_or_missing_count']}")
    return report


def run(args):
    dataset_path = args.dataset.resolve()
    modes = modes_for(args.mode)
    dataset, records = load_plan(dataset_path, modes)
    output = fresh_output(args.output_dir)
    export_files(output, dataset_path, modes, records)
    predictions, rows = [], []
    for record in records:
        prediction = predict_rule(record["request"]) if record["request"] is not None else None
        if record["request"] is not None:
            predictions.append({"request_id": record["request_id"], "request_sha256": record["request_sha256"],
                                "prediction": prediction})
        rows.append(evaluate_record(dataset, record, prediction))
    write_jsonl(output/"predictions.jsonl", predictions)
    report = write_results(output, records, rows, "bilingual_rules_v1",
        {"dataset_sha256": file_sha256(dataset_path), "modes": modes})
    return 0 if report["completed_without_input_errors"] else 1


def export(args):
    modes = modes_for(args.mode)
    _, records = load_plan(args.dataset.resolve(), modes)
    output = fresh_output(args.output_dir)
    export_files(output, args.dataset.resolve(), modes, records)
    print(f"Exported {sum(record['request'] is not None for record in records)}/{len(records)} public requests to {output}")
    return 1 if any(record["context_error"] for record in records) else 0


def load_export(folder):
    manifest = read_json(folder/"manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "conditional_task_update_requests"
            or manifest.get("context_builder") != BUILDER):
        raise ValueError("Unsupported request export")
    dataset_path = Path(manifest["dataset_path"])
    if file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Source dataset changed after request export")
    for name, key in (("requests.jsonl", "requests_sha256"), ("request_index.json", "request_index_sha256")):
        if file_sha256(folder/name) != manifest[key]:
            raise ValueError(f"Frozen public input changed: {name}")
    dataset, records = load_plan(dataset_path, manifest["modes"])
    requests = [strict_json(line) for line in (folder/"requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_requests = [record["request"] for record in records if record["request"] is not None]
    expected_index = {record["request_id"]: record["request_sha256"] for record in records}
    if (requests != expected_requests or read_json(folder/"request_index.json") != expected_index
            or manifest["request_ids"] != list(expected_index) or manifest["planned_records"] != len(records)
            or manifest["context_errors"] != {record["request_id"]: record["context_error"] for record in records if record["context_error"]}):
        raise ValueError("Public request roster or generated context changed")
    return dataset, records, manifest


def score(args):
    """Missing predictions remain errors in the original planned denominator."""
    dataset, records, manifest = load_export(args.requests_dir.resolve())
    planned = {record["request_id"]: record for record in records}
    submitted = {}
    for line_number, line in enumerate(args.predictions.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = strict_json(line)
        exact_keys(value, ("request_id", "request_sha256", "prediction"), f"Prediction line {line_number}")
        identity = value["request_id"]
        if not isinstance(identity, str) or identity not in planned or identity in submitted:
            raise ValueError("Unknown or duplicate prediction request ID")
        submitted[identity] = value
    # Validate all file-level inputs before creating output. Per-response errors
    # are scored below and cannot remove that response's original case slot.
    output = fresh_output(args.output_dir)
    rows = []
    for record in records:
        value = submitted.get(record["request_id"])
        error = ("Missing prediction" if value is None else
                 "Prediction request hash mismatch" if value["request_sha256"] != record["request_sha256"] else None)
        rows.append(evaluate_record(dataset, record, value["prediction"] if value else None, error))
    report = write_results(output, records, rows, args.updater_name,
        {"requests_dir": str(args.requests_dir.resolve()), "requests_sha256": manifest["requests_sha256"],
         "dataset_sha256": manifest["dataset_sha256"], "predictions_sha256": file_sha256(args.predictions)})
    return 0 if report["completed_without_input_errors"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("run", "CPU: run the labelled rule baseline"), ("export", "Export public requests without reference answers")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        command.add_argument("--mode", choices=("all",)+MODES, default="all")
    scoring = commands.add_parser("score", help="Score external JSONL predictions with fixed denominators")
    scoring.add_argument("--requests-dir", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--updater-name", default="external_predictions")
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return {"run": run, "export": export, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
