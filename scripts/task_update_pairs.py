"""Run/export/score controlled task-update pairs with the existing model client."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mivln.rule_task_updater import predict_rule
from mivln.task_update import MODES, exact_keys
from mivln.task_update_eval import evaluate_record
from mivln.task_update_pairs import PROTOCOL, SPLITS, annotate_decision, build_plan, summarize_pairs
from scripts import task_update as base


DEFAULT_CONFIG = base.ROOT / "configs/task_update_context_pairs_v1.json"


def load_plan(path, modes, split):
    return build_plan(base.read_json(path), modes, split)


def export_files(output, path, modes, records, split):
    # The HTTP client consumes exactly the existing public format. Pair labels,
    # expected decisions and dev/test assignments are NEVER sent to the model.
    base.export_files(output, path, modes, records)
    manifest = base.read_json(output / "manifest.json")
    manifest["paired_protocol"] = {"name": PROTOCOL, "split": split}
    base.write_json(output / "manifest.json", manifest)


def write_report(output, records, rows, updater, provenance):
    if [record["request_id"] for record in records] != [row["request_id"] for row in rows]:
        raise ValueError("Paired report changed the planned request roster")
    rows = [annotate_decision(record, row) for record, row in zip(records, rows)]
    report = base.write_results(output, records, rows, updater, provenance)
    report["paired"] = summarize_pairs(records, rows)
    base.write_json(output / "summary.json", report)
    fields = ("request_id", "case_id", "family_id", "split", "language", "mode", "context_id", "order",
              "decision_category", "decision_correct", "decision_state_exact", "correct",
              "unsupported_guess", "unsupported_guess_hit", "ambiguity_execution", "error")
    with (output / "decision_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    for mode, stats in report["paired"]["by_mode"].items():
        print(f"{mode}: {stats['decision_correct_count']}/{stats['planned']} supported decisions; "
              f"unsupported guesses: {stats['unsupported_guess_count']}, "
              f"lucky task matches: {stats['unsupported_guess_hit_count']}")
    return report


def run(args):
    modes = base.modes_for(args.mode)
    dataset, records = load_plan(args.dataset, modes, args.split)
    output = base.fresh_output(args.output_dir)
    export_files(output, args.dataset, modes, records, args.split)
    rows, predictions = [], []
    for record in records:
        prediction = predict_rule(record["request"]) if record["request"] is not None else None
        if record["request"] is not None:
            predictions.append({"request_id": record["request_id"], "request_sha256": record["request_sha256"],
                                "prediction": prediction})
        rows.append(evaluate_record(dataset, record, prediction))
    base.write_jsonl(output / "predictions.jsonl", predictions)
    report = write_report(output, records, rows, "bilingual_rules_v1",
        {"dataset_sha256": base.file_sha256(args.dataset), "modes": modes, "split": args.split})
    return 0 if report["completed_without_input_errors"] else 1


def export(args):
    modes = base.modes_for(args.mode)
    _, records = load_plan(args.dataset, modes, args.split)
    output = base.fresh_output(args.output_dir)
    export_files(output, args.dataset, modes, records, args.split)
    print(f"Exported {sum(record['request'] is not None for record in records)}/{len(records)} paired requests")
    return 1 if any(record["context_error"] for record in records) else 0


def score(args):
    manifest = base.read_json(args.requests_dir / "manifest.json")
    protocol = manifest.get("paired_protocol")
    exact_keys(protocol, ("name", "split"), "Paired evaluation protocol")
    if protocol["name"] != PROTOCOL or protocol["split"] not in SPLITS:
        raise ValueError("Unsupported paired evaluation export")
    # Rebuild the entire frozen roster from its hashed source. The shared scorer
    # retains missing/invalid responses and checks hashes before creating output.
    return base.score(args,
        plan_loader=lambda path, modes: load_plan(path, modes, protocol["split"]),
        report_writer=write_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "export"):
        command = commands.add_parser(name)
        command.add_argument("--config", dest="dataset", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--split", choices=SPLITS, default="all")
        command.add_argument("--mode", choices=("all",) + MODES, default="all")
    scoring = commands.add_parser("score")
    scoring.add_argument("--requests-dir", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--updater-name", default="external_predictions")
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return {"run": run, "export": export, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
