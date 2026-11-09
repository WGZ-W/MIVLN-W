"""Offline five-candidate wording/order study with frozen navigation provenance.

Every source task yields three English wordings times two candidate orders. Only
the public export is needed for inference; scoring independently rebuilds the
private plan and replays raw predictions through the existing task executor.
No simulation or navigation success is inferred from these offline predictions.
Existing navigation engines and their code hashes remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
from types import SimpleNamespace

from mivln import TaskState
from mivln.navigation_updates import apply_navigation_prediction, navigation_request
from mivln.task_update import exact_keys, validate_request
from mivln.task_update_prompts import fingerprint, system_prompt
from scripts import language_navigation as single, language_navigation_candidates as five
from scripts import language_navigation_dataset as batch, predict_task_updates as client
from scripts import task_update as base, task_update_ablation as audit
from scripts.hssd_targets import sha256


PROTOCOL = "hssd_language_wording_order_v1"
WORDINGS = ("original", "paraphrase_1", "paraphrase_2")
ORDERS = ("ascending", "reversed")
DEFAULT_WORDINGS = base.ROOT / "configs/hssd_language_wordings_v1.json"
RUN_FILES = ("run.json", "system_prompt.txt", "model_inputs.jsonl", "predictions.jsonl", "responses.jsonl")


def code_hashes():
    return {**five.code_hashes(), **{name: sha256(base.ROOT / name) for name in
        ("scripts/language_navigation_robustness.py", "scripts/task_update_ablation.py",
         "mivln/regions.py")}}


def source_tasks(source, native_audit=False):
    """Require complete five-candidate evidence, not successful model answers.

    Native mesh/image audits happen once at preparation. Later offline scoring
    freezes and verifies their consumed evidence hashes, requiring no Habitat.
    """
    outer, inner = five.validate(source)
    tasks = []
    for task in inner["tasks"]:
        prepared = source / "batch" / task["preparation_dir"]
        _, config = single.load_preparation(prepared)
        directory = batch.result_dir(source / "batch", task)
        batch.bind_run(directory, prepared, inner["controls"])
        if native_audit and not single.check_experiment(directory)["integrity_passed"]:
            raise ValueError("Source five-candidate task lacks complete evidence: " + task["task_id"])
        if len(config["language_candidates"]) != 5:
            raise ValueError("Use the fixed five-candidate source")
        tasks.append((task["task_id"], config))
    return outer, inner, tasks


def validate_wordings(value, catalog):
    """Bind authored synonyms to existing descriptions; never extend aliases."""
    exact_keys(value, ("schema_version", "kind", "language", "provenance", "templates", "goals"), "Wording set")
    if (value["schema_version"] != 1 or value["kind"] != "hssd_explicit_replacement_wordings_v1"
            or value["language"] != "en" or not value["provenance"].strip()):
        raise ValueError("Unsupported explicit English wording set")
    exact_keys(value["templates"], WORDINGS[1:], "Paraphrase templates")
    for template in value["templates"].values():
        if not isinstance(template, str) or template.count("{goal}") != 1:
            raise ValueError("Paraphrase template must contain exactly one {goal}")
        if "{" in template.replace("{goal}", "") or "}" in template.replace("{goal}", ""):
            raise ValueError("Only the goal placeholder is allowed")
    if set(value["goals"]) != {c["goal_id"] for c in catalog}:
        raise ValueError("Wordings must cover exactly the reviewed five candidates")
    for candidate in catalog:
        phrases = value["goals"][candidate["goal_id"]]
        exact_keys(phrases, ("source_description", *WORDINGS[1:]), "Goal phrases")
        if phrases["source_description"] != candidate["description"]:
            raise ValueError("Paraphrase source differs from reviewed description")
        if any(not isinstance(text, str) or not text.strip() for text in phrases.values()) or len(set(phrases.values())) != 3:
            raise ValueError("Use distinct, nonempty original and paraphrased descriptions")
    return value


def make_records(tasks, wordings):
    """Build opaque public identities plus separately held evaluation references.

    Each trial is independent and starts from the source's pre-intervention
    policy state; predictions do not carry across these six controlled variants.
    This is deliberately not another history/state or dialogue rollout study.
    """
    records = []
    for identity, config in tasks:
        catalog = sorted(deepcopy(config["language_candidates"]), key=lambda c: c["goal_id"])
        validate_wordings(wordings, catalog)
        event = config["events"][0]
        before = TaskState(config["initial_goal"])
        after = deepcopy(before)
        after.apply_update(event["task_update"])
        target = event["task_update"]["new_goal"]
        messages = {"original": event["utterance"], **{name: wordings["templates"][name].format(
            goal=wordings["goals"][target][name]) for name in WORDINGS[1:]}}
        if len(set(messages.values())) != 3:
            raise ValueError("Wording variants must be distinct")
        for wording in WORDINGS:
            if any(candidate["goal_id"] in messages[wording] for candidate in catalog):
                raise ValueError("Messages must describe objects without exposing target IDs")
            for order in ORDERS:
                candidates = catalog if order == "ascending" else list(reversed(catalog))
                request = navigation_request(event["trigger"]["value"], messages[wording], candidates, before)
                # Include task identity only inside an opaque hash. Even two
                # source tasks with identical public text retain separate slots.
                request["request_id"] = "rob_" + fingerprint([identity, request])[:24]
                request = validate_request(request)
                records.append({"task_id": identity, "wording": wording, "order": order,
                    "target_position": next(i + 1 for i, c in enumerate(candidates) if c["goal_id"] == target),
                    "request": request, "request_sha256": fingerprint(request), "before_state": before.to_dict(),
                    "expected_prediction": {"status": "ok", "task_update": deepcopy(event["task_update"])},
                    "expected_state": after.to_dict()})
    if len({row["request"]["request_id"] for row in records}) != len(records):
        raise ValueError("Duplicate robustness request identity")
    return records


def inputs_hashes(source, inner, wordings_path):
    return {**five.frozen_inputs(source / "batch", inner),
            str(source / "manifest.json"): sha256(source / "manifest.json"),
            str(wordings_path): sha256(wordings_path)}


def generate_args(public, output, controls, api_key_env="TASK_UPDATE_API_KEY", prepare_only=False):
    return SimpleNamespace(requests_dir=public, output_dir=output, model=controls["model"],
        base_url=controls["base_url"], temperature=controls["temperature"], max_tokens=controls["max_tokens"],
        timeout=controls["timeout_seconds"], thinking_mode=controls["thinking_mode"],
        prompt_variant="baseline", api_key_env=api_key_env, prepare_only=prepare_only)


def prepare(args):
    source, output, path = args.source_dir.resolve(), args.output_dir.resolve(), args.wordings.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh robustness preparation directory")
    outer, inner, tasks = source_tasks(source, native_audit=True)
    batch.disjoint_output(output, [source, Path(outer["baseline_dir"])])
    records = make_records(tasks, base.read_json(path))
    inputs = inputs_hashes(source, inner, path)
    base.fresh_output(output)
    public, private = output / "public", output / "private"
    public.mkdir()
    private.mkdir()
    base.write_json(private / "records.json", records)
    base.write_jsonl(public / "requests.jsonl", [r["request"] for r in records])
    index = {r["request"]["request_id"]: r["request_sha256"] for r in records}
    base.write_json(public / "request_index.json", index)
    (public / "instructions.txt").write_text(base.INSTRUCTIONS, encoding="utf-8")
    base.write_json(public / "manifest.json", {"schema_version": 1, "kind": "conditional_task_update_requests",
        "experiment_protocol": PROTOCOL, "planned_records": len(records), "task_count": len(tasks),
        "modes": ["state"], "request_ids": list(index), "requests_sha256": sha256(public / "requests.jsonl"),
        "request_index_sha256": sha256(public / "request_index.json"), "context_errors": {},
        "instructions_sha256": sha256(public / "instructions.txt"), "controls": inner["controls"],
        "system_prompt_sha256": batch.prompt_hash(), "experiment_code_sha256": code_hashes()})
    base.write_json(output / "manifest.json", {"schema_version": 1, "kind": PROTOCOL,
        "source_dir": str(source), "wordings_path": str(path), "code_sha256": code_hashes(),
        "inputs_sha256": inputs, "private_records_sha256": sha256(private / "records.json"),
        "public_manifest_sha256": sha256(public / "manifest.json"), "controls": inner["controls"],
        "task_count": len(tasks), "planned_records": len(records), "wordings": list(WORDINGS), "orders": list(ORDERS),
        "created_at": batch.now(), "scope": "Independent explicit English updates; authored development paraphrases; no navigation claims"})
    # Preview actual bodies with the same shared client used for inference. A
    # preview contains no fabricated predictions and cannot be scored as a run.
    with redirect_stdout(io.StringIO()):
        client.generate(generate_args(public, output / "prompt_plan", inner["controls"], prepare_only=True))
    checks = [evaluate(row, fingerprint(row["request"]), single.encode(row["expected_prediction"]), None) for row in records]
    assert all(row["correct"] for row in checks)
    base.write_json(output / "cpu_check.json", {"planned_records": len(records), "public_requests_valid": True,
        "synthetic_reference_replay_correct": len(checks), "no_model_calls": True, "no_renderer": True,
        "note": "Executor/serialization check using declared references; NOT model performance"})
    batch.write_csv(private / "wording_review.csv", [{"task_id": r["task_id"], "wording": r["wording"],
        "order": r["order"], "message": r["request"]["message"],
        "expected_goal": r["expected_prediction"]["task_update"]["new_goal"]} for r in records])
    print(f"Prepared {len(tasks)} tasks x 3 wordings x 2 orders = {len(records)} NEW requests. No model/GPU calls.")
    return 0


def load_public(public):
    """Inference reads ONLY this export, never private records/source episodes."""
    manifest, index, requests, instructions = client.load_public_requests(public)
    batch.validate_controls(manifest["controls"])
    if (manifest.get("experiment_protocol") != PROTOCOL or manifest["experiment_code_sha256"] != code_hashes()
            or manifest["planned_records"] != manifest["task_count"] * 6 or manifest["modes"] != ["state"]
            or manifest["context_errors"] or instructions != base.INSTRUCTIONS
            or sha256(public / "instructions.txt") != manifest["instructions_sha256"]
            or manifest["system_prompt_sha256"] != batch.prompt_hash()):
        raise ValueError("Public robustness plan, controls, prompt or code changed")
    return manifest, index, requests, instructions


def load_prepared(folder):
    manifest = base.read_json(folder / "manifest.json")
    if manifest.get("kind") != PROTOCOL or manifest.get("schema_version") != 1 or manifest["code_sha256"] != code_hashes():
        raise ValueError("Robustness protocol or code changed")
    source, path = Path(manifest["source_dir"]), Path(manifest["wordings_path"])
    _, inner, tasks = source_tasks(source)
    if inputs_hashes(source, inner, path) != manifest["inputs_sha256"]:
        raise ValueError("Frozen source evidence or wordings changed")
    expected = make_records(tasks, base.read_json(path))
    private = folder / "private/records.json"
    if sha256(private) != manifest["private_records_sha256"] or base.read_json(private) != expected:
        raise ValueError("Private references differ from independently rebuilt plan")
    public, index, requests, _ = load_public(folder / "public")
    if (sha256(folder / "public/manifest.json") != manifest["public_manifest_sha256"]
            or list(requests.values()) != [r["request"] for r in expected]
            or index != {r["request"]["request_id"]: r["request_sha256"] for r in expected}
            or manifest["wordings"] != list(WORDINGS) or manifest["orders"] != list(ORDERS)
            or manifest["controls"] != inner["controls"] or public["controls"] != inner["controls"]
            or manifest["task_count"] != len(tasks) or manifest["planned_records"] != len(expected)):
        raise ValueError("Frozen request roster, controls or denominator changed")
    return manifest, expected


def predict(args):
    public, output = args.requests_dir.resolve(), args.output_dir.resolve()
    manifest, _, _, _ = load_public(public)
    batch.disjoint_output(output, [public])
    code = client.generate(generate_args(public, output, manifest["controls"], args.api_key_env))
    # Seal raw outcomes as well as bodies, including call errors and interrupted
    # tails. A generic client run without this seal cannot silently enter scoring.
    base.write_json(output / "sealed_run.json", {"kind": PROTOCOL,
        "public_manifest_sha256": sha256(public / "manifest.json"), "code_sha256": code_hashes(),
        "files_sha256": {name: sha256(output / name) for name in RUN_FILES}})
    return code


def evaluate(record, digest, raw, call_error):
    """Apply the actual raw prediction transactionally; reference is score-only."""
    request = record["request"]
    if digest != record["request_sha256"]:
        raise ValueError("Prediction is attached to a different request")
    state = TaskState(**deepcopy(record["before_state"]))
    after, transition = apply_navigation_prediction(state, request, raw, call_error=call_error)
    prediction = transition["prediction"]
    update = prediction.get("task_update", {}) if prediction else {}
    expected = record["expected_prediction"]["task_update"]
    exact = prediction == record["expected_prediction"]
    state_exact = after.to_dict() == record["expected_state"]
    operation = update.get("operation") == "replace_goal"
    return {"request_id": request["request_id"], "task_id": record["task_id"], "wording": record["wording"],
        "order": record["order"], "target_position": record["target_position"], "raw_prediction": raw,
        "prediction": prediction, "after_state": after.to_dict(), "outcome": transition["outcome"],
        "schema_valid": transition["schema_valid"], "application_valid": transition["application_valid"],
        "operation_correct": operation, "old_goal_correct": operation and update.get("old_goal") == expected["old_goal"],
        "new_goal_correct": operation and update.get("new_goal") == expected["new_goal"],
        "prediction_exact": exact, "state_exact": state_exact,
        "correct": bool(exact and state_exact and transition["outcome"] == "applied"),
        "selected_distractor": operation and update.get("new_goal") not in (expected["old_goal"], expected["new_goal"]),
        "call_error": call_error, "error": transition["error"]}


def stats(rows):
    count = len(rows)
    return {"planned": count, "correct_count": sum(r["correct"] for r in rows),
        "correct_rate": sum(r["correct"] for r in rows) / count,
        "prediction_exact_count": sum(r["prediction_exact"] for r in rows),
        "state_exact_count": sum(r["state_exact"] for r in rows),
        "schema_invalid_count": sum(not r["schema_valid"] for r in rows),
        "call_failure_count": sum(r["call_error"] is not None for r in rows),
        "distractor_selection_count": sum(r["selected_distractor"] for r in rows)}


def group_result(rows):
    """Equal wrong answers count as consistent, but never as all-correct.

    Missing/malformed answers cannot gain consistency credit by sharing null.
    Compare normalized symbolic prediction AND resulting policy state, rather
    than superficial JSON whitespace or raw text equality.
    """
    consistent = all(r["schema_valid"] for r in rows) and len({fingerprint(
        [r["prediction"], r["after_state"]]) for r in rows}) == 1
    return {"planned_members": len(rows), "consistent": consistent,
            "all_correct": all(r["correct"] for r in rows),
            "correctness_changed": len({r["correct"] for r in rows}) > 1}


def score(args):
    prepared, run_dir, output = args.prepared_dir.resolve(), args.run_dir.resolve(), args.output_dir.resolve()
    manifest, records = load_prepared(prepared)
    public, index, requests, instructions = load_public(prepared / "public")
    seal = base.read_json(run_dir / "sealed_run.json")
    if (seal.get("kind") != PROTOCOL or seal["code_sha256"] != code_hashes()
            or seal["public_manifest_sha256"] != sha256(prepared / "public/manifest.json")
            or seal["files_sha256"] != {name: sha256(run_dir / name) for name in RUN_FILES}):
        raise ValueError("Sealed raw model run changed or is incomplete")
    audited = audit.audit_run(run_dir, "baseline", public, index, requests, instructions)
    run = audited["run"]
    if audited["legacy_reconstruction"] or any(run[key] != value for key, value in manifest["controls"].items()):
        raise ValueError("Inference controls differ from the frozen navigation baseline")
    responses = {r["request_id"]: r for r in single.lines(run_dir / "responses.jsonl")}
    rows = [evaluate(r, index[r["request"]["request_id"]], audited["raw"][r["request"]["request_id"]],
                     responses[r["request"]["request_id"]]["error"]) for r in records]
    order_pairs, wording_groups, task_groups = [], [], []
    task_ids = list(dict.fromkeys(r["task_id"] for r in records))
    for task in task_ids:
        task_rows = [r for r in rows if r["task_id"] == task]
        task_groups.append({"task_id": task, **group_result(task_rows)})
        for wording in WORDINGS:
            pair = [r for r in task_rows if r["wording"] == wording]
            order_pairs.append({"task_id": task, "wording": wording,
                "target_position_changed": pair[0]["target_position"] != pair[1]["target_position"], **group_result(pair)})
        for order in ORDERS:
            wording_groups.append({"task_id": task, "order": order,
                                  **group_result([r for r in task_rows if r["order"] == order])})
    counts = lambda groups: {"planned": len(groups), "consistent_count": sum(g["consistent"] for g in groups),
        "all_correct_count": sum(g["all_correct"] for g in groups),
        "correctness_changed_count": sum(g["correctness_changed"] for g in groups)}
    report = {"kind": PROTOCOL, "statistics": stats(rows), "task_count": len(task_ids),
        "by_wording": {name: stats([r for r in rows if r["wording"] == name]) for name in WORDINGS},
        "by_order": {name: stats([r for r in rows if r["order"] == name]) for name in ORDERS},
        "by_target_position": {str(p): stats([r for r in rows if r["target_position"] == p])
                               for p in sorted({r["target_position"] for r in rows})},
        "order_pairs": counts(order_pairs), "wording_groups": counts(wording_groups),
        "task_groups": counts(task_groups), "controls": manifest["controls"],
        "actual_attempted": run["attempted"], "received": run["received"], "failed": run["failed"],
        "interrupted": run["interrupted"], "integrity_passed": True,
        "all_correct": all(r["correct"] for r in rows), "code_sha256": code_hashes(),
        "sources": {"prepared_dir": str(prepared), "run_dir": str(run_dir),
            "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
            "sealed_run_sha256": sha256(run_dir / "sealed_run.json")},
        "limitations": ["Authored English development paraphrases on ten reused base tasks in one scene",
            "Sixty requests are correlated variants, not sixty independent navigation tasks",
            "Ascending/reversed order leaves the middle candidate in the same position",
            "Offline update/state evaluation only; no new navigation, visual grounding, refusal or history/state comparison"]}
    batch.disjoint_output(output, [prepared / "public", prepared / "private", run_dir])
    base.fresh_output(output)
    base.write_json(output / "summary.json", report)
    base.write_jsonl(output / "cases.jsonl", rows)
    flat = [{k: v for k, v in row.items() if k not in ("prediction", "after_state")} for row in rows]
    batch.write_csv(output / "cases.csv", flat)
    for name, groups in (("order_pairs", order_pairs), ("wording_groups", wording_groups), ("task_groups", task_groups)):
        base.write_jsonl(output / (name + ".jsonl"), groups)
        batch.write_csv(output / (name + ".csv"), groups)
    print(f"Updates/states {report['statistics']['correct_count']}/{len(rows)}; "
          f"order pairs both correct {report['order_pairs']['all_correct_count']}/{len(order_pairs)}; "
          f"all-six tasks correct {report['task_groups']['all_correct_count']}/{len(task_groups)}.")
    return 0 if report["all_correct"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="CPU Habitat audit: freeze 3 wordings x 2 orders")
    preparing.add_argument("--source-dir", type=Path, required=True)
    preparing.add_argument("--wordings", type=Path, default=DEFAULT_WORDINGS)
    predicting = commands.add_parser("predict", help="Public inputs only: NEW raw model calls")
    predicting.add_argument("--requests-dir", type=Path, required=True)
    predicting.add_argument("--api-key-env", default="TASK_UPDATE_API_KEY")
    scoring = commands.add_parser("score", help="CPU standard library: audit raw calls and paired consistency")
    scoring.add_argument("--prepared-dir", type=Path, required=True)
    scoring.add_argument("--run-dir", type=Path, required=True)
    for command in commands.choices.values():
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return {"prepare": prepare, "predict": predict, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
