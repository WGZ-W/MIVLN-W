"""Prepare reviewed three-task cases, exercise a CPU V loop and audit raw logs.

Preparation preserves HSSD geometry/reference timings. CPU execution deliberately
uses a separate synthetic fixture; no command here renders Habitat or calls a
model. Old experiment code and source artifacts are read-only dependencies.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from mivln import benchmark_navigation as navigation
from mivln import language_annotations as annotations
from mivln.navigation_fixture import CONDITIONS, SCOPE, FreePlaneBackend, fixture_case, privileged_policy
from mivln.task_update import strict_json
from scripts.benchmark_manifest import PROJECT_ROOT, sha256, within
from scripts.language_annotations import load_bundle, read_json, check_sources
from scripts.public_inputs import fresh_output, write_json, write_lines


KIND = "reviewed_three_task_v_navigation_preparation_v1"
TIMINGS = (("early", .25), ("middle", .5), ("late", .75))
CODE = ("mivln/benchmark_navigation.py", "mivln/navigation_fixture.py", "scripts/benchmark_navigation.py")


def code_bindings():
    return {name: sha256(PROJECT_ROOT / name) for name in CODE}


def build_preparation(annotation_dir):
    """Deterministically derive 3 types x 3 timings, retaining failed case slots.

    The timing is an external action index selected by cumulative travel on the
    frozen A reference, shared across types and policies. It is not recomputed
    from the agent's trajectory. A different prefix can STOP before delivery and
    must then fail intervention success. Old remaining-distance triggers stay
    unchanged in historical configs.
    """
    from scripts import hssd_dataset
    from scripts.intervention_timing import check_reference
    from scripts.subgoal import reference_trigger

    annotation_dir = annotation_dir.resolve()
    dataset = load_bundle(annotation_dir)
    if not annotations.summary(dataset)["all_language_approved"]:
        raise ValueError("All required language annotations must be approved before navigation preparation")
    records = annotations.replay(dataset)
    bindings = dict(dataset["source_files"])

    def bind(path):
        path = Path(path).resolve()
        digest = sha256(path)
        if str(path) in bindings and bindings[str(path)] != digest:
            raise ValueError("Bound source changed during preparation")
        bindings[str(path)] = digest
        return read_json(path)

    # Preserve the full authority chain, not just a mutable pointer to its tip.
    directory = annotation_dir
    while True:
        seal = bind(directory / "manifest.json")
        bind(directory / "annotations.json")
        if seal["parent"] is None:
            break
        directory = Path(seal["parent"]["path"]).parent
    source = Path(dataset["source_dir"])
    roster = bind(source / "manifest.json")
    indexed = {task["task_id"]: task for task in roster["tasks"]}
    cases, configs = [], {}
    for base in dataset["base_tasks"]:
        task = indexed[base["source_task_id"]]
        config = bind(base["source_config"]["path"])
        reference = within(source, task["output_subdir"]) / "reference"
        error, triggers, rows = None, {}, []
        try:
            episode, metrics, rows = [bind(reference / name) for name in ("episode.json", "metrics.json", "trajectory.json")]
            diagnostic = bind(reference / "terminal_diagnostics.json")
            for entry in diagnostic["files"]:
                path = within(reference, entry["path"])
                if sha256(path) != entry["sha256"]:
                    raise ValueError("Reference terminal instance bytes changed")
                bindings[str(path)] = entry["sha256"]
            rows = check_reference(config, reference)
            hssd_dataset.same_task(config, episode)
            hssd_dataset.check_episode_records(config, config, "control", reference, metrics, rows)
            terminal = hssd_dataset.terminal_record(config, metrics, rows, reference, "reference")
            if not terminal["passed"]:
                raise ValueError(terminal["error"] or "Reference terminal evidence failed")
            for before, after in zip(rows, rows[1:]):
                if not math.isclose(math.dist(before["position"], after["position"]), after["distance_m"], abs_tol=1e-5):
                    raise ValueError("Reference cumulative travel differs from measured positions")
            triggers = {name: reference_trigger(rows, fraction) for name, fraction in TIMINGS}
            if len(set(triggers.values())) != len(TIMINGS):
                raise ValueError("Reference cannot support three distinct delivery slots")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            error = str(exc)
        for family in navigation.TASK_TYPES:
            record = records[base["source_task_id"] + "_" + family]
            initial = records[record["initial_annotation_id"]]
            for timing, fraction in TIMINGS:
                identity = f"{base['source_task_id']}_{family}_{timing}"
                item = {"case_id": identity, "base_task_id": base["base_task_id"],
                    "source_task_id": base["source_task_id"], "task_type": family, "timing": timing,
                    "prepared": error is None, "error": error, "config": None, "config_sha256": None}
                if error is None:
                    variant = deepcopy(config)
                    variant.update(navigation_protocol=navigation.PROTOCOL, task_type=family, timing=timing,
                        base_task_id=base["base_task_id"], source_task_id=base["source_task_id"],
                        episode_id=identity, initial_instruction=initial["text"],
                        yaw_tolerance_degrees=config["terminal_alignment"]["yaw_tolerance_degrees"],
                        trigger_protocol="external_step_from_cumulative_reference_travel_v1",
                        trigger_reference={"path": str(reference / "trajectory.json"),
                            "sha256": bindings[str((reference / "trajectory.json").resolve())],
                            "fraction": fraction, "travel_m": sum(row["distance_m"] for row in rows),
                            "reference_pose_before": {key: rows[triggers[timing]][key] for key in ("position", "rotation_xyzw")}},
                        language_binding={"manifest": str(annotation_dir / "manifest.json"),
                            "sha256": bindings[str(annotation_dir / "manifest.json")],
                            "dataset_id": dataset["dataset_id"], "revision_id": dataset["revision_id"],
                            "initial_annotation_id": initial["annotation_id"], "intervention_annotation_id": record["annotation_id"]},
                        execution_scope="hssd_preparation_only", split=None, new_navigation_executed=False)
                    variant["events"] = [{"event_id": identity + "_event", "trigger": {"type": "step", "value": triggers[timing]},
                        "utterance": record["text"], "task_update": deepcopy(record["semantic_label"])}]
                    if family == "cancel_task":
                        variant["cancel_protocol"] = "cancel_task_v1"
                    navigation.validate_case(variant)
                    item["config"] = "configs/" + identity + ".json"
                    configs[item["config"]] = variant
                cases.append(item)
    check_sources(bindings)
    return {"schema_version": 1, "kind": KIND, "protocol": navigation.PROTOCOL,
        "annotation_dir": str(annotation_dir), "source_files": bindings,
        "base_tasks": len(dataset["base_tasks"]), "planned_cases": len(cases),
        "prepared_cases": len(configs), "cases": cases,
        "trigger_protocol": "external_step_from_cumulative_reference_travel_v1",
        "habitat_run_entry_implemented": False, "hssd_navigation_executed": False,
        "release_ready": False, "split_status": "not_frozen"}, configs


def prepare(args):
    output = fresh_output(args.output_dir, args.annotations_dir)
    manifest, configs = build_preparation(args.annotations_dir)
    output.mkdir(parents=True)
    for name, config in configs.items():
        write_json(output / name, config)
    for case in manifest["cases"]:
        if case["prepared"]:
            case["config_sha256"] = sha256(output / case["config"])
    manifest["code_bindings"] = code_bindings()
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {manifest['prepared_cases']}/{manifest['planned_cases']} reviewed HSSD cases; no navigation executed")
    return 0 if len(configs) == manifest["planned_cases"] else 1


def load_preparation(directory):
    """Rebuild source-bound cases to reject re-sealed geometry/language edits."""
    directory = directory.resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest.get("kind") != KIND or manifest.get("code_bindings") != code_bindings():
        raise ValueError("Preparation kind/code binding mismatch; prepare a new version after code changes")
    expected, configs = build_preparation(Path(manifest["annotation_dir"]))
    actual = deepcopy(manifest)
    actual.pop("code_bindings")
    for case in actual["cases"]:
        if case["prepared"]:
            path = within(directory, case["config"])
            if sha256(path) != case["config_sha256"] or read_json(path) != configs[case["config"]]:
                raise ValueError("Prepared case differs from the reviewed origin/reference")
            case["config_sha256"] = None
    if actual != expected:
        raise ValueError("Preparation roster/source binding differs from its reconstructed origin")
    return manifest


def write_episode(directory, config, result):
    """Public files are offline audit assets, not a mountable online input pool."""
    directory.mkdir(parents=True)
    for name, value in (("config", config), ("trajectory", result["trajectory"]), ("events", result["events"]),
                        ("metrics", result["metrics"]), ("final_task_state", result["final_task_state"])):
        write_json(directory / "private" / (name + ".json"), value)
    write_json(directory / "private/run_status.json", {"interface_error": result["interface_error"]})
    write_lines(directory / "private/policy_records.jsonl", result["policy_records"])
    write_lines(directory / "public/requests.jsonl", [row["request"] for row in result["policy_records"]])
    write_lines(directory / "public/actions.jsonl", [row["action"] for row in result["policy_records"] if row["action"] is not None])
    for identity, data in result["assets"].items():
        path = directory / "public/assets" / (identity + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def lines(path):
    return [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_episode(directory, expected):
    """Re-execute saved raw outputs; ignore cached metrics and final-state claims.

    Fixture RGB, poses, measurements and V histories must reproduce exactly.
    A missing/invalid policy response stays failed, with the actual input retained.
    No privileged policy is called during audit, only the saved responses.
    """
    config = read_json(directory / "private/config.json")
    if config != expected:
        raise ValueError("CPU case changed its reviewed language or declared fixture geometry")
    records = lines(directory / "private/policy_records.jsonl")
    trajectory = read_json(directory / "private/trajectory.json")
    if not records or not trajectory:
        raise ValueError("Missing policy input/trajectory")
    responses = iter(records)

    def replay(request, assets):
        record = next(responses)
        if record["request"] != request:
            raise ValueError("Public input differs from replayed causal V history")
        return record["raw_response"]

    acquisitions = [row["rgb_asset_id"] for row in trajectory if row["action"] != "STOP"]
    result = navigation.run_episode(config, FreePlaneBackend(config), replay,
        session_id=records[0]["request"]["session_id"], asset_ids=acquisitions)
    if result["trajectory"] != trajectory or result["events"] != read_json(directory / "private/events.json"):
        raise ValueError("Saved poses, measurements or delivery differ from raw-action replay")
    if result["policy_records"] != records:
        raise ValueError("Accepted actions/errors differ from the preserved raw responses")
    if lines(directory / "public/requests.jsonl") != [row["request"] for row in records]:
        raise ValueError("Published inputs differ from actual policy calls")
    if lines(directory / "public/actions.jsonl") != [row["action"] for row in records if row["action"] is not None]:
        raise ValueError("Published actions differ from actual raw responses")
    assets_dir = directory / "public/assets"
    if {path.name for path in assets_dir.iterdir()} != {identity + ".png" for identity in result["assets"]}:
        raise ValueError("Unexpected, missing or future RGB asset in the published fixture")
    for identity, data in result["assets"].items():
        path = assets_dir / (identity + ".png")
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError("Actual policy RGB differs from replayed capture bytes")
    if read_json(directory / "private/final_task_state.json") != result["final_task_state"]:
        raise ValueError("Evaluator state does not match replay")
    if read_json(directory / "private/run_status.json") != {"interface_error": result["interface_error"]}:
        raise ValueError("Interface status does not match raw outputs")
    return result["metrics"], result


def select_cases(manifest, base_task, timing):
    if timing not in ("early", "middle", "late", "all"):
        raise ValueError("Unknown timing selection; empty batches cannot pass acceptance")
    tasks = {row["source_task_id"] for row in manifest["cases"]}
    if base_task != "all" and base_task not in tasks:
        raise ValueError("Unknown base task selection")
    selected = [row for row in manifest["cases"] if (base_task == "all" or row["source_task_id"] == base_task)
                and (timing == "all" or row["timing"] == timing)]
    if not selected:
        raise ValueError("A run must retain a nonempty planned task denominator")
    return selected


def check_run(directory):
    """Audit every planned condition with fixed denominators and prefix pairing."""
    manifest = read_json(directory / "manifest.json")
    if manifest.get("kind") != "reviewed_three_task_cpu_run_v1" or manifest.get("code_bindings") != code_bindings():
        raise ValueError("CPU run kind/code bindings changed")
    source = Path(manifest["preparation_dir"])
    if sha256(source / "manifest.json") != manifest["preparation_sha256"]:
        raise ValueError("Preparation manifest changed")
    prepared = load_preparation(source)
    selected = select_cases(prepared, manifest["base_task"], manifest["timing"])
    planned = [{"case_id": case["case_id"], "condition": condition,
                "directory": "episodes/" + case["case_id"] + "/" + condition}
               for case in selected for condition in CONDITIONS]
    if manifest["planned_episodes"] != planned:
        raise ValueError("Planned task/condition denominator changed")
    cases = {case["case_id"]: case for case in selected}
    rows, results, pairing_errors = [], {}, []
    for slot in planned:
        case = cases[slot["case_id"]]
        row = {**slot, "task_type": case["task_type"], "task_success": False, "evidence_valid": False, "passed": False}
        try:
            if not case["prepared"]:
                raise ValueError("Unprepared source case: " + str(case["error"]))
            expected = fixture_case(read_json(within(source, case["config"])))
            metrics, result = audit_episode(within(directory, slot["directory"]), expected)
            row.update(metrics)
            condition = slot["condition"]
            expected_success = condition == "follow" or condition == "delayed" and case["task_type"] != "cancel_task"
            # Negative controls must fail for their declared reason, not merely
            # for any unrelated failure. Ignore still reaches A; Wrong executes
            # an invalid completion at k; Invalid preserves the failed k input.
            k = expected["events"][0]["trigger"]["value"]
            actions = result["trajectory"][1:]
            if condition == "ignore":
                valid_negative = (metrics.get("stopped_step") is not None and not metrics["semantic_violations"]
                    and navigation.goal_complete(expected, result["trajectory"][-1]["measurements"], expected["initial_goal"]))
            elif condition == "wrong":
                valid_negative = bool(len(actions) > k and actions[k]["action"] == "COMPLETE_SUBGOAL"
                    and {"step": k, "reason": "invalid_subgoal_completion"} in metrics["semantic_violations"])
            elif condition == "invalid":
                last = result["policy_records"][-1]
                valid_negative = bool(last["request"]["step"] == k and last["request"]["current_intervention"]
                    and last["error"] and last["action"] is None and len(actions) == k)
            elif condition == "delayed":
                cancellation = metrics.get("cancellation", {})
                valid_negative = bool(cancellation.get("stop_delay_actions") == 1
                    and cancellation.get("post_cancel_path_length_m") == 0.)
            else:
                valid_negative = False
            valid_negative = bool(valid_negative and metrics["intervention_received"]
                and (bool(result["interface_error"]) if condition == "invalid" else not result["interface_error"]))
            row["passed"] = bool(metrics["evidence_valid"] and metrics["task_success"] == expected_success
                                 and (expected_success or valid_negative))
            results[(slot["case_id"], condition)] = result
        except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration, OverflowError) as error:
            row.update(error=str(error), passed=False, task_success=False, evidence_valid=False)
        rows.append(row)
    for case in selected:
        follow = results.get((case["case_id"], "follow"))
        if follow is None:
            continue
        k = fixture_case(read_json(within(source, case["config"])))["events"][0]["trigger"]["value"]
        for condition in CONDITIONS[1:]:
            result = results.get((case["case_id"], condition))
            if result is None:
                continue
            physical = lambda value: [{key: row[key] for key in ("step", "action", "position", "rotation_xyzw", "distance_m", "collided")}
                                      for row in value["trajectory"][:k+1]]
            images = lambda value: [row["request"]["observation"]["rgb"]["sha256"] for row in value["policy_records"][:k+1]]
            if physical(follow) != physical(result) or images(follow) != images(result):
                pairing_errors.append(case["case_id"] + ": changed prefix for " + condition)
    return {"passed": len(rows) == len(planned) and all(row["passed"] for row in rows) and not pairing_errors,
        "execution_scope": SCOPE, "planned_cases": len(selected), "planned_episodes": len(planned),
        "conditions": list(CONDITIONS), "episodes": rows, "pairing_errors": pairing_errors,
        "accepted_episode_slots": sum(row["passed"] for row in rows),
        "successful_task_episodes": sum(row["task_success"] for row in rows),
        "policy_privileges": ["private_correct_update", "goal_positions", "synthetic_pose"],
        "real_model_called": False, "habitat_rendered": False, "hssd_navigation_executed": False, "release_ready": False}


def cpu(args):
    source = args.input_dir.resolve()
    output = fresh_output(args.output_dir, source)
    manifest = load_preparation(source)
    selected = select_cases(manifest, args.base_task, args.timing)
    planned = [{"case_id": case["case_id"], "condition": condition,
                "directory": "episodes/" + case["case_id"] + "/" + condition}
               for case in selected for condition in CONDITIONS]
    output.mkdir(parents=True)
    write_json(output / "manifest.json", {"kind": "reviewed_three_task_cpu_run_v1", "execution_scope": SCOPE,
        "preparation_dir": str(source), "preparation_sha256": sha256(source / "manifest.json"),
        "base_task": args.base_task, "timing": args.timing, "planned_episodes": planned, "code_bindings": code_bindings()})
    by_id = {case["case_id"]: case for case in selected}
    for slot in planned:
        case, folder = by_id[slot["case_id"]], output / slot["directory"]
        try:
            if not case["prepared"]:
                raise ValueError(case["error"])
            config = fixture_case(read_json(within(source, case["config"])))
            backend = FreePlaneBackend(config)
            result = navigation.run_episode(config, backend, privileged_policy(config, backend, slot["condition"]))
            write_episode(folder, config, result)
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
            write_json(folder / "private/slot_error.json", {"error": str(error)})
    report = check_run(output)
    write_json(output / "summary.json", report)
    print(f"Synthetic V CPU run: passed={report['passed']}; {report['planned_cases']} cases, {report['planned_episodes']} planned episodes; no HSSD/model execution")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "cpu", "audit"):
        child = sub.add_parser(command)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "prepare":
            child.add_argument("--annotations-dir", type=Path, default=PROJECT_ROOT / "outputs/benchmark_language_annotations_002")
        else:
            child.add_argument("--input-dir", type=Path, required=True)
        if command == "cpu":
            child.add_argument("--base-task", default="task_001")
            child.add_argument("--timing", choices=("early", "middle", "late", "all"), default="middle")
    args = parser.parse_args()
    try:
        if args.command in ("prepare", "cpu"):
            return globals()[args.command](args)
        from scripts.benchmark_audit import snapshot
        output = fresh_output(args.output_dir, args.input_dir)
        before = snapshot({"run": args.input_dir})
        report = check_run(args.input_dir)
        after = snapshot({"run": args.input_dir})
        report["source_files_unchanged"] = before == after
        report["passed"] = report["passed"] and before == after
        output.mkdir(parents=True)
        write_json(output / "audit.json", report)
        write_json(output / "source_bindings.json", {"before": before, "after": after})
        print(f"Read-only V CPU audit: passed={report['passed']}, source unchanged={before == after}")
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        print(f"Benchmark navigation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
