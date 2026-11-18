"""Run reviewed strong-relatedness pilots with a fixed prefix and a native audit.

The human review authorizes the six exact language drafts for a feasibility
pilot. Oracle uses the correct private update and geometry; Ignore continues
toward A. Both execute the same saved prefix through the ordinary action loop.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys

from mivln import benchmark_navigation as navigation
from scripts import hssd_targets as targets
from scripts import hssd_visual_navigation as legacy
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import PROJECT_ROOT, read_json, sha256, within
from scripts.hssd_visual_dataset import disjoint_output, require_disjoint
from scripts.intervention_relatedness import digest
from scripts.public_inputs import write_json, write_lines
from scripts.relatedness_geometry import pose_matches


KIND = "hssd_strong_middle_navigation_preparation_v1"
RUN_KIND = "hssd_strong_middle_navigation_run_v1"
FAMILIES = ("replacement", "subgoal")
CONDITIONS = legacy.CONDITIONS
MOTION = ("move_forward", "turn_left", "turn_right")


def bindings():
    return {**legacy.bindings(), **{name: sha256(PROJECT_ROOT / name) for name in
            ("scripts/strong_relatedness_navigation.py", "scripts/relatedness_geometry.py",
             "scripts/hssd_targets.py", "scripts/intervention_relatedness.py")}}


def check_review(review, cases_file, cases):
    """Use the exact six reviewer declarations, including each draft's hash."""
    if (review.get("kind") != "strong_relatedness_pilot_target_grounding_review_v1"
            or type(review.get("schema_version")) is not int or review["schema_version"] != 1
            or review["review_scope"] != "initial_and_intervention_instruction_target_reference"
            or not review["reviewer"].strip() or review["source_cases_sha256"] != sha256(cases_file)):
        raise ValueError("Target-reference review does not bind this draft package")
    expected = {(row["case_id"], digest(row["config"])) for row in cases}
    decisions = review["decisions"]
    if (len(decisions) != len(expected) or len(expected) != 6
            or review["approved_target_reference_cases"] != len(expected)
            or review["planned_cases"] != len(expected)
            or any(row["decision"] != "approve" for row in decisions)
            or {(row["case_id"], row["draft_sha256"]) for row in decisions} != expected):
        raise ValueError("All six exact draft configurations require approval")


def validate_prefix(prefix, trigger, result):
    """Check the physical prefix through pose P, immediately before action k."""
    trajectory, records = result["trajectory"], result["policy_records"]
    if len(trajectory) <= trigger+1 or len(records) < trigger+1:
        raise ValueError("Episode did not execute the intervention action")
    if [row["action"] for row in trajectory[1:trigger+1]] != prefix["actions"]:
        raise ValueError("Executed actions differ from the fixed intervention prefix")
    if not pose_matches(trajectory[trigger], prefix["pose_before_delivery"]):
        raise ValueError("Actual pre-action intervention pose differs from the prepared pose")
    # RGB k belongs to P, before any resulting branch action. Asset IDs differ
    # across sessions, so compare the physical trace and the RGB byte hashes.
    physical = [{key: row[key] for key in ("step", "action", "position", "rotation_xyzw", "collided", "distance_m")}
                for row in trajectory[:trigger+1]]
    images = [row["request"]["observation"]["rgb"]["sha256"] for row in records[:trigger+1]]
    return physical, images


def build(cases_dir, review_dir, catalog_path, preview_path, prepared_dir):
    """Reconstruct one pilot-scoped catalog and the fixed denominator of 12.

    The catalog retains the photographed candidate roster and geometry so its
    fingerprint still binds the original preview. Human language approval is
    translated into the legacy catalog's approved flag for these six pilot
    targets only. Runtime instance masks still decide actual visibility.
    """
    cases_file = cases_dir / "cases.json"
    summary_file = cases_dir / "summary.json"
    seal_file = cases_dir / "source_bindings.json"
    review_file = review_dir / "review.json"
    source_files = {str(path.resolve()): sha256(path) for path in
                    (cases_file, summary_file, seal_file, review_file, catalog_path, preview_path)}
    seal = read_json(seal_file)
    targets.check_sources([{"path": path, "sha256": value} for path, value in seal["files"].items()])
    for path in (catalog_path, preview_path):
        if seal["files"].get(str(path.resolve())) != source_files[str(path.resolve())]:
            raise ValueError("Catalog or preview is outside the bound draft sources")
    package, summary, review = (read_json(path) for path in (cases_file, summary_file, review_file))
    if (package.get("kind") != "strong_relatedness_middle_drafts_v1" or summary["passed"] is not True
            or summary["draft_cases"] != 6 or Path(review["source_cases"]).is_absolute()
            or (review_dir / review["source_cases"]).resolve() != cases_file.resolve()):
        raise ValueError("Require the complete six-case pilot and its target-reference review")
    cases = package["cases"]
    if len(cases) != 6 or any(row["status"] != "draft" or row["relatedness_level"] != "strong" for row in cases):
        raise ValueError("Missing or invalid strong-relatedness pilot cases")
    check_review(review, cases_file, cases)
    relations = package["relations"]["pairs"]
    relation_index = {row["pair_id"]: row for row in relations}
    prefixes = {row["pair_id"]: row["prefix"] for row in package["geometry"] if row["status"] == "prepared"}
    if (len(relations) != 3 or len(relation_index) != 3 or set(prefixes) != set(relation_index)
            or {(row["relation_id"], row["task_type"]) for row in cases}
            != {(key, family) for key in relation_index for family in FAMILIES}):
        raise ValueError("Require one replacement and one subgoal for each geometry-verified relation")
    catalog, preview = read_json(catalog_path), read_json(preview_path)
    if (preview["candidate_identity"] != targets.candidate_identity(catalog)
            or preview["catalog_sha256"] != source_files[str(catalog_path.resolve())]):
        raise ValueError("Preview does not bind the source catalog")
    by_goal = {target["target_id"]: target for target in catalog["targets"]}
    anchor_configs, configs, slots = {}, {}, []
    for row in cases:
        relation_id, family = row["relation_id"], row["task_type"]
        case_id = f"{relation_id}_{family}_middle"
        if not re.fullmatch(r"strong_pair_[0-9]+", relation_id) or row["case_id"] != case_id:
            raise ValueError("Unexpected pilot case identity")
        relation, prefix = relation_index[relation_id], prefixes[relation_id]
        config = deepcopy(row["config"])
        navigation.validate_case(config)
        a, b = relation["initial_goal"], relation["inserted_or_replacement_goal"]
        event = config["events"][0]
        k = event["trigger"]["value"]
        if (config["episode_id"] != case_id or config["task_type"] != family or config["timing"] != "middle"
                or config["initial_goal"] != a or set(config["goal_positions"]) != {a, b}
                or event["task_update"]["new_goal" if family == "replacement" else "subgoal"] != b
                or k != prefix["delivered_step"] or len(prefix["actions"]) != k
                or any(action not in MOTION for action in prefix["actions"])
                or not pose_matches(prefix["pose_before_delivery"], prefix["pose_before_delivery"])):
            raise ValueError(f"Draft semantics or fixed prefix differs from pilot plan: {case_id}")
        shared = {key: value for key, value in config.items() if key not in ("episode_id", "task_type", "events")}
        if anchor_configs.setdefault(relation_id, shared) != shared:
            raise ValueError(f"Task types must share the same scene, start, goals and action budget: {relation_id}")
        for goal, description in ((a, relation["initial_description_draft"]), (b, relation["updated_description_draft"])):
            target, photo = by_goal[goal], preview["targets"][goal]
            keys = ("template_id", "source_instance_index")
            if (any(target[key] != config["goal_metadata"][goal][key] or target[key] != photo[key] for key in keys)
                    or target["stop_position"] != config["goal_positions"][goal]
                    or photo["stop_position"] != target["stop_position"] or photo["same_category_pixels"] <= 0):
                raise ValueError(f"Pilot goal does not match the photographed instance and stop point: {goal}")
            annotation = {"status": "approved", "description": description, "reviewer": review["reviewer"],
                          "review_scope": "strong_pilot_target_reference_only",
                          "preview": {"path": str(preview_path.resolve()), "sha256": source_files[str(preview_path.resolve())]}}
            if target["review"]["status"] == "approved" and target["review"] != annotation:
                raise ValueError(f"Conflicting target descriptions in pilot: {goal}")
            target["review"] = annotation
        config.update(execution_scope=legacy.SCOPE, pilot_review_scope="target_reference_only",
                      terminal_alignment={"enabled": True, "yaw_tolerance_degrees": config["yaw_tolerance_degrees"],
                                          "max_turn_actions": 24},
                      terminal_visibility={"enabled": True, "min_pixels": 64})
        # Filled after serialization, so both generation and read-only loading
        # bind the exact derived catalog consumed by the existing backend.
        config["catalog_source"] = {"path": str(prepared_dir / "catalog.json"), "sha256": None}
        relative = f"configs/{case_id}.json"
        configs[relative] = config
        slots += [{"case_id": case_id, "relation_id": relation_id, "task_type": family, "condition": condition,
                   "config": relative, "directory": f"episodes/{case_id}/{condition}"} for condition in CONDITIONS]
    # Keep the candidate fingerprint intact. The legacy fingerprint includes
    # sources and all non-review target fields. Bind the extra review outside it.
    catalog["pilot_review_binding"] = {"path": str(review_file.resolve()),
                                       "sha256": source_files[str(review_file.resolve())]}
    targets.validate_catalog(catalog)
    manifest = {"kind": KIND, "cases_dir": str(cases_dir.resolve()), "review_dir": str(review_dir.resolve()),
                "catalog_source": str(catalog_path.resolve()), "preview_source": str(preview_path.resolve()),
                "source_files": source_files, "code_bindings": bindings(), "prefixes": prefixes,
                "planned_cases": len(cases), "slots": slots, "release_ready": False}
    return manifest, configs, catalog, seal


def prepare(args):
    # Check overlap before constructing any output. Frozen draft sources include
    # the earlier run, scene and target preview files alongside the review.
    seal = read_json(args.cases_dir / "source_bindings.json")
    output = disjoint_output(args.output_dir, [args.cases_dir, args.review_dir, args.previews.parent,
                                               args.catalog.parent, *map(Path, seal["files"])])
    manifest, configs, catalog, _ = build(args.cases_dir, args.review_dir, args.catalog, args.previews, output)
    output.mkdir(parents=True)
    write_json(output / "catalog.json", catalog)
    manifest["catalog_sha256"] = sha256(output / "catalog.json")
    for relative, config in configs.items():
        config["catalog_source"]["sha256"] = manifest["catalog_sha256"]
        write_json(output / relative, config)
    manifest["config_hashes"] = {name: sha256(output / name) for name in configs}
    write_json(output / "manifest.json", manifest)
    print("Prepared six reviewed cases and twelve Oracle/Ignore slots.")
    return 0


def load_prepared(directory):
    directory = directory.resolve()
    actual = read_json(directory / "manifest.json")
    if actual.get("kind") != KIND or actual.get("code_bindings") != bindings():
        raise ValueError("Pilot preparation kind/code binding changed")
    expected, configs, catalog, seal = build(Path(actual["cases_dir"]), Path(actual["review_dir"]),
                                             Path(actual["catalog_source"]), Path(actual["preview_source"]), directory)
    catalog_file = directory / "catalog.json"
    if read_json(catalog_file) != catalog:
        raise ValueError("Prepared pilot catalog differs from validated sources")
    expected["catalog_sha256"] = sha256(catalog_file)
    expected["config_hashes"] = {}
    for relative, config in configs.items():
        config["catalog_source"]["sha256"] = expected["catalog_sha256"]
        path = within(directory, relative)
        if read_json(path) != config:
            raise ValueError("Prepared pilot configuration differs from the reviewed draft")
        expected["config_hashes"][relative] = sha256(path)
    if actual != expected:
        raise ValueError("Prepared manifest or source bindings changed")
    protected = [directory, Path(actual["cases_dir"]), Path(actual["review_dir"]),
                 Path(actual["preview_source"]).parent, Path(actual["catalog_source"]).parent, *map(Path, seal["files"])]
    return actual, protected


def episode(args):
    prepared, protected = load_prepared(args.input_dir)
    slot = next((row for row in prepared["slots"] if (row["case_id"], row["condition"])
                 == (args.case_id, args.condition)), None)
    if slot is None:
        raise ValueError("Unknown pilot case/condition slot")
    config = read_json(within(args.input_dir, slot["config"]))
    output = disjoint_output(args.output_dir, protected)
    output.mkdir(parents=True)
    prefix = prepared["prefixes"][slot["relation_id"]]["actions"]
    backend = None
    try:
        backend = legacy.HabitatBackend(config, args.gpu_id, output)
        policy = legacy.OraclePolicy(config, backend, slot["condition"])

        def callback(request, assets):
            step = request["step"]
            if step < len(prefix):
                action = prefix[step]
                policy.trace.append({"step": step, "phase": "fixed_prefix", "action": action,
                                     "policy_goal": config["initial_goal"]})
                return json.dumps({"action": action})
            return policy(request, assets)

        result = navigation.run_episode(config, backend, callback)
        legacy.write_result(output, config, result)
        write_lines(output / "private/policy_trace.jsonl", policy.trace)
        return int(result["interface_error"] is not None)
    except Exception as error:
        write_json(output / "private/slot_error.json", {"error": f"{type(error).__name__}: {error}"})
        return 1
    finally:
        if backend is not None:
            backend.close()


def check_run(directory):
    """Audit raw responses, actual poses and masks independently of the policy."""
    manifest = read_json(directory / "manifest.json")
    if manifest.get("kind") != RUN_KIND or manifest.get("code_bindings") != bindings():
        raise ValueError("Pilot run kind/code binding changed")
    prepared_dir = Path(manifest["prepared_dir"])
    if sha256(prepared_dir / "manifest.json") != manifest["prepared_sha256"]:
        raise ValueError("Bound pilot preparation changed")
    prepared, protected = load_prepared(prepared_dir)
    require_disjoint(directory, protected)
    statuses = read_json(directory / "execution_status.json")
    slots = prepared["slots"]
    keys = ("case_id", "condition")
    if (manifest["slots"] != slots or len(statuses) != len(slots)
            or any(any(status[key] != slot[key] for key in keys) for slot, status in zip(slots, statuses))):
        raise ValueError("Execution denominator or order differs from the prepared plan")
    rows, prefixes, pairing_errors = [], {}, []
    for slot, status in zip(slots, statuses):
        row = {**slot, "passed": False, "task_success": False, "evidence_valid": False}
        try:
            if status["state"] != "completed" or status["returncode"] != 0:
                raise ValueError(f"Child did not complete: {status}")
            config = read_json(within(prepared_dir, slot["config"]))
            folder = within(directory, slot["directory"])
            if read_json(folder / "private/backend.json")["gpu_id"] != manifest["gpu_id"]:
                raise ValueError("Episode GPU differs from the planned run")
            metrics, result = legacy.audit_episode(folder, config)
            row.update(metrics)
            k = config["events"][0]["trigger"]["value"]
            actual_prefix = validate_prefix(prepared["prefixes"][slot["relation_id"]], k, result)
            relation = slot["relation_id"]
            if prefixes.setdefault(relation, actual_prefix) != actual_prefix:
                pairing_errors.append(f"{slot['case_id']} / {slot['condition']}: physical or RGB prefix differs")
            valid = (metrics["evidence_valid"] and metrics["intervention_received"]
                     and metrics["budget_satisfied"] and metrics["stopped_step"] is not None
                     and not metrics["semantic_violations"] and metrics["interface_error"] is None)
            if slot["condition"] == "oracle":
                row["passed"] = bool(valid and metrics["task_success"])
            else:
                at_a = navigation.goal_complete(config, result["trajectory"][-1]["measurements"], config["initial_goal"])
                row["passed"] = bool(valid and not metrics["task_success"] and at_a)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, OverflowError) as error:
            row.update(error=str(error), passed=False, evidence_valid=False)
        rows.append(row)
    return {"passed": all(row["passed"] for row in rows) and not pairing_errors,
            "planned_cases": prepared["planned_cases"], "planned_episodes": len(slots),
            "accepted_episode_slots": sum(row["passed"] for row in rows), "episodes": rows,
            "pairing_errors": pairing_errors, "real_model_called": False,
            "oracle_privileges": ["fixed_prefix", "correct_private_update", "geodesic_follower", "instance_centres"],
            "release_ready": False}


def run(args):
    prepared, protected = load_prepared(args.input_dir)
    output = disjoint_output(args.output_dir, protected)
    output.mkdir(parents=True)
    slots = prepared["slots"]
    statuses = [{"case_id": slot["case_id"], "condition": slot["condition"], "state": "pending", "returncode": None}
                for slot in slots]
    write_json(output / "manifest.json", {"kind": RUN_KIND, "prepared_dir": str(args.input_dir.resolve()),
               "prepared_sha256": sha256(args.input_dir / "manifest.json"), "slots": slots,
               "gpu_id": args.gpu_id, "code_bindings": bindings()})
    status_file = output / "execution_status.json"
    write_json(status_file, statuses)
    (output / "logs").mkdir()
    for slot, status in zip(slots, statuses):
        command = [sys.executable, "-B", "-m", "scripts.strong_relatedness_navigation", "episode",
                   "--input-dir", str(args.input_dir.resolve()), "--case-id", slot["case_id"],
                   "--condition", slot["condition"], "--gpu-id", str(args.gpu_id),
                   "--output-dir", str(within(output, slot["directory"]))]
        status["state"] = "running"
        write_json(status_file, statuses)
        try:
            log = output / "logs" / f"{slot['case_id']}_{slot['condition']}.log"
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status.update(state="completed" if completed.returncode == 0 else "failed", returncode=completed.returncode)
        except KeyboardInterrupt:
            status["state"] = "interrupted"
            write_json(status_file, statuses)
            raise
        except OSError as error:
            status.update(state="failed", error=str(error))
        write_json(status_file, statuses)
    report = check_run(output)
    write_json(output / "summary.json", report)
    print(f"Accepted {report['accepted_episode_slots']}/{report['planned_episodes']} pilot slots")
    return 0 if report["passed"] else 1


def audit(args):
    manifest = read_json(args.input_dir / "manifest.json")
    _, protected = load_prepared(Path(manifest["prepared_dir"]))
    output = disjoint_output(args.output_dir, [args.input_dir, *protected])
    roots = {"run": args.input_dir, "prepared": Path(manifest["prepared_dir"])}
    before = snapshot(roots)
    report = check_run(args.input_dir)
    report["audited_files_unchanged"] = before == snapshot(roots)
    report["passed"] = report["passed"] and report["audited_files_unchanged"]
    output.mkdir(parents=True)
    write_json(output / "audit.json", report)
    print(f"Audit passed={report['passed']}; accepted={report['accepted_episode_slots']}/{report['planned_episodes']}")
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Validate review and prepare twelve slots without rendering")
    for name in ("cases-dir", "review-dir", "catalog", "previews", "output-dir"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("run", "episode", "audit"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--input-dir", type=Path, required=True)
        cmd.add_argument("--output-dir", type=Path, required=True)
        if name != "audit":
            cmd.add_argument("--gpu-id", type=int, default=0)
        if name == "episode":
            cmd.add_argument("--case-id", required=True)
            cmd.add_argument("--condition", choices=CONDITIONS, required=True)
    args = parser.parse_args()
    try:
        return {"prepare": prepare, "run": run, "episode": episode, "audit": audit}[args.command](args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, ImportError) as error:
        print(f"Strong-relatedness navigation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
