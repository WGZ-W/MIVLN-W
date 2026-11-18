"""Build private middle-timing drafts from human-approved HSSD object pairs.

Uses CPU navmesh actions for prefix replay and reference planning. The geometry
supports collection review. Language, terminal visibility and real navigation
still require their own validation before these drafts can enter the benchmark.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import math
from pathlib import Path
import sys

from mivln.benchmark_navigation import PROTOCOL, validate_case
from scripts.benchmark_manifest import PROJECT_ROOT, read_json, sha256
from scripts.hssd_targets import approved_targets, candidate_identity, check_preview_record
from scripts.hssd_visual_backend import NativeReplay
from scripts.hssd_visual_dataset import disjoint_output
from scripts.intervention_relatedness import digest
from scripts.navigation_scene import project_path
from scripts.public_inputs import write_json
from scripts.relatedness_geometry import measure_prefix, pose_matches
from scripts.strong_relatedness_candidates import difficulty, image_link
from scripts.subgoal import reference_trigger


FAMILIES = ("replacement", "subgoal")
SCENE_FIELDS = ("scene_id", "scene_sha256", "navmesh_sha256")
MOTION = ("move_forward", "turn_left", "turn_right")


def paired_distances(replay, point_a, point_b):
    """Distances in metres from the pre-action pose P and between stop points."""
    _, np, _, legacy, _ = replay.modules
    path = legacy.shortest_path(replay.pathfinder, np.asarray(point_b, dtype=np.float32),
                                np.asarray(point_a, dtype=np.float32))
    p_b, p_a = replay.distance(point_b), replay.distance(point_a)
    b_a = float(path.geodesic_distance) if path is not None else None
    if any(d is None or not math.isfinite(d) or d < 0 for d in (p_b, b_a, p_a)):
        raise ValueError("A required route segment is unreachable on the source navmesh")
    via = p_b + b_a
    return {"p_to_b_m": p_b, "b_to_a_m": b_a, "p_to_a_m": p_a,
            "via_b_m": via, "extra_over_direct_m": via - p_a}


def compare_distances(family, measured, baseline):
    """Compare with the old B at the exact same start, A stop and event pose."""
    if baseline is None:
        return {"status": "no_shared_prefix_comparator", "components": {}}
    keys = ("p_to_b_m",) if family == "replacement" else ("p_to_b_m", "b_to_a_m", "via_b_m")
    checks = {key: difficulty(measured[key], baseline[key], 2.0, .25) for key in keys}
    return {"status": "matched" if all(check["distance_match"] for check in checks.values())
            else "distance_mismatch", "components": checks}


def make_case(config, pair, family, base_id, goals, step):
    """One human intervention before action k. Subgoal B completes before A resumes."""
    a, b = pair["initial_goal"], pair["inserted_or_replacement_goal"]
    a_text, b_text = pair["initial_description_draft"], pair["updated_description_draft"]
    case_id = f"{pair['pair_id']}_{family}_middle"
    if family == "replacement":
        utterance = f"Cancel the previous destination. Go to {b_text} instead."
        update = {"operation": "replace_goal", "old_goal": a, "new_goal": b}
    elif family == "subgoal":
        utterance = f"First go to {b_text}. Then resume going to {a_text}."
        update = {"operation": "insert_subgoal", "subgoal": b, "resume_goal": a}
    else:
        raise ValueError("Unsupported pilot task type")
    # Copy only shared scene/agent parameters. Old catalog, language and event
    # bindings described different target pairs and cannot authorize this draft.
    parameters = (*SCENE_FIELDS, "scene", "scene_dataset_config", "scene_dataset_config_sha256",
                  "navmesh", "agent_height", "agent_radius", "seed", "start_position", "start_rotation_xyzw",
                  "forward_step_m", "turn_angle_degrees", "max_steps", "success_radius_m",
                  "sensor_height", "resolution", "semantic_sensor")
    draft = {key: deepcopy(config[key]) for key in parameters}
    draft.update(episode_id=case_id, base_task_id=base_id, task_type=family, timing="middle",
                 navigation_protocol=PROTOCOL, execution_scope="strong_relatedness_draft_only",
                 language_status="pending_human_review", initial_goal=a,
                 initial_instruction=f"Go to {a_text} and stop beside it.", goal_positions=deepcopy(goals),
                 yaw_tolerance_degrees=config.get("yaw_tolerance_degrees", 7.5),
                 events=[{"event_id": case_id + "_event", "utterance": utterance,
                          "trigger": {"type": "step", "value": step}, "task_update": update}])
    return validate_case(draft)


def plan_new_prefix(config, point, replay):
    """Derive middle from half the travelled metres on a CPU native A reference."""
    habitat, np = replay.modules[:2]
    replay.pathfinder.seed(config["seed"])
    follower = habitat.nav.GreedyGeodesicFollower(replay.pathfinder, replay.agent,
                                                  goal_radius=config["success_radius_m"], stop_key="STOP")
    goal = np.asarray(point, dtype=np.float32)
    rows = [{**replay.pose(), "action": "INITIALIZE", "distance_m": 0.0}]
    for _ in range(config["max_steps"]):
        action = follower.next_action_along(goal)
        if action != "STOP":
            if action not in MOTION:
                raise ValueError(f"Invalid reference action: {action}")
            replay.act(action)
        pose = replay.pose()
        rows.append({**pose, "action": action, "distance_m": math.dist(rows[-1]["position"], pose["position"])})
        if action == "STOP":
            return reference_trigger(rows, .5), rows
    raise ValueError("Reference to A did not stop within the budget")


def initial_plan(pair, catalog, geometry, bind):
    """Read a source-bound historical prefix or arrange a new one for A."""
    config = catalog["scene_config"]
    case_id = pair["reference_case_id"]
    if case_id is None:
        return config, None, None, None
    base = geometry[case_id]
    if base["geometry_evidence_status"] != "verified" or base["classification"]["decision"] != "approve":
        raise ValueError("A historical comparator requires reviewed native evidence")
    folder = Path(base["source_episode"])
    names = ("config.json", "trajectory.json", "events.json")
    previous, trajectory, events = [bind(folder / name, base["evidence_files"][name]) for name in names]
    if any(previous[key] != config[key] for key in SCENE_FIELDS) or previous["initial_goal"] != pair["initial_goal"]:
        raise ValueError("Historical prefix scene/mesh or original goal differs from approved pair")
    return previous, base, trajectory, events


def measure_pair(pair, catalog, targets, geometry, bind):
    config, base, trajectory, events = initial_plan(pair, catalog, geometry, bind)
    a, b = pair["initial_goal"], pair["inserted_or_replacement_goal"]
    # Keep the original A stop unchanged when comparing with an existing case.
    # The full-scene preview stop is only a photo site for that older A.
    a_point = config["goal_positions"][a] if base else targets[a]["stop_position"]
    b_point = targets[b]["stop_position"]
    replay = NativeReplay(config)
    try:
        if base:
            observed = measure_prefix(config, trajectory, events, replay)
            if any(base[key] != value for key, value in observed.items()):
                raise ValueError("Historical evidence differs from native replay")
            step = observed["delivered_step"]
            old_b = config["events"][0]["task_update"]["new_goal"]
            baseline = paired_distances(replay, a_point, config["goal_positions"][old_b])
            base_id = base["base_task_id"]
            provenance = "audited_middle_prefix_replay"
        else:
            step, trajectory = plan_new_prefix(config, a_point, replay)
            replay.close()
            replay = None
            replay = NativeReplay(config)
            for row in trajectory[1:step+1]:
                replay.act(row["action"])
            if not pose_matches(replay.pose(), trajectory[step]):
                raise ValueError("New reference prefix did not reproduce its delivery pose")
            baseline = None
            base_id = f"{config['scene_id']}@{config['scene_sha256']}/{pair['pair_id']}_pilot"
            provenance = "new_cpu_reference_half_travel"
        measured = paired_distances(replay, a_point, b_point)
        prefix = {"protocol": provenance, "delivered_step": step, "pose_before_delivery": replay.pose(),
                  "reference_case_id": pair["reference_case_id"], "actions": [row["action"] for row in trajectory[1:step+1]]}
        if base:
            prefix["comparator_relatedness"] = base["classification"]["relatedness_level"]
        if not base:
            prefix["reference_trajectory"] = trajectory
        drafts = {}
        for family in FAMILIES:
            draft = make_case(config, pair, family, base_id, {a: a_point, b: b_point}, step)
            draft["goal_metadata"] = {
                goal: {key: targets[goal][key] for key in ("template_id", "source_instance_index", "category")}
                for goal in (a, b)
            }
            drafts[family] = draft
        return drafts, prefix, measured, baseline
    finally:
        if replay is not None:
            replay.close()


def run(args):
    files = {}

    def bind(path, expected=None, *, parse=True):
        path = Path(path).resolve()
        value = sha256(path)
        old = files.setdefault(str(path), value)
        if value != old or (expected is not None and value != expected):
            raise ValueError(f"Evidence changed: {path}")
        return read_json(path) if parse else None

    approvals = bind(args.pairs)
    if approvals.get("schema_version") != 1 or approvals.get("review_scope") != "approved_strong_instance_relation_only":
        raise ValueError("Expected explicitly reviewed instance relations")
    pairs = approvals["pairs"]
    if not pairs or len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("Require a nonempty, unique approved pair roster")
    catalog, preview = bind(args.catalog), bind(args.previews)
    scene = catalog["scene_config"]
    if any(scene[key] != approvals[key] for key in ("scene_id", "scene_sha256")):
        raise ValueError("Approved relations belong to another scene")
    if (preview["candidate_identity"] != candidate_identity(catalog)
            or preview["catalog_sha256"] != files[str(args.catalog.resolve())]
            or any(preview["scene_provenance"][key] != scene[key] for key in SCENE_FIELDS[1:])):
        raise ValueError("Close-up package belongs to a different catalog or mesh")
    approved_targets(catalog, minimum=0)
    for item in catalog["sources"]:
        bind(project_path(item["path"]), item["sha256"], parse=False)
    targets = {row["target_id"]: row for row in catalog["targets"]}
    package = bind(args.geometry_dir / "geometry.json")
    seal = bind(args.geometry_dir / "source_bindings.json")
    if (package["kind"] != "relatedness_native_geometry_v1"
            or bind(args.geometry_dir / "summary.json")["passed"] is not True
            or seal["source_files_unchanged"] is not True):
        raise ValueError("Require successful historical geometry evidence")
    for name, expected in seal["code_bindings"].items():
        bind(PROJECT_ROOT / name, expected, parse=False)
    historical_roots = {name: Path(path).resolve() for name, path in seal["roots"].items()}
    geometry = {row["case_id"]: row for row in package["rows"]}
    if len(geometry) != len(package["rows"]):
        raise ValueError("Duplicate historical case IDs")
    output = disjoint_output(args.output_dir, [args.geometry_dir, args.previews.parent,
                                              *historical_roots.values(), *map(Path, files)])

    def bind_episode(path, expected):
        # The geometry report declares the episode, while the snapshot binds
        # its exact hot-path files to the original accepted navigation roots.
        resolved = path.resolve()
        if not any(root in resolved.parents and
                   seal["before"][name].get(resolved.relative_to(root).as_posix()) == expected
                   for name, root in historical_roots.items()):
            raise ValueError(f"Episode file is outside the audited snapshot: {resolved}")
        return bind(resolved, expected)

    cases, evidence = [], []
    index = ["# 强相关介入任务草案", "", "以下语言和几何供审核，关系批准范围为具体实例对。", ""]
    for pair in pairs:
        pair_id = pair["pair_id"]
        photos = {}
        try:
            a, b = pair["initial_goal"], pair["inserted_or_replacement_goal"]
            if a == b or set(pair["rgb_sha256"]) != {a, b}:
                raise ValueError("Require two distinct, image-bound targets")
            for goal in (a, b):
                record, target = preview["targets"][goal], targets[goal]
                if any(record[key] != target[key] for key in ("template_id", "source_instance_index", "stop_position")):
                    raise ValueError(f"Preview target binding mismatch: {goal}")
                check_preview_record(record, goal)
                for item in record["files"]:
                    bind(project_path(item["path"]), item["sha256"], parse=False)
                rgb = next(project_path(item["path"]) for item in record["files"] if Path(item["path"]).name == "rgb.png")
                bind(rgb, pair["rgb_sha256"][goal], parse=False)
                photos[goal] = image_link(rgb, output)
            drafts, prefix, measured, baseline = measure_pair(pair, catalog, targets, geometry, bind_episode)
            evidence.append({"pair_id": pair_id, "prefix": prefix, "distances_m": measured, "baseline_m": baseline,
                             "status": "prepared"})
            index += [f"## {pair_id} · {a} → {b}", "", f"![{a}]({photos[a]})", "", f"![{b}]({photos[b]})", "",
                      f"介入步 {prefix['delivered_step']}，P→B {measured['p_to_b_m']:.2f} m，B→A {measured['b_to_a_m']:.2f} m，"
                      f"P→A {measured['p_to_a_m']:.2f} m，额外行程 {measured['extra_over_direct_m']:.2f} m。", ""]
            for family, draft in drafts.items():
                match = compare_distances(family, measured, baseline)
                cases.append({"case_id": draft["episode_id"], "relation_id": pair_id, "task_type": family,
                              "relatedness_level": "strong", "status": "draft", "distance_comparison": match,
                              "restart_pair_id": draft["episode_id"] + "_continue_vs_restart_memory", "config": draft})
                index += [f"- {family} · {match['status']}", f"  - 初始指令 {draft['initial_instruction']}",
                          f"  - 介入指令 {draft['events'][0]['utterance']}"]
            index.append("")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            evidence.append({"pair_id": pair_id, "status": "failed", "error": str(error)})
            index += [f"## {pair_id}", "", f"几何准备失败。原因：{error}", ""]
            for family in FAMILIES:
                cases.append({"case_id": f"{pair_id}_{family}_middle", "relation_id": pair_id, "task_type": family,
                              "relatedness_level": "strong", "status": "blocked", "config": None})

    unchanged = all(sha256(Path(path)) == expected for path, expected in files.items())
    prepared = [row for row in cases if row["status"] == "draft"]
    summary = {"passed": unchanged and len(prepared) == len(cases), "approved_relations": len(pairs),
               "planned_cases": len(cases), "draft_cases": len(prepared), "language_approved_cases": 0,
               "distance_matched_by_type": {family: sum(row["task_type"] == family
                                             and row["distance_comparison"]["status"] == "matched" for row in prepared)
                                             for family in FAMILIES},
               "cases_without_comparator": sum(row["distance_comparison"]["status"] == "no_shared_prefix_comparator"
                                                for row in prepared),
               "source_files_unchanged": unchanged, "navigation_episodes_executed": 0, "release_ready": False}
    output.mkdir(parents=True)
    write_json(output / "cases.json", {"kind": "strong_relatedness_middle_drafts_v1", "cases": cases,
                                      "relations": approvals, "geometry": evidence})
    write_json(output / "summary.json", summary)
    write_json(output / "source_bindings.json", {"files": files, "producer_sha256": sha256(Path(__file__))})
    (output / "review_index.md").write_text("\n".join(index), encoding="utf-8")
    with (output / "language_review_sheet.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case_id", "draft_sha256", "initial_instruction", "intervention_instruction",
                         "decision", "reviewer", "note"))
        for row in prepared:
            draft = row["config"]
            writer.writerow((row["case_id"], digest(draft), draft["initial_instruction"],
                             draft["events"][0]["utterance"], "pending", "", ""))
    print(summary)
    return 0 if summary["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=PROJECT_ROOT / "configs/hssd_strong_pairs_102344280.json")
    for name in ("catalog", "previews", "geometry-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    try:
        return run(parser.parse_args())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"Strong relatedness pilot failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
