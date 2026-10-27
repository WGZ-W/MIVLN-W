"""Prepare, preview, and explicitly review instance-grounded HSSD stopping points."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys

from scripts.intervention_timing import load_json, write_json
from scripts.navigation_scene import PROJECT_ROOT, project_path


def sha256(path: Path) -> str:
    """Hash files in chunks; scene assets can be much larger than JSON reports."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def candidate_identity(catalog: dict) -> str:
    """Reviews may change; preview evidence must still bind the exact candidate set."""
    return fingerprint({
        "scene_config": catalog["scene_config"], "sources": catalog["sources"],
        "targets": [{key: value for key, value in target.items() if key != "review"}
                    for target in catalog["targets"]],
    })


def check_sources(sources: list) -> None:
    for source in sources:
        path = project_path(source["path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError(f"Source missing or changed: {path}")


def check_preview_record(record: dict, target_id: str) -> None:
    """A decision must refer to the complete, unchanged target evidence bundle."""
    files = record["files"]
    expected = {"rgb.png", "depth.npy", "semantic.npy", "category_mask.png"}
    if len(files) != 4 or {Path(item["path"]).name for item in files} != expected:
        raise ValueError("Preview must include RGB, depth, semantic IDs and category mask")
    if any(Path(item["path"]).parent.name != target_id for item in files):
        raise ValueError("Preview files belong to a different target")
    check_sources(files)


def approved_targets(catalog: dict, minimum: int = 3) -> list:
    """No implicit approval from class IDs, old metadata, or nonzero pixel counts."""
    if catalog.get("schema_version") != 1 or catalog.get("kind") != "hssd_target_catalog":
        raise ValueError("Unsupported HSSD target catalog")
    identities, names, approved, instances = set(), set(), [], set()
    for target in catalog["targets"]:
        identifier = target["target_id"]
        if not re.fullmatch(r"target_[0-9]+", identifier) or identifier in identities:
            raise ValueError("Target identifiers must be safe and unique")
        identities.add(identifier)
        point = target["stop_position"]
        if len(point) != 3 or not all(math.isfinite(v) for v in point):
            raise ValueError("Stopping points must have three finite world coordinates")
        review = target["review"]
        if review["status"] not in ("pending", "approved", "rejected"):
            raise ValueError("Unknown target review status")
        if review["status"] != "approved":
            continue
        description = " ".join(review.get("description", "").split()).casefold()
        if not description or not review.get("reviewer", "").strip() or not review.get("preview"):
            raise ValueError("Approved targets require a description, reviewer, and preview evidence")
        instance = (target["template_id"], target["source_instance_index"])
        if description in names or instance in instances:
            raise ValueError("Approved descriptions and object instances must be distinct")
        names.add(description)
        instances.add(instance)
        approved.append(target)
    if len(approved) < minimum:
        raise ValueError(f"Need {minimum} explicitly approved distinct targets; found {len(approved)}")
    return approved


def validate_catalog(catalog: dict, minimum: int = 3) -> list:
    """Check raw sources and review evidence without importing Habitat or using a GPU."""
    targets = approved_targets(catalog, minimum)
    check_sources(catalog["sources"])
    identity = candidate_identity(catalog)
    for target in targets:
        review = target["review"]
        preview = review["preview"]
        check_sources([preview])
        evidence = load_json(project_path(preview["path"]))
        if evidence["candidate_identity"] != identity:
            raise ValueError("Review preview belongs to a different candidate set")
        check_preview_record(evidence["targets"][target["target_id"]], target["target_id"])
        if evidence["targets"][target["target_id"]]["same_category_pixels"] <= 0:
            raise ValueError("Reviewed target has no rendered category pixels")
    return targets


def load_geometry(config: dict):
    """Only construct a CPU PathFinder; Simulator construction belongs to preview/run."""
    import habitat_sim
    import numpy as np
    from scripts.run_navigation import load_navigation_mesh, scene_provenance, validate_config

    validate_config(config)
    if not config.get("scene_dataset_config"):
        raise ValueError("An HSSD scene configuration is required")
    provenance = scene_provenance(config, project_path(config["scene"]))
    pathfinder = habitat_sim.PathFinder()
    load_navigation_mesh(pathfinder, config)
    return pathfinder, {**provenance, "versions": {
        "python": platform.python_version(), "habitat_sim": habitat_sim.__version__,
        "numpy": np.__version__,
    }}


def stopping_point(pathfinder, obj: dict, anchor: list, preferred=None):
    """Sample outside the XZ world AABB, on the anchor's reachable floor.

    AABB separation is only a geometry filter: a nearby point can be across a wall.
    The later preview/review gate is mandatory before this point becomes a target.
    """
    import habitat_sim
    import numpy as np

    low, high = (np.asarray(obj["world_aabb"][key], dtype=float) for key in ("min", "max"))
    anchor = np.asarray(anchor, dtype=np.float32)
    queries = []
    if preferred is not None:
        queries.append(np.asarray(preferred, dtype=np.float32))
    for t in np.linspace(0.1, 0.9, 9):
        x, z = (low + t * (high - low))[[0, 2]]
        queries.extend(np.array([qx, anchor[1], qz], dtype=np.float32) for qx, qz in (
            (low[0] - 0.6, z), (high[0] + 0.6, z), (x, low[2] - 0.6), (x, high[2] + 0.6)))
    choices = []
    for index, query in enumerate(queries):
        point = np.asarray(pathfinder.snap_point(query), dtype=np.float32)
        if (not np.isfinite(point).all() or not pathfinder.is_navigable(point)
                or abs(float(point[1] - anchor[1])) > 0.3
                or np.linalg.norm((point - query)[[0, 2]]) > 0.15):
            continue
        gap = float(np.linalg.norm(np.maximum(np.maximum(low[[0, 2]] - point[[0, 2]],
                                                         point[[0, 2]] - high[[0, 2]]), 0)))
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = anchor, point
        if (not 0.4 <= gap <= 0.7 or not pathfinder.find_path(path)
                or not math.isfinite(path.geodesic_distance) or path.geodesic_distance <= 1.0):
            continue
        record = {"stop_position": point.tolist(), "geometry": {
            "anchor_distance_m": float(path.geodesic_distance), "xz_aabb_gap_m": gap,
            "island": pathfinder.get_island(point)}}
        if preferred is not None and index == 0:
            return record
        choices.append(record)
    return min(choices, key=lambda item: item["geometry"]["anchor_distance_m"]) if choices else None


def prepare(args) -> int:
    config = load_json(args.base_config)
    config["semantic_sensor"] = True
    pathfinder, provenance = load_geometry(config)
    source = args.source_dir.resolve()
    metrics, assets = load_json(source / "metrics.json"), load_json(source / "assets.json")
    if not metrics["success"] or not assets["passed"] or metrics["scene_id"] != config["scene_id"]:
        raise ValueError("Use a successful smoke check for the configured scene")
    if project_path(assets["dataset_config"]) != project_path(config["scene_dataset_config"]):
        raise ValueError("Smoke assets and navigation configuration use different datasets")
    for key in ("scene", "scene_dataset_config", "navmesh"):
        if key == "navmesh":
            if sha256(source / "scene.navmesh") != provenance["navmesh_sha256"]:
                raise ValueError("Smoke navigation mesh differs from the configured mesh")
        elif assets["input_sha256"].get(provenance[key]) != provenance[f"{key}_sha256"]:
            raise ValueError(f"Smoke input hash differs: {key}")
    source_objects = load_json(source / "source_objects.json")
    targets, rejected = [], []
    for obj in load_json(source / "objects.json"):
        if obj["category"] not in args.categories:
            continue
        matches = [entry for entry in source_objects if entry["template_id"] == obj["template_id"]]
        reason = None
        if obj["category_labels_agree"] is not True:
            reason = "metadata/lexicon categories are not both available and equal"
        elif len(matches) != 1:
            # A runtime ID is not a stable substitute for an ambiguous source index.
            reason = "template has multiple or missing source instances; explicit mapping required"
        elif matches[0]["category_id"] != obj["semantic_id"]:
            reason = "runtime category differs from the source template"
        if reason:
            rejected.append({"template_id": obj["template_id"], "reason": reason})
            continue
        source_index = matches[0]["source_instance_index"]
        preferred = next((config["goal_positions"][name] for name, item in config.get("goal_metadata", {}).items()
                          if item["template_id"] == obj["template_id"]
                          and item["source_instance_index"] == source_index), None)
        point = stopping_point(pathfinder, obj, config["start_position"], preferred)
        if point is None:
            rejected.append({"template_id": obj["template_id"], "reason": "no reachable stopping candidate"})
            continue
        targets.append({
            "target_id": f"target_{source_index:03d}", "source_instance_index": source_index,
            "template_id": obj["template_id"], "category": obj["category"],
            "semantic_id": obj["semantic_id"], "display_name": obj["display_name"],
            "world_aabb": obj["world_aabb"], **point, "review": {"status": "pending"},
        })
    if not targets:
        raise ValueError("No eligible semantic stopping candidates")
    paths = [args.base_config.resolve(), *[source / name for name in
             ("metrics.json", "assets.json", "objects.json", "source_objects.json")],
             *[Path(provenance[key]) for key in ("scene", "scene_dataset_config", "navmesh")]]
    catalog = {"schema_version": 1, "kind": "hssd_target_catalog", "scene_config": config,
               "sources": [{"path": str(path), "sha256": sha256(path)} for path in paths],
               "targets": sorted(targets, key=lambda item: item["target_id"]), "rejected": rejected,
               "note": "Geometry candidates only; category masks do not establish object-instance visibility"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "catalog.json", catalog)
    for target in catalog["targets"]:
        print(target["target_id"], target["display_name"], target["stop_position"], "PENDING REVIEW")
    return 0


def preview(args) -> int:
    """Teleport only for offline evidence capture; this is not a navigation episode."""
    import habitat_sim
    import numpy as np
    from PIL import Image
    from habitat_sim.utils.common import quat_from_coeffs
    from scripts.check_hssd import export_objects
    from scripts.run_navigation import make_configuration, load_navigation_mesh

    catalog = load_json(args.catalog)
    approved_targets(catalog, minimum=0)
    check_sources(catalog["sources"])
    config = catalog["scene_config"]
    _, provenance = load_geometry(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"catalog": str(args.catalog.resolve()), "catalog_sha256": sha256(args.catalog),
              "candidate_identity": candidate_identity(catalog), "scene_provenance": provenance, "targets": {}}
    with habitat_sim.Simulator(make_configuration(config, project_path(config["scene"]), args.gpu_id)) as sim:
        sim.seed(config["seed"])
        load_navigation_mesh(sim.pathfinder, config)
        objects = export_objects(sim, {}, [])
        if len(objects) != provenance["expected_rigid_objects"]:
            raise RuntimeError("Furniture count differs from the source scene")
        for target in catalog["targets"]:
            matching = [obj for obj in objects if obj["template_id"] == target["template_id"]]
            if (len(matching) != 1 or matching[0]["semantic_id"] != target["semantic_id"]
                    or not np.allclose(matching[0]["world_aabb"]["center"], target["world_aabb"]["center"], atol=1e-4)):
                raise ValueError("Preview instance does not match the catalog template/pose")
            point = np.array(target["stop_position"], dtype=np.float32)
            if not sim.pathfinder.is_navigable(point):
                raise ValueError("Preview stopping point is not navigable")
            center = np.asarray(target["world_aabb"]["center"])
            direction = center - point
            yaw = math.atan2(-direction[0], -direction[2])
            rotation = quat_from_coeffs(np.array([0, math.sin(yaw / 2), 0, math.cos(yaw / 2)]))
            state = habitat_sim.AgentState()
            state.position, state.rotation = point, rotation
            agent = sim.initialize_agent(0, state)
            state = agent.get_state()
            # World yaw followed by local X pitch keeps the camera upright (zero roll).
            # Sensor poses are only changed for preview; navigation retains its normal camera.
            for sensor in state.sensor_states.values():
                delta = center - np.asarray(sensor.position)
                pitch = math.atan2(delta[1], np.linalg.norm(delta[[0, 2]]))
                sensor.rotation = rotation * quat_from_coeffs(np.array([math.sin(pitch / 2), 0, 0, math.cos(pitch / 2)]))
            agent.set_state(state, reset_sensors=False, infer_sensor_states=False)
            observations = sim.get_sensor_observations()
            rgb, depth, semantic = (observations[key] for key in ("rgb", "depth", "semantic"))
            if (rgb.shape[:2] != depth.shape or semantic.shape != depth.shape
                    or not np.isfinite(depth).all() or not np.any(depth > 0)):
                raise RuntimeError("Invalid preview sensor output")
            folder = args.output_dir / target["target_id"]
            folder.mkdir()
            Image.fromarray(rgb[:, :, :3]).save(folder / "rgb.png")
            np.save(folder / "depth.npy", depth)
            np.save(folder / "semantic.npy", semantic)
            mask = semantic == target["semantic_id"]
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "category_mask.png")
            report["targets"][target["target_id"]] = {
                "same_category_pixels": int(mask.sum()), "manually_verified": False,
                "stop_position": point.tolist(), "template_id": target["template_id"],
                "source_instance_index": target["source_instance_index"],
                "camera_position": np.asarray(state.sensor_states["rgb"].position).tolist(),
                "camera_rotation_xyzw": [*state.sensor_states["rgb"].rotation.imag.tolist(),
                                         float(state.sensor_states["rgb"].rotation.real)],
                "files": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in sorted(folder.iterdir())],
            }
    write_json(args.output_dir / "preview.json", report)
    print(f"Preview complete: {args.output_dir}; explicit review is still required")
    return 0


def review(args) -> int:
    """Produce a new catalog version, preserving the original and earlier decisions."""
    catalog, evidence = load_json(args.catalog), load_json(args.previews)
    approved_targets(catalog, minimum=0)
    check_sources(catalog["sources"])
    if evidence["candidate_identity"] != candidate_identity(catalog):
        raise ValueError("Preview evidence is for different objects or stopping points")
    matches = [target for target in catalog["targets"] if target["target_id"] == args.target_id]
    if len(matches) != 1 or matches[0]["review"]["status"] != "pending":
        raise ValueError("Select one pending target; existing decisions are not overwritten")
    target = matches[0]
    record = evidence["targets"][args.target_id]
    check_preview_record(record, args.target_id)
    if args.decision == "approve" and (not args.description or record["same_category_pixels"] <= 0):
        raise ValueError("Approval requires a visually grounded description and rendered category pixels")
    target["review"] = {"status": "approved" if args.decision == "approve" else "rejected",
                        "description": args.description, "reviewer": args.reviewer, "note": args.note,
                        "reviewed_at": datetime.now(timezone.utc).isoformat(),
                        "preview": {"path": str(args.previews.resolve()), "sha256": sha256(args.previews)}}
    approved_targets(catalog, minimum=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.decision}: {args.target_id} -> {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="CPU: create unreviewed geometry candidates")
    preparation.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "outputs/hssd_smoke_001")
    preparation.add_argument("--base-config", type=Path, default=PROJECT_ROOT / "configs/simulator/hssd_navigation_001.json")
    preparation.add_argument("--categories", nargs="+", default=["bed", "couch"])
    preparation.add_argument("--output-dir", type=Path, required=True)
    rendering = commands.add_parser("preview", help="GPU: render candidate evidence; never auto-approve")
    rendering.add_argument("--output-dir", type=Path, required=True)
    rendering.add_argument("--gpu-id", type=int, default=0)
    reviewing = commands.add_parser("review", help="Record one explicit approve/reject decision in a NEW file")
    reviewing.add_argument("--previews", type=Path, required=True)
    reviewing.add_argument("--target-id", required=True)
    reviewing.add_argument("--decision", choices=("approve", "reject"), required=True)
    reviewing.add_argument("--description", help="Unique visual referring expression, not just a retailer product name")
    reviewing.add_argument("--reviewer", required=True)
    reviewing.add_argument("--note", default="")
    reviewing.add_argument("--output", type=Path, required=True)
    validating = commands.add_parser("validate", help="CPU: verify at least three approved targets and evidence")
    for command in (rendering, reviewing, validating):
        command.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "preview":
        if args.gpu_id < 0:
            parser.error("--gpu-id must be non-negative")
        return preview(args)
    if args.command == "review":
        if not args.reviewer.strip():
            parser.error("--reviewer must be non-empty")
        return review(args)
    targets = validate_catalog(load_json(args.catalog))
    print(f"Catalog valid: {len(targets)} explicitly reviewed targets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
