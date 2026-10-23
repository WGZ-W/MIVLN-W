"""Audit one HSSD scene and inspect RGB-D, class semantics, objects, and navigation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inspect_assets(dataset_config: Path, scene_id: str) -> tuple:
    """Check declared dependencies of an original, rigid HSSD scene using only CPU I/O."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", scene_id):
        raise ValueError("scene-id must be a scene identifier, not a filesystem path")
    dataset_config = dataset_config.resolve()
    root = dataset_config.parent
    issues = []
    checked = {}

    def check_file(path):
        path = path.resolve()
        if path not in checked:
            error = None
            if not path.is_file():
                error = "missing file"
            else:
                with path.open("rb") as stream:
                    header = stream.read(80)
                if not header:
                    error = "empty file"
                elif header.startswith(LFS_POINTER):
                    error = "Git LFS pointer; payload has not been downloaded"
                elif path.suffix.lower() == ".glb" and not header.startswith(b"glTF"):
                    error = "invalid GLB header"
            checked[path] = error is None
            if error:
                issues.append({"path": str(path), "error": error})
        return checked[path]

    def load(path):
        if not check_file(path):
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            issues.append({"path": str(path), "error": str(error)})
            return {}

    def check_dependencies(path, template):
        for key in ("render_asset", "collision_asset"):
            asset = template.get(key)
            if asset and asset.lower() != "none":
                check_file(path.parent / asset)
        for annotation in template.get("user_defined", {}).values():
            if isinstance(annotation, dict) and annotation.get("mesh_filepath"):
                check_file(path.parent / annotation["mesh_filepath"])

    dataset = load(dataset_config)
    scene_file = root / "scenes" / f"{scene_id}.scene_instance.json"
    scene = load(scene_file)
    if scene.get("articulated_object_instances"):
        issues.append({"path": str(scene_file), "error": "This check supports original rigid scenes; articulated scenes need separate handling"})
    stage_name = scene.get("stage_instance", {}).get("template_name")
    if stage_name:
        stage_path = root / (stage_name if stage_name.endswith(".stage_config.json") else stage_name + ".stage_config.json")
        check_dependencies(stage_path, load(stage_path))
    else:
        issues.append({"path": str(scene_file), "error": "Missing stage instance"})
    descriptor = dataset.get("scene_instances", {}).get("default_attributes", {}).get("semantic_scene_instance")
    lexicon_name = dataset.get("semantic_scene_descriptor_instances", {}).get(descriptor)
    lexicon = load(root / lexicon_name) if lexicon_name else {}
    classes = {entry["id"]: entry["name"] for entry in lexicon.get("classes", [])}
    if not classes:
        issues.append({"path": str(dataset_config), "error": "No semantic class lexicon found"})
    region_path = root / "semantics" / "scenes" / f"{scene_id}.semantic_config.json"
    region_data = load(region_path) if region_path.is_file() else {}
    regions = region_data.get("region_annotations", [])
    index = defaultdict(list)
    for pattern in dataset.get("objects", {}).get("paths", {}).get(".json", []):
        for directory in root.glob(pattern):
            for path in directory.glob("*.object_config.json"):
                name = path.name.removesuffix(".object_config.json")
                if path not in index[name]:
                    index[name].append(path)
    metadata_path = root / "metadata" / "objects.json"
    metadata = load(metadata_path) if metadata_path.is_file() else {}
    category_path = root / "metadata" / "hssd_obj_semantics_condensed.csv"
    metadata_categories = {}
    if category_path.is_file() and check_file(category_path):
        with category_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            category_key = next((key for key in reader.fieldnames or [] if "CONDENSED" in key), None)
            if category_key:
                metadata_categories = {row["Object Hash"]: row[category_key].strip()
                                       for row in reader if row.get("Object Hash") and row.get(category_key)}
    objects = []
    for instance_index, instance in enumerate(scene.get("object_instances", [])):
        name = instance["template_name"]
        template_id = Path(name).name.removesuffix(".object_config.json")
        candidates = index.get(template_id, [])
        if len(candidates) != 1:
            issues.append({"path": name, "error": f"Expected one object template, found {len(candidates)}"})
            continue
        path = candidates[0]
        template = load(path)
        check_dependencies(path, template)
        semantic_id = template.get("semantic_id")
        lexicon_category = classes.get(semantic_id)
        metadata_category = metadata_categories.get(template_id)
        objects.append({
            "source_instance_index": instance_index, "template_id": template_id,
            "template_path": str(path.resolve()), "category_id": semantic_id,
            "category": metadata_category or lexicon_category or "unknown",
            "category_source": "metadata_csv" if metadata_category else "lexicon" if lexicon_category else "unavailable",
            "lexicon_category": lexicon_category, "metadata_category": metadata_category,
            "category_labels_agree": (metadata_category == lexicon_category
                                      if metadata_category is not None and lexicon_category is not None else None),
            "display_name": metadata.get(template_id, {}).get("name"),
            "source_translation": instance.get("translation"),
            "motion_type": instance.get("motion_type", "UNSPECIFIED"),
        })
    if not objects:
        issues.append({"path": str(scene_file), "error": "No object instances resolved"})
    revision = None
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        revision = result.stdout.strip() if result.returncode == 0 else None
    provenance_paths = [dataset_config, scene_file, region_path, metadata_path, category_path]
    if lexicon_name:
        provenance_paths.append(root / lexicon_name)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in provenance_paths if path.is_file() and checked.get(path.resolve())}
    report = {
        "passed": not issues, "scope": "Dependencies declared by the selected scene, not the entire dataset",
        "dataset_config": str(dataset_config), "scene_id": scene_id, "scene_file": str(scene_file),
        "dataset_revision": revision, "input_sha256": hashes,
        "declared_object_instances": len(scene.get("object_instances", [])), "resolved_object_instances": len(objects),
        "checked_files": len(checked), "category_count": len(classes), "region_annotation_count": len(regions),
        "object_category_counts": dict(sorted(Counter(obj["category"] for obj in objects).items())),
        "unlabeled_template_instances": sum(obj["category_id"] is None for obj in objects),
        "unknown_template_category_ids": sorted({obj["category_id"] for obj in objects
                                                  if obj["category_id"] is not None and obj["category_id"] not in classes}),
        "category_label_disagreements": [obj["source_instance_index"] for obj in objects
                                         if obj["category_labels_agree"] is False],
        "semantics_note": "Raw template IDs and lexicon labels may disagree with object metadata; use instance records and visual review for target selection",
        "issues": issues,
    }
    return report, objects, regions, classes


def make_configuration(args):
    import habitat_sim

    simulator = habitat_sim.SimulatorConfiguration()
    simulator.scene_dataset_config_file = str(args.dataset_config.resolve())
    simulator.scene_id = args.scene_id
    simulator.gpu_device_id = args.gpu_id
    simulator.enable_physics = True
    sensors = []
    for name, kind in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH),
                       ("semantic", habitat_sim.SensorType.SEMANTIC)):
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = name
        sensor.sensor_type = kind
        sensor.resolution = [args.resolution, args.resolution]
        sensor.position = [0.0, args.sensor_height, 0.0]
        sensors.append(sensor)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height = args.agent_height
    agent.radius = args.agent_radius
    agent.sensor_specifications = sensors
    agent.action_space = {"turn_right": habitat_sim.agent.ActionSpec(
        "turn_right", habitat_sim.agent.ActuationSpec(amount=360.0 / args.views))}
    return habitat_sim.Configuration(simulator, [agent])


def export_objects(sim, classes: dict, source_objects: list) -> list:
    import habitat_sim
    import numpy as np

    manager = sim.get_rigid_object_manager()
    source_by_template = {obj["template_id"]: obj for obj in source_objects}
    objects = []
    for handle in sorted(manager.get_object_handles()):
        obj = manager.get_object_by_handle(handle)
        # ManagedRigidObject.aabb is LOCAL. Transform it before exporting a
        # world-space center; the object origin is not a safe navigation goal.
        bounds = habitat_sim.geo.get_transformed_bb(obj.aabb, obj.transformation)
        semantic_id = int(obj.semantic_id)
        template_id = Path(obj.creation_attributes.handle).name.removesuffix(".object_config.json")
        source = source_by_template.get(template_id, {})
        objects.append({
            "object_id": int(obj.object_id), "instance_handle": handle,
            "template_handle": obj.creation_attributes.handle, "template_id": template_id,
            "semantic_id": semantic_id, "lexicon_category": classes.get(semantic_id),
            "category": source.get("category", "unknown"), "category_source": source.get("category_source", "unavailable"),
            "metadata_category": source.get("metadata_category"), "display_name": source.get("display_name"),
            "category_labels_agree": source.get("category_labels_agree"),
            "translation": np.asarray(obj.translation).tolist(),
            "world_aabb": {"min": np.asarray(bounds.min).tolist(), "max": np.asarray(bounds.max).tolist(),
                           "center": np.asarray(bounds.center()).tolist(), "size": np.asarray(bounds.size()).tolist()},
        })
    return objects


def run_scene(args, assets: dict, classes: dict, source_objects: list) -> dict:
    import habitat_sim
    import numpy as np
    from PIL import Image
    from scripts.check_habitat import pose

    output = args.output_dir
    for name in ("rgb", "depth", "depth_preview", "semantic", "semantic_preview"):
        (output / name).mkdir(parents=True, exist_ok=True)
    with habitat_sim.Simulator(make_configuration(args)) as sim:
        sim.seed(args.seed)
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = args.agent_height
        settings.agent_radius = args.agent_radius
        settings.include_static_objects = True
        if not sim.recompute_navmesh(sim.pathfinder, settings) or not sim.pathfinder.is_loaded:
            raise RuntimeError("Failed to build a navigation mesh including static furniture")
        if not sim.pathfinder.save_nav_mesh(str(output / "scene.navmesh")):
            raise RuntimeError("Failed to save the navigation mesh")
        sim.pathfinder.seed(args.seed)
        island = max(range(sim.pathfinder.num_islands), key=sim.pathfinder.island_area)
        start = np.asarray(sim.pathfinder.get_random_navigable_point(island_index=island), dtype=np.float32)
        if not np.isfinite(start).all() or not sim.pathfinder.is_navigable(start):
            raise RuntimeError("Could not sample a valid start on the largest island")
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        for _ in range(100):
            path.requested_end = sim.pathfinder.get_random_navigable_point(island_index=island)
            if (sim.pathfinder.find_path(path) and math.isfinite(path.geodesic_distance)
                    and path.geodesic_distance > 1.0):
                break
        else:
            raise RuntimeError("No reachable reference point more than one meter away")
        write_json(output / "shortest_path.json", {
            "start": start.tolist(), "goal": np.asarray(path.requested_end).tolist(),
            "distance_m": float(path.geodesic_distance),
            "points": [np.asarray(point).tolist() for point in path.points],
        })
        state = habitat_sim.AgentState()
        state.position = start
        agent = sim.initialize_agent(0, state)
        objects = export_objects(sim, classes, source_objects)
        write_json(output / "objects.json", objects)
        trajectory = []
        seen = set()
        palette = np.random.default_rng(args.seed).integers(0, 256, size=(65536, 3), dtype=np.uint8)
        palette[0] = 0
        for frame in range(args.views):
            observations = sim.get_sensor_observations() if frame == 0 else sim.step("turn_right")
            rgb, depth, semantic = (observations[name] for name in ("rgb", "depth", "semantic"))
            if (rgb.shape[:2] != depth.shape or semantic.shape != depth.shape
                    or not np.isfinite(depth).all() or not np.any(depth > 0)):
                raise RuntimeError("Invalid RGB-D or semantic sensor output")
            seen.update(int(value) for value in np.unique(semantic))
            filename = f"{frame:06d}"
            Image.fromarray(rgb[:, :, :3]).save(output / "rgb" / f"{filename}.png")
            np.save(output / "depth" / f"{filename}.npy", depth)
            Image.fromarray((np.clip(depth / 10.0, 0.0, 1.0) * 255).astype(np.uint8)).save(
                output / "depth_preview" / f"{filename}.png")
            np.save(output / "semantic" / f"{filename}.npy", semantic)
            Image.fromarray(palette[semantic.astype(np.int64) % len(palette)]).save(
                output / "semantic_preview" / f"{filename}.png")
            trajectory.append({"step": frame, "action": "INITIALIZE" if frame == 0 else "turn_right", **pose(agent)})
        write_json(output / "trajectory.json", trajectory)
        visible_classes = sorted(value for value in seen if value in classes and value != 0)
        checks = {
            "all_scene_objects_loaded": (len(objects) == assets["declared_object_instances"]
                                         and Counter(obj["template_id"] for obj in objects)
                                         == Counter(obj["template_id"] for obj in source_objects)),
            "navigation_mesh_loaded": bool(sim.pathfinder.is_loaded),
            "navigation_area_positive": sim.pathfinder.navigable_area > 0,
            "reference_path_found": path.geodesic_distance > 1.0,
            "rgb_depth_valid": True, "semantic_ids_rendered": any(value > 0 and value < 2 ** 31 for value in seen),
        }
        metrics = {
            "success": all(checks.values()), "checks": checks,
            "scene_id": args.scene_id, "gpu_device_id": args.gpu_id, "dataset_revision": assets["dataset_revision"],
            "views": args.views, "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
            "semantic_shape": list(semantic.shape), "loaded_rigid_objects": len(objects),
            "runtime_semantic_objects": len(sim.semantic_scene.objects),
            "runtime_semantic_regions": len(sim.semantic_scene.regions),
            "region_annotation_count": assets["region_annotation_count"],
            "region_annotations_source": "dataset semantics/scenes file; exported separately from runtime regions",
            "visible_semantic_ids": sorted(seen),
            "visible_lexicon_labels": [{"id": value, "name": classes[value]} for value in visible_classes],
            "semantic_labels_consistent": not assets["category_label_disagreements"]
            and not assets["unknown_template_category_ids"] and assets["unlabeled_template_instances"] == 0,
            "semantics_note": assets["semantics_note"],
            "navmesh": {"include_static_objects": True, "agent_height": args.agent_height,
                        "agent_radius": args.agent_radius, "navigable_area_m2": sim.pathfinder.navigable_area,
                        "island_count": sim.pathfinder.num_islands, "sampled_island": island},
            "versions": {"python": platform.python_version(), "habitat_sim": habitat_sim.__version__, "numpy": np.__version__},
        }
        write_json(output / "metrics.json", metrics)
        return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path,
                        default=PROJECT_ROOT / "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json")
    parser.add_argument("--scene-id", default="102344280")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/hssd_smoke_001")
    parser.add_argument("--assets-only", action="store_true", help="Audit selected scene assets without Habitat or a GPU")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--sensor-height", type=float, default=1.5)
    parser.add_argument("--agent-height", type=float, default=1.5)
    parser.add_argument("--agent-radius", type=float, default=0.1)
    args = parser.parse_args()
    if (args.gpu_id < 0 or args.resolution < 1 or args.views < 1
            or any(not math.isfinite(value) or value <= 0
                   for value in (args.sensor_height, args.agent_height, args.agent_radius))):
        parser.error("Invalid GPU index, view count, resolution, or sensor/agent dimensions")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Choose a new directory.")
    assets, source_objects, regions, classes = inspect_assets(args.dataset_config, args.scene_id)
    write_json(args.output_dir / "assets.json", assets)
    write_json(args.output_dir / "source_objects.json", source_objects)
    write_json(args.output_dir / "regions.json", regions)
    write_json(args.output_dir / "semantic_classes.json", classes)
    write_json(args.output_dir / "config.json", {key: str(value) if isinstance(value, Path) else value
                                               for key, value in vars(args).items()})
    print(json.dumps(assets, indent=2))
    if not assets["passed"] or args.assets_only:
        return 0 if assets["passed"] else 1
    try:
        metrics = run_scene(args, assets, classes, source_objects)
    except Exception as error:
        write_json(args.output_dir / "metrics.json", {"success": False, "error": str(error)})
        raise
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to: {args.output_dir}")
    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
