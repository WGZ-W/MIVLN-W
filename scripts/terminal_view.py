"""Opt-in Oracle terminal facing and evaluator-only object-instance visibility."""

from __future__ import annotations

import json
import math
from pathlib import Path


def enabled(config: dict, key: str) -> bool:
    return config.get(key, {}).get("enabled", False)


def validate(config: dict, goals: dict) -> None:
    """Keep legacy point navigation unchanged; new settings are explicit contracts."""
    for key in ("terminal_alignment", "terminal_visibility"):
        settings = config.get(key, {})
        if not isinstance(settings, dict) or type(settings.get("enabled", False)) is not bool:
            raise ValueError(f"{key} must contain a boolean enabled flag")
    alignment = config.get("terminal_alignment", {})
    if enabled(config, "terminal_alignment"):
        tolerance = alignment.get("yaw_tolerance_degrees", 7.5)
        limit = alignment.get("max_turn_actions", 24)
        if (not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance)
                or not config["turn_angle_degrees"] / 2 <= tolerance <= 45):
            raise ValueError("Yaw tolerance must be between half a turn action and 45 degrees")
        if type(limit) is not int or limit < 1:
            raise ValueError("max_turn_actions must be positive")
    if enabled(config, "terminal_visibility"):
        pixels = config["terminal_visibility"].get("min_pixels", 64)
        if type(pixels) is not int or not 1 <= pixels <= math.prod(config["resolution"]):
            raise ValueError("min_pixels must fit the sensor resolution")
        if config.get("semantic_sensor") is not True:
            raise ValueError("Instance visibility requires semantic_sensor")
    if enabled(config, "terminal_alignment") or enabled(config, "terminal_visibility"):
        if not config.get("scene_dataset_config"):
            raise ValueError("Terminal object evaluation currently requires a rigid HSSD scene")
        for goal in goals:
            metadata = config.get("goal_metadata", {}).get(goal, {})
            if (not isinstance(metadata.get("template_id"), str) or not metadata["template_id"]
                    or type(metadata.get("source_instance_index")) is not int or metadata["source_instance_index"] < 0):
                raise ValueError(f"Terminal evaluation requires a stable source instance: {goal}")


def resolve_instances(sim, scene: Path, metadata: dict) -> dict:
    """Resolve source indices to unique runtime templates, never category IDs."""
    import habitat_sim
    import numpy as np

    instances = json.loads(scene.read_text(encoding="utf-8"))["object_instances"]
    template_ids = [Path(value["template_name"]).name.removesuffix(".object_config.json") for value in instances]
    manager = sim.get_rigid_object_manager()
    objects = [manager.get_object_by_handle(handle) for handle in manager.get_object_handles()]
    resolved = {}
    for goal, value in metadata.items():
        index, template = value["source_instance_index"], value["template_id"]
        if index >= len(template_ids) or template_ids[index] != template or template_ids.count(template) != 1:
            raise ValueError(f"Ambiguous or mismatched source instance: {goal}")
        matches = [obj for obj in objects if Path(obj.creation_attributes.handle).name.removesuffix(".object_config.json") == template]
        if len(matches) != 1:
            raise ValueError(f"Expected one runtime object for {goal}, found {len(matches)}")
        obj = matches[0]
        bounds = habitat_sim.geo.get_transformed_bb(obj.aabb, obj.transformation)
        resolved[goal] = {"object": obj, "template_id": template, "source_instance_index": index,
                          "object_id": int(obj.object_id), "world_center": np.asarray(bounds.center()).tolist()}
    return resolved


def yaw_action(state, center, tolerance: float):
    """Choose an actual 15-degree-style action in world XZ; Habitat faces local -Z.

    Positive signed error is a left turn about +Y. The tolerance must be at
    least half the discrete turn size to avoid oscillating around the target.
    """
    import numpy as np
    from habitat_sim.utils.common import quat_rotate_vector

    forward = quat_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0]))[[0, 2]]
    desired = np.asarray(center)[[0, 2]] - np.asarray(state.position)[[0, 2]]
    if np.linalg.norm(desired) < 1e-8:
        return "STOP", 0.0
    error = math.degrees(math.atan2(forward[1] * desired[0] - forward[0] * desired[1],
                                    float(np.dot(forward, desired))))
    if abs(error) <= tolerance + 1e-5:
        return "STOP", error
    return ("turn_left" if error > 0 else "turn_right"), error


def pose_snapshot(agent):
    import numpy as np

    state = agent.get_state()
    return {"position": np.asarray(state.position).tolist(),
            "rotation_xyzw": [*state.rotation.imag.tolist(), float(state.rotation.real)]}


def measure_visibility(sim, agent, target: dict, min_pixels: int):
    """Render a target-specific label AFTER execution and restore it in finally.

    Existing semantic frames remain class-labelled and are saved before this
    diagnostic. Relabeling changes no geometry, pose, action, or simulation time.
    Even when seven objects share class 27, this mask includes only the selected
    rigid instance. No observation from this diagnostic is supplied to the policy.
    """
    import numpy as np

    before = pose_snapshot(agent)
    observations = sim.get_sensor_observations()
    original = np.asarray(observations["semantic"]).copy()
    rgb = np.asarray(observations["rgb"])[..., :3].copy()
    if original.ndim != 2 or not np.issubdtype(original.dtype, np.integer):
        raise ValueError("Invalid semantic sensor output for terminal diagnostic")
    manager = sim.get_rigid_object_manager()
    used = {int(manager.get_object_by_handle(handle).semantic_id) for handle in manager.get_object_handles()}
    used.update(int(value) for value in np.unique(original))
    # Stay below the common 16-bit sentinel as well as within the renderer's
    # uint32 range. Availability is checked against objects AND visible stage IDs.
    marker = next((value for value in range(65534, 0, -1) if value not in used), None)
    if marker is None:
        raise RuntimeError("No unused diagnostic semantic label")
    obj = target["object"]
    old_label = int(obj.semantic_id)
    try:
        obj.semantic_id = marker
        labelled = np.asarray(sim.get_sensor_observations()["semantic"]).copy()
        if labelled.shape != original.shape:
            raise RuntimeError("Diagnostic changed semantic resolution")
        mask = labelled == marker
    finally:
        # Exception safety is essential: category labels must not leak into later
        # observations, paired controls, or an Ignore rollout's policy state.
        obj.semantic_id = old_label
    restored = np.asarray(sim.get_sensor_observations()["semantic"]).copy()
    after = pose_snapshot(agent)
    if int(obj.semantic_id) != old_label or not np.array_equal(original, restored) or before != after:
        raise RuntimeError("Terminal diagnostic failed to restore labels or preserve pose")
    pixels = int(mask.sum())
    report = {"evaluated": True, "method": "isolated_instance_semantic_relabel_v1",
              "object_id": target["object_id"], "template_id": target["template_id"],
              "source_instance_index": target["source_instance_index"], "original_category_id": old_label,
              "diagnostic_label": marker, "instance_pixels": pixels, "min_pixels": min_pixels,
              "instance_visible": pixels >= min_pixels, "labels_restored": True,
              "pose_unchanged": True, "pose": before}
    return report, mask, rgb
