"""Simulator-independent scene identity shared by navigation and reference checks."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    return (PROJECT_ROOT / value).resolve()


def navmesh_path(config: dict) -> Path:
    return project_path(config["navmesh"]) if config.get("navmesh") else project_path(config["scene"]).with_suffix(".navmesh")


def check_scene_match(config: dict, reference: dict) -> None:
    """Reject cross-scene/mesh comparisons, retaining old GLB configuration support."""
    for key in ("scene", "scene_dataset_config"):
        left, right = config.get(key), reference.get(key)
        if (project_path(left) if left else None) != (project_path(right) if right else None):
            raise ValueError(f"Reference scene configuration differs: {key}")
    if navmesh_path(config) != navmesh_path(reference):
        raise ValueError("Reference scene configuration differs: navmesh")
    for key, default in (("scene_id", None), ("agent_height", 1.5), ("agent_radius", 0.1),
                         ("semantic_sensor", False), ("terminal_alignment", None), ("terminal_visibility", None)):
        if config.get(key, default) != reference.get(key, default):
            raise ValueError(f"Reference scene configuration differs: {key}")
    for key in ("navmesh_sha256", "scene_sha256", "scene_dataset_config_sha256", "dataset_revision"):
        if config.get(key) is not None and config[key] != reference.get(key):
            raise ValueError(f"Reference scene provenance differs: {key}")
