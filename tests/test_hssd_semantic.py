"""Catalog review gates and three-condition dataset regressions, without GPU use."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import hssd_dataset as dataset
from scripts import hssd_targets as targets
from scripts.intervention_timing import load_json, write_json


def catalog_fixture(root, scene_config=None):
    """Synthetic evidence is isolated in a temp directory, never in the real catalog."""
    catalog = {"schema_version": 1, "kind": "hssd_target_catalog", "scene_config": scene_config or {},
               "sources": [], "targets": [{
                   "target_id": f"target_{index:03d}", "source_instance_index": index,
                   "template_id": f"template_{index}", "category": "bed", "semantic_id": 27,
                   "display_name": f"Bed {index}", "stop_position": [x, 0, 0],
                   "world_aabb": {"min": [x, 0, 0], "max": [x + 1, 1, 1], "center": [x + .5, .5, .5]},
                   "review": {"status": "pending"},
               } for index, x in enumerate((2, -2, 5))]}
    evidence = {"candidate_identity": targets.candidate_identity(catalog), "targets": {}}
    for target in catalog["targets"]:
        folder = root / "previews" / target["target_id"]
        folder.mkdir(parents=True)
        files = []
        for filename in ("rgb.png", "depth.npy", "semantic.npy", "category_mask.png"):
            path = folder / filename
            path.write_bytes(f"Synthetic unit-test evidence {target['target_id']} {filename}".encode())
            files.append({"path": str(path), "sha256": targets.sha256(path)})
        evidence["targets"][target["target_id"]] = {"files": files, "same_category_pixels": 10}
    preview = root / "previews/preview.json"
    write_json(preview, evidence)
    for index, target in enumerate(catalog["targets"]):
        target["review"] = {"status": "approved", "reviewer": "unit-test fixture",
                            "description": f"the visually distinct bed {index}",
                            "preview": {"path": str(preview), "sha256": targets.sha256(preview)}}
    return catalog, preview


class CatalogReviewTests(unittest.TestCase):
    def test_pending_targets_cannot_be_used_for_formal_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog, _ = catalog_fixture(Path(temporary))
            for target in catalog["targets"]:
                target["review"] = {"status": "pending"}
            with self.assertRaisesRegex(ValueError, "explicitly approved"):
                targets.validate_catalog(catalog)
            root = Path(temporary)
            path = root / "pending.json"
            write_json(path, catalog)
            with patch.object(dataset, "load_geometry") as geometry:
                with self.assertRaises(ValueError):
                    dataset.prepare(SimpleNamespace(catalog=path, output_dir=root / "formal"))
                geometry.assert_not_called()
            self.assertFalse((root / "formal").exists())

    def test_approved_catalog_requires_unique_instance_and_language(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog, _ = catalog_fixture(Path(temporary))
            self.assertEqual(len(targets.validate_catalog(catalog)), 3)
            for field in ("description", "reviewer", "preview"):
                changed = deepcopy(catalog)
                changed["targets"][0]["review"].pop(field)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    targets.approved_targets(changed)
            catalog["targets"][1]["review"]["description"] = catalog["targets"][0]["review"]["description"]
            with self.assertRaisesRegex(ValueError, "distinct"):
                targets.approved_targets(catalog)

    def test_changed_stop_or_preview_payload_invalidates_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog, _ = catalog_fixture(Path(temporary))
            moved = deepcopy(catalog)
            moved["targets"][0]["stop_position"][0] += 1
            with self.assertRaisesRegex(ValueError, "different candidate"):
                targets.validate_catalog(moved)
            image = Path(catalog["targets"][0]["review"]["preview"]["path"]).parent / "target_000/rgb.png"
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                targets.validate_catalog(catalog)

    def test_review_creates_new_version_and_preserves_prior_decisions(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            catalog, preview = catalog_fixture(root)
            for target in catalog["targets"]:
                target["review"] = {"status": "pending"}
            original = root / "catalog.json"
            write_json(original, catalog)
            first = root / "reviewed_1.json"
            args = SimpleNamespace(catalog=original, previews=preview, target_id="target_000", decision="approve",
                                   description="the first bed", reviewer="tester", note="RGB checked", output=first)
            targets.review(args)
            self.assertEqual(load_json(original)["targets"][0]["review"]["status"], "pending")
            args.catalog, args.output, args.target_id = first, root / "reviewed_2.json", "target_001"
            args.description = "the second bed"
            targets.review(args)
            reviewed = load_json(args.output)
            self.assertEqual([item["review"]["status"] for item in reviewed["targets"]], ["approved", "approved", "pending"])
            with self.assertRaises(FileExistsError):
                targets.review(args)
            args.catalog = args.output
            args.output = root / "overwrite-decision.json"
            with self.assertRaisesRegex(ValueError, "pending"):
                targets.review(args)

    def test_zero_pixels_or_incomplete_preview_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, preview = catalog_fixture(root)
            catalog["targets"][0]["review"] = {"status": "pending"}
            path = root / "catalog.json"
            write_json(path, catalog)
            args = SimpleNamespace(catalog=path, previews=preview, target_id="target_000", decision="approve",
                                   description="the first bed", reviewer="tester", note="", output=root / "result.json")
            evidence = load_json(preview)
            evidence["targets"]["target_000"]["same_category_pixels"] = 0
            write_json(preview, evidence)
            with self.assertRaisesRegex(ValueError, "pixels"):
                targets.review(args)
            evidence["targets"]["target_000"]["same_category_pixels"] = 10
            evidence["targets"]["target_000"]["files"] = []
            write_json(preview, evidence)
            with self.assertRaisesRegex(ValueError, "RGB"):
                targets.review(args)

    def test_task_construction_does_not_retain_old_semantic_metadata(self):
        template = {"scene": "scene.json", "goal_metadata": {"old": "bad"}, "goal_source": {"old": "bad"}}
        with tempfile.TemporaryDirectory() as temporary:
            catalog, _ = catalog_fixture(Path(temporary))
            a, b = catalog["targets"][:2]
            task = dataset.make_task(template, a, b, [0, 0, 0], 0, 42, "task_001", {"path": "reviewed.json"})
            self.assertEqual(set(task["goal_positions"]), {a["target_id"], b["target_id"]})
            self.assertNotIn("old", task["goal_metadata"])
            self.assertNotIn("goal_source", task)
            self.assertIn(b["review"]["description"], task["events"][0]["utterance"])


class DatasetAccountingTests(unittest.TestCase):
    def test_preparation_freezes_inputs_and_never_overwrites_a_collection(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            catalog, _ = catalog_fixture(root)
            source = root / "catalog.json"
            write_json(source, catalog)
            source_record = {"path": str(source), "sha256": targets.sha256(source)}
            directed = [(a, b) for a in catalog["targets"] for b in catalog["targets"] if a is not b]
            sampled = []
            for index in range(10):
                a, b = directed[index % len(directed)]
                name = f"task_{index + 1:03d}"
                config = dataset.make_task({}, a, b, [index, 0, 0], 0, index, name, source_record)
                sampled.append((config, {"task_id": name, "target_a": a["target_id"], "target_b": b["target_id"],
                    "base_config": f"configs/{name}/base.json", "reference_config": f"configs/{name}/reference.json",
                    "output_subdir": f"tasks/{name}"}))
            args = SimpleNamespace(catalog=source, output_dir=root / "collection", count=10, seed=42, max_attempts=100)
            with patch.object(dataset, "load_geometry", return_value=(None, {})), patch.object(dataset, "sample_tasks", return_value=sampled):
                self.assertEqual(dataset.prepare(args), 0)
                with self.assertRaises(FileExistsError):
                    dataset.prepare(args)
            manifest = load_json(args.output_dir / "manifest.json")
            dataset.validate_inputs(args.output_dir, manifest)
            self.assertEqual(manifest["expected_episodes"], 100)
            for task in manifest["tasks"]:
                self.assertEqual(load_json(args.output_dir / task["reference_config"])["events"], [])
            path = args.output_dir / manifest["tasks"][0]["reference_config"]
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                dataset.validate_inputs(args.output_dir, manifest)

    def test_missing_results_keep_all_task_condition_denominators(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            write_json(root / "manifest.json", {"schema_version": 1, "kind": "hssd_semantic_dataset", "task_count": 2,
                "catalog_source": {"path": str(root / "missing.json"), "sha256": "0" * 64}, "tasks": [{
                    "task_id": f"task_{index:03d}", "base_config": f"configs/task_{index:03d}/base.json",
                    "base_config_sha256": "0" * 64, "reference_config": f"configs/task_{index:03d}/reference.json",
                    "reference_config_sha256": "0" * 64, "output_subdir": f"tasks/task_{index:03d}",
                } for index in (1, 2)]})
            result = dataset.summarize(root)
            self.assertFalse(result["passed"])
            self.assertEqual(result["total_tasks"], 2)
            self.assertEqual(result["total_pairs"], 6)
            self.assertEqual(result["total_negative_controls"], 6)
            self.assertEqual(result["expected_episodes"], 20)
            self.assertEqual(result["observed_episode_metrics"], 0)
            self.assertEqual(result["by_timing"]["early"]["intervention"]["total"], 2)
            self.assertEqual(result["conditions"]["intervention"]["dynamic_success_rate"], 0)
            self.assertIsNone(result["conditions"]["intervention"]["mean_path_length_m"])
            self.assertEqual(len((root / "cases.csv").read_text().splitlines()), 19)

    def test_manifest_paths_cannot_escape_the_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for value in ("../outside", str(root.parent / "outside"), "."):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    dataset.within(root, value)


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires Habitat imports; simulation is CPU fake")
class SemanticPipelineTests(unittest.TestCase):
    def setup_pipeline(self, root, stack, ignore_follow=False):
        import numpy as np
        import test_navigation_integration as fixtures
        from scripts import run_navigation as navigation
        from scripts.intervention_timing import prepare as prepare_timing
        from scripts.run_intervention import check_pair
        from scripts.run_ignore_intervention import check_negative

        fixture = fixtures.NavigationIntegrationTests()
        fixture.setUp()
        template, scene = fixture.hssd_config(root)
        template["resolution"] = [4, 4]
        catalog, _ = catalog_fixture(root, template)
        source = root / "catalog.json"
        write_json(source, catalog)
        source_record = {"path": str(source), "sha256": targets.sha256(source)}
        output = root / "collection"
        tasks = []
        for index in (1, 2):
            task_id = f"task_{index:03d}"
            config = dataset.make_task(template, *catalog["targets"][:2], [0, 0, 0], 0, 42, task_id, source_record)
            reference = deepcopy(config)
            reference["events"] = []
            record = {"task_id": task_id, "output_subdir": f"tasks/{task_id}"}
            for key, value in (("base_config", config), ("reference_config", reference)):
                record[key] = f"configs/{task_id}/{key}.json"
                write_json(output / record[key], value)
                record[f"{key}_sha256"] = targets.sha256(output / record[key])
            tasks.append(record)
        write_json(output / "manifest.json", {"schema_version": 1, "kind": "hssd_semantic_dataset",
                                               "task_count": 2, "catalog_source": source_record, "tasks": tasks})
        real_save_frame = navigation.save_frame
        instances = fixture.simulator_patches(stack, ignore_new_goal=ignore_follow, explicit_mesh=True)
        observations = {"rgb": np.zeros((4, 4, 4), dtype=np.uint8), "depth": np.ones((4, 4), dtype=np.float32),
                        "semantic": np.full((4, 4), 27, dtype=np.uint32)}
        stack.enter_context(patch.object(navigation, "save_frame", lambda output, frame, obs, required=False:
                                         real_save_frame(output, frame, observations, required)))

        class FakePathFinder:
            def load_nav_mesh(self, path):
                return True

            def is_navigable(self, position):
                return True

            def find_path(self, path):
                path.geodesic_distance = float(np.linalg.norm(path.requested_end - path.requested_start))
                return True

        stack.enter_context(patch.object(navigation.habitat_sim, "PathFinder", FakePathFinder))

        def execute(command, log):
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("CPU fake child; no GPU simulation")
            module = command[command.index("-m") + 1]
            def argument(name):
                return Path(command[command.index(name) + 1])
            destination = argument("--output-dir")
            if module == "scripts.intervention_timing":
                return prepare_timing(SimpleNamespace(output_dir=destination, base_config=argument("--base-config"),
                    reference_dir=argument("--reference-dir"), keep_target=True, min_separation=2.0,
                    min_route_distance=1.0, max_distance=20.0, seed=42))
            config = load_json(argument("--config"))
            if module == "scripts.run_navigation":
                return 0 if navigation.run(config, scene, 0, destination)["success"] else 1
            if module == "scripts.run_intervention":
                control = deepcopy(config)
                control["events"] = []
                navigation.run(control, scene, 0, destination / "control")
                navigation.run(config, scene, 0, destination / "intervention")
                result = check_pair(destination, config["events"][0])
                write_json(destination / "pair_summary.json", result)
                return 0 if result["passed"] else 1
            navigation.run(config, scene, 0, destination, intervention_policy="ignore")
            result = check_negative(destination, config, argument("--reference-dir"))
            write_json(destination / "negative_check.json", result)
            return 0 if result["negative_check_passed"] else 1
        return output, instances, execute

    def test_complete_pipeline_and_tampered_records_are_rechecked(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            output, instances, execute = self.setup_pipeline(Path(temporary), stack)
            stack.enter_context(patch.object(dataset, "execute_stage", execute))
            self.assertEqual(dataset.run(SimpleNamespace(output_dir=output, gpu_id=0)), 0)
            result = load_json(output / "summary.json")
            self.assertEqual(result["passed_tasks"], 2)
            self.assertEqual(result["passed_pairs"], 6)
            self.assertEqual(result["passed_negative_controls"], 6)
            self.assertEqual(result["observed_episode_metrics"], 20)
            self.assertEqual(result["conditions"]["ignore"]["dynamic_successes"], 0)
            self.assertEqual(len(instances), 20)
            with self.assertRaises(FileExistsError):
                dataset.run(SimpleNamespace(output_dir=output, gpu_id=0))
            # Ignore task failure is insufficient: an actually delivered event is required.
            path = output / "tasks/task_001/timing/early/ignore/events.json"
            write_json(path, [])
            result = dataset.summarize(output)
            self.assertFalse(result["passed"])
            self.assertEqual(result["passed_negative_controls"], 5)
            # A changed raw prefix payload is not hidden by unchanged trajectory filenames.
            image = output / "tasks/task_002/timing/late/intervention/rgb/000000.png"
            image.write_bytes(b"tampered image")
            result = dataset.summarize(output)
            self.assertEqual(result["passed_pairs"], 5)
            self.assertEqual(result["failed_tasks"], ["task_001", "task_002"])
            # Post-intervention payloads are also required, not just the shared prefix.
            episode = output / "tasks/task_002/timing/middle/intervention"
            last = load_json(episode / "trajectory.json")[-1]
            (episode / last["frames"]["depth"]).unlink()
            self.assertEqual(dataset.summarize(output)["passed_pairs"], 4)
            # Malformed numeric fields fail validation without crashing grouped statistics.
            path = output / "tasks/task_001/timing/middle/intervention/metrics.json"
            metrics = load_json(path)
            metrics["path_length_m"] = "not a number"
            write_json(path, metrics)
            self.assertFalse(dataset.summarize(output)["passed"])

    def test_failed_reference_kept_and_next_task_still_runs(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            output, _, execute = self.setup_pipeline(Path(temporary), stack)
            def fail_first(command, log):
                if "task_001" in str(log):
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text("simulated failed reference")
                    return 1
                return execute(command, log)
            stack.enter_context(patch.object(dataset, "execute_stage", fail_first))
            self.assertEqual(dataset.run(SimpleNamespace(output_dir=output, gpu_id=0)), 1)
            result = load_json(output / "summary.json")
            self.assertEqual(result["passed_tasks"], 1)
            self.assertEqual(result["total_pairs"], 6)
            self.assertEqual(result["passed_pairs"], 3)
            self.assertEqual(result["total_negative_controls"], 6)
            self.assertEqual(result["failed_tasks"], ["task_001"])

    def test_follow_failure_does_not_disable_valid_negative_controls(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            output, _, execute = self.setup_pipeline(Path(temporary), stack, ignore_follow=True)
            stack.enter_context(patch.object(dataset, "execute_stage", execute))
            self.assertEqual(dataset.run(SimpleNamespace(output_dir=output, gpu_id=0)), 1)
            result = load_json(output / "summary.json")
            self.assertEqual(result["passed_pairs"], 0)
            self.assertEqual(result["passed_negative_controls"], 6)

    def test_changed_frozen_config_stops_before_first_child(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output, instances, _ = self.setup_pipeline(Path(temporary), stack)
            path = output / "configs/task_001/base_config.json"
            config = load_json(path)
            config["goal_positions"]["target_001"] = [100, 0, 0]
            write_json(path, config)
            with patch.object(dataset, "execute_stage") as execute:
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    dataset.run(SimpleNamespace(output_dir=output, gpu_id=0))
                execute.assert_not_called()
            self.assertEqual(instances, [])


@unittest.skipUnless(importlib.util.find_spec("habitat_sim"), "Requires Habitat imports; no GPU rendering")
class SemanticGeometryTests(unittest.TestCase):
    def test_sampler_is_reproducible_balanced_and_never_resamples_object_goals(self):
        import numpy as np
        from scripts import run_navigation as navigation

        class Pathfinder:
            def seed(self, value):
                self.random = random.Random(value)

            def get_island(self, position):
                return 0

            def is_navigable(self, position):
                return True

            def get_random_navigable_point(self, island_index):
                return [self.random.uniform(-8, 8), 0, self.random.uniform(-8, 8)]

        def path(pathfinder, start, end):
            return SimpleNamespace(geodesic_distance=float(np.linalg.norm(end - start)), points=[start, end])

        template = {"scene": "fixture.glb", "success_radius_m": .3, "max_steps": 100,
                    "sensor_height": 1.5, "resolution": [4, 4], "forward_step_m": .25, "turn_angle_degrees": 15}
        with tempfile.TemporaryDirectory() as temporary, patch.object(navigation, "shortest_path", path):
            catalog, _ = catalog_fixture(Path(temporary))
            kwargs = (Pathfinder(), template, catalog["targets"], 10, 42, 1000, {"path": "fixture"})
            first = dataset.sample_tasks(*kwargs)
            second = dataset.sample_tasks(*kwargs)
            self.assertEqual(first, second)
            directed = {(row["target_a"], row["target_b"]) for _, row in first}
            self.assertEqual(len(directed), 6)
            self.assertTrue(all((b, a) in directed for a, b in directed))
            positions = {target["target_id"]: target["stop_position"] for target in catalog["targets"]}
            for config, row in first:
                self.assertEqual(config["goal_positions"], {row["target_a"]: positions[row["target_a"]],
                                                         row["target_b"]: positions[row["target_b"]]})
            broken = deepcopy(catalog["targets"])
            for target in broken:
                target["stop_position"] = [0, 0, 0]
            with self.assertRaisesRegex(ValueError, "overlapping"):
                dataset.sample_tasks(Pathfinder(), template, broken, 10, 42, 1000, {})

    def test_preview_captures_upright_sensor_poses_without_auto_approval(self):
        import numpy as np
        from habitat_sim.utils.common import quat_rotate_vector
        from scripts import run_navigation as navigation
        from scripts import check_hssd

        captured = []

        class Simulator:
            def __init__(self, configuration):
                self.pathfinder = SimpleNamespace(is_navigable=lambda point: True)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def seed(self, seed):
                pass

            def initialize_agent(self, index, state):
                state.sensor_states = {name: SimpleNamespace(position=state.position + [0, 1.5, 0], rotation=state.rotation)
                                       for name in ("rgb", "depth", "semantic")}
                def set_state(updated, **kwargs):
                    captured.append(updated)
                return SimpleNamespace(get_state=lambda: state, set_state=set_state)

            def get_sensor_observations(self):
                return {"rgb": np.zeros((4, 4, 4), dtype=np.uint8), "depth": np.ones((4, 4), dtype=np.float32),
                        "semantic": np.full((4, 4), 27, dtype=np.uint32)}

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            catalog, _ = catalog_fixture(root, {"scene": "fixture.json", "seed": 42})
            for target in catalog["targets"]:
                target["review"] = {"status": "pending"}
            path = root / "catalog.json"
            write_json(path, catalog)
            objects = [{"template_id": target["template_id"], "semantic_id": 27, "world_aabb": target["world_aabb"]}
                       for target in catalog["targets"]]
            stack.enter_context(patch.object(targets, "load_geometry", return_value=(None, {"expected_rigid_objects": 3})))
            stack.enter_context(patch.object(navigation, "make_configuration", return_value=None))
            stack.enter_context(patch.object(navigation, "load_navigation_mesh", return_value={}))
            stack.enter_context(patch.object(navigation.habitat_sim, "Simulator", Simulator))
            stack.enter_context(patch.object(check_hssd, "export_objects", return_value=objects))
            args = SimpleNamespace(catalog=path, output_dir=root / "new_previews", gpu_id=0)
            self.assertEqual(targets.preview(args), 0)
            self.assertEqual(len(captured), 3)
            for state, target in zip(captured, catalog["targets"]):
                sensor = state.sensor_states["rgb"]
                wanted = np.asarray(target["world_aabb"]["center"]) - sensor.position
                np.testing.assert_allclose(quat_rotate_vector(sensor.rotation, np.array([0, 0, -1])),
                                           wanted / np.linalg.norm(wanted), atol=1e-6)
                self.assertAlmostEqual(quat_rotate_vector(sensor.rotation, np.array([1, 0, 0]))[1], 0)
            self.assertTrue(all(target["review"]["status"] == "pending" for target in load_json(path)["targets"]))
            report = load_json(args.output_dir / "preview.json")
            for identifier, evidence in report["targets"].items():
                targets.check_preview_record(evidence, identifier)
                self.assertFalse(evidence["manually_verified"])
            with self.assertRaises(FileExistsError):
                targets.preview(args)


if __name__ == "__main__":
    unittest.main()
