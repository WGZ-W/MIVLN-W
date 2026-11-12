"""Fixed-denominator inventory and read-only audit regressions (no GPU)."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import benchmark_audit as audit, benchmark_manifest as manifest


class BenchmarkInventoryTests(unittest.TestCase):
    def fixture(self, root):
        """One base, seven configurations, 22 slots; no synthetic success claim."""
        sources = {name: root / name for name in manifest.PROFILES}
        task = {"task_id": "task_001", "base_config": "configs/task_001/base.json",
                "reference_config": "configs/task_001/reference.json", "output_subdir": "tasks/task_001"}
        config = {"scene_id": "scene_1", "scene_sha256": "a" * 64, "initial_goal": "A", "seed": 7}
        audit.write_json(sources["replacement"] / task["base_config"], config)
        audit.write_json(sources["replacement"] / task["reference_config"], config)
        audit.write_json(sources["replacement"] / "manifest.json", {
            "tasks": [task], "task_count": 1, "expected_episodes": 10})
        for timing in manifest.TIMINGS:
            audit.write_json(sources["replacement"] / f"tasks/task_001/timing/configs/{timing}.json", config)
        for family in ("avoid", "subgoal"):
            records = []
            for timing in (["early"] if family == "avoid" else manifest.TIMINGS):
                suffix = f"/{timing}" if family == "subgoal" else ""
                record = {"task_id": "task_001", "case_id": f"task_001_{timing}", "timing": timing,
                          "output_subdir": f"tasks/task_001{suffix}",
                          "preparation_dir": f"preparations/task_001{suffix}", "prepared": True}
                cfg = sources[family] / record["preparation_dir"] / "config.json"
                audit.write_json(cfg, config)
                record["config_sha256"] = manifest.sha256(cfg)
                audit.write_json(sources[family] / record["output_subdir"] / "experiment.json", {
                    "source_config": str(cfg), "config_sha256": record["config_sha256"]})
                records.append(record)
            audit.write_json(sources[family] / "manifest.json", {
                "source_dir": str(sources["replacement"]),
                "source_manifest_sha256": manifest.sha256(sources["replacement"] / "manifest.json"),
                "tasks": [task] if family == "subgoal" else records,
                "cases": records, "task_count": 1, "case_count": 3,
                "expected_episodes": 3 if family == "avoid" else 9,
                "trigger_protocol": manifest.PROFILES[family][1],
                "timings": list(manifest.TIMINGS), "timing_progress": manifest.TIMINGS})
        for family, path in sources.items():
            audit.write_json(path / "run_status.json", {"task_001": {"attempted": True, "returncode": 0}})
            audit.write_json(path / "summary.json", {"passed": True, "invented_success_count": 1000})
        return sources

    def test_shared_identity_fixed_denominators_and_unknown_release_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            inventory = manifest.build_inventory(self.fixture(Path(temp)))
            counts = manifest.coverage(inventory)
            self.assertEqual((counts["base_tasks"], counts["scenes"], counts["intervention_configurations"], counts["planned_episodes"]),
                             (1, 1, 7, 22))
            self.assertEqual(len({row["base_task_id"] for row in inventory["cases"]}), 1)
            self.assertEqual(len({row["episode_id"] for row in inventory["episodes"]}), 22)
            self.assertTrue(all(row["split"] is None for row in inventory["cases"]))
            self.assertTrue(all(row["manual_review_status"] == "not_recorded" for row in inventory["cases"]))
            self.assertNotEqual(inventory["cases"][0]["trigger_protocol"], inventory["cases"][-1]["trigger_protocol"])
            self.assertTrue(all(row["audit_status"] == "not_audited" for row in inventory["episodes"]))

    def test_missing_family_preserves_its_planned_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            sources = self.fixture(Path(temp))
            (sources["avoid"] / "manifest.json").unlink()
            inventory = manifest.build_inventory(sources)
            self.assertEqual(len(inventory["episodes"]), 22)
            self.assertEqual(len([row for row in inventory["episodes"] if row["task_type"] == "avoid"]), 3)
            self.assertTrue(audit.family_errors(inventory, "avoid"))

    def test_unprepared_and_missing_metrics_never_disappear(self):
        with tempfile.TemporaryDirectory() as temp:
            sources = self.fixture(Path(temp))
            path = sources["avoid"] / "manifest.json"
            data = manifest.read_json(path)
            data["tasks"][0].update(prepared=False, preparation_error="No admissible region")
            audit.write_json(path, data)
            inventory = manifest.build_inventory(sources)
            audit.read_metrics(inventory)
            self.assertEqual(len(inventory["episodes"]), 22)
            self.assertFalse(any(row["metrics_available"] for row in inventory["episodes"]))
            self.assertFalse(next(row for row in inventory["cases"] if row["task_type"] == "avoid")["prepared"])

    def test_different_parent_source_is_not_merged_as_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            sources = self.fixture(Path(temp))
            path = sources["subgoal"] / "manifest.json"
            data = manifest.read_json(path)
            data["source_dir"] = str(Path(temp) / "different_source")
            audit.write_json(path, data)
            inventory = manifest.build_inventory(sources)
            self.assertTrue(audit.family_errors(inventory, "subgoal"))
            self.assertEqual(len(inventory["episodes"]), 22)

    def test_reject_duplicate_canonical_identity_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            sources = self.fixture(Path(temp))
            path = sources["replacement"] / "manifest.json"
            data = manifest.read_json(path)
            data["tasks"].append(deepcopy(data["tasks"][0]))
            audit.write_json(path, data)
            with self.assertRaises(ValueError):
                manifest.build_inventory(sources)
            with self.assertRaises(ValueError):
                manifest.within(sources["replacement"], "../outside")

    def test_unsafe_derived_slot_stays_rejected_without_following_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            sources = self.fixture(Path(temp))
            path = sources["avoid"] / "manifest.json"
            data = manifest.read_json(path)
            data["tasks"][0]["output_subdir"] = "../outside"
            audit.write_json(path, data)
            inventory = manifest.build_inventory(sources)
            self.assertEqual(len(inventory["episodes"]), 22)
            self.assertTrue(audit.family_errors(inventory, "avoid"))
            self.assertTrue(all(sources["avoid"] in Path(row["source_dir"]).parents
                                for row in inventory["episodes"] if row["task_type"] == "avoid"))

    def test_only_audited_successes_count_and_ignore_failure_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            inventory = manifest.build_inventory(self.fixture(Path(temp)))
            rows = [row for row in inventory["episodes"] if row["task_type"] == "avoid"]
            for row in rows:
                row["reported_dynamic_success"] = row["condition"] != "ignore"
                audit.finish_episode(row, row["condition"] != "control")
            counts = manifest.coverage(inventory)["by_task_type"]["avoid"]["conditions"]
            self.assertEqual(counts["control"]["verified_task_successes"], 0)
            self.assertEqual(counts["follow"]["verified_task_successes"], 1)
            self.assertEqual(counts["ignore"]["accepted"], 1)
            self.assertEqual(counts["ignore"]["verified_expected_failures"], 1)

    def test_nonzero_exit_requires_fresh_native_reaudit_not_cached_success(self):
        status = {"attempted": True, "returncode": 1}
        fresh = {"passed": True, "conditions": [{"native_projection_checks": []}]}
        self.assertEqual(audit.run_completed(status, fresh, True), (False, False))
        fresh["conditions"][0]["native_projection_checks"] = [{"endpoint_error_m": 0, "simulator_created": False}]
        self.assertEqual(audit.run_completed(status, fresh, True), (True, True))
        self.assertEqual(audit.run_completed(status, fresh, False), (False, False))
        self.assertEqual(audit.run_completed({**status, "returncode": 2}, fresh, True), (False, False))
        self.assertEqual(status["returncode"], 1)

    def test_child_binding_rejects_valid_but_swapped_experiment(self):
        from scripts import avoid_dataset, avoid_region
        with tempfile.TemporaryDirectory() as temp:
            sources = self.fixture(Path(temp))
            inventory = manifest.build_inventory(sources)
            fresh = {"passed": True, "pairing_errors": [], "conditions": [
                {"condition": condition, "passed": True, "error": None} for condition in manifest.CONDITIONS]}
            path = sources["avoid"] / "tasks/task_001/experiment.json"
            audit.write_json(path, {"source_config": "/different/config.json", "config_sha256": "wrong"})
            with patch.object(avoid_dataset, "validate_batch"), patch.object(avoid_region, "check_experiment", return_value=fresh):
                audit.audit_derived(inventory, "avoid", sources["avoid"])
            rows = [row for row in inventory["episodes"] if row["task_type"] == "avoid"]
            self.assertTrue(all(row["audit_status"] == "rejected" for row in rows))
            self.assertIn("different manifest slot", rows[0]["audit_error"])

    def test_snapshot_detects_frame_changes_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            root.mkdir()
            (root / "rgb.bin").write_bytes(b"old")
            before = audit.snapshot({"source": root})
            (root / "rgb.bin").write_bytes(b"new")
            self.assertNotEqual(before, audit.snapshot({"source": root}))
            (root / "linked").symlink_to(root / "rgb.bin")
            with self.assertRaises(ValueError):
                audit.snapshot({"source": root})

    def test_output_must_be_fresh_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            root.mkdir()
            for output in (root, root / "audit", root.parent):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    audit.validate_output(output, {"source": root})
            output = Path(temp) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                audit.validate_output(output, {"source": root})

    def test_index_only_exports_all_slots_without_claiming_validation(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            sources = self.fixture(root)
            before = audit.snapshot(sources)
            args = SimpleNamespace(replacement_dir=sources["replacement"], avoid_dir=sources["avoid"],
                subgoal_dir=sources["subgoal"], reaudit_dir=root / "no_historical_evidence",
                output_dir=root / "audit", index_only=True)
            self.assertEqual(audit.run(args), 0)
            report = manifest.read_json(args.output_dir / "summary.json")
            self.assertFalse(report["passed"])
            self.assertEqual(report["coverage"]["planned_episodes"], 22)
            self.assertEqual(len((args.output_dir / "episode_index.jsonl").read_text().splitlines()), 22)
            self.assertEqual(before, audit.snapshot(sources))
            rows = [json.loads(line) for line in (args.output_dir / "episode_index.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["audit_status"] == "not_audited" for row in rows))


if __name__ == "__main__":
    unittest.main()
