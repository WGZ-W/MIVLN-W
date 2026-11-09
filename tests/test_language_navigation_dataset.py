"""Batch orchestration checks; synthetic audit scores are not model evidence.

The existing single-task tests exercise the physics/model boundary. These tests
use real frozen source/preparation artifacts but stub the final native audit and
child launcher so failure accounting can be checked without Habitat or an API.
"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import language_navigation_dataset as batch, language_navigation as single
from scripts import task_update as base
from scripts.hssd_targets import sha256
from tests.test_language_navigation import source_fixture


def collection(root, count=3):
    """Clone the orchestration fixture, not a set of independent HSSD tasks."""
    source = source_fixture(root)
    manifest = base.read_json(source / "manifest.json")
    first = manifest["tasks"][0]
    for index in range(2, count + 1):
        identity = f"task_{index:03d}"
        shutil.copytree(source / "configs/task_001", source / f"configs/{identity}")
        shutil.copytree(source / "tasks/task_001", source / f"tasks/{identity}")
        item = {key: value.replace("task_001", identity) if isinstance(value, str) else value
                for key, value in first.items()}
        for key in ("base_config", "reference_config"):
            item[key + "_sha256"] = sha256(source / item[key])
        manifest["tasks"].append(item)
    manifest["task_count"] = count
    base.write_json(source / "manifest.json", manifest)
    return source


def args_for(root, source, reuse=()):
    return SimpleNamespace(source_dir=source, output_dir=root / "batch", timing="middle",
        reuse_run=list(reuse), model="qwen3.5-9b", base_url="http://127.0.0.1:8000/v1",
        thinking_mode="off", temperature=0.0, max_tokens=512, timeout=120.0)


def fake_audit(directory, correct=True, integrity=True):
    """Explicit synthetic result, independent of the production audit code."""
    return {"conditions": [{"condition": name, "integrity_passed": integrity,
        "dynamic_success": name != "ignore" and (correct or name == "oracle"),
        "joint_success": name != "ignore" and (correct or name == "oracle"),
        "ignore_expected_failure": name == "ignore", "steps": 5, "path_length_m": 1.0,
        "navigation_error_m": .1, "error": None} for name in batch.CONDITIONS],
        "integrity_passed": integrity, "prefix_matched": integrity, "prefix_error": None,
        "model_update_correct": correct, "model_pipeline_success": correct and integrity,
        "model_outcome": "applied" if correct else "invalid_prediction", "model_callback_seconds": .25,
        "model_failure_stage": None if correct else "update", "passed": correct and integrity}


def fake_run(directory, prepared, controls, gpu=2):
    """Small opaque raw files for snapshot/binding tests, never native evidence."""
    directory.mkdir(parents=True)
    _, config = single.load_preparation(prepared)
    for name in single.JOURNAL_FILES:
        (directory / name).write_text("fixture", encoding="utf-8")
    for condition in batch.CONDITIONS:
        branch = directory / condition
        branch.mkdir()
        for name in batch.BRANCH_FILES:
            base.write_json(branch / name, {})
        base.write_json(branch / "episode.json", {**config, "gpu_device_id": gpu})
        (branch / "frame.bin").write_bytes(b"sensor fixture")
        (branch / "mask.bin").write_bytes(b"mask fixture")
        base.write_json(branch / "trajectory.json", [{"frames": {"rgb": "frame.bin"}}])
        base.write_json(branch / "terminal_diagnostics.json", {"files": [{"path": "mask.bin"}]})
    base.write_json(directory / "model/policy_updates.json", [])
    base.write_json(directory / "run.json", {"kind": single.PROTOCOL, "prepared_dir": str(prepared),
        "prepared_manifest_sha256": sha256(prepared / "manifest.json"), "controls": controls,
        "gpu_id": gpu, "system_prompt_sha256": batch.prompt_hash(), "code_sha256": single.code_hashes()})


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "batch"
        self.source = collection(self.root)
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.audit = contexts.enter_context(patch.object(single, "check_experiment", side_effect=fake_audit))
        contexts.enter_context(redirect_stdout(io.StringIO()))

    def prepare(self, reuse=()):
        args = args_for(self.root, self.source, reuse)
        batch.prepare(args)
        return args

    def old_run(self, identity="task_001", gpu=0):
        prepared = self.root / (identity + "_prepared")
        single.prepare(SimpleNamespace(source_dir=self.source, task_id=identity,
                                      timing="middle", output_dir=prepared))
        directory = self.root / (identity + "_run")
        fake_run(directory, prepared, batch.controls_from(args_for(self.root, self.source)), gpu)
        return directory

    def run_batch(self, launcher):
        # These imports remain inside production run because core environments
        # need no Habitat. Stub only native preflight and process launch here.
        import sys
        native = SimpleNamespace(validate_config=lambda config: None,
                                 scene_provenance=lambda config, scene: None)
        with patch.dict(sys.modules, {"scripts.run_navigation": native}), patch.object(batch, "execute_stage", side_effect=launcher):
            return batch.run(SimpleNamespace(output_dir=self.output, gpu_id=2, api_key_env="KEY"))

    def launcher(self, codes=None):
        commands = []

        def launch(command, log):
            commands.append(command)
            destination = Path(command[command.index("--output-dir") + 1])
            prepared = Path(command[command.index("--prepared-dir") + 1])
            fake_run(destination, prepared, batch.controls_from(args_for(self.root, self.source)))
            return (codes or {}).get(destination.name, 0)

        return commands, launch

    def test_prepare_freezes_whole_roster_and_controls_without_api(self):
        with patch.object(single.client, "complete_payload") as call:
            self.prepare()
        call.assert_not_called()
        manifest = batch.validate_batch(self.output)
        self.assertEqual(manifest["planned_episodes"], 9)
        self.assertEqual(manifest["new_model_calls"], 3)
        self.assertEqual(manifest["controls"]["max_tokens"], 512)
        self.assertEqual(self.audit.call_count, 0)

    def test_prepare_never_overwrites(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_reuse_is_read_only_and_video_addition_does_not_change_snapshot(self):
        old = self.old_run()
        before = batch.evidence_hashes(old)
        self.prepare([old])
        (old / "oracle/trajectory.mp4").write_bytes(b"optional later export")
        manifest = batch.validate_batch(self.output)
        self.assertEqual(batch.evidence_hashes(old), before)
        self.assertEqual(manifest["new_episodes"], 6)
        report = batch.summarize(self.output)
        self.assertEqual(report["audited_episodes"], 3)
        self.assertEqual(report["not_run_tasks"], 2)
        self.assertTrue(report["results_provisional"])
        self.assertAlmostEqual(report["model_pipeline_success_rate"], 1 / 3)

    def test_honestly_failed_model_can_be_reused(self):
        old = self.old_run()
        self.audit.side_effect = lambda directory: fake_audit(directory, correct=False)
        self.prepare([old])
        report = batch.summarize(self.output)
        self.assertEqual(report["audited_tasks"], 1)
        self.assertEqual(report["model_update_correct_count"], 0)
        self.assertEqual(report["model_failure_stages"]["update"], 1)

    def test_invalid_reuse_is_rejected_before_creating_output(self):
        old = self.old_run()
        self.audit.side_effect = lambda directory: fake_audit(directory, integrity=False)
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.prepare([old])
        self.assertFalse(self.output.exists())

    def test_duplicate_reuse_rejected(self):
        old = self.old_run()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare([old, old])
        self.assertFalse(self.output.exists())

    def test_changed_model_controls_cannot_be_imported(self):
        old = self.old_run()
        record = base.read_json(old / "run.json")
        record["controls"]["thinking_mode"] = "on"
        base.write_json(old / "run.json", record)
        with self.assertRaisesRegex(ValueError, "controls"):
            self.prepare([old])

    def test_branch_gpu_mismatch_cannot_be_imported(self):
        old = self.old_run()
        episode = base.read_json(old / "model/episode.json")
        episode["gpu_device_id"] = 3
        base.write_json(old / "model/episode.json", episode)
        with self.assertRaisesRegex(ValueError, "GPU"):
            self.prepare([old])

    def test_raw_evidence_tamper_blocks_run_but_is_reported_with_fixed_denominator(self):
        old = self.old_run()
        self.prepare([old])
        (old / "model/frame.bin").write_bytes(b"changed sensor")
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.run_batch(lambda *args: self.fail("must not launch"))
        report = batch.summarize(self.output)
        self.assertEqual(report["planned_episodes"], 9)
        self.assertEqual(report["audited_episodes"], 0)
        self.assertEqual(report["model_failure_stages"]["evidence"], 1)

    def test_path_escape_in_sensor_attachment_rejected(self):
        old = self.old_run()
        base.write_json(old / "model/trajectory.json", [{"frames": {"rgb": "../oracle/frame.bin"}}])
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare([old])

    def test_denominator_and_roster_cannot_be_reduced(self):
        self.prepare()
        path = self.output / "manifest.json"
        original = base.read_json(path)
        for key, value in (("tasks", original["tasks"][:2]), ("planned_episodes", 6)):
            changed = deepcopy(original)
            changed[key] = value
            base.write_json(path, changed)
            with self.assertRaises(ValueError):
                batch.validate_batch(self.output)
        base.write_json(path, original)

    def test_run_skips_reuse_continues_nonzero_and_records_gpu_per_task(self):
        old = self.old_run(gpu=0)
        self.prepare([old])
        commands, launcher = self.launcher({"task_002": 1})
        self.assertEqual(self.run_batch(launcher), 0)
        self.assertEqual(len(commands), 2)
        self.assertEqual([Path(c[c.index("--output-dir") + 1]).name for c in commands], ["task_002", "task_003"])
        self.assertTrue(all(c[c.index("--max-tokens") + 1] == "512" for c in commands))
        status = base.read_json(self.output / "run_status.json")
        self.assertEqual([s["returncode"] for s in status], [None, 1, 0])
        report = batch.summarize(self.output)
        self.assertTrue(report["passed"])
        self.assertEqual(report["gpu_ids"], [0, 2])
        self.assertEqual(report["callback_latency"]["new"]["count"], 2)
        with self.assertRaises(FileExistsError):
            self.run_batch(launcher)

    def test_real_model_failure_remains_in_final_scores(self):
        self.prepare()
        commands, launcher = self.launcher({"task_002": 1})
        self.audit.side_effect = lambda path: fake_audit(path, correct=path.name != "task_002")
        self.assertEqual(self.run_batch(launcher), 1)
        report = batch.summarize(self.output)
        self.assertEqual(len(commands), 3)
        self.assertTrue(report["integrity_passed"])
        self.assertFalse(report["results_provisional"])
        self.assertEqual(report["model_pipeline_success_count"], 2)
        self.assertAlmostEqual(report["model_pipeline_success_rate"], 2 / 3)
        self.assertEqual(len(single.lines(self.output / "cases.jsonl")), 9)
        self.assertEqual(len(single.lines(self.output / "tasks.jsonl")), 3)

    def test_missing_and_invalid_results_keep_all_slots(self):
        self.prepare()
        def missing(command, log):
            return 1
        self.assertEqual(self.run_batch(missing), 1)
        report = batch.summarize(self.output)
        self.assertEqual(report["planned_episodes"], 9)
        self.assertEqual(report["model_failure_stages"]["evidence"], 3)
        self.assertEqual(report["by_condition"]["model"]["joint_success_rate"], 0)

    def test_interrupt_keeps_pending_tasks_and_prevents_silent_retry(self):
        self.prepare()
        def interrupt(command, log):
            raise KeyboardInterrupt
        self.assertEqual(self.run_batch(interrupt), 1)
        status = base.read_json(self.output / "run_status.json")
        self.assertEqual([s["status"] for s in status], ["interrupted", "pending", "pending"])
        with self.assertRaises(FileExistsError):
            self.run_batch(interrupt)

    def test_occupied_future_output_stops_before_first_child(self):
        self.prepare()
        (self.output / "tasks/task_003").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.run_batch(lambda *args: self.fail("must not launch"))
        self.assertFalse((self.output / "batch_run.json").exists())

    def test_videos_read_reuse_but_write_only_batch_destinations(self):
        old = self.old_run()
        self.prepare([old])
        with patch.object(batch, "export_video", return_value=Path("fixture.mp4")) as export:
            result = batch.videos(SimpleNamespace(output_dir=self.output, task_id="task_001", fps=10.0))
        self.assertEqual(result, 0)
        self.assertEqual(export.call_count, 3)
        for call in export.call_args_list:
            self.assertTrue(str(call.args[0]).startswith(str(old)))
            self.assertTrue(str(call.kwargs["output"]).startswith(str(self.output / "videos/task_001")))


if __name__ == "__main__":
    unittest.main()
