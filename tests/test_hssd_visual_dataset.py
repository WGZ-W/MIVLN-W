"""Full-roster V batch tests with isolated synthetic sources and fake GPUs."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import hssd_visual_dataset as batch
from scripts import hssd_visual_navigation as single
from scripts import benchmark_navigation as source
from scripts import language_annotations as language
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import sha256
from scripts.public_inputs import write_json


def source_fixture(root, failed_second=False):
    """Two synthetic base identities sharing geometry, never actual HSSD data."""
    from tests.test_benchmark_navigation import reviewed_fixture
    from tests.test_language_annotations import LanguageAnnotationTests
    _, raw = reviewed_fixture(root)
    manifest = language.read_json(raw / 'manifest.json')
    second = deepcopy(manifest['tasks'][0])
    second.update(task_id='task_002', output_subdir='tasks/task_002')
    for key in ('base_config', 'reference_config'):
        old = raw / second[key]
        second[key] = second[key].replace('task_001', 'task_002')
        config = language.read_json(old)
        config['episode_id'] = 'task_002'
        write_json(raw / second[key], config)
        second[key+'_sha256'] = sha256(raw / second[key])
    shutil.copytree(raw / 'tasks/task_001/reference', raw / 'tasks/task_002/reference')
    episode = language.read_json(raw / 'tasks/task_002/reference/episode.json')
    episode['episode_id'] = 'task_002'
    write_json(raw / 'tasks/task_002/reference/episode.json', episode)
    manifest['tasks'].append(second)
    manifest['task_count'] = 2
    write_json(raw / 'manifest.json', manifest)
    draft = root / 'two_draft'
    language.prepare(SimpleNamespace(source_dir=raw, output_dir=draft))
    approved = LanguageAnnotationTests().approve(root, draft, name='two_approved')
    if failed_second:
        (raw / 'tasks/task_002/reference/trajectory.json').unlink()
    original = root / 'source_prepared'
    source.prepare(SimpleNamespace(annotations_dir=approved, output_dir=original))
    return original


def prepare_args(root, source_dir, reused=None):
    return SimpleNamespace(source_dir=source_dir, reuse_run=reused or [], output_dir=root / 'prepared')


class BatchPreparationTests(unittest.TestCase):
    def test_prepare_keeps_full_roster_three_types_two_conditions_and_provisional_scores(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            original = source_fixture(root)
            before = snapshot({'source': original})
            args = prepare_args(root, original)
            self.assertEqual(batch.prepare(args), 0)
            manifest, _ = batch.validate_prepared(args.output_dir)
            self.assertEqual(manifest['task_count'], 2)
            self.assertEqual(manifest['planned_cases'], 6)
            self.assertEqual(manifest['planned_episodes'], 12)
            self.assertEqual(manifest['new_episodes'], 12)
            self.assertEqual(len(manifest['episode_plan']), 12)
            report = language.read_json(args.output_dir / 'preparation_summary.json')
            self.assertTrue(report['results_provisional'])
            self.assertFalse(report['passed'])
            self.assertEqual(report['oracle_macro_success_rate'], 0.)
            self.assertEqual(before, snapshot({'source': original}))

    def test_missing_reference_keeps_failed_base_and_all_twelve_slots(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = prepare_args(root, source_fixture(root, failed_second=True))
            self.assertEqual(batch.prepare(args), 1)
            manifest, _ = batch.validate_prepared(args.output_dir)
            self.assertIsNotNone(manifest['tasks'][1]['preparation_error'])
            self.assertEqual(len(manifest['episode_plan']), 12)
            report = language.read_json(args.output_dir / 'preparation_summary.json')
            self.assertEqual(sum(row['audit_status'] == 'preparation_failed' for row in report['episodes']), 6)

    def test_task_or_condition_removal_cannot_shrink_frozen_denominator(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = prepare_args(root, source_fixture(root))
            batch.prepare(args)
            path = args.output_dir / 'manifest.json'
            original = path.read_bytes()
            for field in ('tasks', 'episode_plan'):
                manifest = language.read_json(path)
                manifest[field].pop()
                write_json(path, manifest)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    batch.validate_prepared(args.output_dir)
                path.write_bytes(original)

    def test_changed_later_config_is_rejected_before_any_child_launch(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = prepare_args(root, source_fixture(root))
            batch.prepare(args)
            manifest = language.read_json(args.output_dir / 'manifest.json')
            folder = args.output_dir / manifest['tasks'][1]['prepared_dir']
            info = language.read_json(folder / 'manifest.json')
            path = folder / info['planned_episodes'][-1]['config']
            config = language.read_json(path)
            config['max_steps'] += 1
            write_json(path, config)
            with patch.object(batch.subprocess, 'run') as launch, self.assertRaises(ValueError):
                batch.run(SimpleNamespace(input_dir=args.output_dir, output_dir=root / 'run', gpu_id=0))
            launch.assert_not_called()
            self.assertFalse((root / 'run').exists())

    def test_occupied_run_output_is_rejected_without_launch_or_rewrite(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = prepare_args(root, source_fixture(root))
            batch.prepare(args)
            output = root / 'run'
            output.mkdir()
            (output / 'existing').write_text('retain')
            with patch.object(batch.subprocess, 'run') as launch, self.assertRaises(FileExistsError):
                batch.run(SimpleNamespace(input_dir=args.output_dir, output_dir=output, gpu_id=0))
            launch.assert_not_called()
            self.assertEqual((output / 'existing').read_text(), 'retain')

    def test_negative_or_boolean_gpu_is_rejected_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for gpu in (-1, True):
                with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                    batch.run(SimpleNamespace(input_dir=root / 'not_loaded', output_dir=root / 'run', gpu_id=gpu))
            self.assertFalse((root / 'run').exists())

    def test_interrupt_and_failed_children_keep_all_task_statuses_and_continue(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args = prepare_args(root, source_fixture(root))
            batch.prepare(args)
            output = root / 'interrupted'
            with patch.object(batch.subprocess, 'run', side_effect=KeyboardInterrupt) as launch:
                self.assertEqual(batch.run(SimpleNamespace(input_dir=args.output_dir, output_dir=output, gpu_id=0)), 1)
            self.assertEqual(launch.call_count, 1)
            statuses = language.read_json(output / 'execution_status.json')
            self.assertEqual([row['state'] for row in statuses], ['interrupted', 'pending'])
            report = language.read_json(output / 'summary.json')
            self.assertEqual(len(report['episodes']), 12)
            self.assertTrue(report['results_provisional'])
            output = root / 'failed'
            with patch.object(batch.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as launch:
                self.assertEqual(batch.run(SimpleNamespace(input_dir=args.output_dir, output_dir=output, gpu_id=0)), 1)
            self.assertEqual(launch.call_count, 2)
            self.assertEqual(len(language.read_json(output / 'summary.json')['episodes']), 12)

    def test_complete_valid_failure_is_scored_without_filtering_or_provisional_success(self):
        tasks = [{'task_id': 'task_001', 'preparation_error': None, 'reuse': None}]
        slots = batch.planned_slots(tasks)
        manifest = {'tasks': tasks, 'task_count': 1, 'planned_cases': 3, 'planned_episodes': 6,
                    'episode_plan': slots, 'reused_episodes': 0, 'new_episodes': 6}
        episodes = [{**slot, 'evidence_valid': True, 'task_success': slot['condition'] == 'oracle', 'passed': True} for slot in slots]
        episodes[0].update(task_success=False, passed=False)
        report = batch.collect(manifest, {'task_001': {'state': 'completed', 'returncode': 0}},
                               {'task_001': {'episodes': episodes, 'pairing_errors': []}})
        self.assertFalse(report['passed'])
        self.assertFalse(report['results_provisional'])
        self.assertEqual(report['valid_evidence_episodes'], 6)
        self.assertEqual(report['accepted_episode_slots'], 5)
        self.assertAlmostEqual(report['oracle_macro_success_rate'], 2/3)


@unittest.skipUnless(importlib.util.find_spec('habitat_sim'), 'Requires native imports; all renderers/children are fake')
class BatchReuseTests(unittest.TestCase):
    def fixture(self, root, stack, reused=True):
        from tests.test_hssd_visual_navigation import fake_renderer
        original = source_fixture(root)
        old_prepared = root / 'old_prepared'
        single.prepare(SimpleNamespace(input_dir=original, output_dir=old_prepared, base_task='task_001'))
        info = single.load_prepared(old_prepared)
        config = language.read_json(old_prepared / info['planned_episodes'][0]['config'])
        instances, _ = fake_renderer(stack, config)
        calls = []
        def child(command, **kwargs):
            module = command[command.index('-m')+1]
            operation = command[command.index('-m')+2]
            self.assertEqual(module, 'scripts.hssd_visual_navigation')
            keys = ['input_dir', 'output_dir', 'gpu_id'] + (['case_id', 'condition'] if operation == 'episode' else [])
            args = {key: command[command.index('--'+key.replace('_', '-'))+1] for key in keys}
            args.update(input_dir=Path(args['input_dir']), output_dir=Path(args['output_dir']), gpu_id=int(args['gpu_id']))
            calls.append((operation, str(args['input_dir'])))
            code = single.episode(SimpleNamespace(**args)) if operation == 'episode' else single.run(SimpleNamespace(**args))
            return SimpleNamespace(returncode=code)
        stack.enter_context(patch.object(batch.subprocess, 'run', side_effect=child))
        old_run = root / 'old_run'
        if reused:
            self.assertEqual(single.run(SimpleNamespace(input_dir=old_prepared, output_dir=old_run, gpu_id=0)), 0)
        args = prepare_args(root, original, [old_run] if reused else [])
        before = snapshot({'old': old_run}) if reused else None
        self.assertEqual(batch.prepare(args), 0)
        if reused:
            self.assertEqual(before, snapshot({'old': old_run}))
        calls.clear()
        return args.output_dir, old_run, calls, instances

    def test_reuse_six_slots_run_only_new_task_and_audit_all_twelve_without_gpu_replay(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, old, calls, instances = self.fixture(root, stack)
            draft = language.read_json(prepared / 'preparation_summary.json')
            self.assertEqual(draft['accepted_episode_slots'], 6)
            self.assertEqual(draft['planned_episodes'], 12)
            self.assertTrue(draft['results_provisional'])
            self.assertFalse(draft['passed'])
            before = snapshot({'old': old, 'prepared': prepared})
            output = root / 'run'
            self.assertEqual(batch.run(SimpleNamespace(input_dir=prepared, output_dir=output, gpu_id=2)), 0)
            self.assertEqual(sum(operation == 'run' for operation, path in calls), 1)
            self.assertEqual(sum(operation == 'episode' for operation, path in calls), 6)
            self.assertEqual(len(instances), 12)
            self.assertFalse((output / 'tasks/task_001/run').exists())
            self.assertEqual(before, snapshot({'old': old, 'prepared': prepared}))
            report = batch.check_run(output)
            self.assertTrue(report['passed'])
            self.assertFalse(report['results_provisional'])
            self.assertEqual(report['accepted_episode_slots'], 12)
            self.assertEqual(report['oracle_macro_success_rate'], 1.)
            self.assertEqual([row['origin'] for row in report['tasks']], ['reused', 'new'])
            self.assertTrue(all(row['accepted'] == 6 and row['oracle_successes'] == 3 for row in report['tasks']))
            for group in report['by_task_type'].values():
                self.assertEqual(group['oracle']['task_successes'], 2)
                self.assertEqual(group['ignore']['accepted'], 2)
                self.assertEqual(group['ignore']['task_successes'], 0)
            before = snapshot({'run': output, 'old': old, 'prepared': prepared})
            audit = root / 'audit'
            with patch('sys.argv', ['hssd_visual_dataset', 'audit', '--input-dir', str(output), '--output-dir', str(audit)]):
                self.assertEqual(batch.main(), 0)
            self.assertTrue(language.read_json(audit / 'audit.json')['source_files_unchanged'])
            self.assertEqual(before, snapshot({'run': output, 'old': old, 'prepared': prepared}))
            self.assertEqual(len(instances), 12)

    def test_changed_reused_bytes_prevent_every_new_child(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, old, calls, _ = self.fixture(root, stack)
            (old / 'logs/extra.txt').write_text('Changed file roster')
            with self.assertRaisesRegex(ValueError, 'file roster/bytes changed'):
                batch.run(SimpleNamespace(input_dir=prepared, output_dir=root / 'run', gpu_id=0))
            self.assertEqual(calls, [])
            self.assertFalse((root / 'run').exists())

    def test_duplicate_reuse_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, old, _, _ = self.fixture(root, stack)
            manifest = language.read_json(prepared / 'manifest.json')
            args = SimpleNamespace(source_dir=Path(manifest['source_dir']), reuse_run=[old, old], output_dir=root / 'duplicate')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                batch.prepare(args)
            self.assertFalse(args.output_dir.exists())

    def test_output_cannot_be_nested_inside_reused_preparation_or_evidence(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, old, _, _ = self.fixture(root, stack)
            manifest = language.read_json(prepared / 'manifest.json')
            for output in (old / 'nested_batch', root / 'old_prepared/nested_batch'):
                args = SimpleNamespace(source_dir=Path(manifest['source_dir']), reuse_run=[old], output_dir=output)
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'overlaps'):
                    batch.prepare(args)
                self.assertFalse(output.exists())

    def test_runtime_pending_or_nonzero_parent_cannot_be_relabelled_as_success(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, old, calls, _ = self.fixture(root, stack)
            output = root / 'run'
            self.assertEqual(batch.run(SimpleNamespace(input_dir=prepared, output_dir=output, gpu_id=0)), 0)
            path = output / 'execution_status.json'
            original = path.read_bytes()
            for state, code in (('pending', None), ('failed', 1)):
                statuses = language.read_json(path)
                statuses[1].update(state=state, returncode=code)
                write_json(path, statuses)
                report = batch.check_run(output)
                self.assertFalse(report['passed'])
                self.assertTrue(report['execution_errors'])
                self.assertEqual(len(report['episodes']), 12)
                path.write_bytes(original)

    def test_missing_new_branch_remains_in_denominator_and_caches_do_not_repair_it(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, _, _, _ = self.fixture(root, stack)
            output = root / 'run'
            self.assertEqual(batch.run(SimpleNamespace(input_dir=prepared, output_dir=output, gpu_id=0)), 0)
            shutil.rmtree(output / 'tasks/task_002/run/episodes/task_002_cancel_task_middle/oracle')
            write_json(output / 'summary.json', {'passed': True, 'accepted_episode_slots': 12})
            report = batch.check_run(output)
            self.assertFalse(report['passed'])
            self.assertEqual(len(report['episodes']), 12)
            self.assertEqual(report['accepted_episode_slots'], 11)


if __name__ == '__main__':
    unittest.main()
