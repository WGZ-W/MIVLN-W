"""Video ordering, audit provenance and explicit reviews; no simulator/model.

The miniature batch/audit below is synthetic test evidence. Encoder mocks never
count as real MP4 acceptance; a separate optional test invokes actual FFmpeg.
"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mivln import benchmark_navigation as navigation
from mivln import video_review
from mivln.navigation_fixture import FreePlaneBackend, privileged_policy
from scripts import export_visual_videos as cli
from scripts import hssd_visual_dataset as batch
from scripts import benchmark_navigation as source
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import sha256
from scripts.public_inputs import write_json
from tests.test_benchmark_navigation import case


def fixture(root, stack, mock_encoder=True):
    """Build real causal V traces with descending IDs, synthetic audit authority."""
    run, prepared, base, audit = [root / name for name in ('run', 'prepared', 'source', 'audit')]
    write_json(base / 'manifest.json', {'fixture': 'synthetic source authority'})
    tasks = [{'task_id': identity, 'reuse': None} for identity in ('task_001', 'task_007')]
    plan = batch.planned_slots(tasks)
    write_json(prepared / 'manifest.json', {'kind': batch.KIND, 'code_bindings': batch.code_bindings(),
        'source_dir': str(base), 'tasks': tasks, 'planned_episodes': 12, 'episode_plan': plan})
    write_json(run / 'manifest.json', {'kind': batch.RUN_KIND, 'code_bindings': batch.code_bindings(),
        'prepared_dir': str(prepared), 'prepared_manifest_sha256': sha256(prepared / 'manifest.json')})
    rows = []
    for slot in plan:
        config = case(slot['task_type'])
        backend = FreePlaneBackend(config)
        result = navigation.run_episode(config, backend,
            privileged_policy(config, backend, 'follow' if slot['condition'] == 'oracle' else 'ignore'),
            asset_ids=[f'{300-index:032x}' for index in range(65)])
        task_run = run / 'tasks' / slot['task_id'] / 'run'
        folder = task_run / 'episodes' / slot['case_id'] / slot['condition']
        source.write_episode(folder, config, result)
        rows.append({**slot, 'source_run': str(task_run), 'integrity_passed': True, 'accepted': True,
            'task_success': result['metrics']['task_success']})
    report = {'passed': True, 'results_provisional': False, 'planned_episodes': 12,
              'accepted_episode_slots': 12, 'episodes': rows}
    write_json(run / 'summary.json', report)
    write_json(audit / 'audit.json', {**report, 'source_files_unchanged': True})
    old = snapshot(cli.roots_for(run, read_json(prepared / 'manifest.json')))
    write_json(audit / 'source_bindings.json', {'before': old, 'after': old})
    stack.enter_context(patch.object(source, 'load_preparation', return_value={'fixture': 'synthetic source'}))
    encoded = {}
    def encode(frames, path, fps, binary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('Synthetic test video ' + path.name).encode())
        encoded[str(path)] = {'codec_name': 'h264', 'width': frames[0]['width'], 'height': frames[0]['height'],
            'nb_read_frames': str(len(frames)), 'avg_frame_rate': str(fps), 'duration': str(len(frames)/fps)}
    if mock_encoder:
        stack.enter_context(patch.object(cli, 'encode', side_effect=encode))
        stack.enter_context(patch.object(cli, 'probe_video', side_effect=lambda path, binary: encoded[str(path)]))
    args = SimpleNamespace(input_dir=run, audit_dir=audit, output_dir=root / 'videos',
        selection='representative', fps=10., ffmpeg=Path(sys.executable), ffprobe=Path(sys.executable))
    return args, report


def read_json(path):
    return json.loads(path.read_text())


def write_sheet(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=video_review.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def approve_rows(videos):
    rows = video_review.worksheet(videos)
    for row in rows:
        row.update(decision='approve', reviewer='Synthetic Reviewer', note='Fixture-only playback declaration, not real review.')
        for key in video_review.CHECKS:
            if row[key] != 'N/A':
                row[key] = 'true'
    return rows


class VideoReviewTests(unittest.TestCase):
    def videos(self):
        return [{'video_id': str(index), 'task_id': 'task_001', 'task_type': family, 'condition': 'ignore',
                 'expected_task_success': False, 'video_sha256': 'a'*64}
                for index, family in enumerate(('replacement', 'subgoal', 'cancel_task'))]

    def test_pending_and_inapplicable_checks_are_not_automatic_approvals(self):
        rows = video_review.worksheet(self.videos())
        self.assertTrue(all(row['decision'] == 'pending' and not row['reviewer'] for row in rows))
        self.assertEqual(rows[2]['target_identity'], 'N/A')
        with self.assertRaises(ValueError):
            video_review.import_sheet(self.videos(), rows)

    def test_explicit_complete_review_preserves_ignore_failure_labels(self):
        videos = self.videos()
        result = video_review.import_sheet(videos, approve_rows(videos))
        self.assertTrue(all(row['expected_task_success'] == 'false' for row in result))
        self.assertTrue(all(video['expected_task_success'] is False for video in videos))

    def test_names_notes_full_playback_and_all_applicable_checks_are_required(self):
        videos = self.videos()
        for field, value in (('reviewer', 'YOUR_REVIEWER_NAME'), ('note', ''), ('watched_full_video', 'false'), ('target_identity', 'false')):
            rows = approve_rows(videos)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                video_review.import_sheet(videos, rows)

    def test_review_cannot_drop_video_change_hash_or_fill_na_as_checked(self):
        videos = self.videos()
        for kind in ('drop', 'hash', 'label', 'na'):
            rows = approve_rows(videos)
            if kind == 'drop': rows.pop()
            elif kind == 'hash': rows[0]['video_sha256'] = 'b'*64
            elif kind == 'label': rows[0]['expected_task_success'] = 'true'
            else: rows[2]['target_identity'] = 'true'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                video_review.import_sheet(videos, rows)

    def test_malformed_review_cells_are_rejected_as_data_errors(self):
        rows = approve_rows(self.videos())
        rows[0]['note'] = None
        with self.assertRaisesRegex(ValueError, 'every review CSV cell'):
            video_review.import_sheet(self.videos(), rows)


class ExportTests(unittest.TestCase):
    def test_eight_selection_frame_order_stop_and_source_snapshots(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args, _ = fixture(root, stack)
            before = snapshot({'run': args.input_dir, 'audit': args.audit_dir})
            self.assertEqual(cli.export(args), 0)
            self.assertEqual(before, snapshot({'run': args.input_dir, 'audit': args.audit_dir}))
            manifest = cli.load_export(args.output_dir, sys.executable)
            self.assertEqual(len(manifest['videos']), 8)
            self.assertEqual(manifest['navigation_planned_episodes'], 12)
            self.assertEqual(cli.status(manifest, args.output_dir)['human_review']['pending'], 8)
            video = manifest['videos'][0]
            frames = source.lines(args.output_dir / video['frames'])
            self.assertEqual([row['frame_index'] for row in frames], list(range(video['frame_count'])))
            self.assertNotEqual([row['asset_id'] for row in frames], sorted(row['asset_id'] for row in frames))
            self.assertEqual(frames[-1]['action'], 'STOP')
            self.assertEqual(frames[-1]['observation_step'], video['frame_count']-1)
            self.assertEqual(sum(row['current_intervention'] is not None for row in frames), 1)

    def test_subtitles_mark_intervention_only_at_actual_slot_and_keep_terminal_stop(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            args, report = fixture(Path(temp), stack)
            prepared = read_json(Path(args.input_dir.parent / 'prepared/manifest.json'))
            _, _, frames = cli.frame_plan(args.input_dir, prepared, report['episodes'][0])
            srt = cli.subtitles(frames, 10.)
            self.assertEqual(srt.count('INTERVENTION:'), 1)
            self.assertIn('00:00:00,400 --> 00:00:00,500\nStep 4', srt)
            self.assertIn('next action: STOP', srt)

    def test_source_change_blocks_export_before_writing(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            args, _ = fixture(Path(temp), stack)
            png = next(args.input_dir.rglob('public/assets/*.png'))
            png.write_bytes(b'Changed actual image')
            with self.assertRaisesRegex(ValueError, 'source bytes/file roster changed'):
                cli.export(args)
            self.assertFalse(args.output_dir.exists())

    def test_incomplete_or_unmatched_audit_is_not_an_export_authority(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            args, _ = fixture(Path(temp), stack)
            path = args.audit_dir / 'audit.json'
            original = path.read_bytes()
            for field, value in (('passed', False), ('results_provisional', True), ('accepted_episode_slots', 11)):
                data = read_json(path)
                data[field] = value
                write_json(path, data)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    cli.export(args)
                path.write_bytes(original)

    def test_output_cannot_change_sealed_source_tree(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            args, _ = fixture(Path(temp), stack)
            args.output_dir = args.input_dir / 'new_videos'
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                cli.export(args)
            self.assertFalse(args.output_dir.exists())

    def test_encoder_failure_keeps_eight_planned_slots_and_no_reviewable_manifest(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _ = fixture(Path(temp), stack)
            with patch.object(cli, 'encode', side_effect=ValueError('Synthetic codec failure')):
                self.assertEqual(cli.export(args), 1)
            report = read_json(args.output_dir / 'export_report.json')
            self.assertEqual(report['planned_videos'], 8)
            self.assertEqual(len(report['planned_video_ids']), 8)
            self.assertEqual(len(report['failures']), 8)
            self.assertFalse((args.output_dir / 'manifest.json').exists())
            self.assertFalse((args.output_dir / 'review_sheet.csv').exists())

    def test_changed_mp4_or_resealed_wrong_subtitle_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _ = fixture(Path(temp), stack)
            self.assertEqual(cli.export(args), 0)
            manifest = read_json(args.output_dir / 'manifest.json')
            video = manifest['videos'][0]
            mp4 = args.output_dir / video['video']
            original = mp4.read_bytes()
            mp4.write_bytes(b'Another movie')
            with self.assertRaisesRegex(ValueError, 'changed'):
                cli.load_export(args.output_dir)
            mp4.write_bytes(original)
            subtitle = args.output_dir / video['subtitles']
            subtitle.write_text('Wrong timing')
            video['subtitles_sha256'] = sha256(subtitle)
            write_json(args.output_dir / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'Subtitle timing'):
                cli.load_export(args.output_dir)

    def test_human_review_is_new_artifact_and_bound_to_unchanged_export(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args, _ = fixture(root, stack)
            self.assertEqual(cli.export(args), 0)
            manifest = cli.load_export(args.output_dir)
            sheet = root / 'reviews.csv'
            write_sheet(sheet, approve_rows(manifest['videos']))
            before = snapshot({'export': args.output_dir})
            review_dir = root / 'review'
            self.assertEqual(cli.review(SimpleNamespace(input_dir=args.output_dir, sheet=sheet, output_dir=review_dir)), 0)
            self.assertEqual(before, snapshot({'export': args.output_dir}))
            self.assertTrue(cli.status(manifest, args.output_dir, review_dir)['all_playback_approved'])
            write_json(args.output_dir / 'manifest.json', {**manifest, 'export_id': 'another-export'})
            with self.assertRaisesRegex(ValueError, 'another or changed export'):
                cli.status(manifest, args.output_dir, review_dir)

    def test_wrong_encoding_frame_count_rate_or_duration_is_rejected(self):
        video = {'frame_count': 10, 'fps': 10., 'width': 8, 'height': 8}
        good = {'codec_name': 'h264', 'nb_read_frames': '10', 'width': 8, 'height': 8, 'avg_frame_rate': '10/1', 'duration': '1'}
        cli.check_media(good, video)
        for field, value in (('nb_read_frames', '9'), ('avg_frame_rate', '0/0'), ('duration', '2')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                cli.check_media({**good, field: value}, video)

    def test_representative_selection_does_not_silently_drop_missing_task(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            args, report = fixture(Path(temp), stack)
            self.assertEqual(len(cli.select_rows(report['episodes'], 'all')), 12)
            with self.assertRaises(ValueError):
                cli.select_rows([r for r in report['episodes'] if r['task_id'] != 'task_007'], 'representative')
            for rows, selection in (([], 'all'), (report['episodes'], 'unknown')):
                with self.assertRaises(ValueError):
                    cli.select_rows(rows, selection)


class RealEncoderTests(unittest.TestCase):
    def test_actual_ffmpeg_outputs_ordered_frame_count_and_dimensions(self):
        try:
            ffmpeg, ffprobe = cli.executable('ffmpeg'), cli.executable('ffprobe')
        except ValueError:
            self.skipTest('FFmpeg/FFprobe unavailable; no dependency installation')
        from mivln.visual_frames import RGBFrame
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = []
            for index in range(3):
                path = root / (str(3-index)+'.png')
                path.write_bytes(RGBFrame.collect('0'*32, index, 8, 8, bytes([index*80, 40, 80])*64).png())
                frames.append({'frame_index': index, 'source_png': str(path), 'source_sha256': sha256(path), 'width': 8, 'height': 8})
            destination = root / 'video.mp4'
            cli.encode(frames, destination, 10., ffmpeg)
            cli.check_media(cli.probe_video(destination, ffprobe), {'frame_count': 3, 'fps': 10., 'width': 8, 'height': 8})


if __name__ == '__main__':
    unittest.main()
