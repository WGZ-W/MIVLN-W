"""Middle HSSD wiring tests: real adapter/API, fake renderer, no GPU.

Synthetic image/instance sources are isolated in temporary directories. Passing
these tests does not establish real HSSD visibility or rendered navigation.
"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import benchmark_navigation as source
from scripts import hssd_visual_navigation as cli
from scripts import hssd_visual_backend as backend_module
from scripts.benchmark_audit import snapshot
from scripts.benchmark_manifest import sha256
from scripts.language_annotations import read_json
from scripts.public_inputs import write_json


def prepared_fixture(root):
    from tests.test_benchmark_navigation import reviewed_fixture
    reviewed, _ = reviewed_fixture(root)
    original = root / 'source_prepared'
    source.prepare(SimpleNamespace(annotations_dir=reviewed, output_dir=original))
    prepared = root / 'hssd_prepared'
    cli.prepare(SimpleNamespace(input_dir=original, output_dir=prepared, base_task='task_001'))
    return prepared, original


class HSSDPreparationTests(unittest.TestCase):
    def test_camera_uses_actual_sensor_hfov_resolution_and_square_pixels(self):
        configuration = SimpleNamespace(agents=[SimpleNamespace(sensor_specifications=[
            SimpleNamespace(uuid='rgb', resolution=[128, 256], hfov=60.)])])
        camera = backend_module.camera_calibration(configuration)
        self.assertEqual((camera['width'], camera['height']), (256, 128))
        self.assertAlmostEqual(camera['fx'], 128/math.tan(math.pi/6))
        self.assertEqual(camera['fx'], camera['fy'])
        configuration.agents[0].sensor_specifications[0].hfov = 180.
        with self.assertRaises(ValueError):
            backend_module.camera_calibration(configuration)

    def test_middle_has_three_cases_six_slots_and_preserves_geometry_language_budget(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            prepared, original = prepared_fixture(Path(temp))
            before = snapshot({'source': original})
            manifest = cli.load_prepared(prepared)
            self.assertEqual(manifest['planned_cases'], 3)
            self.assertEqual(len(manifest['planned_episodes']), 6)
            self.assertEqual(manifest['trigger_step'], 4)
            for slot in manifest['planned_episodes']:
                config = read_json(prepared / slot['config'])
                old = read_json(Path(config['source_prepared_config']['path']))
                for key in ('start_position', 'start_rotation_xyzw', 'goal_positions', 'max_steps', 'initial_instruction', 'events', 'language_binding'):
                    self.assertEqual(config[key], old[key])
                self.assertEqual(config['execution_scope'], backend_module.SCOPE)
            self.assertEqual(before, snapshot({'source': original}))

    def test_preparation_rejects_all_tasks_and_occupied_output(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, original = prepared_fixture(root)
            with self.assertRaises(ValueError):
                cli.build(original, 'all')
            with self.assertRaises(FileExistsError):
                cli.prepare(SimpleNamespace(input_dir=original, output_dir=prepared, base_task='task_001'))

    def test_resealed_hssd_config_cannot_replace_reviewed_language(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            prepared, _ = prepared_fixture(Path(temp))
            manifest = read_json(prepared / 'manifest.json')
            path = prepared / manifest['planned_episodes'][0]['config']
            config = read_json(path)
            config['initial_instruction'] = 'An unrelated task.'
            write_json(path, config)
            for slot in manifest['planned_episodes']:
                if slot['config'] == str(path.relative_to(prepared)):
                    slot['config_sha256'] = sha256(path)
            write_json(prepared / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'reviewed source'):
                cli.load_prepared(prepared)

    def test_interruption_retains_running_and_pending_slots_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            prepared, _ = prepared_fixture(root)
            args = SimpleNamespace(input_dir=prepared, output_dir=root / 'run', gpu_id=0)
            with patch.object(cli.subprocess, 'run', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                cli.run(args)
            statuses = read_json(args.output_dir / 'execution_status.json')
            self.assertEqual(len(statuses), 6)
            self.assertEqual(statuses[0]['state'], 'interrupted')
            self.assertTrue(all(row['state'] == 'pending' for row in statuses[1:]))
            with self.assertRaises(FileExistsError):
                cli.run(args)


def fake_renderer(stack, config, *, bad_rgb=False):
    """Real quaternions and discrete motion; instance diagnostic uses real code."""
    habitat, np, quat, legacy, terminal = backend_module.native_modules()
    from scripts.subgoal import reviewed_centers
    centres = reviewed_centers(config, config['goal_positions'])
    instances = []
    objects = [SimpleNamespace(object_id=index, semantic_id=27) for index in range(3)]

    class Agent:
        def __init__(self, state):
            self.state = SimpleNamespace(position=state.position.copy(), rotation=state.rotation)

        def get_state(self):
            return self.state

        def act(self, action):
            angle = 2*math.atan2(float(self.state.rotation.imag[1]), float(self.state.rotation.real))
            if action == 'move_forward':
                self.state.position += np.asarray([-math.sin(angle), 0., -math.cos(angle)]) * config['forward_step_m']
            elif action in ('turn_left', 'turn_right'):
                angle += math.radians(config['turn_angle_degrees']) * (1 if action == 'turn_left' else -1)
                self.state.rotation = quat(np.asarray([0., math.sin(angle/2), 0., math.cos(angle/2)]))
            return False

    class Simulator:
        def __init__(self, configuration):
            instances.append(self)
            self.initializations, self.steps, self.closed = 0, [], False
            self.pathfinder = SimpleNamespace(is_loaded=True, seed=lambda seed: None,
                load_nav_mesh=lambda path: True, is_navigable=lambda position: True,
                nav_mesh_settings=SimpleNamespace(agent_height=1.5, agent_radius=.1, include_static_objects=True),
                navigable_area=10., num_islands=1)

        def seed(self, value):
            pass

        def get_rigid_object_manager(self):
            return SimpleNamespace(get_object_handles=lambda: list(range(3)), get_object_by_handle=lambda key: objects[key])

        def initialize_agent(self, index, state):
            self.initializations += 1
            self.agent = Agent(state)
            return self.agent

        def get_sensor_observations(self):
            height, width = config['resolution']
            rgb = np.zeros((height, width, 4), dtype=np.uint8)
            rgb[..., 0] = int(round(float(self.agent.state.position[0])*30)) % 256
            rgb[..., 1] = 80
            semantic = np.full((height, width), 27, dtype=np.uint32)
            for index, obj in enumerate(objects):
                if obj.semantic_id != 27:
                    semantic.reshape(-1)[index*4:index*4+4] = obj.semantic_id
            return {'rgb': rgb.astype(np.float32) if bad_rgb else rgb, 'semantic': semantic}

        def step(self, action):
            self.steps.append(action)
            self.agent.act(action)
            return {**self.get_sensor_observations(), 'collided': False}

        def close(self):
            self.closed = True

    class Follower:
        def __init__(self, pathfinder, agent, goal_radius, stop_key):
            self.agent, self.radius = agent, goal_radius

        def next_action_along(self, goal):
            if np.linalg.norm(goal-self.agent.state.position) <= self.radius:
                return 'STOP'
            action, _ = terminal.yaw_action(self.agent.state, goal, config['yaw_tolerance_degrees'])
            return 'move_forward' if action == 'STOP' else action

    class Replay:
        def __init__(self, case):
            self.agent = Agent(SimpleNamespace(position=np.asarray(case['start_position'], dtype=np.float32),
                rotation=quat(np.asarray(case['start_rotation_xyzw']))))

        def pose(self):
            return terminal.pose_snapshot(self.agent)

        def act(self, action):
            return self.agent.act(action) if action not in ('STOP', 'COMPLETE_SUBGOAL') else False

        def distance(self, point):
            return float(np.linalg.norm(np.asarray(point, dtype=np.float32)-self.agent.state.position))

        def yaw(self, centre):
            return terminal.yaw_action(self.agent.state, centre, config['yaw_tolerance_degrees'])[1]

        def close(self):
            pass

    def resolve(sim, scene, metadata):
        return {goal: {'object': objects[value['source_instance_index']], **value,
            'world_center': centres[goal], 'object_id': value['source_instance_index']} for goal, value in metadata.items()}

    stack.enter_context(patch.object(habitat, 'Simulator', Simulator))
    stack.enter_context(patch.object(habitat.nav, 'GreedyGeodesicFollower', Follower))
    stack.enter_context(patch.object(legacy, 'shortest_path', lambda pathfinder, start, goal:
        SimpleNamespace(geodesic_distance=float(np.linalg.norm(goal-start)))))
    stack.enter_context(patch.object(terminal, 'resolve_instances', resolve))
    stack.enter_context(patch.object(cli, 'NativeReplay', Replay))
    return instances, objects


@unittest.skipUnless(importlib.util.find_spec('habitat_sim'), 'Requires native imports; renderer and mesh replay are fake')
class HSSDWiringTests(unittest.TestCase):
    def fixture(self, root, stack, *, fail_first=False, bad_rgb=False, early_stop=False):
        prepared, _ = prepared_fixture(root)
        manifest = cli.load_prepared(prepared)
        config = read_json(prepared / manifest['planned_episodes'][0]['config'])
        instances, objects = fake_renderer(stack, config, bad_rgb=bad_rgb)
        calls = []
        real_policy = cli.OraclePolicy
        if early_stop:
            class EarlyPolicy:
                def __init__(self, cfg, backend, condition):
                    self.trace = []
                    self.delegate = real_policy(cfg, backend, condition)
                    self.condition = condition
                def __call__(self, request, assets):
                    return '{"action":"STOP"}' if self.condition == 'oracle' else self.delegate(request, assets)
            stack.enter_context(patch.object(cli, 'OraclePolicy', EarlyPolicy))

        def child(command, **kwargs):
            calls.append(command)
            if fail_first and len(calls) == 1:
                return SimpleNamespace(returncode=1)
            arguments = {key: command[command.index('--'+key.replace('_', '-'))+1]
                         for key in ('input_dir', 'output_dir', 'case_id', 'condition', 'gpu_id')}
            arguments.update(input_dir=Path(arguments['input_dir']), output_dir=Path(arguments['output_dir']), gpu_id=int(arguments['gpu_id']))
            return SimpleNamespace(returncode=cli.episode(SimpleNamespace(**arguments)))
        stack.enter_context(patch.object(cli.subprocess, 'run', side_effect=child))
        args = SimpleNamespace(input_dir=prepared, output_dir=root / 'run', gpu_id=0)
        return args, instances, objects, calls

    def test_six_real_adapter_branches_and_no_render_audit_preserve_pose_labels_and_inputs(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args, instances, objects, calls = self.fixture(root, stack)
            self.assertEqual(cli.run(args), 0)
            self.assertEqual(len(calls), 6)
            self.assertEqual(len(instances), 6)
            self.assertTrue(all(sim.initializations == 1 and sim.closed for sim in instances))
            self.assertTrue(all(obj.semantic_id == 27 for obj in objects))
            before = snapshot({'run': args.output_dir})
            report = cli.check_run(args.output_dir)
            self.assertTrue(report['passed'])
            self.assertEqual(report['accepted_episode_slots'], 6)
            self.assertEqual(sum(row['task_success'] for row in report['episodes']), 3)
            self.assertEqual(before, snapshot({'run': args.output_dir}))
            self.assertEqual(len(instances), 6, 'Audit must not create a rendered Simulator')
            requests = source.lines(args.output_dir / 'episodes/task_001_subgoal_middle/oracle/public/requests.jsonl')
            self.assertTrue(any(row['current_intervention'] for row in requests))
            for row in requests:
                for forbidden in ('task_update', 'goal_positions', 'measurements', 'instance_mask', 'source_instance_index'):
                    self.assertNotIn(forbidden, json.dumps(row))

    def test_measured_subgoal_completion_restores_a_and_cancel_stop_is_immediate(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _, _ = self.fixture(Path(temp), stack)
            self.assertEqual(cli.run(args), 0)
            report = cli.check_run(args.output_dir)
            subgoal = next(row for row in report['episodes'] if row['task_type'] == 'subgoal' and row['condition'] == 'oracle')
            self.assertEqual(len(subgoal['completions']), 1)
            self.assertTrue(subgoal['completions'][0]['valid'])
            self.assertEqual(subgoal['final_task_state']['active_goal'], 'target_000')
            cancel = next(row for row in report['episodes'] if row['task_type'] == 'cancel_task' and row['condition'] == 'oracle')
            self.assertEqual(cancel['cancellation']['stop_delay_actions'], 0)
            self.assertEqual(cancel['cancellation']['post_cancel_path_length_m'], 0.)

    def test_native_prefix_rehearsal_rejects_source_motion_that_does_not_match_actions(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temp)
            args, instances, _, _ = self.fixture(root, stack)
            # The older hand-authored source fixture moves +X with identity
            # orientation. The independent replay correctly moves toward -Z.
            output = root / 'geometry'
            self.assertEqual(cli.geometry(SimpleNamespace(input_dir=args.input_dir, output_dir=output)), 1)
            report = read_json(output / 'geometry.json')
            self.assertFalse(report['passed'])
            self.assertFalse(report['simulator_created'])
            self.assertEqual(len(instances), 0)

    def test_failed_child_does_not_skip_remaining_five_slots(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _, calls = self.fixture(Path(temp), stack, fail_first=True)
            self.assertEqual(cli.run(args), 1)
            self.assertEqual(len(calls), 6)
            report = read_json(args.output_dir / 'summary.json')
            self.assertEqual(len(report['episodes']), 6)
            self.assertEqual(report['accepted_episode_slots'], 5)

    def test_invalid_rgb_fails_without_inventing_policy_input_or_success(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, instances, _, calls = self.fixture(Path(temp), stack, bad_rgb=True)
            self.assertEqual(cli.run(args), 1)
            self.assertEqual(len(calls), 6)
            self.assertTrue(all(sim.closed for sim in instances))
            self.assertEqual(read_json(args.output_dir / 'summary.json')['accepted_episode_slots'], 0)
            self.assertEqual(len(list(args.output_dir.glob('episodes/*/*/private/slot_error.json'))), 6)

    def test_early_stop_is_failed_task_even_if_child_exits_successfully(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _, _ = self.fixture(Path(temp), stack, early_stop=True)
            self.assertEqual(cli.run(args), 1)
            report = read_json(args.output_dir / 'summary.json')
            oracle = [row for row in report['episodes'] if row['condition'] == 'oracle']
            self.assertTrue(all(not row['task_success'] and not row['passed'] for row in oracle))

    def test_caches_cannot_override_real_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _, _ = self.fixture(Path(temp), stack)
            self.assertEqual(cli.run(args), 0)
            for path in args.output_dir.glob('episodes/*/*/private/metrics.json'):
                write_json(path, {'task_success': True})
            write_json(args.output_dir / 'summary.json', {'passed': False})
            self.assertTrue(cli.check_run(args.output_dir)['passed'])

    def test_changed_png_mask_pose_or_future_message_fails_read_only_audit(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, _, _, _ = self.fixture(Path(temp), stack)
            self.assertEqual(cli.run(args), 0)
            folder = args.output_dir / 'episodes/task_001_subgoal_middle/oracle'
            for kind in ('rgb', 'mask', 'pose', 'message'):
                if kind == 'rgb':
                    path = next((folder / 'public/assets').glob('*.png'))
                elif kind == 'mask':
                    path = next((folder / 'private/diagnostics').glob('*/*/terminal_instance_mask.npy'))
                else:
                    path = folder / ('private/trajectory.json' if kind == 'pose' else 'private/policy_records.jsonl')
                original = path.read_bytes()
                if kind in ('rgb', 'mask'):
                    path.write_bytes(b'Changed actual bytes')
                elif kind == 'pose':
                    rows = read_json(path)
                    rows[1]['position'][0] += .5
                    write_json(path, rows)
                else:
                    records = source.lines(path)
                    records[0]['request']['current_intervention'] = {'delivered_step': 0, 'message': 'Future message.'}
                    source.write_lines(path, records)
                self.assertFalse(cli.check_run(args.output_dir)['passed'], kind)
                path.write_bytes(original)
            self.assertTrue(cli.check_run(args.output_dir)['passed'])

    def test_cancel_logical_stop_never_calls_physical_simulator_step(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()):
            args, instances, _, _ = self.fixture(Path(temp), stack)
            self.assertEqual(cli.run(args), 0)
            for sim in instances:
                self.assertNotIn('STOP', sim.steps)
                self.assertNotIn('COMPLETE_SUBGOAL', sim.steps)


if __name__ == '__main__':
    unittest.main()
