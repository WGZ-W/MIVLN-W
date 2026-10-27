# MIVLN

MIVLN is a minimal research scaffold for **mixed-initiative vision-and-language navigation**. It focuses first on the task logic needed when a robot is navigating autonomously and a human changes the task during execution.

The core task logic is simulator-independent, with optional Habitat-Sim navigation experiments:

```text
Go to the kitchen
        ↓
intervention at step 5
        ↓
Instead, go to the bedroom
        ↓
update task state → stop at bedroom → evaluate success
```

## Included components

- `TaskState`: the authoritative dynamic task state.
- `EventEngine`: deterministic delivery of intervention events.
- `Evaluator`: metrics for the final active goal and intervention response.
- `scripts/run_demo.py`: a complete target-replacement example.
- `scripts/run_intervention.py`: matched Habitat control/target-replacement branches and automatic validation.
- `tests/`: dependency-free unit tests for the core logic.

Habitat is intentionally not required yet. The core task semantics can therefore be tested on Windows before connecting them to a Linux simulator.

## Quick start

Python 3.9 or newer is supported, including the existing Python 3.9 Habitat environment.

```powershell
python -m unittest discover -s tests -v
python -m scripts.run_demo
```

The demo writes a reproducible sample to `outputs/demo_001/` containing the episode, delivered events, trajectory, and metrics.

For all environment installation, scene download, execution, video export, and migration commands, see
[`docs/migration_commands.md`](docs/migration_commands.md).

## Hand-authored task cases

[`configs/task_cases_v0.json`](configs/task_cases_v0.json) contains five task definitions: kitchen to study, bedroom to living room, refrigerator to sofa, a later kitchen-to-study intervention, and a kitchen task without intervention. Cases 001, 004, and 005 share the same initial task for timing and control comparisons.

Each case uses the demo's episode fields and adds `expected_final_goal` as an evaluator-only reference. Goals are symbolic identifiers; scene locations and reachability still need to be established in Habitat. The step-15 event is a placeholder for late intervention and does not yet establish proximity to the original goal. Triggers use zero-based steps and arrive before that step's action.

These cases define task semantics. `scripts/run_demo.py` continues to read only `configs/demo_episode.json`.

## Habitat navigation check

In the separate `habitat` environment, run a fixed navigation episode with the Oracle shortest-path follower:

```bash
python -m scripts.run_navigation --gpu-id 0
```

The configuration is [`configs/simulator/navigation_001.json`](configs/simulator/navigation_001.json).
The script saves each RGB-D frame, poses, actions, the reference path, and success/SPL metrics to
`outputs/navigation_001/`, and uses `ffmpeg` to export `trajectory.mp4` at 10 FPS by default.
Existing runs can also be exported without starting Habitat:

```bash
python -m scripts.export_video --episode-dir outputs/navigation_001
```

See [`docs/navigation_001.md`](docs/navigation_001.md) for setup and acceptance criteria.
Both the simulator scripts and the MIVLN core support the existing Python 3.9 Habitat environment.

## Target replacement in Habitat

Run matched control and intervention branches in the configured `habitat` environment:

```bash
python -m scripts.run_intervention --gpu-id 0 --no-video
```

Both branches initially navigate to point A. Before action 20, the intervention branch
receives a structured goal replacement and returns to the starting point B without
resetting the simulator. Results are saved under `outputs/intervention_001/`.
`pair_summary.json` must report `passed: true`; STOP success uses measured geodesic distance.
This is an Oracle integration check with privileged goal coordinates and task updates,
not a learned language-following agent. See [`docs/intervention_001.md`](docs/intervention_001.md)
for commands, video export, output definitions, and acceptance criteria.

After this pair passes, compare early/middle/late interventions toward a shared
independent target B. The preparation step samples and validates B using only the
navigation mesh, then saves all three configurations:

```bash
python -m scripts.intervention_timing prepare --output-dir outputs/intervention_timing_001
python -m scripts.intervention_timing run --output-dir outputs/intervention_timing_001 --gpu-id 0 --no-video
```

See [`docs/intervention_timing.md`](docs/intervention_timing.md) for trigger definitions,
summary metrics, video export, and repeat runs.

For a small collection of different start/A/B combinations, prepare 10 fixed tasks
and run the reference plus three paired intervention timings for each task:

```bash
python -m scripts.debug_dataset prepare --count 10 --output-dir outputs/debug_dataset_001
python -m scripts.debug_dataset run --output-dir outputs/debug_dataset_001 --gpu-id 0
python -m scripts.debug_dataset summarize --output-dir outputs/debug_dataset_001
```

Failed tasks remain in the reports. See [`docs/debug_dataset.md`](docs/debug_dataset.md)
for geometry filters, logs, video review, and reproduction commands.

An Ignore Intervention negative control keeps the policy's target at A while the
evaluator updates its independent task state to B. It checks that stopping at the
cancelled target is rejected, with no compliant-action latency reported. Use
`scripts.run_ignore_intervention` with a saved event configuration and an existing
matching control directory. See [`docs/ignore_intervention.md`](docs/ignore_intervention.md)
for the single-episode command and expected-failure acceptance checks.

## HSSD scene preparation

After downloading HSSD, check one complete furnished scene through its scene dataset configuration:

```bash
python -m scripts.check_hssd --assets-only --output-dir outputs/hssd_assets_001
python -m scripts.check_hssd --scene-id 102344280 --gpu-id 0 --output-dir outputs/hssd_smoke_001
```

The first command needs no GPU; the second requires the `habitat` environment.
It saves RGB-D, raw semantic IDs, object metadata/world bounds, room annotations,
and a navigation mesh including static furniture. Asset availability and semantic
label consistency are reported separately; metadata and lexicon labels can disagree.
See [`docs/hssd_scene_check.md`](docs/hssd_scene_check.md) for acceptance checks and next steps.

After the target close-ups have been reviewed, run the first actual HSSD navigation
to the bed's configured stopping point:

```bash
python -m scripts.run_navigation --config configs/simulator/hssd_navigation_001.json --gpu-id 0 --output-dir outputs/hssd_navigation_001 --no-video
```

This configuration loads the complete scene instance, verifies input hashes and
the furniture-inclusive navigation mesh dimensions, and records raw semantic frames.
It is still a privileged Oracle point-goal check, not a language-following or
object-visibility benchmark. See [`docs/hssd_navigation_001.md`](docs/hssd_navigation_001.md).

For the next multi-task experiment, use the review-gated semantic workflow:

```bash
python -m scripts.hssd_targets prepare --output-dir outputs/hssd_catalog_001
python -m scripts.hssd_targets preview --catalog outputs/hssd_catalog_001/catalog.json --output-dir outputs/hssd_catalog_preview_001 --gpu-id 0
```

Candidates are never auto-approved. After explicit visual review of at least three
distinct objects, `scripts.hssd_dataset` can freeze 10 semantic tasks and run
reference plus early/middle/late control/follow/ignore conditions (100 episodes).
See [`docs/hssd_semantic_dataset.md`](docs/hssd_semantic_dataset.md) for versioned
review decisions, preparation, batch execution, failure accounting, and summaries.

## Project layout

```text
MIVLN/
├── configs/             # Reproducible episode definitions
├── docs/                # Research and task specifications
├── mivln/               # Simulator-independent task logic
├── scripts/             # Executable experiments and utilities
├── tests/               # Unit tests
├── pyproject.toml
└── README.md
```

## Development order

1. Keep the target-replacement demo and tests passing.
2. Connect the same interfaces to a single Habitat navigation episode.
3. Add avoid-region constraints.
4. Add temporary subgoals and task resumption.
5. Generate a small validated benchmark before training models.

The detailed Chinese execution checklist is available in
[`docs/step_by_step_roadmap.md`](docs/step_by_step_roadmap.md).

The first Habitat integration should replace only the toy trajectory and goal controller. `TaskState`, `EventEngine`, the event schema, and the evaluator ground truth should remain simulator-independent.

