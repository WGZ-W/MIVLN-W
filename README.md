# MIVLN

MIVLN is a minimal research scaffold for **mixed-initiative vision-and-language navigation**. It focuses first on the task logic needed when a robot is navigating autonomously and a human changes the task during execution.

The current milestone is deliberately simulator-independent:

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
- `tests/`: dependency-free unit tests for the core logic.

Habitat is intentionally not required yet. The core task semantics can therefore be tested on Windows before connecting them to a Linux simulator.

## Quick start

Python 3.10 or newer is recommended.

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
python -m scripts.run_navigation --gpu-id 7
```

The configuration is [`configs/simulator/navigation_001.json`](configs/simulator/navigation_001.json).
The script saves each RGB-D frame, poses, actions, the reference path, and success/SPL metrics to
`outputs/navigation_001/`, and uses `ffmpeg` to export `trajectory.mp4` at 10 FPS by default.
Existing runs can also be exported without starting Habitat:

```bash
python -m scripts.export_video --episode-dir outputs/navigation_001
```

See [`docs/navigation_001.md`](docs/navigation_001.md) for setup and acceptance criteria.
This simulator check supports the existing Python 3.9 Habitat environment independently of the MIVLN package.

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

