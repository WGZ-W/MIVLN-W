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

