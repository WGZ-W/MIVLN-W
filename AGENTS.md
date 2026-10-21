# Repository Guidelines

## Project Structure & Module Organization

- `mivln/`: simulator-independent task state, intervention delivery, and evaluation.
- `scripts/`: executable demos, Habitat checks, navigation experiments, and video export.
- `configs/`: JSON task definitions; `configs/simulator/` contains scene coordinates and navigation parameters.
- `tests/`: standard-library unit tests and simulation-loop tests using a fake simulator.
- `docs/`: task semantics, installation instructions, and experiment acceptance criteria.
- `latex/` and `resource/`: paper files and reference PDFs.
- `data/` and `outputs/`: downloaded scenes and generated experiment artifacts; both are Git-ignored.

## Build, Test, and Development Commands

Run commands from the repository root using Python 3.9+. The core has no third-party dependencies; Habitat experiments require the separately configured `habitat` environment.

```bash
python -m pip install -e .                 # Optional editable installation
python -m unittest discover -s tests -v    # Run all available tests
python -m scripts.run_demo                # Run the deterministic text demo
python -m scripts.check_habitat --gpu-id 0 # Check rendering and navigation
python -m scripts.run_intervention --gpu-id 0 --no-video
```

The last command runs matched control/intervention branches. Use the assigned GPU index and a fresh `--output-dir` when repeating experiments. Video export requires FFmpeg. See `docs/intervention_timing.md` for the three-timing experiment.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, and `PascalCase` classes. Follow existing type annotations and `from __future__ import annotations` to preserve Python 3.9 compatibility. Keep core modules independent of Habitat and NumPy. Use descriptive JSON identifiers such as `episode_id` and `event_id`. No formatter or linter is currently configured.

## Testing Guidelines

Use `unittest`, with files named `test_*.py` and methods named `test_<behavior>`. Habitat-dependent loop tests skip when Habitat is unavailable and use a fake simulator when installed. No numerical coverage threshold is configured. Add regression tests for changed task semantics, especially event timing, cancelled goals, STOP requirements, and timeout handling.

## Commit & Pull Request Guidelines

Follow existing prefixed commit subjects, e.g. `feat: add Habitat navigation checks and video export`. PRs should explain the behavior change, include validation commands and results, and link relevant issues when applicable. For simulator changes, report scene/configuration, seed, GPU selection, and acceptance metrics; include representative frames or videos when visual behavior changes. Exclude downloaded scenes, credentials, and experiment outputs.

## Experiment Invariants

Deliver each event once, before its action. Preserve pose and history across goal replacement. Evaluate STOP using measured distance to the active goal. Document privileged Oracle inputs explicitly; do not present Oracle success as language-understanding performance.
