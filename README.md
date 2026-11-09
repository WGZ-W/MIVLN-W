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

To compare terminal facing against an existing frozen collection without resampling:

```bash
python -m scripts.hssd_dataset derive-facing --source-dir outputs/hssd_dataset_001 --output-dir outputs/hssd_dataset_facing_001
```

This opt-in variant turns toward the policy's object before STOP and evaluates
instance visibility separately from point-goal success. Use its `smoke_config.json`
first; see [`docs/hssd_terminal_facing.md`](docs/hssd_terminal_facing.md). The original
collection and its success definition remain unchanged.

## Experimental avoid-region intervention

The next single-case entry preserves the original goal while adding one forbidden
floor region. Preparation uses CPU mesh queries; execution runs matched control,
follow, and Ignore branches on the assigned GPU:

```bash
python -m scripts.avoid_region prepare --reference-dir outputs/hssd_dataset_facing_001/tasks/task_001/reference --output-dir outputs/hssd_avoid_prepare_002
python -m scripts.avoid_region run --config outputs/hssd_avoid_prepare_002/config.json --gpu-id 0 --output-dir outputs/hssd_avoid_001
python -m scripts.avoid_region summarize --output-dir outputs/hssd_avoid_001
```

Preparation `_002` already exists locally; skip its command for this run. Ignore
must reach the same goal but fail because its executed path violates the newly
effective constraint. The single-case GPU experiment has passed. This remains a
privileged geometry experiment. See [`docs/avoid_region.md`](docs/avoid_region.md)
for boundary semantics, planner limitations, metrics and video commands.

For the fixed 10-task collection, `scripts.avoid_dataset` preserves preparation
failures and all 30 planned episode slots, and independently audits raw evidence:

```bash
python -m scripts.avoid_dataset prepare --source-dir outputs/hssd_dataset_facing_001 --output-dir outputs/hssd_avoid_batch_001
python -m scripts.avoid_dataset run --output-dir outputs/hssd_avoid_batch_001 --gpu-id 0
python -m scripts.avoid_dataset summarize --output-dir outputs/hssd_avoid_batch_001
python -m scripts.avoid_dataset videos --output-dir outputs/hssd_avoid_batch_001
```

The local batch is complete: 10/10 tasks and all 30 episodes passed their acceptance
checks, with 20 normal-condition successes and 10 expected Ignore failures. All 30
videos have been exported. The commands above document the workflow; use a fresh
directory when repeating an experiment. See [`docs/avoid_dataset.md`](docs/avoid_dataset.md)
for failure accounting and [`the acceptance record`](docs/hssd_avoid_results_2026-10-29.md)
for measured results and next steps.

## Experimental temporary subgoal and resumption

`scripts.subgoal` keeps original goal A suspended while visiting B, then restores
A. It introduces a counted logical `COMPLETE_SUBGOAL` action with independent
distance, heading and instance evidence; final `STOP` still ends the episode.
The single-case and ten-task early/middle/late GPU experiments have passed
independent acceptance locally. Existing results can be re-audited on CPU:

```bash
python -m scripts.subgoal_dataset summarize --output-dir outputs/hssd_subgoal_batch_001
python -m scripts.subgoal_dataset summarize --output-dir outputs/hssd_subgoal_timing_001
```

Control reaches A, Follow completes B then stops at A, and Ignore reaches A but
fails the missing-B requirement. All ten early cases (30 episodes) and thirty
early/middle/late cases (90 episodes) passed re-audit, and all 120 videos were exported.
A historical displacement-audit rejection was resolved by an exact native
navmesh action replay; original exit codes and failed summaries are preserved.
The batch keeps preparation failures,
missing episodes and failed child processes in its denominators and independently
replays raw evidence. See [`docs/subgoal.md`](docs/subgoal.md) for the single-case
protocol and [`docs/subgoal_dataset.md`](docs/subgoal_dataset.md) for the batch,
negative tests, commands and complete stage acceptance criteria. Measured results,
the audit correction and remaining manual video review are documented in
[`the acceptance report`](docs/hssd_subgoal_results_2026-11-03.md).

## Independent language task-update development

The dependency-free `scripts.task_update` entry evaluates goal replacement,
avoid-region addition and subgoal insertion without Habitat or a GPU:

```bash
python -m scripts.task_update run --mode all --output-dir outputs/task_update_rules_002
python -m scripts.task_update export --mode all --output-dir outputs/task_update_requests_002
```

The first command runs a limited bilingual **rule baseline** in latest-message,
history and prior-state modes. The second exports public requests without reference
answers or region geometry. A `score` command imports external model JSONL responses
and preserves missing/invalid cases in the denominator. Eighteen synthetic
development cases exercise the contracts; their scores are not held-out model
performance. Prior policy context is built from public language using a fixed
rule builder, while reference states remain evaluator-only.
`python -m scripts.predict_task_updates` can generate raw predictions from a
configured Chat Completions compatible service; it requires a base URL and model
identifier. A first Qwen3.5-9B development run completed all 54 requests: exact
update/state scores were 15/18, 15/18 and 16/18 across the three modes.
These are synthetic development results, with ambiguity errors and unsupported
context guesses; see [the audit report](docs/qwen35_task_update_results_2026-11-05.md).
See [`docs/task_update.md`](docs/task_update.md) for schemas,
scope, metrics, prediction-generation and model-import commands.
For local Qwen3.5-9B weights, see [`the deployment guide`](docs/qwen35_local.md);
the client supports `--thinking-mode off` for compatible chat templates.

Controlled context/order pairs are available through `python -m scripts.task_update_pairs`:
144 synthetic cases expand to 432 requests, split into dev/test families with
216 requests each. The evaluator separates task-label matches from supported
decisions, including missing-context abstentions and lucky guesses. See
[`the paired evaluation guide`](docs/task_update_pairs.md) for frozen inputs,
category metrics and commands. The 216-request Qwen dev run has been audited:
supported-decision scores are 34/72 (latest), 21/72 (history), and 52/72 (state).
State handles all 48 clear dev updates but still fails ambiguity rejection.
The 216-request test run is also audited: scores are 36/72, 25/72 and 46/72;
state handles 46/48 clear updates but correctly rejects none of the 24 ambiguous
or unknown region requests. See [dev results](docs/qwen35_pairs_dev_results_2026-11-05.md)
and [test results and next steps](docs/qwen35_pairs_test_results_2026-11-05.md).
Controlled prompt ablations are available through `python -m scripts.task_update_ablation`:
prepare inputs without inference, then audit and rescore baseline, history-timeline,
and refusal-check runs with identical model controls and candidates. See
[the ablation commands](docs/task_update_ablation.md). The history condition has
been audited: history improves from 21/72 to 39/72, with regression on explicit
replacement and unresolved ambiguity rejection; see [the results](docs/qwen35_history_ablation_dev_results_2026-11-06.md).
All three dev conditions are now audited. Refusal instructions yield 35/72,
27/72 and 53/72 across latest/history/state, improving unknown-object refusal
while leaving ambiguity and missing-context refusal unresolved. See
[the three-condition analysis](docs/qwen35_prompt_ablation_dev_results_2026-11-06.md).

Sequential clear-update experiments are available through `python -m scripts.task_update_rollout`.
Each branch carries its own applied model updates; history receives its own execution
feedback, and state receives its resulting symbolic state. Twenty-four synthetic
dialogues provide six updates and one completion notice each, totaling 288 model
requests across history/state. CPU rules, failure injection and causal replay are
validated. The real 288-request Qwen run is now audited: exact updates and states
are 144/144 for state and 124/144 for history; entire dialogues are 24/24 and 19/24.
Five history operation errors leave persistent state errors. See
[the rollout results](docs/qwen35_rollout_dev_results_2026-11-06.md) and
[the rollout guide](docs/task_update_rollout.md).

Language updates can now drive the HSSD navigation loop through
`python -m scripts.language_navigation`. A reviewed task is run as matched
Oracle/model/Ignore branches; the model receives only public candidates and
policy state, and failed predictions never fall back to labels. CPU preparation,
causal auditing and 19 regression tests are complete. The real three-episode
smoke run passed: Oracle/model reach the sofa and Ignore correctly fails at the
old goal. Correct model updates yield identical Oracle/model trajectories and
video hashes with the shared navigator. See [the smoke results](docs/hssd_language_results_2026-11-09.md)
and [the navigation integration guide](docs/language_navigation.md).

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

