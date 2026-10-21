# MIVLN Task Definition — Version 0.1

## Research setting

The robot navigates autonomously toward a language-specified goal. During execution, a human supervisor may issue an asynchronous natural-language intervention that changes the active task. Version 0.1 isolates task updating from perception and low-level navigation.

## Episode inputs

- An initial natural-language instruction.
- A structured initial goal used as evaluator ground truth.
- A sequence of observations and robot poses supplied later by a simulator.
- Zero or one intervention event.

The agent sees the natural-language instruction. The evaluator consumes the paired structured task update and must not use the agent's own interpretation as ground truth.

## Action space

The first simulator-backed version will use:

- `MOVE_FORWARD` by 0.25 m;
- `TURN_LEFT` by 15 degrees;
- `TURN_RIGHT` by 15 degrees;
- `STOP`.

The text demo abstracts away these low-level actions. The Habitat runner executes
movement and turn actions, preserves the robot state across interventions, and
records the active goal and measured pose at every step.

## Supported intervention

Version 0.1 supports one operation:

```text
replace_goal(old_goal, new_goal)
```

Example:

```text
Initial: Go to the kitchen.
Intervention: Instead, go to the bedroom.
```

After delivery, `bedroom` becomes the active goal and `kitchen` is recorded as cancelled.

## Trigger semantics

The initial event engine supports deterministic step triggers. An event with trigger step `k` is delivered before the agent chooses its action for step `k`. Each event is delivered exactly once.

## Success and failure

An episode succeeds when all of the following hold:

1. The robot issues `STOP` at the final active goal.
2. The stopped goal matches the evaluator's structured ground truth.
3. The episode remains within the maximum step limit.

Stopping at a cancelled goal, timing out, or failing to stop is a failure.

In the Habitat integration, STOP is grounded in the measured geodesic distance
from the final pose to each configured goal. A policy's intended-goal label alone
cannot establish success. The default success radius is 0.3 meters.

The first paired simulator experiment uses an Oracle controller that deliberately
reads privileged goal positions and structured task updates. It validates task
semantics and navigation plumbing; it does not implement the language-only agent
interface described above. See `docs/intervention_001.md` for this experiment.

## Initial metrics

- `dynamic_success`: whether the final effective goal was completed.
- `intervention_received`: whether a scheduled intervention was delivered.
- `goal_update_compliance`: whether the robot stopped at the post-intervention goal.
- `response_latency_steps`: steps between intervention delivery and the first action directed at the new goal.
- `path_length`: accumulated trajectory length supplied by the controller or simulator.

## Out of scope for Version 0.1

- Multiple interventions in one episode;
- avoid-region constraints;
- temporary subgoals and resumption;
- speech input;
- moving people or objects;
- learned language parsing;
- end-to-end VLN training;
- live human-in-the-loop interfaces.

