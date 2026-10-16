"""Run the simulator-independent target-replacement milestone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mivln import EpisodeEvaluator, EventEngine, InterventionEvent, TaskState


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_demo() -> dict[str, Any]:
    episode = load_json(PROJECT_ROOT / "configs" / "demo_episode.json")
    events = [InterventionEvent.from_dict(item) for item in episode["events"]]
    task_state = TaskState(active_goal=episode["initial_goal"])
    event_engine = EventEngine(events)

    trajectory: list[dict[str, Any]] = []
    delivered_events: list[dict[str, Any]] = []
    intervention_step: int | None = None
    first_compliant_action_step: int | None = None
    stopped_goal: str | None = None
    path_length = 0.0

    # This deterministic controller is a placeholder for Habitat. It records
    # the goal the robot is pursuing and stops at the updated goal on step 11.
    for step_index in range(episode["max_steps"]):
        delivered = event_engine.step(step_index, task_state)
        if delivered:
            intervention_step = step_index
            delivered_events.extend(event.to_dict() for event in delivered)

        intended_goal = task_state.active_goal
        if intervention_step is not None and first_compliant_action_step is None:
            first_compliant_action_step = step_index

        action = "STOP" if intended_goal == "bedroom" and step_index == 11 else "MOVE"
        if action == "MOVE":
            path_length += 0.25
        else:
            stopped_goal = intended_goal

        trajectory.append(
            {
                "step": step_index,
                "action": action,
                "intended_goal": intended_goal,
                "task_state": task_state.to_dict(),
            }
        )
        if action == "STOP":
            break

    metrics = EpisodeEvaluator().evaluate(
        task_state=task_state,
        stopped_goal=stopped_goal,
        intervention_step=intervention_step,
        first_compliant_action_step=first_compliant_action_step,
        path_length=path_length,
        steps=len(trajectory),
        max_steps=episode["max_steps"],
    )

    output_dir = PROJECT_ROOT / "outputs" / episode["episode_id"]
    write_json(output_dir / "episode.json", episode)
    write_json(output_dir / "events.json", delivered_events)
    write_json(output_dir / "trajectory.json", trajectory)
    write_json(output_dir / "metrics.json", metrics.to_dict())

    print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))
    print(f"Artifacts written to: {output_dir}")
    return metrics.to_dict()


if __name__ == "__main__":
    run_demo()

