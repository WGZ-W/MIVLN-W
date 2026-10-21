"""Initial simulator-independent MIVLN metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .task_state import TaskState


@dataclass(frozen=True)
class EpisodeMetrics:
    dynamic_success: bool
    intervention_received: bool
    goal_update_compliance: bool
    response_latency_steps: int | None
    path_length: float
    steps: int
    final_goal: str
    stopped_goal: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EpisodeEvaluator:
    def evaluate_navigation(
        self,
        *,
        task_state: TaskState,
        stopped: bool,
        goal_distances_m: Mapping[str, float | None],
        success_radius_m: float,
        intervention_step: int | None,
        first_compliant_action_step: int | None,
        path_length: float,
        steps: int,
        max_steps: int,
    ) -> EpisodeMetrics:
        """Ground STOP in measured geodesic distances, not the policy's goal label.

        Unreachable goals have distance None. If success regions overlap, the
        current active goal takes precedence over other nearby goals.
        """

        if not math.isfinite(success_radius_m) or success_radius_m <= 0:
            raise ValueError("success_radius_m must be finite and positive")
        nearby = [
            goal for goal, distance in goal_distances_m.items()
            if distance is not None and math.isfinite(distance)
            and 0 <= distance <= success_radius_m
        ]
        stopped_goal = None
        if stopped and nearby:
            stopped_goal = (
                task_state.active_goal if task_state.active_goal in nearby
                else min(nearby, key=lambda goal: goal_distances_m[goal])
            )
        return self.evaluate(
            task_state=task_state,
            stopped_goal=stopped_goal,
            intervention_step=intervention_step,
            first_compliant_action_step=first_compliant_action_step,
            path_length=path_length,
            steps=steps,
            max_steps=max_steps,
        )

    def evaluate(
        self,
        *,
        task_state: TaskState,
        stopped_goal: str | None,
        intervention_step: int | None,
        first_compliant_action_step: int | None,
        path_length: float,
        steps: int,
        max_steps: int,
    ) -> EpisodeMetrics:
        """Evaluate an episode against the structured final task state."""

        intervention_received = intervention_step is not None
        compliance = stopped_goal == task_state.active_goal
        within_budget = steps <= max_steps
        dynamic_success = bool(stopped_goal) and compliance and within_budget

        latency: int | None = None
        if intervention_received and first_compliant_action_step is not None:
            latency = max(0, first_compliant_action_step - intervention_step)

        return EpisodeMetrics(
            dynamic_success=dynamic_success,
            intervention_received=intervention_received,
            goal_update_compliance=compliance,
            response_latency_steps=latency,
            path_length=round(float(path_length), 4),
            steps=steps,
            final_goal=task_state.active_goal,
            stopped_goal=stopped_goal,
        )

