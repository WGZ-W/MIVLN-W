"""Initial simulator-independent MIVLN metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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

