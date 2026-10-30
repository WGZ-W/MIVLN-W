"""Structured task state independent of any navigation simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .regions import validate_region


@dataclass
class TaskState:
    """Authoritative state produced by applying structured task updates."""

    active_goal: str
    suspended_goals: list[str] = field(default_factory=list)
    active_constraints: list[dict[str, Any]] = field(default_factory=list)
    completed_subgoals: list[str] = field(default_factory=list)
    cancelled_goals: list[str] = field(default_factory=list)
    status: str = "active"

    def apply_update(self, update: Mapping[str, Any]) -> None:
        """Apply a validated structured update to the task state.

        Goal replacement retains its v0.1 semantics. Experimental extensions add
        a floor constraint or temporarily suspend the goal for a single subgoal.
        """

        operation = update.get("operation")
        if operation == "insert_subgoal":
            subgoal = update.get("subgoal")
            if not isinstance(subgoal, str) or not subgoal.strip() or subgoal == self.active_goal:
                raise ValueError("insert_subgoal requires a different, non-empty subgoal")
            if update.get("resume_goal", self.active_goal) != self.active_goal:
                raise ValueError("resume_goal must match the current active goal")
            if self.suspended_goals or subgoal in self.completed_subgoals:
                raise ValueError("This protocol supports one outstanding temporary subgoal")
            self.suspended_goals.append(self.active_goal)
            self.active_goal = subgoal
            self.status = "active"
            return
        if operation == "add_avoid_region":
            region = validate_region(update.get("region"))
            if any(item["region_id"] == region["region_id"] for item in self.active_constraints):
                raise ValueError(f"Duplicate avoid region: {region['region_id']}")
            self.active_constraints.append(region)
            return
        if operation != "replace_goal":
            raise ValueError(f"Unsupported task operation: {operation!r}")

        old_goal = update.get("old_goal")
        new_goal = update.get("new_goal")
        if not isinstance(new_goal, str) or not new_goal.strip():
            raise ValueError("replace_goal requires a non-empty 'new_goal'")
        if old_goal is not None and old_goal != self.active_goal:
            raise ValueError(
                f"Cannot replace {old_goal!r}; active goal is {self.active_goal!r}"
            )

        if self.active_goal not in self.cancelled_goals:
            self.cancelled_goals.append(self.active_goal)
        self.active_goal = new_goal
        self.status = "active"

    def complete_subgoal(self, goal: str) -> None:
        """Complete the temporary goal and restore the original without cancelling it.

        This is an internal completion transition, not a human task-update opcode.
        Policy callers may record their intent; evaluator callers MUST independently
        validate measured completion evidence before invoking this method.
        """
        if not self.suspended_goals or goal != self.active_goal:
            raise ValueError("Only the active, suspended-task subgoal can be completed")
        if goal in self.completed_subgoals:
            raise ValueError("Subgoal was already completed")
        self.completed_subgoals.append(goal)
        self.active_goal = self.suspended_goals.pop()
        self.status = "active"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot."""

        return asdict(self)

