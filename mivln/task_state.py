"""Structured task state independent of any navigation simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


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

        Only target replacement is part of the v0.1 public task definition.
        The explicit dispatcher makes future operations easy to add without
        changing the event engine or simulator adapter.
        """

        operation = update.get("operation")
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot."""

        return asdict(self)

