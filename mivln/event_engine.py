"""Deterministic intervention event delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .task_state import TaskState


@dataclass(frozen=True)
class InterventionEvent:
    event_id: str
    trigger_step: int
    utterance: str
    task_update: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterventionEvent":
        trigger = value.get("trigger", {})
        if trigger.get("type") != "step":
            raise ValueError("Version 0.1 supports only step-based triggers")

        trigger_step = trigger.get("value")
        if not isinstance(trigger_step, int) or trigger_step < 0:
            raise ValueError("Trigger step must be a non-negative integer")

        event_id = value.get("event_id")
        utterance = value.get("utterance")
        task_update = value.get("task_update")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Event requires a non-empty 'event_id'")
        if not isinstance(utterance, str) or not utterance:
            raise ValueError("Event requires a non-empty 'utterance'")
        if not isinstance(task_update, Mapping):
            raise ValueError("Event requires a structured 'task_update'")

        return cls(event_id, trigger_step, utterance, task_update)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trigger": {"type": "step", "value": self.trigger_step},
            "utterance": self.utterance,
            "task_update": dict(self.task_update),
        }


class EventEngine:
    """Deliver each scheduled event once and update evaluator ground truth."""

    def __init__(self, events: Iterable[InterventionEvent]) -> None:
        self._events = sorted(events, key=lambda event: event.trigger_step)
        event_ids = [event.event_id for event in self._events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Event IDs must be unique")
        self._delivered_ids: set[str] = set()

    def step(self, step_index: int, task_state: TaskState) -> list[InterventionEvent]:
        """Deliver all due events before the action at ``step_index``."""

        delivered: list[InterventionEvent] = []
        for event in self._events:
            if event.event_id in self._delivered_ids:
                continue
            if event.trigger_step <= step_index:
                task_state.apply_update(event.task_update)
                self._delivered_ids.add(event.event_id)
                delivered.append(event)
        return delivered

    @property
    def delivered_ids(self) -> frozenset[str]:
        return frozenset(self._delivered_ids)

