"""Measured completion predicates for the single temporary-subgoal protocol."""

from __future__ import annotations

import math


COMPLETE_SUBGOAL = "COMPLETE_SUBGOAL"


def completion_is_valid(distance_m, radius_m, yaw_degrees, tolerance_degrees, diagnostic) -> bool:
    """Unknown/failed diagnostics cannot grant completion or restore evaluator A.

    The caller supplies independently measured geodesic distance, heading, and
    instance visibility. A policy's intended goal or claimed completion is not an
    input to this predicate. Stable instance identity is checked by the adapter.
    """
    return bool(distance_m is not None and math.isfinite(distance_m) and 0 <= distance_m <= radius_m
                and yaw_degrees is not None and math.isfinite(yaw_degrees)
                and abs(yaw_degrees) <= tolerance_degrees+1e-5
                and diagnostic.get("evaluated") is True and diagnostic.get("instance_visible") is True
                and diagnostic.get("labels_restored") is True and diagnostic.get("pose_unchanged") is True)


def sequence_satisfied(root_goal, subgoal, state, completions) -> bool:
    """Require one measured B completion and evaluator restoration to A.

    Final STOP, distance to A, the action budget and constraints are additional
    conditions enforced by EpisodeEvaluator. This predicate alone never makes
    an episode successful.
    """
    return bool(len(completions) == 1 and completions[0].get("valid") is True
                and completions[0].get("goal_id") == subgoal
                and state.active_goal == root_goal and not state.suspended_goals
                and state.completed_subgoals == [subgoal])
