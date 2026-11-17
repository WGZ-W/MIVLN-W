"""Explicit human playback declarations, separate from navigation success.

Approving an Ignore video confirms its expected failure is faithfully shown;
it never changes the evaluator result. Encoding checks never approve playback.
"""

from __future__ import annotations

from .language_annotations import person


COMMON = ("watched_full_video", "rgb_quality", "intervention_timing", "matches_recorded_actions")
SPECIFIC = {"replacement": ("target_identity", "replacement_behavior"),
            "subgoal": ("target_identity", "subgoal_and_resumption"), "cancel_task": ("cancel_behavior",)}
CHECKS = (*COMMON, "target_identity", "replacement_behavior", "subgoal_and_resumption", "cancel_behavior")
LOCKED = ("video_id", "task_id", "task_type", "condition", "expected_task_success", "video_sha256")
COLUMNS = (*LOCKED, "decision", "reviewer", "note", *CHECKS)


def worksheet(videos):
    """Generate pending rows with explicit N/A for inapplicable semantic checks."""
    result = []
    for video in videos:
        required = {*COMMON, *SPECIFIC[video["task_type"]]}
        result.append({**{key: str(video[key]).lower() if type(video[key]) is bool else video[key] for key in LOCKED},
            "decision": "pending", "reviewer": "", "note": "",
            **{key: "false" if key in required else "N/A" for key in CHECKS}})
    return result


def import_sheet(videos, rows):
    """Require the whole video roster, real names and explicit decisions.

    This records claimed review identities, without authenticating them or
    proving that the reviewer really watched a player. Labels/MP4 hashes cannot
    be changed by a review sheet, and no missing row is silently discarded.
    Notes are optional strings; an empty CSV cell is valid, a missing cell is not.
    """
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ValueError("Playback review rows must be dictionaries")
    if any(any(type(value) is not str for value in row.values()) for row in rows):
        raise ValueError("Fill every review CSV cell; missing cells are not decisions")
    expected = {row["video_id"]: row for row in worksheet(videos)}
    ids = [row.get("video_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Keep every exported video exactly once in the review sheet")
    decisions = []
    for row in rows:
        identity = row["video_id"]
        original = expected[identity]
        if set(row) != set(COLUMNS) or any(row[key] != original[key] for key in LOCKED):
            raise ValueError(f"{identity}: video identity/label/hash fields are immutable")
        if row["decision"] == "pending":
            if row != original:
                raise ValueError(f"{identity}: set a decision for a filled review row")
            decisions.append(dict(row))
            continue
        if row["decision"] not in ("approve", "reject", "needs_revision"):
            raise ValueError(f"{identity}: unknown review decision")
        person(row["reviewer"])
        required = {*COMMON, *SPECIFIC[row["task_type"]]}
        for key in CHECKS:
            if key not in required and row[key] != "N/A":
                raise ValueError(f"{identity}: keep inapplicable checks as N/A")
            if key in required and row[key] not in ("true", "false"):
                raise ValueError(f"{identity}: checklist values must be true/false")
        if row["decision"] == "approve" and any(row[key] != "true" for key in required):
            raise ValueError(f"{identity}: approval requires full playback and all applicable checks")
        decisions.append(dict(row))
    if all(row["decision"] == "pending" for row in decisions):
        raise ValueError("Blank pending templates are not human playback reviews")
    return decisions
