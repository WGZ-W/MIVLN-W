"""Controlled context/order pairs and information-aware decision diagnostics.

Task references stay identical across context modes. A separate decision metric
expects missing-context abstention when the public view cannot identify the old
or resume goal. Thus a lucky task-label match is visible without being rewarded
as a supported decision. This module never invokes a model or a simulator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import random
import re
from string import Formatter

from .task_update import exact_keys, candidates, public_state
from .task_update_eval import digest, prepare_records, validate_dataset


PROTOCOL = "task_update_context_pairs_v1"
SPLITS = ("dev", "test", "all")
OPERATIONS = ("replace_goal", "insert_subgoal", "add_avoid_region")
REFUSALS = ("ambiguous_goal", "ambiguous_region", "unknown_goal", "unknown_region")


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Pair family/context identifiers must be simple strings")


def names_for(values, catalog, id_key):
    """Names used to author language must exist in the public alias catalog."""
    ids = candidates(catalog, id_key)
    if not isinstance(values, dict) or set(values) != ids:
        raise ValueError("Language names must cover the public catalog")
    for item in catalog:
        names = values[item[id_key]]
        exact_keys(names, ("zh", "en"), "Bilingual candidate names")
        aliases = {alias.casefold() for alias in item["aliases"]}
        for name in names.values():
            if not isinstance(name, str) or name.casefold() not in aliases:
                raise ValueError("Authored language name is not a public alias")
    return ids


def validate_config(config):
    """Validate the authored design before deriving any evaluation records."""
    exact_keys(config, ("schema_version", "kind", "purpose", "seed", "candidate_goals",
        "candidate_regions", "region_registry", "goal_names", "region_names", "contexts", "families"), "Pair configuration")
    if (type(config["schema_version"]) is not int or config["schema_version"] != 1
            or config["kind"] != "task_update_context_pairs"
            or type(config["seed"]) is not int or not isinstance(config["purpose"], str) or not config["purpose"].strip()):
        raise ValueError("Unsupported pair design or missing purpose/seed")
    goals = names_for(config["goal_names"], config["candidate_goals"], "goal_id")
    regions = names_for(config["region_names"], config["candidate_regions"], "region_id")
    if len(regions) != 2:
        raise ValueError("This protocol requires exactly two region candidates")
    if not isinstance(config["contexts"], list) or len(config["contexts"]) < 2:
        raise ValueError("At least two different prior goal contexts are required")
    context_ids, active_goals = set(), set()
    for context in config["contexts"]:
        exact_keys(context, ("context_id", "initial_goal", "active_goal", "prior_region"), "Pair context")
        identity(context["context_id"])
        if (context["context_id"] in context_ids or context["initial_goal"] not in goals
                or context["active_goal"] not in goals or context["active_goal"] in active_goals
                or context["initial_goal"] == context["active_goal"] or context["prior_region"] not in regions):
            raise ValueError("Context must have a unique ID/active goal and a genuine previous replacement")
        context_ids.add(context["context_id"])
        active_goals.add(context["active_goal"])
    if not isinstance(config["families"], list) or not config["families"]:
        raise ValueError("Pair design needs authored families")
    family_ids = set()
    for family in config["families"]:
        exact_keys(family, ("family_id", "split", "kind", "target_id", "requires_context", "messages"), "Pair family")
        identity(family["family_id"])
        kind, target = family["kind"], family["target_id"]
        if (family["family_id"] in family_ids or family["split"] not in ("dev", "test")
                or kind not in OPERATIONS + REFUSALS or type(family["requires_context"]) is not bool):
            raise ValueError("Duplicate family or invalid split/kind/context flag")
        family_ids.add(family["family_id"])
        if kind in ("replace_goal", "insert_subgoal"):
            if target not in goals or target in active_goals:
                raise ValueError("New/temporary target must differ from every active context goal")
        elif kind == "add_avoid_region":
            if target not in regions or family["requires_context"]:
                raise ValueError("Avoid controls must explicitly identify a known region")
        elif target is not None or family["requires_context"]:
            raise ValueError("Refusal families cannot have targets or missing-context labels")
        exact_keys(family["messages"], ("zh", "en"), "Bilingual messages")
        for message in family["messages"].values():
            if not isinstance(message, str) or not message.strip():
                raise ValueError("Pair messages must be nonempty")
            parts = list(Formatter().parse(message))
            if any(spec or conversion for _, _, spec, conversion in parts):
                raise ValueError("Message placeholders cannot use formatting or conversions")
            fields = [field for _, field, _, _ in parts if field is not None]
            if any(field != "active_goal" for field in fields):
                raise ValueError("Messages accept only the active_goal placeholder")
            explicit_goal = kind in ("replace_goal", "insert_subgoal") and not family["requires_context"]
            if bool(fields) != explicit_goal:
                raise ValueError("Explicit goal updates must name the active goal; implicit updates must not")
    return deepcopy(config)


def reference_prediction(family, active_goal):
    """Author labels from the controlled design, independently of rule parsing."""
    kind, target = family["kind"], family["target_id"]
    if kind in REFUSALS:
        return {"status": "abstain", "reason": kind}
    update = {"operation": kind}
    if kind == "replace_goal":
        update.update(old_goal=active_goal, new_goal=target)
    elif kind == "insert_subgoal":
        update.update(subgoal=target, resume_goal=active_goal)
    else:
        update["region_id"] = target
    return {"status": "ok", "task_update": update}


def build_plan(config, modes, split="all"):
    """Expand matched contexts with two catalog orders and opaque public IDs.

    The same catalog order is used throughout each context contrast. Reverse
    order controls change only the catalogs. Family/language/split annotations
    stay evaluator-owned, never entering the public request. The common context
    builder still derives old policy state from public language, without labels.
    """
    config = validate_config(config)
    if split not in SPLITS:
        raise ValueError("Unknown pair split")
    cases, metadata = [], {}
    goal_catalog = config["candidate_goals"]
    for family_index, family in enumerate(config["families"]):
        rotation = family_index % len(goal_catalog)
        rotated = goal_catalog[rotation:] + goal_catalog[:rotation]
        for language, message in family["messages"].items():
            for context in config["contexts"]:
                initial, active = context["initial_goal"], context["active_goal"]
                # Avoid controls always add the OTHER region, so the old
                # constraint can be retained without a duplicate-region error.
                region = context["prior_region"]
                if family["kind"] == "add_avoid_region":
                    region = next(item["region_id"] for item in config["candidate_regions"]
                                  if item["region_id"] != family["target_id"])
                initial_name = config["goal_names"][initial][language]
                active_name = config["goal_names"][active][language]
                region_name = config["region_names"][region][language]
                instruction = f"前往{initial_name}。" if language == "zh" else f"Go to the {initial_name}."
                history = ([f"改去{active_name}。", f"避开{region_name}。"] if language == "zh" else
                           [f"Go to the {active_name} instead.", f"Avoid the {region_name}."])
                for order in (0, 1):
                    # No group name, target, split or expected refusal is encoded
                    # in the model-visible ID; it cannot serve as an answer hint.
                    case_id = "r" + digest([config["seed"], family["family_id"], language,
                                           context["context_id"], order])[:24]
                    cases.append({"case_id": case_id, "group_id": family["family_id"], "language": language,
                        "initial_instruction": instruction, "history": history,
                        "message": message.format(active_goal=active_name),
                        "reference": {"initial_goal": initial, "context_updates": [
                            {"operation": "replace_goal", "old_goal": initial, "new_goal": active},
                            {"operation": "add_avoid_region", "region_id": region}],
                            "prediction": reference_prediction(family, active)}})
                    metadata[case_id] = {"family_id": family["family_id"], "split": family["split"],
                        "context_id": context["context_id"], "order": order,
                        "requires_context": family["requires_context"],
                        "goal_order": deepcopy(rotated if order == 0 else list(reversed(rotated))),
                        "region_order": deepcopy(config["candidate_regions"] if order == 0 else list(reversed(config["candidate_regions"])))}
    random.Random(config["seed"]).shuffle(cases)
    selected = [case for case in cases if split == "all" or metadata[case["case_id"]]["split"] == split]
    dataset = validate_dataset({"schema_version": 1, "kind": "task_update_development_set",
        "purpose": config["purpose"], "candidate_goals": config["candidate_goals"],
        "candidate_regions": config["candidate_regions"], "region_registry": config["region_registry"], "cases": selected})
    records = prepare_records(dataset, modes)
    for record in records:
        meta = deepcopy(metadata[record["case"]["case_id"]])
        request = record["request"]
        if request is not None:
            request["candidate_goals"] = meta.pop("goal_order")
            request["candidate_regions"] = meta.pop("region_order")
            record["request_sha256"] = digest(request)
        else:
            meta.pop("goal_order")
            meta.pop("region_order")
        record["pair_metadata"] = meta
    return dataset, records


def annotate_decision(record, row):
    """Keep task-label correctness separate from a supported public decision."""
    meta = record["pair_metadata"]
    reference = record["case"]["reference"]["prediction"]
    missing = meta["requires_context"] and record["mode"] == "latest"
    expected = {"status": "abstain", "reason": "missing_context"} if missing else reference
    category = expected.get("reason", "actionable")
    expected_state = public_state(record["expected_before"] if expected["status"] == "abstain" else record["expected_after"])
    state_correct = row["application_valid"] and row["after_state"] == expected_state
    prediction = row["prediction"]
    attempted = bool(row["schema_valid"] and prediction["status"] == "ok")
    return {**row, **meta, "decision_category": category, "expected_decision": deepcopy(expected),
        "decision_state_exact": bool(state_correct),
        "decision_correct": bool(row["context_correct"] and state_correct and prediction == expected),
        "unsupported_guess": missing and attempted,
        "unsupported_guess_hit": bool(missing and attempted and row["correct"]),
        "ambiguity_execution": category in ("ambiguous_goal", "ambiguous_region") and attempted}


def decision_statistics(rows):
    count = len(rows)
    correct = sum(row["decision_correct"] for row in rows)
    result = {"planned": count, "decision_correct_count": correct,
              "decision_correct_rate": correct / count if count else None}
    partitions = {
        "actionable": [row for row in rows if row["decision_category"] == "actionable"],
        "missing_context": [row for row in rows if row["decision_category"] == "missing_context"],
        "ambiguity": [row for row in rows if row["decision_category"] in ("ambiguous_goal", "ambiguous_region")],
        "unknown": [row for row in rows if row["decision_category"] in ("unknown_goal", "unknown_region")],
    }
    for category, selected in partitions.items():
        total = sum(row["decision_correct"] for row in selected)
        result[category + "_planned"] = len(selected)
        result[category + "_correct_count"] = total
        result[category + "_correct_rate"] = total / len(selected) if selected else None
    # Use eligible categories as denominators: e.g. ambiguity execution is a
    # fraction of ambiguous requests, not diluted by all the clear instructions.
    for key, category in (("unsupported_guess", "missing_context"), ("ambiguity_execution", "ambiguity")):
        total = sum(row[key] is True for row in partitions[category])
        denominator = len(partitions[category])
        result[key + "_count"] = total
        result[key + "_rate"] = total / denominator if denominator else None
    result["unsupported_guess_hit_count"] = sum(row["unsupported_guess_hit"] for row in rows)
    return result


def summarize_pairs(records, rows):
    """Matched-set denominators include missing/invalid predictions unchanged."""
    by_id = {row["request_id"]: row for row in rows}
    if len(by_id) != len(records) or set(by_id) != {record["request_id"] for record in records}:
        raise ValueError("Pair evaluation requires the entire planned roster")
    context_groups, order_groups = defaultdict(list), defaultdict(list)
    first_positions = Counter()
    for record in records:
        row = by_id[record["request_id"]]
        meta = record["pair_metadata"]
        order_groups[(row["mode"], meta["family_id"], row["language"], meta["context_id"])].append(row)
        if meta["requires_context"]:
            context_groups[(row["mode"], meta["family_id"], row["language"], meta["order"])].append(row)
        if record["request"] is not None:
            first_positions[record["request"]["candidate_goals"][0]["goal_id"]] += 1
    context_results = []
    for key, group in context_groups.items():
        # Report latest as safe-abstention groups, not context-following ability:
        # no hidden reference goal is available in that public input view.
        slot = "old_goal" if group[0]["expected_kind"] == "replace_goal" else "resume_goal"
        if (len(group) < 2 or len({row["context_id"] for row in group}) != len(group)
                or len({row["reference_prediction"]["task_update"][slot] for row in group}) != len(group)):
            raise ValueError("Context contrast must contain distinct prior reference goals")
        predicted = [row["prediction"].get("task_update", {}).get(slot) if row["prediction"] else None for row in group]
        context_results.append({"mode": key[0], "family_id": key[1], "language": key[2], "order": key[3],
            "members": [row["request_id"] for row in group],
            "all_decisions_correct": all(row["decision_correct"] for row in group),
            "constant_reference_guess": all(value is not None for value in predicted) and len(set(predicted)) == 1})
    order_results = []
    for key, group in order_groups.items():
        if len(group) != 2 or {row["order"] for row in group} != {0, 1}:
            raise ValueError("Every order control must retain both planned variants")
        agreement = all(row["schema_valid"] for row in group) and group[0]["prediction"] == group[1]["prediction"]
        order_results.append({"mode": key[0], "family_id": key[1], "language": key[2], "context_id": key[3],
            "members": [row["request_id"] for row in group], "prediction_agreement": bool(agreement),
            "both_decisions_correct": all(row["decision_correct"] for row in group)})

    def group_stats(groups, fields):
        return {"planned_groups": len(groups), **{field + "_count": sum(group[field] for group in groups) for field in fields}}

    return {"protocol": PROTOCOL, "statistics": decision_statistics(rows),
        "by_mode": {mode: decision_statistics([row for row in rows if row["mode"] == mode]) for mode in sorted({row["mode"] for row in rows})},
        "by_category": {category: decision_statistics([row for row in rows if row["decision_category"] == category])
                        for category in sorted({row["decision_category"] for row in rows})},
        "by_split": {split: decision_statistics([row for row in rows if row["split"] == split]) for split in sorted({row["split"] for row in rows})},
        "by_split_mode": {split: {mode: decision_statistics([row for row in rows if row["split"] == split and row["mode"] == mode])
            for mode in sorted({row["mode"] for row in rows if row["split"] == split})} for split in sorted({row["split"] for row in rows})},
        "by_mode_category": {mode: {category: decision_statistics([row for row in rows if row["mode"] == mode and row["decision_category"] == category])
            for category in sorted({row["decision_category"] for row in rows if row["mode"] == mode})} for mode in sorted({row["mode"] for row in rows})},
        "first_candidate_counts": dict(sorted(first_positions.items())),
        "context_groups_by_mode": {mode: group_stats([group for group in context_results if group["mode"] == mode],
            ("all_decisions_correct", "constant_reference_guess")) for mode in sorted({group["mode"] for group in context_results})},
        "order_groups_by_mode": {mode: group_stats([group for group in order_results if group["mode"] == mode],
            ("prediction_agreement", "both_decisions_correct")) for mode in sorted({group["mode"] for group in order_results})},
        "context_groups": context_results, "order_groups": order_results,
        "limitations": ["Synthetic controlled test, not navigation or unseen-task generalization",
            "Two order variants per family, not all possible permutations",
            "Task-reference matches under missing context can be lucky guesses",
            "Prior state comes from the fixed public-language rule builder",
            "Retention of untouched fields remains a shared executor guarantee"]}
