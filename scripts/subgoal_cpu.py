"""Native no-sensor geometry rehearsal for a prepared B-then-A task.

Uses SceneGraph, Agent and PathFinder only: no Simulator, renderer, sensors or
physics world. A passing rehearsal does NOT establish target visibility or
replace GPU acceptance. It helps catch action-budget and trigger problems before
the user runs the existing renderer-backed experiment.
"""

from __future__ import annotations

import math
from pathlib import Path

from scripts import subgoal
from scripts.intervention_timing import load_json


def check_case(config_path):
    """Rehearse the complete counted action sequence without resetting pose."""
    import habitat_sim
    import numpy as np
    from habitat_sim.utils.common import quat_from_coeffs, quat_to_coeffs
    from scripts.terminal_view import yaw_action

    prepared = subgoal.validate_preparation(config_path)
    config = load_json(config_path)
    reference = load_json(Path(prepared["reference_dir"])/"trajectory.json")
    pathfinder = subgoal.pathfinder_for(config)
    graph = habitat_sim.SceneGraph()
    specification = habitat_sim.agent.AgentConfiguration()
    specification.sensor_specifications = []
    specification.height, specification.radius = config["agent_height"], config["agent_radius"]
    specification.action_space = {name: habitat_sim.agent.ActionSpec(name,
        habitat_sim.agent.ActuationSpec(amount=amount)) for name, amount in (
            ("move_forward", config["forward_step_m"]), ("turn_left", config["turn_angle_degrees"]),
            ("turn_right", config["turn_angle_degrees"]))}
    agent = habitat_sim.Agent(graph.get_root_node().create_child(), specification,
        controls=habitat_sim.agent.ObjectControls(move_filter_fn=pathfinder.try_step))
    state = habitat_sim.AgentState()
    state.position = np.asarray(config["start_position"], dtype=np.float32)
    state.rotation = quat_from_coeffs(np.asarray(config["start_rotation_xyzw"]))
    agent.set_state(state)
    root, temporary = prepared["root_goal"], prepared["subgoal"]
    active, inserted, stages, turns, counted = root, False, [], 0, 0

    def follower():
        return habitat_sim.nav.GreedyGeodesicFollower(pathfinder, agent,
            goal_radius=config["success_radius_m"], stop_key="STOP")

    navigator = follower()
    report = {"geometry_passed": False, "simulator_created": False, "visibility_evaluated": False,
              "scope": "Native geometry and heading only; GPU instance visibility remains required",
              "stages": stages, "error": None}
    try:
        for step in range(config["max_steps"]):
            if step <= prepared["trigger_step"]:
                pose = agent.get_state()
                expected = reference[step]
                if (not np.allclose(pose.position, expected["position"], atol=1e-5, rtol=0)
                        or not np.allclose(quat_to_coeffs(pose.rotation), expected["rotation_xyzw"], atol=1e-5, rtol=0)):
                    raise ValueError("CPU execution differs from source before intervention")
            if step == prepared["trigger_step"]:
                active, inserted, turns = temporary, True, 0
                navigator = follower()
            goal = np.asarray(config["goal_positions"][active], dtype=np.float32)
            action = navigator.next_action_along(goal)
            yaw = None
            if action == "STOP":
                action, yaw = yaw_action(agent.get_state(), prepared["target_centers"][active],
                    config["terminal_alignment"]["yaw_tolerance_degrees"])
                if action != "STOP":
                    turns += 1
                    if turns > config["terminal_alignment"]["max_turn_actions"]:
                        raise ValueError("Terminal alignment exceeded per-goal budget")
            counted = step+1
            if action == "STOP":
                distance = subgoal.distance_between(pathfinder, agent.get_state().position, goal)
                if (distance is None or not 0 <= distance <= config["success_radius_m"]
                        or not math.isfinite(yaw)
                        or abs(yaw) > config["terminal_alignment"]["yaw_tolerance_degrees"]+1e-5):
                    raise ValueError("Stage completion geometry failed")
                stages.append({"goal_id": active, "step": step, "distance_m": distance, "yaw_error_degrees": yaw})
                if not inserted:
                    raise ValueError("Initial goal stopped before insertion")
                if active == temporary:
                    # This iteration is the counted nonmoving completion action.
                    # Rebuild only the follower, keeping the same Agent and pose.
                    active, turns = root, 0
                    navigator = follower()
                    continue
                report["geometry_passed"] = [stage["goal_id"] for stage in stages] == [temporary, root]
                break
            agent.act(action)
            if not pathfinder.is_navigable(agent.get_state().position):
                raise ValueError("Native action left the frozen navigation mesh")
        if not report["geometry_passed"]:
            raise ValueError("Budget exhausted without completing B then stopping at A")
    except (ValueError, RuntimeError, habitat_sim.errors.GreedyFollowerError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        agent.close()
    report["total_counted_steps"] = counted
    return report
