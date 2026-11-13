"""Explicitly synthetic navigation backend and privileged regression policies.

The obstacle-free plane, images and visibility checks are fixtures, not HSSD
rendering or evidence about a real instance. Policies intentionally receive
private geometry/update access through their setup; they are not V baselines.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math


CONDITIONS = ("follow", "ignore", "delayed", "wrong", "invalid")
SCOPE = "synthetic_free_plane_v1"


def fixture_case(source_case):
    """Bind approved text/labels to a separate, disclosed synthetic geometry.

    The reference is eight 0.25 m forward actions followed by STOP. Cumulative
    travel fractions .25/.50/.75 produce external pre-action steps 2/4/6. Source
    HSSD trigger/geometry remain in the preparation and are never called executed.
    """
    config = deepcopy(source_case)
    root = config["initial_goal"]
    pair = next(goal for goal in config["goal_positions"] if goal != root)
    config.update(execution_scope=SCOPE, max_steps=64, start_position=[0., 0., 0.],
        start_rotation_xyzw=[0., 0., 0., 1.], forward_step_m=.25,
        turn_angle_degrees=90., success_radius_m=.13, yaw_tolerance_degrees=1.,
        goal_positions={root: [0., 0., -2.], pair: [1., 0., -1.]},
        trigger_protocol="synthetic_cumulative_reference_travel_external_step_v1")
    config["events"][0]["trigger"]["value"] = {"early": 2, "middle": 4, "late": 6}[config["timing"]]
    return config


class FreePlaneBackend:
    """Metre/degree kinematics with a -Z forward camera and no obstacles.

    Positive yaw turns left about +Y. Synthetic target centres sit 0.5 m beyond
    the stopping point along -Z; all terminal headings consequently face -Z.
    RGB is a deterministic pose-dependent test pattern, with no task labels.
    """
    camera = {"width": 8, "height": 8, "fx": 4., "fy": 4., "cx": 3.5, "cy": 3.5}

    def __init__(self, config):
        if config.get("execution_scope") != SCOPE:
            raise ValueError("The free-plane backend only accepts explicit synthetic cases")
        self.config = deepcopy(config)
        self.position = config["start_position"][:]
        rotation = config["start_rotation_xyzw"]
        self.yaw = math.degrees(2*math.atan2(rotation[1], rotation[3]))

    def pose(self):
        angle = math.radians(self.yaw)/2
        return {"position": self.position[:], "rotation_xyzw": [0., math.sin(angle), 0., math.cos(angle)]}

    def act(self, action):
        if action == "move_forward":
            angle, distance = math.radians(self.yaw), self.config["forward_step_m"]
            self.position[0] -= math.sin(angle)*distance
            self.position[2] -= math.cos(angle)*distance
        elif action in ("turn_left", "turn_right"):
            self.yaw += self.config["turn_angle_degrees"] * (1 if action == "turn_left" else -1)
        elif action not in ("STOP", "COMPLETE_SUBGOAL"):
            raise ValueError("Unknown physical fixture action")
        return False

    def yaw_to(self, point):
        dx, dz = point[0]-self.position[0], point[2]-self.position[2]
        desired = math.degrees(math.atan2(-dx, -dz))
        return (desired-self.yaw+180) % 360-180

    def measurements(self):
        result = {}
        for goal, point in self.config["goal_positions"].items():
            centre = [point[0], point[1], point[2]-.5]
            yaw = self.yaw_to(centre)
            result[goal] = {"distance_m": math.dist(self.position, point), "yaw_error_degrees": yaw,
                "diagnostic": {"evaluated": True, "instance_visible": abs(yaw) <= 45,
                    "labels_restored": True, "pose_unchanged": True, "scope": SCOPE}}
        return result

    def rgb(self):
        # Quantized continuous pose is used only to make changed observations
        # detectable in replay. Pixels do not encode labels or evaluator state.
        colour = [int(round((value+8)*16)) % 256 for value in self.position]
        colour[1] = int(round(self.yaw)) % 256
        return bytes(colour)*64


def privileged_policy(config, backend, condition):
    """Controlled loop test with private setup, never language/image understanding.

    A public message's arrival selects when to apply the privately supplied label.
    The policy's goal/completion memory is separate from evaluator truth. The
    delayed variant turns once before responding; that is legal for reaching
    tasks but violates immediate cancellation. Wrong sends a parseable yet
    semantically invalid completion; invalid sends malformed action content.
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown synthetic policy condition")
    state = {"goal": config["initial_goal"], "updated": False, "completed": False, "perturbed": False}

    def callback(request, assets):
        message = request["current_intervention"] or (request["message_history"][0] if request["message_history"] else None)
        if message and condition != "ignore":
            if not state["perturbed"] and condition in ("delayed", "wrong", "invalid"):
                state["perturbed"] = True
                if condition == "invalid":
                    return '{"action":"STOP","extra":true}'
                return json.dumps({"action": "turn_left" if condition == "delayed" else "COMPLETE_SUBGOAL"})
            if config["task_type"] == "cancel_task":
                return '{"action":"STOP"}'
            if not state["updated"]:
                update = config["events"][0]["task_update"]
                state["goal"] = update.get("new_goal", update.get("subgoal"))
                state["updated"] = True
        goal = state["goal"]
        point = config["goal_positions"][goal]
        close = math.dist(backend.position, point) <= config["success_radius_m"]
        # Cardinal-axis waypoints match the fixture's 90-degree turns. A direct
        # diagonal bearing would oscillate forever between two discrete headings.
        waypoint = ([point[0], point[1], point[2]-.5] if close else
                    [point[0], point[1], backend.position[2]] if abs(point[0]-backend.position[0]) > config["success_radius_m"]
                    else point)
        yaw = backend.yaw_to(waypoint)
        if abs(yaw) > config["yaw_tolerance_degrees"]:
            return json.dumps({"action": "turn_left" if yaw > 0 else "turn_right"})
        if not close:
            return '{"action":"move_forward"}'
        if config["task_type"] == "subgoal" and state["updated"] and not state["completed"]:
            state.update(goal=config["initial_goal"], completed=True)
            return '{"action":"COMPLETE_SUBGOAL"}'
        return '{"action":"STOP"}'
    return callback
