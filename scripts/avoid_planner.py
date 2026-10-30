"""Small deterministic Oracle roadmap for one floor-space constraint.

Native navigation queries are CPU-only. Nothing here creates a Simulator or
changes the original mesh. Edges are Habitat geodesic polylines, rejected when
they intersect the forbidden footprint. This finite roadmap is deliberately
incomplete: no route means preparation/planning failure, not global infeasibility.
"""

from __future__ import annotations

import heapq
import math

from mivln.regions import segment_intersects_region, validate_region, violating_regions


class AvoidPlanningError(RuntimeError):
    """A safe route/action could not be produced; never fall back through a region."""


def validate_settings(config):
    settings = config.get("avoid_planner", {})
    if not isinstance(settings, dict):
        raise ValueError("avoid_planner must be an object")
    for key, default in (("clearance_m", .35), ("waypoint_radius_m", .2)):
        value = settings.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"avoid_planner.{key} must be finite and positive")
    # Following may stop short of an intermediate waypoint. The extra margin
    # leaves room for that tolerance and the quantized 0.25 m action space.
    if settings.get("clearance_m", .35) < settings.get("waypoint_radius_m", .2) + .1:
        raise ValueError("Avoid clearance must exceed waypoint radius by at least 0.1 m")


def plan_route(pathfinder, start, goal, region, radius_m=.1, clearance_m=.35):
    """Find the shortest legal route in a bounded, deterministic guard roadmap."""
    import habitat_sim
    import numpy as np

    validate_region(region)
    for point in (start, goal):
        if not pathfinder.is_navigable(np.asarray(point, dtype=np.float32)):
            raise AvoidPlanningError("Route endpoint is not navigable")
        if segment_intersects_region(point, point, region, radius_m):
            raise AvoidPlanningError("Route endpoint is in the inflated avoid region")
    nodes = [np.asarray(start, dtype=np.float32), np.asarray(goal, dtype=np.float32)]
    low, high = region["bounds_xz"]
    # Corners and side midpoints at three offsets handle common furniture
    # obstructions without claiming the finite roadmap is a complete planner.
    for offset in (radius_m + clearance_m + .02, radius_m + clearance_m + .4,
                   radius_m + clearance_m + .8):
        xs = (low[0] - offset, (low[0] + high[0]) / 2, high[0] + offset)
        zs = (low[1] - offset, (low[1] + high[1]) / 2, high[1] + offset)
        for i, x in enumerate(xs):
            for j, z in enumerate(zs):
                if i == j == 1:
                    continue
                requested = np.array([x, region["floor_y"], z], dtype=np.float32)
                point = pathfinder.snap_point(requested)
                if (not np.isfinite(point).all() or np.linalg.norm(point - requested) > .2
                        or not pathfinder.is_navigable(point)
                        or segment_intersects_region(point, point, region, radius_m + clearance_m)
                        or any(np.linalg.norm(point - old) < .05 for old in nodes)):
                    continue
                nodes.append(np.asarray(point, dtype=np.float32))
    adjacency = [[] for _ in nodes]
    edges = {}
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = a, nodes[j]
            if not pathfinder.find_path(path) or not math.isfinite(path.geodesic_distance):
                continue
            points = [np.asarray(point).tolist() for point in path.points]
            # Keep the entire geodesic edge away from the true boundary, not
            # just its endpoints. Waypoint tolerance can cut inside a corner.
            if len(points) < 2 or any(segment_intersects_region(p, q, region, radius_m + clearance_m - .02)
                                       for p, q in zip(points, points[1:])):
                continue
            adjacency[i].append((j, float(path.geodesic_distance)))
            adjacency[j].append((i, float(path.geodesic_distance)))
            edges[i, j], edges[j, i] = points, list(reversed(points))
    distances, parents, queue = {0: 0.0}, {}, [(0.0, 0)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distances[node]:
            continue
        if node == 1:
            chain = [1]
            while chain[-1] != 0:
                chain.append(parents[chain[-1]])
            chain.reverse()
            points = []
            for a, b in zip(chain, chain[1:]):
                points.extend(edges[a, b] if not points else edges[a, b][1:])
            return {"method": "geodesic_guard_roadmap_v1", "distance_m": cost,
                    "points": points, "waypoints": [nodes[index].tolist() for index in chain[1:]],
                    "node_count": len(nodes), "region_id": region["region_id"],
                    "clearance_m": clearance_m, "agent_radius_m": radius_m}
        for neighbor, weight in adjacency[node]:
            updated = cost + weight
            if updated < distances.get(neighbor, float("inf")):
                distances[neighbor], parents[neighbor] = updated, node
                heapq.heappush(queue, (updated, neighbor))
    raise AvoidPlanningError("No safe route in the finite guard roadmap")


class AvoidFollower:
    """Follow legal guard waypoints using ordinary Habitat movement actions.

    Reaching a guard waypoint advances the controller, not task state, and does
    not emit a STOP action. Before any forward action, check Habitat's predicted
    filtered segment. Actual executed segments are evaluated separately.
    """

    def __init__(self, sim, agent, goal, constraints, config):
        import habitat_sim
        import numpy as np

        if len(constraints) != 1:
            raise AvoidPlanningError("The experimental planner supports exactly one avoid region")
        self.sim, self.agent, self.config, self.constraints = sim, agent, config, constraints
        self.route = plan_route(sim.pathfinder, agent.get_state().position, goal, constraints[0],
                                config.get("agent_radius", .1), config.get("avoid_planner", {}).get("clearance_m", .35))
        self.waypoints = [np.asarray(point, dtype=np.float32) for point in self.route["waypoints"]]
        self.index, self.follower = 0, None

    def next_action_along(self, goal):
        import habitat_sim
        import numpy as np
        from habitat_sim.utils.common import quat_rotate_vector

        while self.index < len(self.waypoints):
            if self.follower is None:
                radius = (self.config["success_radius_m"] if self.index == len(self.waypoints) - 1
                          else self.config.get("avoid_planner", {}).get("waypoint_radius_m", .2))
                self.follower = habitat_sim.nav.GreedyGeodesicFollower(
                    self.sim.pathfinder, self.agent, goal_radius=radius, stop_key="STOP")
            action = self.follower.next_action_along(self.waypoints[self.index])
            if action == "STOP" and self.index < len(self.waypoints) - 1:
                self.index, self.follower = self.index + 1, None
                continue
            if action == "move_forward":
                state = self.agent.get_state()
                desired = state.position + quat_rotate_vector(state.rotation, np.array([0., 0., -self.config["forward_step_m"]]))
                predicted = self.sim.pathfinder.try_step(state.position, desired)
                if (not np.isfinite(predicted).all() or violating_regions(state.position, predicted,
                        self.constraints, self.config.get("agent_radius", .1))):
                    raise AvoidPlanningError("Discrete forward action would intersect avoid region")
            return action
        raise AvoidPlanningError("Guard waypoint controller exhausted without STOP")
