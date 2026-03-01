from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from ..core.types import State, Trajectory
from ..registry import register_task

AABB = Tuple[float, float, float, float, float, float]  # xmin,ymin,zmin,xmax,ymax,zmax

@dataclass
class FridgeReachConfig:
    ee_radius: float = 0.035
    goal_tol: float = 0.05
    step_size: float = 0.05
    w_collide: float = 1000.0

class FridgeReachEE3D:
    name = "fridge_reach_ee_3d"

    def __init__(self, cfg: FridgeReachConfig | None = None):
        self.cfg = cfg or FridgeReachConfig()

    def reset(self, rng) -> State:
        # Workspace inside fridge (meters): width x depth x height
        bounds: AABB = (0.0, 0.0, 0.0, 0.8, 0.6, 0.8)
        
        # Obstacles: a shelf slab + a couple clutter boxes
        obstacles: List[AABB] = [
            # shelf slab (thin in z)
            (0.02, 0.02, 0.35, 0.78, 0.58, 0.38),
            # bottle/carton
            (0.35, 0.22, 0.38, 0.43, 0.30, 0.65),
            # side bin wall protrusion
            (0.68, 0.00, 0.00, 0.80, 0.10, 0.80),
        ]

        # Start near opening (front is y ~ 0.6), goal deeper in
        ee_pos = np.array([0.10, 0.55, 0.55], dtype=float)
        goal_pos = np.array([0.55, 0.25, 0.55], dtype=float)

        return {
            "ee_pos": ee_pos,
            "goal_pos": goal_pos,
            "ee_radius": float(self.cfg.ee_radius),
            "obstacles": obstacles,
            "bounds": bounds,
            "collided": False,
            "step_size": float(self.cfg.step_size),
            "w_step": 0.01,
        }

    def is_success(self, state: State) -> bool:
        if state.get("collided", False):
            return False
        d = float(np.linalg.norm(state["ee_pos"] - state["goal_pos"]))
        return d < self.cfg.goal_tol

    def score(self, traj: Trajectory) -> float:
        final = traj.states[-1]
        dist = float(np.linalg.norm(final["ee_pos"] - final["goal_pos"]))
        collided = any(info.get("collided", False) for info in traj.infos)
        step_cost = 0.0
        for info in traj.infos:
            step_cost += float(info.get("step_cost", 0.0))
        return dist + (self.cfg.w_collide if collided else 0.0) + step_cost

@register_task("fridge_reach_ee_3d")
def _make_task():
    return FridgeReachEE3D()