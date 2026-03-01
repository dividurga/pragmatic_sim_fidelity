from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.types import State, Action, Trajectory
from ...registry import register_sim
from .collision import sphere_aabb_collides

@dataclass
class Geom3DConfig:
    micro_steps: int = 6  # prevents "tunneling" through thin obstacles

class Geom3DSimulator:
    name = "geom3d"

    def __init__(self, cfg: Geom3DConfig | None = None):
        self.cfg = cfg or Geom3DConfig()
        self._state: State | None = None

    def reset(self, state: State) -> None:
        self._state = copy.deepcopy(state)

    def get_state(self) -> State:
        assert self._state is not None
        return copy.deepcopy(self._state)

    def _collides(self, pos: np.ndarray) -> bool:
        assert self._state is not None
        r = float(self._state["ee_radius"])
        for aabb in self._state["obstacles"]:
            aabb = np.asarray(aabb, dtype=float)
            if sphere_aabb_collides(pos, r, aabb):
                return True
        return False

    def step(self, action: Action, task) -> State:
        assert self._state is not None
        if action.kind != "move_ee":
            # ignore other actions in this cheap sim
            return self.get_state()

        start = np.asarray(self._state["ee_pos"], dtype=float)
        delta = np.array([action.params["dx"], action.params["dy"], action.params["dz"]], dtype=float)
        bounds = np.asarray(self._state["bounds"], dtype=float)  # [xmin,ymin,zmin,xmax,ymax,zmax]
        alpha = float(self._state.get("w_step", 0.01))
        step_cost = alpha * float(delta @ delta)
        self._state["last_step_cost"] = step_cost
        collided = False
        pos = start.copy()
        for i in range(self.cfg.micro_steps):
            pos = start + (i + 1) / self.cfg.micro_steps * delta
            pos = np.minimum(np.maximum(pos, bounds[:3]), bounds[3:])
            if self._collides(pos):
                collided = True
                break

        self._state["ee_pos"] = pos
        self._state["collided"] = bool(collided)
        return self.get_state()

    def rollout(self, state0: State, actions: List[Action], task) -> Trajectory:
        self.reset(state0)
        states = [self.get_state()]
        infos = []
        terminated = False

        for a in actions:
            s = self.step(a, task)
            infos.append({
    "collided": bool(s.get("collided", False)),
    "step_cost": float(s.get("last_step_cost", 0.0)),
})
            states.append(s)
            if s.get("collided", False):
                terminated = True
                break

        return Trajectory(states=states, infos=infos, terminated=terminated)

@register_sim("geom3d")
def _make_sim():
    return Geom3DSimulator()