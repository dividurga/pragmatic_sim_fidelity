from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.types import State, Action, Trajectory
from ...registry import register_sim


@dataclass
class Geom3DConfig:
    micro_steps: int = 6  # prevents "tunneling" in joint-space


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

    def _extract_dq(self, action: Action) -> np.ndarray:
        if "dq" in action.params:
            dq = np.asarray(action.params["dq"], dtype=float).reshape(-1)
            if dq.shape != (7,):
                raise ValueError(f"Expected dq shape (7,), got {dq.shape}")
            return dq

        keys = [f"dq{i}" for i in range(7)]
        if all(k in action.params for k in keys):
            return np.array([float(action.params[k]) for k in keys], dtype=float)

        raise ValueError("delta_q action missing params: provide 'dq' (len 7) or dq0..dq6")

    def step(self, action: Action, task) -> State:
        """
        Expects task.step(state, dq_inc)->(next_state, info)
        where dq_inc is a small 7D joint increment direction/command.
        """
        assert self._state is not None

        if action.kind != "delta_q":
            return self.get_state()

        dq = self._extract_dq(action)

        # Store step cost for the *overall* action (not per micro step)
        alpha = float(self._state.get("w_step", 0.01))
        self._state["last_step_cost"] = alpha * float(dq @ dq)

        # Incremental micro-steps
        dq_inc = dq / float(self.cfg.micro_steps)

        s = self.get_state()
        collided = False
        for _ in range(self.cfg.micro_steps):
            s, info = task.step(s, dq_inc)
            if bool(info.get("collided", False)) or bool(s.get("collided", False)):
                collided = True
                break

        self._state = s
        self._state["collided"] = bool(collided)
        return self.get_state()

    def rollout(self, state0: State, actions: List[Action], task) -> Trajectory:
        self.reset(state0)
        states = [self.get_state()]
        infos = []
        terminated = False

        for a in actions:
            s = self.step(a, task)
            infos.append(
                {
                    "collided": bool(s.get("collided", False)),
                    "step_cost": float(s.get("last_step_cost", 0.0)),
                }
            )
            states.append(s)
            if s.get("collided", False):
                terminated = True
                break

        return Trajectory(states=states, infos=infos, terminated=terminated)


@register_sim("geom3d")
def _make_sim():
    return Geom3DSimulator()