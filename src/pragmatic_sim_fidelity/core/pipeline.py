"""Pipeline for running episodes of a task with a planner and simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .planner import Planner
from .simulator import Simulator
from .task import TaskSpec
from .types import Action, State


@dataclass
class EpisodeResult:
    """Result of running an episode."""

    states: List[State]
    actions: List[Action]
    success: bool
    steps: int


def _fmt(v):
    """Format a vector as a numpy array of floats."""
    return np.array(v, dtype=np.float64)


def run_episode(
    task: TaskSpec,
    planner: Planner,
    plan_sim: Simulator,
    exec_sim: Simulator,
    rng,
    max_steps: int = 300,
    mpc_execute_k: int = 1,
) -> EpisodeResult:
    """Run an episode of a task with a planner and simulator."""
    s0 = task.reset(rng)
    exec_sim.reset(s0)

    states: List[State] = [exec_sim.get_state()]
    actions_taken: List[Action] = []

    for t in range(max_steps):
        s = exec_sim.get_state()
        if task.is_success(s):
            return EpisodeResult(states, actions_taken, True, t)

        plan = planner.plan(s, task, plan_sim, rng)
        if not plan:
            return EpisodeResult(states, actions_taken, False, t)
        s = exec_sim.get_state()
        ee = _fmt(s["ee_pos"])
        goal = _fmt(s["goal_pos"])
        dist = float(np.linalg.norm(ee - goal))
        coll = bool(s.get("collided", False))

        print(f"[t={t:02d}] dist={dist:.3f} ee={ee.round(3)} \
                goal={goal.round(3)} collided={coll}")
        for a in plan[:mpc_execute_k]:
            exec_sim.step(a, task)
            actions_taken.append(a)
            states.append(exec_sim.get_state())
            if task.is_success(states[-1]):
                return EpisodeResult(states, actions_taken, True, t + 1)

    return EpisodeResult(states, actions_taken, task.is_success(states[-1]), max_steps)
