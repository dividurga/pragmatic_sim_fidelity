"""A very simple random shooting planner that samples random action sequences and picks
the best one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from ..core.simulator import Simulator
from ..core.task import TaskSpec
from ..core.types import Action, State
from ..registry import register_planner

ActionSampler = Callable[[object, TaskSpec, State, int], List[Action]]


@dataclass
class RandomShooting:
    """Random shooting planner that samples random action sequences and picks the best
    one."""

    name: str = "random_shooting"
    horizon: int = 20
    num_samples: int = 512
    sampler: ActionSampler | None = None

    def plan(self, state: State, task: TaskSpec, sim: Simulator, rng) -> List[Action]:
        """Randomly shoot a number of action sequences and return the best one according
        to the task's score function."""
        if self.sampler is None:
            raise ValueError("RandomShooting requires sampler=...")

        best_cost = float("inf")
        best_seq: List[Action] = []

        for _ in range(self.num_samples):
            seq = self.sampler(rng, task, state, self.horizon)
            traj = sim.rollout(state, seq, task)
            cost = task.score(traj)
            if cost < best_cost:
                best_cost = cost
                best_seq = seq

        return best_seq


@register_planner("random_shooting")
def _make_planner():
    return RandomShooting()
