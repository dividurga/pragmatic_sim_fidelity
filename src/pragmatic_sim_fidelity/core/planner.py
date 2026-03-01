from __future__ import annotations
from typing import Protocol, List
from .types import State, Action
from .task import TaskSpec
from .simulator import Simulator

class Planner(Protocol):
    name: str
    def plan(self, state: State, task: TaskSpec, sim: Simulator, rng) -> List[Action]:
        ...