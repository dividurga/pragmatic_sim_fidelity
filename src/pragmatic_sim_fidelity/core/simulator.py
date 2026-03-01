from __future__ import annotations
from typing import Protocol, List
from .types import State, Action, Trajectory
from .task import TaskSpec

class Simulator(Protocol):
    name: str

    def reset(self, state: State) -> None:
        ...

    def get_state(self) -> State:
        ...

    def step(self, action: Action, task: TaskSpec) -> State:
        ...

    def rollout(self, state0: State, actions: List[Action], task: TaskSpec) -> Trajectory:
        ...