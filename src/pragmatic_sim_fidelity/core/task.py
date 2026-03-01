from __future__ import annotations
from typing import Protocol
from .types import State, Trajectory

class TaskSpec(Protocol):
    name: str

    def reset(self, rng) -> State:
        ...

    def is_success(self, state: State) -> bool:
        ...

    def score(self, traj: Trajectory) -> float:
        """Lower is better."""
        ...