"""Minimal Simulator interface."""

from __future__ import annotations

from typing import List, Protocol

from .task import TaskSpec
from .types import Action, State, Trajectory


class Simulator(Protocol):
    """Simulator interface."""

    name: str

    def reset(self, state: State) -> None:
        """Reset the simulator to a given state."""
        raise NotImplementedError

    def get_state(self) -> State:
        """Get the current state of the simulator."""
        raise NotImplementedError

    def step(self, action: Action, task: TaskSpec) -> State:
        """Step the simulator with a given action and return the next state."""
        raise NotImplementedError

    def rollout(
        self, state0: State, actions: List[Action], task: TaskSpec
    ) -> Trajectory:
        """Rollout a trajectory from a given initial state and sequence of actions and
        return the trajectory as a list of states."""
        raise NotImplementedError
