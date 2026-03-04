"""Minimal interface for a task specification."""

from __future__ import annotations

from typing import Protocol

from .types import State, Trajectory


class TaskSpec(Protocol):
    """Task specification interface."""

    name: str

    def reset(self, rng) -> State:
        """Reset the task and return the initial state."""
        raise NotImplementedError

    def is_success(self, state: State) -> bool:
        """Return True if the given state is a success state for the task."""
        raise NotImplementedError

    def score(self, traj: Trajectory) -> float:
        """Return a score for a given trajectory of states for the task."""
        raise NotImplementedError
