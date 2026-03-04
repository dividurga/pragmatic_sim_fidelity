"""Planner interface and implementations."""

from __future__ import annotations

from typing import List, Protocol

from .simulator import Simulator
from .task import TaskSpec
from .types import Action, State


class Planner(Protocol):
    """Planner interface."""

    name: str

    def plan(self, state: State, task: TaskSpec, sim: Simulator, rng) -> List[Action]:
        """Plan a sequence of actions to solve a task from a given state and return the
        plan as a list of actions."""
        raise NotImplementedError
