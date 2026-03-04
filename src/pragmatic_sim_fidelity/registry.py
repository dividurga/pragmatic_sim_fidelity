"""Registry for tasks, sims, and planners."""

from __future__ import annotations

from typing import Callable, Dict

from .core.planner import Planner
from .core.simulator import Simulator
from .core.task import TaskSpec

TASKS: Dict[str, Callable[[], TaskSpec]] = {}
SIMS: Dict[str, Callable[[], Simulator]] = {}
PLANNERS: Dict[str, Callable[[], Planner]] = {}


def register_task(name: str):
    """Register a task specification with a given name."""

    def deco(factory: Callable[[], TaskSpec]):
        TASKS[name] = factory
        return factory

    return deco


def register_sim(name: str):
    """Register a simulator with a given name."""

    def deco(factory: Callable[[], Simulator]):
        SIMS[name] = factory
        return factory

    return deco


def register_planner(name: str):
    """Register a planner with a given name."""

    def deco(factory: Callable[[], Planner]):
        PLANNERS[name] = factory
        return factory

    return deco
