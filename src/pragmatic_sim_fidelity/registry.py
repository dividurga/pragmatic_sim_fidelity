from __future__ import annotations
from typing import Callable, Dict
from .core.task import TaskSpec
from .core.simulator import Simulator
from .core.planner import Planner

TASKS: Dict[str, Callable[[], TaskSpec]] = {}
SIMS: Dict[str, Callable[[], Simulator]] = {}
PLANNERS: Dict[str, Callable[[], Planner]] = {}

def register_task(name: str):
    def deco(factory: Callable[[], TaskSpec]):
        TASKS[name] = factory
        return factory
    return deco

def register_sim(name: str):
    def deco(factory: Callable[[], Simulator]):
        SIMS[name] = factory
        return factory
    return deco

def register_planner(name: str):
    def deco(factory: Callable[[], Planner]):
        PLANNERS[name] = factory
        return factory
    return deco