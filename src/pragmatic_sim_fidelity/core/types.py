from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

State = Dict[str, Any]

@dataclass(frozen=True)
class Action:
    kind: str
    params: Dict[str, Any]

@dataclass
class Trajectory:
    states: List[State]
    infos: List[Dict[str, Any]]
    terminated: bool