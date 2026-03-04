"""Core data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

State = Dict[str, Any]


@dataclass(frozen=True)
class Action:
    """Action data type."""

    kind: str
    params: Dict[str, Any]


@dataclass
class Trajectory:
    """Trajectory data type."""

    states: List[State]
    infos: List[Dict[str, Any]]
    terminated: bool
