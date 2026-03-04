"""Sampler for delta_q action sequences for Franka joint-space planning."""

from __future__ import annotations

from typing import List

from ..core.types import Action


def sample_delta_q_sequence_gaussian(
    rng,
    horizon: int,
    sigma: float = 0.3,
) -> List[Action]:
    """
    Baseline sampler for Franka joint-space planning:
      dq ~ N(0, sigma^2 I)
    Returns horizon actions of kind='delta_q' with params={'dq': (7,)}.
    """
    seq: List[Action] = []
    for _ in range(horizon):
        dq = rng.normal(0.02, sigma, size=(7,))
        seq.append(Action(kind="delta_q", params={"dq": dq}))
    return seq
